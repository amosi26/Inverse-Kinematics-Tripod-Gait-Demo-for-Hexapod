"""Reusable all-six-leg body pose IK solver helpers."""

import numpy as np

from hexapod_ik.body.body_pose import body_rotation_matrix, foot_world_to_leg_local, is_body_pose_within_limits, neutral_foot_positions_body
from hexapod_ik.config.robot_config import IK_JOINT_LIMITS_DEG, LEG_MOUNT_POSITIONS_BODY
from hexapod_ik.kinematics.leg_ik import solve_ik_to_target


def _angles_within_limits(angles_deg):
    coxa_min, coxa_max = IK_JOINT_LIMITS_DEG["coxa"]
    femur_min, femur_max = IK_JOINT_LIMITS_DEG["femur"]
    tibia_min, tibia_max = IK_JOINT_LIMITS_DEG["tibia"]
    return (
        coxa_min <= angles_deg[0] <= coxa_max
        and femur_min <= angles_deg[1] <= femur_max
        and tibia_min <= angles_deg[2] <= tibia_max
    )


def _build_neutral_world_foot_positions():
    neutral_body_position_world = np.array([0.0, 0.0, 0.0], dtype=float)
    r_neutral = body_rotation_matrix(roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0)

    neutral_foot_body = neutral_foot_positions_body()
    neutral_foot_world = {}
    for leg_name, foot_body in neutral_foot_body.items():
        foot_body = np.array(foot_body, dtype=float)
        neutral_foot_world[leg_name] = neutral_body_position_world + r_neutral @ foot_body
    return neutral_foot_world


def solve_body_pose_ik(
    body_position_world,
    roll_deg=0.0,
    pitch_deg=0.0,
    yaw_deg=0.0,
    max_allowed_error=0.10,
):
    """Solve IK for all six legs for a requested body pose with fixed neutral world feet."""
    body_position_world = np.array(body_position_world, dtype=float)
    roll_deg = float(roll_deg)
    pitch_deg = float(pitch_deg)
    yaw_deg = float(yaw_deg)
    max_allowed_error = float(max_allowed_error)

    leg_names = list(LEG_MOUNT_POSITIONS_BODY.keys())
    if not is_body_pose_within_limits(body_position_world, roll_deg, pitch_deg, yaw_deg):
        return {
            "accepted": False,
            "reason": "outside_body_pose_limits",
            "joint_angles_deg": {},
            "foot_leg_local_targets": {},
            "final_errors": {},
            "failed_legs": leg_names,
            "max_final_error": 0.0,
            "per_leg": {},
        }

    fixed_neutral_foot_world = _build_neutral_world_foot_positions()

    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    joint_angles_deg = {}
    foot_leg_local_targets = {}
    final_errors = {}
    failed_legs = []
    per_leg = {}

    max_final_error = 0.0

    for leg_name, foot_world in fixed_neutral_foot_world.items():
        foot_leg_local = foot_world_to_leg_local(
            leg_name,
            foot_world,
            body_position_world,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
        )

        start_angles = (90.0, 90.0, 90.0)
        final_angles, _angle_history, _ee_history, converged, iterations, final_error = solve_ik_to_target(
            start_angles,
            foot_leg_local,
            alpha=0.02,
            tol=0.05,
            max_iters=6000,
            damping=0.1,
        )

        final_error = float(final_error)
        joint_limits_ok = _angles_within_limits(final_angles)
        error_ok = final_error <= max_allowed_error
        leg_passed = bool(converged and joint_limits_ok and error_ok)

        joint_angles_deg[leg_name] = final_angles
        foot_leg_local_targets[leg_name] = foot_leg_local
        final_errors[leg_name] = final_error
        max_final_error = max(max_final_error, final_error)

        per_leg[leg_name] = {
            "foot_world": foot_world,
            "foot_leg_local": foot_leg_local,
            "joint_angles_deg": final_angles,
            "ik_converged": bool(converged),
            "ik_iterations": int(iterations),
            "ik_final_error": final_error,
            "joint_limits_ok": bool(joint_limits_ok),
            "error_ok": bool(error_ok),
            "passed": leg_passed,
        }

        if not leg_passed:
            failed_legs.append(leg_name)

    accepted = len(failed_legs) == 0
    reason = "solved" if accepted else "ik_infeasible"

    return {
        "accepted": accepted,
        "reason": reason,
        "joint_angles_deg": joint_angles_deg,
        "foot_leg_local_targets": foot_leg_local_targets,
        "final_errors": final_errors,
        "failed_legs": failed_legs,
        "max_final_error": max_final_error,
        "per_leg": per_leg,
    }
