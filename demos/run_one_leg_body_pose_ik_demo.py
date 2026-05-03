"""Minimal one-leg body-pose IK demo (RF leg only)."""

import numpy as np

from hexapod_ik.body.body_pose import body_rotation_matrix, foot_world_to_leg_local, rotation_z
from hexapod_ik.config.robot_config import (
    CONTROL_DT_SEC,
    IK_JOINT_LIMITS_DEG,
    LEG_MOUNT_POSITIONS_BODY,
    LEG_MOUNT_YAWS_DEG_BODY,
    MAX_JOINT_SPEED_DEG_PER_SEC,
)
from hexapod_ik.kinematics.leg_ik import solve_ik_to_target


def main():
    leg_name = "RF"

    body_position_world = np.array([0.0, 0.0, 4.0])
    roll_rad = 0.0
    pitch_rad = 0.0
    yaw_rad = 0.0

    # Desired RF target in RF leg-local frame: outward (+X local) and down (-Z local).
    foot_leg_desired = np.array([3.0, 0.0, -4.0])

    # Build a consistent world target via the completed transform chain direction:
    # leg-local -> body -> world.
    leg_mount_body = np.array(LEG_MOUNT_POSITIONS_BODY[leg_name], dtype=float)
    leg_yaw_rad = np.deg2rad(LEG_MOUNT_YAWS_DEG_BODY[leg_name])
    foot_body = leg_mount_body + rotation_z(leg_yaw_rad) @ foot_leg_desired

    r_body = body_rotation_matrix(roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad)
    foot_world = body_position_world + r_body @ foot_body

    foot_leg_local = foot_world_to_leg_local(
        leg_name,
        foot_world,
        body_position_world,
        roll_rad,
        pitch_rad,
        yaw_rad,
    )

    print(f"leg_name: {leg_name}")
    print(
        "body_pose: "
        f"position={body_position_world}, roll_rad={roll_rad}, pitch_rad={pitch_rad}, yaw_rad={yaw_rad}"
    )
    print(f"foot_world: {foot_world}")
    print(f"foot_leg_local: {foot_leg_local}")
    print("ik_limit_mode: internal math-angle limits (IK_JOINT_LIMITS_DEG)")

    # Use an in-range initial seed for this demo so safety checks reflect solver behavior.
    start_angles = (90.0, 90.0, 90.0)
    final_angles, angle_history, _ee_history, converged, iterations, final_error = solve_ik_to_target(
        start_angles,
        foot_leg_local,
        alpha=0.02,
        tol=0.05,
        max_iters=6000,
        damping=0.1,
    )

    print(f"ik_converged: {converged}")
    print(f"ik_iterations: {iterations}")
    print(f"ik_final_error: {final_error:.6f}")
    print(f"joint_angles_deg: {final_angles}")

    history = np.array(angle_history, dtype=float)
    coxa_min, coxa_max = IK_JOINT_LIMITS_DEG["coxa"]
    femur_min, femur_max = IK_JOINT_LIMITS_DEG["femur"]
    tibia_min, tibia_max = IK_JOINT_LIMITS_DEG["tibia"]
    within_limits = (
        np.all((history[:, 0] >= coxa_min) & (history[:, 0] <= coxa_max))
        and np.all((history[:, 1] >= femur_min) & (history[:, 1] <= femur_max))
        and np.all((history[:, 2] >= tibia_min) & (history[:, 2] <= tibia_max))
    )

    max_step_deg = MAX_JOINT_SPEED_DEG_PER_SEC * CONTROL_DT_SEC
    max_observed_step_deg = 0.0
    step_limit_ok = True
    if len(history) > 1:
        deltas = np.abs(np.diff(history, axis=0))
        max_observed_step_deg = float(np.max(deltas))
        step_limit_ok = max_observed_step_deg <= (max_step_deg + 1e-9)

    print(f"joint_limits_ok: {within_limits}")
    print(
        f"max_step_limit_ok: {step_limit_ok} "
        f"(max_observed_step_deg={max_observed_step_deg:.6f}, configured_max_step_deg={max_step_deg:.6f})"
    )


if __name__ == "__main__":
    main()
