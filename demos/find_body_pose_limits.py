"""Empirical body pose constraint sweep for current IK settings.

These limits are empirical based on the current neutral stance, IK solver,
joint limits, and convergence settings.

They are not final hardware-safe limits.
Hardware testing and CAD collision checks are still required.
"""

import numpy as np

from hexapod_ik.body.body_pose import body_rotation_matrix, foot_world_to_leg_local, neutral_foot_positions_body
from hexapod_ik.config.robot_config import IK_JOINT_LIMITS_DEG
from hexapod_ik.kinematics.leg_ik import solve_ik_to_target

MAX_ALLOWED_IK_ERROR = 0.10


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


def pose_solves_all_legs(body_position_world, roll_deg, pitch_deg, yaw_deg, fixed_neutral_foot_world):
    roll_rad = np.deg2rad(float(roll_deg))
    pitch_rad = np.deg2rad(float(pitch_deg))
    yaw_rad = np.deg2rad(float(yaw_deg))
    body_position_world = np.array(body_position_world, dtype=float)

    per_leg_results = []
    max_final_error = 0.0
    worst_leg = None

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
        error_ok = final_error <= MAX_ALLOWED_IK_ERROR

        if final_error >= max_final_error:
            max_final_error = final_error
            worst_leg = leg_name

        per_leg_results.append(
            {
                "leg_name": leg_name,
                "foot_world": foot_world,
                "foot_leg_local": foot_leg_local,
                "joint_angles_deg": final_angles,
                "ik_converged": bool(converged),
                "ik_iterations": int(iterations),
                "ik_final_error": final_error,
                "joint_limits_ok": bool(joint_limits_ok),
                "error_ok": bool(error_ok),
            }
        )

    passed = all(
        leg_result["ik_converged"] and leg_result["joint_limits_ok"] and leg_result["error_ok"]
        for leg_result in per_leg_results
    )

    return passed, max_final_error, worst_leg, per_leg_results


def _frange_with_zero(start, stop, step):
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    if 0.0 not in values:
        values.insert(0, 0.0)
    return values


def _sweep_direction(name, values, pose_builder, fixed_neutral_foot_world):
    print("=" * 90)
    print(f"Sweep: {name}")

    tested_values = []
    first_failing_value = None
    last_passing_value = 0.0

    for value in values:
        body_position_world, roll_deg, pitch_deg, yaw_deg = pose_builder(value)
        passed, max_final_error, worst_leg, _per_leg_results = pose_solves_all_legs(
            body_position_world,
            roll_deg,
            pitch_deg,
            yaw_deg,
            fixed_neutral_foot_world,
        )

        tested_values.append(value)
        status = "PASS" if passed else "FAIL"
        print(
            f"  value={value:>6.2f} -> {status} "
            f"(max_final_error={max_final_error:.6f}, worst_leg={worst_leg})"
        )

        if passed:
            last_passing_value = value
        else:
            first_failing_value = value
            break

    print(f"  tested_values: {tested_values}")
    print(f"  first_failing_value: {first_failing_value}")
    print(f"  last_converged_value: {last_passing_value}")

    return {
        "name": name,
        "tested_values": tested_values,
        "first_failing_value": first_failing_value,
        "last_converged_value": last_passing_value,
    }


def main():
    fixed_neutral_foot_world = _build_neutral_world_foot_positions()

    z_values = _frange_with_zero(0.0, 1.5, 0.1)
    angle_values = _frange_with_zero(0.0, 20.0, 1.0)

    sweeps = [
        (
            "z_offset_positive",
            z_values,
            lambda v: (np.array([0.0, 0.0, v], dtype=float), 0.0, 0.0, 0.0),
        ),
        (
            "z_offset_negative",
            z_values,
            lambda v: (np.array([0.0, 0.0, -v], dtype=float), 0.0, 0.0, 0.0),
        ),
        (
            "roll_positive_deg",
            angle_values,
            lambda v: (np.array([0.0, 0.0, 0.0], dtype=float), v, 0.0, 0.0),
        ),
        (
            "roll_negative_deg",
            angle_values,
            lambda v: (np.array([0.0, 0.0, 0.0], dtype=float), -v, 0.0, 0.0),
        ),
        (
            "pitch_positive_deg",
            angle_values,
            lambda v: (np.array([0.0, 0.0, 0.0], dtype=float), 0.0, v, 0.0),
        ),
        (
            "pitch_negative_deg",
            angle_values,
            lambda v: (np.array([0.0, 0.0, 0.0], dtype=float), 0.0, -v, 0.0),
        ),
        (
            "yaw_positive_deg",
            angle_values,
            lambda v: (np.array([0.0, 0.0, 0.0], dtype=float), 0.0, 0.0, v),
        ),
        (
            "yaw_negative_deg",
            angle_values,
            lambda v: (np.array([0.0, 0.0, 0.0], dtype=float), 0.0, 0.0, -v),
        ),
    ]

    sweep_results = {}
    for name, values, pose_builder in sweeps:
        sweep_results[name] = _sweep_direction(name, values, pose_builder, fixed_neutral_foot_world)

    body_pose_limits = {
        "z_offset_min": -sweep_results["z_offset_negative"]["last_converged_value"],
        "z_offset_max": sweep_results["z_offset_positive"]["last_converged_value"],
        "roll_deg_min": -sweep_results["roll_negative_deg"]["last_converged_value"],
        "roll_deg_max": sweep_results["roll_positive_deg"]["last_converged_value"],
        "pitch_deg_min": -sweep_results["pitch_negative_deg"]["last_converged_value"],
        "pitch_deg_max": sweep_results["pitch_positive_deg"]["last_converged_value"],
        "yaw_deg_min": -sweep_results["yaw_negative_deg"]["last_converged_value"],
        "yaw_deg_max": sweep_results["yaw_positive_deg"]["last_converged_value"],
    }

    print("=" * 90)
    print("Recommended conservative constraints (empirical diagnostic output):")
    print("BODY_POSE_LIMITS = {")
    print(f"    \"z_offset_min\": {body_pose_limits['z_offset_min']},")
    print(f"    \"z_offset_max\": {body_pose_limits['z_offset_max']},")
    print(f"    \"roll_deg_min\": {body_pose_limits['roll_deg_min']},")
    print(f"    \"roll_deg_max\": {body_pose_limits['roll_deg_max']},")
    print(f"    \"pitch_deg_min\": {body_pose_limits['pitch_deg_min']},")
    print(f"    \"pitch_deg_max\": {body_pose_limits['pitch_deg_max']},")
    print(f"    \"yaw_deg_min\": {body_pose_limits['yaw_deg_min']},")
    print(f"    \"yaw_deg_max\": {body_pose_limits['yaw_deg_max']},")
    print("}")

    print("\nFirst failing values by sweep direction:")
    for name in [
        "z_offset_positive",
        "z_offset_negative",
        "roll_positive_deg",
        "roll_negative_deg",
        "pitch_positive_deg",
        "pitch_negative_deg",
        "yaw_positive_deg",
        "yaw_negative_deg",
    ]:
        result = sweep_results[name]
        print(f"  {name}: {result['first_failing_value']}")


if __name__ == "__main__":
    main()
