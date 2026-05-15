"""Check neutral stance IK convergence for all six legs."""

import numpy as np

from hexapod_ik.body.body_pose import foot_body_to_leg_local, neutral_foot_positions_body
from hexapod_ik.config.robot_config import IK_JOINT_LIMITS_DEG, NEUTRAL_FOOT_LEG_LOCAL
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


def main():
    neutral_target = np.array(NEUTRAL_FOOT_LEG_LOCAL, dtype=float)
    footprint_body = neutral_foot_positions_body()

    tested = 0
    converged_count = 0
    max_final_error = 0.0
    all_final_angles_in_limits = True
    all_recovered_neutral = True
    converged_legs = []
    failed_legs = []

    for leg_name in footprint_body:
        tested += 1

        foot_body = np.array(footprint_body[leg_name], dtype=float)
        recovered_leg_local = foot_body_to_leg_local(leg_name, foot_body)
        recovered_ok = np.allclose(recovered_leg_local, neutral_target, atol=1e-9)
        all_recovered_neutral = all_recovered_neutral and recovered_ok

        start_angles = (90.0, 90.0, 90.0)
        final_angles, _angle_history, _ee_history, converged, iterations, final_error = solve_ik_to_target(
            start_angles,
            recovered_leg_local,
            alpha=0.02,
            tol=0.05,
            max_iters=6000,
            damping=0.1,
        )

        in_limits = _angles_within_limits(final_angles)
        all_final_angles_in_limits = all_final_angles_in_limits and in_limits

        if converged:
            converged_count += 1
            converged_legs.append(leg_name)
        else:
            failed_legs.append(leg_name)

        max_final_error = max(max_final_error, float(final_error))

        print(f"leg_name: {leg_name}")
        print(f"neutral_foot_body: {foot_body}")
        print(f"recovered_leg_local: {recovered_leg_local}")
        print(f"recovered_matches_neutral: {recovered_ok}")
        print(f"joint_angles_deg: {final_angles}")
        print(f"converged: {converged}")
        print(f"iterations: {iterations}")
        print(f"final_error: {final_error:.6f}")
        print(f"final_angles_within_ik_limits: {in_limits}")
        print("-" * 60)

    print("summary:")
    print(f"legs_tested: {tested}")
    print(f"legs_converged: {converged_count}")
    print(f"converged_legs: {converged_legs}")
    print(f"failed_legs: {failed_legs}")
    print(f"max_final_error: {max_final_error:.6f}")
    print(f"all_recovered_neutral_target: {all_recovered_neutral}")
    print(f"all_final_joint_angles_within_ik_limits: {all_final_angles_in_limits}")


if __name__ == "__main__":
    main()
