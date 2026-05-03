"""Console demo for constrained leg IK behavior."""

import numpy as np

from hexapod_ik.kinematics.leg_ik import (
    START_T1_DEG,
    START_T2_DEG,
    START_T3_DEG,
    random_reachable_target,
    solve_ik_to_target,
)


def main():
    rng = np.random.default_rng(7)
    num_targets = 5

    t1 = START_T1_DEG
    t2 = START_T2_DEG
    t3 = START_T3_DEG

    print("Running constrained leg IK demo...")
    for idx in range(num_targets):
        target = random_reachable_target(rng)
        final_angles, _angle_history, _ee_history, converged, iterations, final_error = solve_ik_to_target(
            (t1, t2, t3),
            target,
        )
        t1, t2, t3 = final_angles
        print(
            f"Target {idx + 1}/{num_targets}: target={target}, "
            f"converged={converged}, iterations={iterations}, final_error={final_error:.4f}, "
            f"angles={tuple(round(a, 3) for a in final_angles)}"
        )


if __name__ == "__main__":
    main()
