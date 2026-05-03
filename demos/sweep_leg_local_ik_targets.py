"""Sweep a small grid of leg-local targets and report IK convergence diagnostics."""

import numpy as np

from hexapod_ik.kinematics.leg_ik import solve_ik_to_target


def main():
    # Match the current one-leg body-pose demo seed.
    start_angles = (90.0, 90.0, 90.0)

    x_values = [3.0, 4.0, 5.0, 6.0]
    y_values = [0.0]
    z_values = [0.0, -1.0, -2.0, -3.0, -4.0]

    rows = []
    for x in x_values:
        for y in y_values:
            for z in z_values:
                target = np.array([x, y, z], dtype=float)
                final_angles, _angle_history, _ee_history, converged, iterations, final_error = solve_ik_to_target(
                    start_angles,
                    target,
                    alpha=0.02,
                    tol=0.05,
                    max_iters=6000,
                    damping=0.1,
                )
                rows.append(
                    {
                        "target": target,
                        "converged": bool(converged),
                        "iterations": int(iterations),
                        "final_error": float(final_error),
                        "angles": tuple(float(a) for a in final_angles),
                    }
                )

    print("target                  converged  iters   final_error   joint_angles_deg")
    print("-" * 84)
    for row in rows:
        target = row["target"]
        print(
            f"[{target[0]:4.1f}, {target[1]:4.1f}, {target[2]:4.1f}]"
            f"    {str(row['converged']):<9}"
            f"  {row['iterations']:5d}"
            f"   {row['final_error']:10.6f}"
            f"   ({row['angles'][0]:8.3f}, {row['angles'][1]:8.3f}, {row['angles'][2]:8.3f})"
        )

    total = len(rows)
    converged_rows = [r for r in rows if r["converged"]]
    best = min(rows, key=lambda r: r["final_error"])

    print("\nSummary")
    print(f"targets_tested: {total}")
    print(f"targets_converged: {len(converged_rows)}")
    print(
        "best_target_by_error: "
        f"{best['target'].tolist()} with final_error={best['final_error']:.6f}, converged={best['converged']}"
    )

    print("converged_targets:")
    if converged_rows:
        for row in converged_rows:
            print(f"  {row['target'].tolist()}")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
