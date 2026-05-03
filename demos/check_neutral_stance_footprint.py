"""Check neutral stance footprint construction in body frame."""

import numpy as np

from hexapod_ik.body.body_pose import foot_body_to_leg_local, neutral_foot_positions_body
from hexapod_ik.config.robot_config import NEUTRAL_FOOT_LEG_LOCAL


def main():
    expected = np.array(NEUTRAL_FOOT_LEG_LOCAL, dtype=float)
    footprint = neutral_foot_positions_body()

    print("leg_name | neutral_foot_body                 | recovered_foot_leg_local")
    print("-" * 78)
    for leg_name, foot_body in footprint.items():
        recovered = foot_body_to_leg_local(leg_name, foot_body)
        print(f"{leg_name:>7} | {np.array(foot_body)} | {recovered}")
        assert np.allclose(recovered, expected, atol=1e-6), (
            f"Neutral stance round-trip failed for {leg_name}: {recovered}"
        )

    print("All neutral stance footprint checks passed.")


if __name__ == "__main__":
    main()
