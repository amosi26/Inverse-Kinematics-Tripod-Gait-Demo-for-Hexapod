"""Quick verification checks for world-frame to leg-local transform wrapper."""

import numpy as np

from hexapod_ik.body.body_pose import body_rotation_matrix, foot_world_to_leg_local, rotation_z
from hexapod_ik.config.robot_config import LEG_MOUNT_POSITIONS_BODY, LEG_MOUNT_YAWS_DEG_BODY


def check_world_to_leg_roundtrip():
    body_position_world = np.array([5.0, -2.0, 1.0])
    pitch_rad = np.deg2rad(10.0)
    roll_rad = np.deg2rad(5.0)
    yaw_rad = np.deg2rad(30.0)

    r_body = body_rotation_matrix(roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad)
    foot_leg_expected = np.array([2.0, 0.0, -3.0])

    print("Check: world -> body -> leg wrapper round trip")
    for leg_name, leg_mount_position in LEG_MOUNT_POSITIONS_BODY.items():
        leg_yaw_rad = np.deg2rad(LEG_MOUNT_YAWS_DEG_BODY[leg_name])

        foot_body = np.array(leg_mount_position, dtype=float) + rotation_z(leg_yaw_rad) @ foot_leg_expected
        foot_world = body_position_world + r_body @ foot_body

        foot_leg_result = foot_world_to_leg_local(
            leg_name,
            foot_world,
            body_position_world,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            yaw_rad=yaw_rad,
        )

        print(f"  {leg_name}: {foot_leg_result}")
        assert np.allclose(foot_leg_result, foot_leg_expected, atol=1e-6), (
            f"Check failed for {leg_name}: {foot_leg_result}"
        )


if __name__ == "__main__":
    check_world_to_leg_roundtrip()
    print("All world-to-leg transform checks passed.")
