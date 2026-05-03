"""Quick verification checks for body-frame to leg-local transform math."""

import numpy as np

from hexapod_ik.body.body_pose import foot_body_to_leg_local, rotation_z
from hexapod_ik.config.robot_config import LEG_MOUNT_POSITIONS_BODY, LEG_MOUNT_YAWS_DEG_BODY


def check_mount_becomes_zero():
    print("Check 1: leg mount -> local origin")
    for leg_name, mount_position in LEG_MOUNT_POSITIONS_BODY.items():
        foot_leg_local = foot_body_to_leg_local(leg_name, mount_position)
        print(f"  {leg_name}: {foot_leg_local}")
        assert np.allclose(foot_leg_local, np.array([0.0, 0.0, 0.0]), atol=1e-6), (
            f"Check 1 failed for {leg_name}: {foot_leg_local}"
        )


def check_outward_becomes_local_x():
    print("Check 2: outward along yaw -> +local X")
    for leg_name, mount_position in LEG_MOUNT_POSITIONS_BODY.items():
        yaw_deg = LEG_MOUNT_YAWS_DEG_BODY[leg_name]
        yaw_rad = np.deg2rad(yaw_deg)
        foot_body = np.array(mount_position, dtype=float) + rotation_z(yaw_rad) @ np.array([2.0, 0.0, 0.0])
        foot_leg_local = foot_body_to_leg_local(leg_name, foot_body)
        print(f"  {leg_name}: {foot_leg_local}")
        assert np.allclose(foot_leg_local, np.array([2.0, 0.0, 0.0]), atol=1e-6), (
            f"Check 2 failed for {leg_name}: {foot_leg_local}"
        )


def check_upward_stays_upward():
    print("Check 3: +body Z offset -> +local Z offset")
    for leg_name, mount_position in LEG_MOUNT_POSITIONS_BODY.items():
        foot_body = np.array(mount_position, dtype=float) + np.array([0.0, 0.0, 3.0])
        foot_leg_local = foot_body_to_leg_local(leg_name, foot_body)
        print(f"  {leg_name}: {foot_leg_local}")
        assert np.allclose(foot_leg_local, np.array([0.0, 0.0, 3.0]), atol=1e-6), (
            f"Check 3 failed for {leg_name}: {foot_leg_local}"
        )


if __name__ == "__main__":
    check_mount_becomes_zero()
    check_outward_becomes_local_x()
    check_upward_stays_upward()
    print("All body-to-leg transform checks passed.")
