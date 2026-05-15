import unittest

import numpy as np

from hexapod_ik.body.body_pose import (
    foot_body_to_leg_local,
    is_body_pose_within_limits,
    neutral_foot_positions_body,
)
from hexapod_ik.config.robot_config import LEG_MOUNT_POSITIONS_BODY, NEUTRAL_FOOT_LEG_LOCAL


class TestBodyPose(unittest.TestCase):
    def test_leg_mount_maps_to_local_origin(self):
        for leg_name, leg_mount_body in LEG_MOUNT_POSITIONS_BODY.items():
            foot_leg_local = foot_body_to_leg_local(leg_name, leg_mount_body)
            self.assertTrue(np.allclose(foot_leg_local, np.zeros(3), atol=1e-9))

    def test_neutral_footprint_roundtrip(self):
        footprint = neutral_foot_positions_body()
        expected_names = {"RF", "LF", "RM", "LM", "RB", "LB"}
        self.assertEqual(set(footprint.keys()), expected_names)

        expected = np.array(NEUTRAL_FOOT_LEG_LOCAL, dtype=float)
        for leg_name, foot_body in footprint.items():
            foot_body = np.array(foot_body, dtype=float)
            self.assertEqual(foot_body.shape, (3,))
            recovered = foot_body_to_leg_local(leg_name, foot_body)
            self.assertTrue(np.allclose(recovered, expected, atol=1e-6))

    def test_body_pose_limits_neutral_passes(self):
        self.assertTrue(is_body_pose_within_limits([0.0, 0.0, 0.0], 0.0, 0.0, 0.0))

    def test_body_pose_limits_z_too_high_fails(self):
        self.assertFalse(is_body_pose_within_limits([0.0, 0.0, 0.6], 0.0, 0.0, 0.0))

    def test_body_pose_limits_z_too_low_fails(self):
        self.assertFalse(is_body_pose_within_limits([0.0, 0.0, -1.1], 0.0, 0.0, 0.0))

    def test_body_pose_limits_roll_outside_fails(self):
        self.assertFalse(is_body_pose_within_limits([0.0, 0.0, 0.0], 8.1, 0.0, 0.0))

    def test_body_pose_limits_pitch_outside_fails(self):
        self.assertFalse(is_body_pose_within_limits([0.0, 0.0, 0.0], 0.0, -5.1, 0.0))

    def test_body_pose_limits_yaw_outside_fails(self):
        self.assertFalse(is_body_pose_within_limits([0.0, 0.0, 0.0], 0.0, 0.0, 5.1))

    def test_body_pose_limits_conservative_combined_pose_passes(self):
        self.assertTrue(is_body_pose_within_limits([0.0, 0.0, 0.25], 3.0, -3.0, 4.0))


if __name__ == "__main__":
    unittest.main()
