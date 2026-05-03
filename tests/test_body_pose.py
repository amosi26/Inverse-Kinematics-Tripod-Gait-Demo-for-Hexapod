import unittest

import numpy as np

from hexapod_ik.body.body_pose import foot_body_to_leg_local
from hexapod_ik.config.robot_config import LEG_MOUNT_POSITIONS_BODY


class TestBodyPose(unittest.TestCase):
    def test_leg_mount_maps_to_local_origin(self):
        for leg_name, leg_mount_body in LEG_MOUNT_POSITIONS_BODY.items():
            foot_leg_local = foot_body_to_leg_local(leg_name, leg_mount_body)
            self.assertTrue(np.allclose(foot_leg_local, np.zeros(3), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
