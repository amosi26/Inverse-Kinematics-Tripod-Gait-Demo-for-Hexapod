import unittest

from hexapod_ik.body.body_pose_ik import solve_body_pose_ik


class TestBodyPoseIK(unittest.TestCase):
    def test_neutral_pose_is_accepted(self):
        result = solve_body_pose_ik([0.0, 0.0, 0.0], 0.0, 0.0, 0.0)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "solved")

    def test_body_up_pose_is_accepted(self):
        result = solve_body_pose_ik([0.0, 0.0, 0.5], 0.0, 0.0, 0.0)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "solved")

    def test_roll_pos_5deg_is_accepted(self):
        result = solve_body_pose_ik([0.0, 0.0, 0.0], 5.0, 0.0, 0.0)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["reason"], "solved")

    def test_pose_outside_limits_is_rejected(self):
        result = solve_body_pose_ik([0.0, 0.0, 0.6], 0.0, 0.0, 0.0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "outside_body_pose_limits")

    def test_combined_small_pose_is_ik_infeasible(self):
        result = solve_body_pose_ik([0.0, 0.0, 0.25], 3.0, -3.0, 5.0)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "ik_infeasible")


if __name__ == "__main__":
    unittest.main()
