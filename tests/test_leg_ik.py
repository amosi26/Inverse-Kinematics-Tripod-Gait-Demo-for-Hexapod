import unittest

import numpy as np

from hexapod_ik.config.robot_config import (
    CONTROL_DT_SEC,
    IK_JOINT_LIMITS_DEG,
    MAX_JOINT_SPEED_DEG_PER_SEC,
)
from hexapod_ik.kinematics.leg_ik import fk_and_jacobian, solve_ik_to_target


class TestLegIKSafety(unittest.TestCase):
    def test_ik_output_within_joint_limits(self):
        start_angles = (0.0, 45.0, 0.0)
        target, _ = fk_and_jacobian(90.0, 80.0, 100.0)

        final_angles, angle_history, *_rest = solve_ik_to_target(
            start_angles,
            target,
            alpha=0.02,
            tol=0.05,
            max_iters=2000,
            damping=0.1,
        )

        all_angles = np.array(angle_history, dtype=float)
        coxa_min, coxa_max = IK_JOINT_LIMITS_DEG["coxa"]
        femur_min, femur_max = IK_JOINT_LIMITS_DEG["femur"]
        tibia_min, tibia_max = IK_JOINT_LIMITS_DEG["tibia"]
        self.assertTrue(np.all(all_angles[:, 0] >= coxa_min))
        self.assertTrue(np.all(all_angles[:, 0] <= coxa_max))
        self.assertTrue(np.all(all_angles[:, 1] >= femur_min))
        self.assertTrue(np.all(all_angles[:, 1] <= femur_max))
        self.assertTrue(np.all(all_angles[:, 2] >= tibia_min))
        self.assertTrue(np.all(all_angles[:, 2] <= tibia_max))

        self.assertGreaterEqual(final_angles[0], coxa_min)
        self.assertLessEqual(final_angles[0], coxa_max)
        self.assertGreaterEqual(final_angles[1], femur_min)
        self.assertLessEqual(final_angles[1], femur_max)
        self.assertGreaterEqual(final_angles[2], tibia_min)
        self.assertLessEqual(final_angles[2], tibia_max)

    def test_ik_step_limited_per_iteration(self):
        start_angles = (0.0, 45.0, 0.0)
        target, _ = fk_and_jacobian(150.0, 150.0, 150.0)

        _final_angles, angle_history, *_rest = solve_ik_to_target(
            start_angles,
            target,
            alpha=0.05,
            tol=0.05,
            max_iters=2000,
            damping=0.1,
        )

        max_step = MAX_JOINT_SPEED_DEG_PER_SEC * CONTROL_DT_SEC
        history = np.array(angle_history, dtype=float)
        if len(history) > 1:
            step_deltas = np.abs(np.diff(history, axis=0))
            self.assertLessEqual(float(np.max(step_deltas)), max_step + 1e-9)

    def test_unreachable_target_returns_initial_angles(self):
        start_angles = (0.0, 45.0, 0.0)
        unreachable_target = np.array([20.0, 0.0, 0.0], dtype=float)

        final_angles, angle_history, ee_history, converged, iterations, _final_error = solve_ik_to_target(
            start_angles,
            unreachable_target,
            alpha=0.02,
            tol=0.05,
            max_iters=200,
            damping=0.1,
        )

        self.assertEqual(tuple(final_angles), start_angles)
        self.assertEqual(angle_history, [start_angles])
        self.assertEqual(len(ee_history), 1)
        self.assertFalse(converged)
        self.assertEqual(iterations, 0)


if __name__ == "__main__":
    unittest.main()
