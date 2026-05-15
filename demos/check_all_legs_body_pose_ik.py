"""Check all-six-leg IK while body pose changes and feet stay fixed in world."""

import numpy as np

from hexapod_ik.body.body_pose import is_body_pose_within_limits
from hexapod_ik.body.body_pose_ik import solve_body_pose_ik


def main():
    poses = [
        {
            "name": "neutral",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "body_up_0p5",
            "body_position_world": [0.0, 0.0, 0.5],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "body_down_0p5",
            "body_position_world": [0.0, 0.0, -0.5],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "roll_pos_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 5.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "roll_neg_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": -5.0,
            "pitch_deg": 0.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "pitch_pos_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 5.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "pitch_neg_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": -5.0,
            "yaw_deg": 0.0,
        },
        {
            "name": "yaw_pos_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 5.0,
        },
        {
            "name": "yaw_neg_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": -5.0,
        },
        {
            "name": "combined_small_pose",
            "body_position_world": [0.0, 0.0, 0.25],
            "roll_deg": 3.0,
            "pitch_deg": -3.0,
            "yaw_deg": 5.0,
        },
    ]

    poses_tested = 0
    poses_skipped_by_limits = 0
    poses_solved_successfully = 0
    poses_failed_inside_limits = 0
    skipped_poses = []
    failed_inside_limits_poses = []
    solved_poses = []
    overall_worst_final_error = 0.0
    all_joint_limits_ok_overall = True

    for pose in poses:
        poses_tested += 1

        pose_name = pose["name"]
        body_position_world = np.array(pose["body_position_world"], dtype=float)
        roll_deg = float(pose["roll_deg"])
        pitch_deg = float(pose["pitch_deg"])
        yaw_deg = float(pose["yaw_deg"])

        print("=" * 80)
        print(f"pose_name: {pose_name}")
        print(f"body_position_world: {body_position_world}")
        print(f"roll_pitch_yaw_deg: ({roll_deg}, {pitch_deg}, {yaw_deg})")
        pose_within_limits = is_body_pose_within_limits(body_position_world, roll_deg, pitch_deg, yaw_deg)
        print(f"within_body_pose_limits: {pose_within_limits}")

        result = solve_body_pose_ik(
            body_position_world=body_position_world,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            max_allowed_error=0.10,
        )

        if result["reason"] == "outside_body_pose_limits":
            poses_skipped_by_limits += 1
            skipped_poses.append(pose_name)
            print("pose_rejected_by_limits: True (skipping IK for this pose)")
            continue

        overall_worst_final_error = max(overall_worst_final_error, float(result["max_final_error"]))

        legs_converged = 0
        pose_all_joint_limits_ok = True
        for leg_name, leg_result in result["per_leg"].items():
            if leg_result["ik_converged"]:
                legs_converged += 1
            pose_all_joint_limits_ok = pose_all_joint_limits_ok and leg_result["joint_limits_ok"]
            all_joint_limits_ok_overall = all_joint_limits_ok_overall and leg_result["joint_limits_ok"]

            print(f"  leg_name: {leg_name}")
            print(f"  fixed_foot_world: {leg_result['foot_world']}")
            print(f"  foot_leg_local: {leg_result['foot_leg_local']}")
            print(f"  joint_angles_deg: {leg_result['joint_angles_deg']}")
            print(f"  ik_converged: {leg_result['ik_converged']}")
            print(f"  ik_iterations: {leg_result['ik_iterations']}")
            print(f"  ik_final_error: {leg_result['ik_final_error']:.6f}")
            print(f"  joint_limits_ok: {leg_result['joint_limits_ok']}")
            print("  " + "-" * 56)

        if result["accepted"]:
            poses_solved_successfully += 1
            solved_poses.append(pose_name)
        else:
            poses_failed_inside_limits += 1
            failed_inside_limits_poses.append(pose_name)

        print("pose_summary:")
        print(f"  accepted: {result['accepted']}")
        print(f"  reason: {result['reason']}")
        print(f"  failed_legs: {result['failed_legs']}")
        print(f"  legs_converged: {legs_converged}/6")
        print(f"  max_final_error: {result['max_final_error']:.6f}")
        print(f"  all_joint_limits_ok: {pose_all_joint_limits_ok}")

    print("=" * 80)
    print("final_summary:")
    print(f"poses_tested: {poses_tested}")
    print(f"poses_skipped_by_limits: {poses_skipped_by_limits}")
    print(f"poses_solved_successfully: {poses_solved_successfully}")
    print(f"poses_failed_inside_limits: {poses_failed_inside_limits}")
    print(f"skipped_poses: {skipped_poses}")
    print(f"solved_poses: {solved_poses}")
    print(f"failed_inside_limits_poses: {failed_inside_limits_poses}")
    print(f"worst_final_error_seen: {overall_worst_final_error:.6f}")
    print(f"all_joint_limits_within_ik_limits: {all_joint_limits_ok_overall}")

    if poses_failed_inside_limits:
        print(
            "Some body poses inside BODY_POSE_LIMITS still failed IK. "
            "This means the requested body pose may exceed the current reachable "
            "workspace or solver settings."
        )
    else:
        print("All inside-limit body poses solved successfully.")


if __name__ == "__main__":
    main()
