"""Run a compact body-pose IK acceptance demo using the reusable solver."""

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
            "name": "roll_pos_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 5.0,
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
            "name": "yaw_pos_5deg",
            "body_position_world": [0.0, 0.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "yaw_deg": 5.0,
        },
        {
            "name": "combined_small_pose",
            "body_position_world": [0.0, 0.0, 0.25],
            "roll_deg": 3.0,
            "pitch_deg": -3.0,
            "yaw_deg": 5.0,
        },
    ]

    for pose in poses:
        result = solve_body_pose_ik(
            body_position_world=pose["body_position_world"],
            roll_deg=pose["roll_deg"],
            pitch_deg=pose["pitch_deg"],
            yaw_deg=pose["yaw_deg"],
            max_allowed_error=0.10,
        )

        print("=" * 72)
        print(f"pose_name: {pose['name']}")
        print(f"accepted: {result['accepted']}")
        print(f"reason: {result['reason']}")
        print(f"failed_legs: {result['failed_legs']}")
        print(f"max_final_error: {result['max_final_error']:.6f}")

        if result["accepted"]:
            print("joint_angles_deg:")
            for leg_name, angles in result["joint_angles_deg"].items():
                print(f"  {leg_name}: {angles}")


if __name__ == "__main__":
    main()
