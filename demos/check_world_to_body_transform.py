"""Quick verification checks for world-frame to body-frame transform math."""

import numpy as np

from hexapod_ik.body.body_pose import body_rotation_matrix, foot_world_to_body


def check_translation_only():
    print("Check 1: translation only, zero rotation")
    body_position_world = np.array([10.0, 20.0, 5.0])
    foot_world = np.array([11.0, 22.0, 2.0])

    foot_body = foot_world_to_body(
        foot_world,
        body_position_world,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=0.0,
    )

    expected = np.array([1.0, 2.0, -3.0])
    print(f"  recovered: {foot_body}")
    assert np.allclose(foot_body, expected, atol=1e-6), f"Check 1 failed: {foot_body}"


def check_yaw_roundtrip():
    print("Check 2: yaw-only round trip")
    body_position_world = np.array([0.0, 0.0, 0.0])
    foot_body_expected = np.array([2.0, 0.0, 0.0])
    yaw_rad = np.deg2rad(90.0)

    r_body = body_rotation_matrix(yaw_rad=yaw_rad)
    foot_world = r_body @ foot_body_expected

    foot_body_recovered = foot_world_to_body(
        foot_world,
        body_position_world,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=yaw_rad,
    )

    print(f"  recovered: {foot_body_recovered}")
    assert np.allclose(foot_body_recovered, foot_body_expected, atol=1e-6), (
        f"Check 2 failed: {foot_body_recovered}"
    )


def check_full_rotation_translation_roundtrip():
    print("Check 3: roll/pitch/yaw + translation round trip")
    body_position_world = np.array([3.0, -2.0, 1.0])
    foot_body_expected = np.array([1.0, 4.0, -2.0])

    pitch_rad = np.deg2rad(10.0)
    roll_rad = np.deg2rad(5.0)
    yaw_rad = np.deg2rad(30.0)

    r_body = body_rotation_matrix(roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad)
    foot_world = body_position_world + r_body @ foot_body_expected

    foot_body_recovered = foot_world_to_body(
        foot_world,
        body_position_world,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
    )

    print(f"  recovered: {foot_body_recovered}")
    assert np.allclose(foot_body_recovered, foot_body_expected, atol=1e-6), (
        f"Check 3 failed: {foot_body_recovered}"
    )


if __name__ == "__main__":
    check_translation_only()
    check_yaw_roundtrip()
    check_full_rotation_translation_roundtrip()
    print("All world-to-body transform checks passed.")
