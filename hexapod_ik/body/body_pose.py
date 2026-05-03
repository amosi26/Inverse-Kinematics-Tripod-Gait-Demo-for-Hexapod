"""Body pose IK helpers.

Planned scope:
- Represent body translation in XYZ and orientation in roll/pitch/yaw.
- Transform world/body-frame foot targets into each leg's local frame.
- Feed leg-local targets to constrained leg IK.
"""

import numpy as np

from hexapod_ik.config.robot_config import (
    LEG_MOUNT_POSITIONS_BODY,
    LEG_MOUNT_YAWS_DEG_BODY,
    NEUTRAL_FOOT_LEG_LOCAL,
)


def rotation_x(theta_rad):
    """Return a 3x3 rotation matrix about +X."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def rotation_y(theta_rad):
    """Return a 3x3 rotation matrix about +Y."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def rotation_z(theta_rad):
    """Return a 3x3 rotation matrix about +Z."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def body_rotation_matrix(roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0):
    """
    Build body orientation matrix that maps body-frame vectors into world-frame vectors.
    """
    return rotation_z(yaw_rad) @ rotation_y(roll_rad) @ rotation_x(pitch_rad)


def foot_world_to_body(foot_world, body_position_world, roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0):
    """
    Convert a world-frame foot target to the body frame.

    world foot target
    -> subtract body position
    -> undo body rotation using R_body.T
    -> foot target in body frame
    """
    foot_world = np.array(foot_world, dtype=float)
    body_position_world = np.array(body_position_world, dtype=float)
    r_body = body_rotation_matrix(roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad)
    foot_body = r_body.T @ (foot_world - body_position_world)
    return foot_body


def foot_body_to_leg_local(leg_name, foot_target_body):
    """
    Convert a body-frame foot target to one leg's local frame.

    body frame target
    -> subtract leg mount position
    -> undo leg mount yaw
    -> leg-local target for IK
    """
    foot_target_body = np.array(foot_target_body, dtype=float)
    leg_mount_position_body = np.array(LEG_MOUNT_POSITIONS_BODY[leg_name], dtype=float)
    leg_mount_yaw_deg = float(LEG_MOUNT_YAWS_DEG_BODY[leg_name])

    foot_relative_body = foot_target_body - leg_mount_position_body

    yaw_rad = np.deg2rad(leg_mount_yaw_deg)
    foot_leg_local = rotation_z(-yaw_rad) @ foot_relative_body
    return foot_leg_local


def foot_world_to_leg_local(
    leg_name,
    foot_world,
    body_position_world,
    roll_rad=0.0,
    pitch_rad=0.0,
    yaw_rad=0.0,
):
    """Convert a world-frame foot target directly to one leg's local frame."""
    foot_body = foot_world_to_body(
        foot_world,
        body_position_world,
        roll_rad=roll_rad,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
    )
    return foot_body_to_leg_local(leg_name, foot_body)


def neutral_foot_positions_body():
    """
    Build neutral stance footprint in body frame for all six legs.

    This applies one shared leg-local neutral target to each leg and maps it
    into body coordinates using fixed body geometry and leg mount yaw angles.
    """
    neutral_leg_local = np.array(NEUTRAL_FOOT_LEG_LOCAL, dtype=float)
    footprint = {}
    for leg_name, mount_position in LEG_MOUNT_POSITIONS_BODY.items():
        mount_position = np.array(mount_position, dtype=float)
        yaw_rad = np.deg2rad(LEG_MOUNT_YAWS_DEG_BODY[leg_name])
        foot_body = mount_position + rotation_z(yaw_rad) @ neutral_leg_local
        footprint[leg_name] = foot_body
    return footprint
