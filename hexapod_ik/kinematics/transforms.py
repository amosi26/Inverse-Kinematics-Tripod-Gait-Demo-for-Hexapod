"""Small transform utilities used by kinematics modules.

This module is intentionally minimal for now. Future body pose IK work can
expand it with body/world and leg-local transform helpers.
"""

import numpy as np


def rot_x(theta_rad):
    """Rotation matrix for X-axis rotation."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(theta_rad):
    """Rotation matrix for Y-axis rotation."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(theta_rad):
    """Rotation matrix for Z-axis rotation."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
