"""Robot-specific constants for leg IK and gait prototyping."""

# Shared default starting pose (degrees). Update these to change all demos that import them.
START_T1_DEG = 0.0
START_T2_DEG = 45.0
START_T3_DEG = -120.0

# Physical link lengths from the current simplified leg model.
L1X = 2.14246204  # coxa length
L2X = 3.01904217  # femur length
L3X = 3.23556539  # tibia length

# Internal mathematical joint limits used by the IK solver.
# These limits are centered around zero for a 180-degree servo travel.
IK_JOINT_LIMITS_DEG = {
    "coxa": (-90.0, 90.0),
    "femur": (-90.0, 90.0),
    "tibia": (-90.0, 90.0),
}

# Future hardware command limits for servo outputs.
# The IK solver should not use these command limits directly.
# Servo mapping will be handled later (likely servo_command = ik_angle + 90).
SERVO_COMMAND_LIMITS_DEG = {
    "coxa": (0.0, 180.0),
    "femur": (0.0, 180.0),
    "tibia": (0.0, 180.0),
}

# Conservative practical servo speed model used by the IK step limiter.
MAX_JOINT_SPEED_DEG_PER_SEC = 90.0
CONTROL_DT_SEC = 0.05

# Body frame convention (right-handed):
# - Origin at the geometric center of the body/base.
# - +X points to the robot's right.
# - +Y points to the robot's front.
# - +Z points upward/out of the page.
# These mount positions are measured from the body center to each coxa/base
# servo axis center, expressed in the body frame.
# TODO: Verify these rough values directly from CAD before hardware testing.
LEG_MOUNT_POSITIONS_BODY = {
    "RF": (1.50, 3.75, 0.0),
    "LF": (-1.50, 3.75, 0.0),
    "RM": (1.87, 0.00, 0.0),
    "LM": (-1.87, 0.00, 0.0),
    "RB": (1.50, -3.75, 0.0),
    "LB": (-1.50, -3.75, 0.0),
}

# Leg mount yaw angles in degrees, measured in the body XY plane about +Z.
# 0 deg points along +X/right, and 90 deg points along +Y/front.
# These define each leg-local frame orientation relative to the body frame.
# Values were computed from servo-center positions using atan2(y, x).
LEG_MOUNT_YAWS_DEG_BODY = {
    "RF": 68.2,
    "LF": 111.8,
    "RM": 0.0,
    "LM": 180.0,
    "RB": -68.2,
    "LB": -111.8,
}

# Default neutral foot target in each leg's local frame.
# +X_leg points outward from the body along the leg mount yaw direction.
# Z is negative because the foot sits below the body.
# This value was chosen because the leg-local IK sweep showed it converges.
# This is a prototype stance and should be validated against CAD/hardware clearance.
NEUTRAL_FOOT_LEG_LOCAL = (6.0, 0.0, -4.0)
