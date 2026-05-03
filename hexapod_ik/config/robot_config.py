"""Robot-specific constants for leg IK and gait prototyping."""

# Shared default starting pose (degrees). Update these to change all demos that import them.
START_T1_DEG = 0.0
START_T2_DEG = 45.0
START_T3_DEG = -120.0

# Physical link lengths from the current simplified leg model.
L1X = 2.14246204  # coxa length
L2X = 3.01904217  # femur length
L3X = 3.23556539  # tibia length

# Physical servo angle range in degrees for each leg joint.
COXA_MIN_DEG = 0.0
COXA_MAX_DEG = 180.0
FEMUR_MIN_DEG = 0.0
FEMUR_MAX_DEG = 180.0
TIBIA_MIN_DEG = 0.0
TIBIA_MAX_DEG = 180.0

# Conservative practical servo speed model used by the IK step limiter.
MAX_JOINT_SPEED_DEG_PER_SEC = 90.0
CONTROL_DT_SEC = 0.05
