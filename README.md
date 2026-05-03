# Hexapod IK

This repository is currently focused on constrained single-leg inverse kinematics (IK) for a hexapod leg.

The current solver keeps practical safety constraints in the loop:
- internal IK math-angle clamping with `IK_JOINT_LIMITS_DEG = [-90, 90]` per joint
- separate `SERVO_COMMAND_LIMITS_DEG = [0, 180]` config for future hardware mapping (not used by IK yet)
- per-iteration joint-rate limiting
- rough reachability checks before iteration

Animations are temporarily deprioritized/removed while development focuses on kinematics reliability and body pose groundwork.

## Current Focus
- Reliable constrained leg IK behavior
- Clean package structure for upcoming body pose math

## Next Planned Work
- Body pose IK using body translation and roll/pitch/yaw transforms
- Converting body/world foot targets into leg-local targets before leg IK

Tripod gait planning exists in the repo, but it is not the immediate next focus.

## Body Pose Status
- Body pose transform utilities exist:
  - world -> body
  - body -> leg-local
  - world -> leg-local wrapper
- Transform demos/checks currently pass.

## Project Layout

```text
hexapod_ik/
    __init__.py

    config/
        __init__.py
        robot_config.py

    kinematics/
        __init__.py
        leg_ik.py
        transforms.py

    gait/
        __init__.py
        swing_stance.py
        tripod_gait.py

    body/
        __init__.py
        body_pose.py

demos/
    __init__.py
    check_body_pose_transform.py
    check_world_to_body_transform.py
    check_world_to_leg_transform.py
    run_leg_ik_demo.py
    run_one_leg_body_pose_ik_demo.py
    run_tripod_demo.py
    sweep_leg_local_ik_targets.py

tests/
    __init__.py
    test_body_pose.py
    test_leg_ik.py

README.md
.gitignore
requirements.txt
```

## Current Known Issue
- The body pose transform chain passes its checks.
- The IK solver currently converges only for some leg-local targets.
- The target `[3.0, 0.0, -4.0]` does not converge under current settings.
- Next work is defining valid neutral foot/home positions based on the actual IK workspace.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run leg IK demo:

```bash
python -m demos.run_leg_ik_demo
```

Run tripod console demo:

```bash
python -m demos.run_tripod_demo
```

Run tests:

```bash
pytest
```
