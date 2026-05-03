# Hexapod IK

This repository is currently focused on constrained single-leg inverse kinematics (IK) for a hexapod leg.

The current solver keeps practical safety constraints in the loop:
- joint angle clamping to physical servo limits
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
    run_leg_ik_demo.py
    run_tripod_demo.py

tests/
    __init__.py
    test_leg_ik.py

README.md
.gitignore
requirements.txt
```

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
python -m unittest -v tests.test_leg_ik
```
