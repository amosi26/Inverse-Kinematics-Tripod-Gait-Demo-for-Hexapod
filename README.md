# Hexapod IK

This repository currently provides a constrained inverse-kinematics foundation for a hexapod, including a reusable body-pose IK layer for all six legs.

## Project Status
- Constrained single-leg IK is implemented and tested.
- Body pose transform layer is implemented (world/body/leg-local).
- Neutral stance footprint is implemented and validated.
- Reusable all-six-leg body pose IK API exists: `solve_body_pose_ik()`.
- Tripod gait integration with body pose IK is planned next.
- Animations are currently removed/deprioritized.

## Current Architecture

```text
hexapod_ik/
    config/
        robot_config.py
    kinematics/
        leg_ik.py
        transforms.py
    body/
        body_pose.py
        body_pose_ik.py
    gait/
        swing_stance.py
        tripod_gait.py

demos/
    run_body_pose_ik_demo.py
    check_all_legs_body_pose_ik.py
    check_all_legs_neutral_ik.py
    check_neutral_stance_footprint.py
    find_body_pose_limits.py
    check_body_pose_transform.py          # low-level diagnostic
    check_world_to_body_transform.py      # low-level diagnostic
    check_world_to_leg_transform.py       # low-level diagnostic
    run_leg_ik_demo.py
    run_tripod_demo.py

tests/
    test_leg_ik.py
    test_body_pose.py
    test_body_pose_ik.py
```

## Key Features
- IK math limits are separate from future servo command limits.
- Per-step joint rate limiter is enforced in IK updates.
- Rough reachability check prevents impossible-target chasing.
- Explicit body frame convention and leg mount geometry/yaw configuration.
- Neutral stance footprint generation and round-trip validation.
- Reusable all-six-leg body pose solver:
  - `hexapod_ik.body.body_pose_ik.solve_body_pose_ik(...)`
  - validates `BODY_POSE_LIMITS`
  - returns `outside_body_pose_limits` or `ik_infeasible` when not accepted

## Known Limitations
- No hardware servo output mapping/control yet.
- No tripod gait integration with `solve_body_pose_ik()` yet.
- No CAD collision checking yet.
- Conservative body pose limits are empirical software constraints.
- Some combined poses can still be rejected as `ik_infeasible` even if inside `BODY_POSE_LIMITS`.

## Commands

Install requirements:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

Run key demos:

```bash
python -m demos.run_body_pose_ik_demo
python -m demos.check_all_legs_body_pose_ik
python -m demos.check_all_legs_neutral_ik
python -m demos.check_neutral_stance_footprint
python -m demos.find_body_pose_limits
```

Low-level transform diagnostics:

```bash
python -m demos.check_body_pose_transform
python -m demos.check_world_to_body_transform
python -m demos.check_world_to_leg_transform
```
