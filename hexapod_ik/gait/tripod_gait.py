"""Tripod gait generation and IK solving using body-pose primitives.

This gait uses the neutral stance footprint from `neutral_foot_positions_body`.
Body frame convention is +X right, +Y forward, +Z up.
Tripod groups alternate swing/stance phases across half-cycles.
Outputs are IK joint angles per leg/frame, not servo commands.
Servo mapping is intentionally kept separate from gait/IK planning.
Body pose limits are validated before IK solving.
"""

from __future__ import annotations

import math

import numpy as np

from hexapod_ik.body.body_pose import (
    body_rotation_matrix,
    foot_world_to_leg_local,
    is_body_pose_within_limits,
    neutral_foot_positions_body,
)
from hexapod_ik.kinematics.leg_ik import solve_ik_to_target


LEG_ORDER = ("RF", "LF", "RM", "LM", "RB", "LB")
TRIPOD_A = ("RF", "LM", "RB")
TRIPOD_B = ("LF", "RM", "LB")


def define_tripod_leg_groups():
    return {
        "tripod_a": TRIPOD_A,
        "tripod_b": TRIPOD_B,
    }


def _phase_fraction(step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return 0.0
    return step_idx / float(total_steps - 1)


def _stance_y_offset(step_idx: int, stance_steps: int, step_length: float) -> float:
    frac = _phase_fraction(step_idx, stance_steps)
    return (step_length / 2.0) - (step_length * frac)


def _swing_y_offset(step_idx: int, swing_steps: int, step_length: float) -> float:
    frac = _phase_fraction(step_idx, swing_steps)
    return (-step_length / 2.0) + (step_length * frac)


def _swing_z_offset(step_idx: int, swing_steps: int, step_height: float) -> float:
    frac = _phase_fraction(step_idx, swing_steps)
    return step_height * math.sin(math.pi * frac)


def build_tripod_foot_targets_body(step_length=1.0, step_height=0.75, stance_steps=10, swing_steps=10):
    neutral_targets = neutral_foot_positions_body()
    groups = define_tripod_leg_groups()

    frames = []

    # Half-cycle 1: tripod A swings, tripod B stances.
    for step_idx in range(max(1, swing_steps)):
        leg_targets_body = {}
        phase_by_leg = {}

        for leg_name in LEG_ORDER:
            neutral = np.array(neutral_targets[leg_name], dtype=float)

            if leg_name in groups["tripod_a"]:
                y_offset = _swing_y_offset(step_idx, max(1, swing_steps), step_length)
                z_offset = _swing_z_offset(step_idx, max(1, swing_steps), step_height)
                phase_by_leg[leg_name] = "swing"
            else:
                y_offset = _stance_y_offset(step_idx, max(1, stance_steps), step_length)
                z_offset = 0.0
                phase_by_leg[leg_name] = "stance"

            leg_targets_body[leg_name] = np.array(
                [neutral[0], neutral[1] + y_offset, neutral[2] + z_offset],
                dtype=float,
            )

        frames.append(
            {
                "leg_targets_body": leg_targets_body,
                "phase_by_leg": phase_by_leg,
            }
        )

    # Half-cycle 2: tripod B swings, tripod A stances.
    for step_idx in range(max(1, swing_steps)):
        leg_targets_body = {}
        phase_by_leg = {}

        for leg_name in LEG_ORDER:
            neutral = np.array(neutral_targets[leg_name], dtype=float)

            if leg_name in groups["tripod_b"]:
                y_offset = _swing_y_offset(step_idx, max(1, swing_steps), step_length)
                z_offset = _swing_z_offset(step_idx, max(1, swing_steps), step_height)
                phase_by_leg[leg_name] = "swing"
            else:
                y_offset = _stance_y_offset(step_idx, max(1, stance_steps), step_length)
                z_offset = 0.0
                phase_by_leg[leg_name] = "stance"

            leg_targets_body[leg_name] = np.array(
                [neutral[0], neutral[1] + y_offset, neutral[2] + z_offset],
                dtype=float,
            )

        frames.append(
            {
                "leg_targets_body": leg_targets_body,
                "phase_by_leg": phase_by_leg,
            }
        )

    return {
        "leg_order": LEG_ORDER,
        "groups": groups,
        "num_steps": len(frames),
        "frames": frames,
    }


def solve_tripod_cycle_ik(
    body_position_world=(0.0, 0.0, 0.0),
    roll_deg=0.0,
    pitch_deg=0.0,
    yaw_deg=0.0,
    step_length=1.0,
    step_height=0.75,
    stance_steps=10,
    swing_steps=10,
    start_angles=(90.0, 90.0, 90.0),
    alpha=0.02,
    tol=0.05,
    max_iters=6000,
    damping=0.1,
    max_allowed_error=0.10,
    verbose=False,
):
    if not is_body_pose_within_limits(body_position_world, roll_deg, pitch_deg, yaw_deg):
        return {
            "accepted": False,
            "reason": "outside_body_pose_limits",
            "leg_order": LEG_ORDER,
            "groups": define_tripod_leg_groups(),
            "num_steps": 0,
            "angle_frames": [],
            "phase_frames": [],
            "failed_steps": [],
            "max_final_error": 0.0,
        }

    cycle = build_tripod_foot_targets_body(
        step_length=step_length,
        step_height=step_height,
        stance_steps=stance_steps,
        swing_steps=swing_steps,
    )

    body_origin = np.array(body_position_world, dtype=float)
    roll_rad = math.radians(roll_deg)
    pitch_rad = math.radians(pitch_deg)
    yaw_rad = math.radians(yaw_deg)
    body_rot = body_rotation_matrix(roll_rad=roll_rad, pitch_rad=pitch_rad, yaw_rad=yaw_rad)

    current_angles = {leg_name: tuple(start_angles) for leg_name in LEG_ORDER}

    angle_frames = []
    phase_frames = []
    failed_steps = []
    max_final_error = 0.0

    for frame_idx, frame in enumerate(cycle["frames"]):
        frame_angles = {}
        frame_phases = {}
        frame_failed_legs = []

        for leg_name in LEG_ORDER:
            foot_body = frame["leg_targets_body"][leg_name]
            foot_world = body_origin + body_rot @ foot_body
            foot_local = foot_world_to_leg_local(
                leg_name=leg_name,
                foot_world=foot_world,
                body_position_world=body_origin,
                roll_rad=roll_rad,
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
            )

            final_angles, _angle_history, _ee_history, converged, _iterations, final_error = solve_ik_to_target(
                current_angles[leg_name],
                foot_local,
                alpha=alpha,
                tol=tol,
                max_iters=max_iters,
                damping=damping,
            )

            current_angles[leg_name] = final_angles
            frame_angles[leg_name] = tuple(float(v) for v in final_angles)
            frame_phases[leg_name] = frame["phase_by_leg"][leg_name]

            max_final_error = max(max_final_error, float(final_error))
            if (not converged) or (final_error > max_allowed_error):
                frame_failed_legs.append(
                    {
                        "leg": leg_name,
                        "phase": frame["phase_by_leg"][leg_name],
                        "final_error": float(final_error),
                        "converged": bool(converged),
                    }
                )

            if verbose:
                print(
                    f"frame={frame_idx + 1}/{cycle['num_steps']} "
                    f"leg={leg_name} phase={frame['phase_by_leg'][leg_name]} "
                    f"converged={converged} err={final_error:.4f}"
                )

        angle_frames.append(frame_angles)
        phase_frames.append(frame_phases)

        if frame_failed_legs:
            failed_steps.append(
                {
                    "frame_index": frame_idx,
                    "failed_legs": frame_failed_legs,
                }
            )

    accepted = len(failed_steps) == 0
    reason = "solved" if accepted else "ik_infeasible"

    return {
        "accepted": accepted,
        "reason": reason,
        "leg_order": cycle["leg_order"],
        "groups": cycle["groups"],
        "num_steps": cycle["num_steps"],
        "target_frames_body": cycle["frames"],
        "angle_frames": angle_frames,
        "phase_frames": phase_frames,
        "failed_steps": failed_steps,
        "max_final_error": max_final_error,
    }


def validate_tripod_gait_solution(gait_solution):
    legs = tuple(gait_solution.get("leg_order", LEG_ORDER))
    expected_legs = set(LEG_ORDER)
    angle_frames = gait_solution.get("angle_frames", [])
    phase_frames = gait_solution.get("phase_frames", [])
    target_frames_body = gait_solution.get("target_frames_body", [])
    failed_steps = gait_solution.get("failed_steps", [])
    max_final_error = float(gait_solution.get("max_final_error", 0.0))

    warnings = []

    all_frames_have_all_legs = True
    for frame in angle_frames:
        if set(frame.keys()) != expected_legs:
            all_frames_have_all_legs = False
            break

    all_ik_converged = bool(gait_solution.get("accepted", False)) and len(failed_steps) == 0

    neutral_targets = neutral_foot_positions_body()
    swing_lift_seen = {"tripod_a": False, "tripod_b": False}
    stance_ground_ok = True
    z_tol = 1e-6
    groups = gait_solution.get("groups", define_tripod_leg_groups())

    if not target_frames_body:
        warnings.append("target_frames_body not available; swing/stance z checks skipped")
        swing_lift_ok = False
        stance_ground_ok = False
    else:
        for frame in target_frames_body:
            leg_targets_body = frame.get("leg_targets_body", {})
            phase_by_leg = frame.get("phase_by_leg", {})
            for leg_name in legs:
                if leg_name not in leg_targets_body or leg_name not in phase_by_leg:
                    continue

                target_z = float(leg_targets_body[leg_name][2])
                neutral_z = float(neutral_targets[leg_name][2])
                phase_name = phase_by_leg[leg_name]

                if phase_name == "stance" and abs(target_z - neutral_z) > z_tol:
                    stance_ground_ok = False

                if phase_name == "swing" and target_z > neutral_z:
                    if leg_name in groups.get("tripod_a", ()):  # pragma: no branch
                        swing_lift_seen["tripod_a"] = True
                    if leg_name in groups.get("tripod_b", ()):  # pragma: no branch
                        swing_lift_seen["tripod_b"] = True

        swing_lift_ok = swing_lift_seen["tripod_a"] and swing_lift_seen["tripod_b"]

    max_joint_delta_deg = 0.0
    for idx in range(len(angle_frames) - 1):
        prev_frame = angle_frames[idx]
        next_frame = angle_frames[idx + 1]
        for leg_name in legs:
            if leg_name not in prev_frame or leg_name not in next_frame:
                continue
            prev_angles = np.array(prev_frame[leg_name], dtype=float)
            next_angles = np.array(next_frame[leg_name], dtype=float)
            frame_delta = float(np.max(np.abs(next_angles - prev_angles)))
            max_joint_delta_deg = max(max_joint_delta_deg, frame_delta)

    if max_joint_delta_deg > 15.0:
        warnings.append(f"Large per-frame joint delta: {max_joint_delta_deg:.3f} deg (> 15 deg)")

    if not all_frames_have_all_legs:
        warnings.append("At least one frame is missing one or more legs")
    if not all_ik_converged:
        warnings.append("IK did not converge for all frames/legs")
    if not swing_lift_ok:
        warnings.append("Swing lift check failed for one or both tripod groups")
    if not stance_ground_ok:
        warnings.append("Stance z deviates from neutral ground height")

    valid = (
        bool(gait_solution.get("accepted", False))
        and all_frames_have_all_legs
        and all_ik_converged
        and swing_lift_ok
        and stance_ground_ok
    )

    return {
        "valid": valid,
        "num_frames": len(angle_frames),
        "legs": legs,
        "all_frames_have_all_legs": all_frames_have_all_legs,
        "all_ik_converged": all_ik_converged,
        "max_final_error": max_final_error,
        "failed_steps": failed_steps,
        "swing_lift_ok": swing_lift_ok,
        "stance_ground_ok": stance_ground_ok,
        "max_joint_delta_deg": max_joint_delta_deg,
        "warnings": warnings,
    }


def print_tripod_gait_validation_report(validation):
    print("tripod_gait_validation:")
    print(f"  valid: {validation['valid']}")
    print(f"  num_frames: {validation['num_frames']}")
    print(f"  legs: {validation['legs']}")
    print(f"  all_ik_converged: {validation['all_ik_converged']}")
    print(f"  max_final_error: {validation['max_final_error']:.6f}")
    print(f"  swing_lift_ok: {validation['swing_lift_ok']}")
    print(f"  stance_ground_ok: {validation['stance_ground_ok']}")
    print(f"  max_joint_delta_deg: {validation['max_joint_delta_deg']:.6f}")
    print(f"  warnings: {validation['warnings']}")
    if validation["failed_steps"]:
        print(f"  failed_steps: {validation['failed_steps']}")


def generate_joint_commands(gait_solution):
    commands = []
    for frame_idx, frame_angles in enumerate(gait_solution["angle_frames"]):
        commands.append(
            {
                "frame_index": frame_idx,
                "leg_angles_deg": {
                    leg_name: tuple(float(v) for v in frame_angles[leg_name])
                    for leg_name in gait_solution["leg_order"]
                },
            }
        )
    return commands


def run_tripod_gait_demo(cycles=1, **kwargs):
    solution = solve_tripod_cycle_ik(**kwargs)
    solution["joint_commands"] = generate_joint_commands(solution)
    solution["validation"] = validate_tripod_gait_solution(solution)

    if cycles <= 1 or not solution["joint_commands"]:
        return solution

    base_commands = solution["joint_commands"]
    repeated_commands = []
    for cycle_idx in range(cycles):
        frame_offset = cycle_idx * len(base_commands)
        for frame in base_commands:
            repeated_commands.append(
                {
                    "frame_index": frame_offset + frame["frame_index"],
                    "leg_angles_deg": dict(frame["leg_angles_deg"]),
                }
            )

    repeated = dict(solution)
    repeated["joint_commands"] = repeated_commands
    repeated["num_steps"] = solution["num_steps"] * cycles
    repeated["validation"] = validate_tripod_gait_solution(repeated)
    return repeated
