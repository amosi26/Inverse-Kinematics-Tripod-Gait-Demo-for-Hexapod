"""Tripod gait planner and IK stepping for a 6-leg layout."""

import numpy as np

from hexapod_ik.gait.swing_stance import foot_trajectory
from hexapod_ik.kinematics.leg_ik import (
    START_T1_DEG,
    START_T2_DEG,
    START_T3_DEG,
    fk_and_jacobian,
    fk_joint_positions,
    solve_ik_to_target,
)


LEG_ORDER = (
    "left_front",
    "left_middle",
    "left_rear",
    "right_front",
    "right_middle",
    "right_rear",
)

TRIPOD_A = ("left_front", "left_rear", "right_middle")
TRIPOD_B = ("right_front", "right_rear", "left_middle")

LEG_BASE_OFFSETS = {
    "left_front": np.array([3.5, 2.5, 0.0]),
    "left_middle": np.array([0.0, 2.5, 0.0]),
    "left_rear": np.array([-3.5, 2.5, 0.0]),
    "right_front": np.array([3.5, -2.5, 0.0]),
    "right_middle": np.array([0.0, -2.5, 0.0]),
    "right_rear": np.array([-3.5, -2.5, 0.0]),
}


def _build_leg_start_angles(base_angles=None):
    """Create per-leg start angles so legs mount perpendicular to the body."""
    if base_angles is None:
        base_angles = (START_T1_DEG, START_T2_DEG, START_T3_DEG)

    _, base_t2, base_t3 = base_angles
    return {
        "left_front": (90.0, base_t2, base_t3),
        "left_middle": (90.0, base_t2, base_t3),
        "left_rear": (90.0, base_t2, base_t3),
        "right_front": (-90.0, base_t2, base_t3),
        "right_middle": (-90.0, base_t2, base_t3),
        "right_rear": (-90.0, base_t2, base_t3),
    }


def define_tripod_leg_groups():
    return {
        "tripod_a": TRIPOD_A,
        "tripod_b": TRIPOD_B,
    }


def _phase_shift_cycle(targets, phase_ids, shift_steps):
    shifted_targets = np.roll(targets, shift_steps, axis=0)
    shifted_phase_ids = np.roll(phase_ids, shift_steps, axis=0)
    return shifted_targets, shifted_phase_ids


def build_leg_trajectories(start_angles=None, phase_shift_steps=None, **trajectory_kwargs):
    leg_start_angles = _build_leg_start_angles(start_angles)

    reference_start_foot, _ = fk_and_jacobian(*leg_start_angles["left_front"])
    reference_targets, _reference_phase_ids = foot_trajectory(
        start_foot_pos=reference_start_foot,
        **trajectory_kwargs,
    )
    num_steps = len(reference_targets)
    if phase_shift_steps is None:
        phase_shift_steps = num_steps // 2

    trajectories = {}
    for leg_name in LEG_ORDER:
        leg_start_foot, _ = fk_and_jacobian(*leg_start_angles[leg_name])
        leg_cycle_targets, leg_cycle_phase_ids = foot_trajectory(
            start_foot_pos=leg_start_foot,
            **trajectory_kwargs,
        )

        if leg_name in TRIPOD_A:
            leg_local_targets = leg_cycle_targets.copy()
            leg_phase_ids = leg_cycle_phase_ids.copy()
        else:
            leg_local_targets, leg_phase_ids = _phase_shift_cycle(
                leg_cycle_targets,
                leg_cycle_phase_ids,
                phase_shift_steps,
            )

        base = LEG_BASE_OFFSETS[leg_name]
        trajectories[leg_name] = {
            "local_targets": leg_local_targets,
            "world_targets": leg_local_targets + base,
            "phase_ids": leg_phase_ids,
        }

    return {
        "leg_order": LEG_ORDER,
        "groups": define_tripod_leg_groups(),
        "base_offsets": LEG_BASE_OFFSETS,
        "start_angles": leg_start_angles,
        "num_steps": num_steps,
        "trajectories": trajectories,
    }


def solve_tripod_cycle_ik(
    start_angles=None,
    alpha=0.02,
    tol=0.05,
    max_iters=6000,
    damping=0.1,
    verbose=False,
    **trajectory_kwargs,
):
    trajectory_data = build_leg_trajectories(
        start_angles=start_angles,
        **trajectory_kwargs,
    )

    current_angles = {
        leg_name: tuple(trajectory_data["start_angles"][leg_name]) for leg_name in LEG_ORDER
    }

    angle_frames = []
    joint_frames_local = []
    joint_frames_world = []
    foot_target_frames_world = []
    phase_frames = []
    solve_stats = []

    for step_idx in range(trajectory_data["num_steps"]):
        step_angles = {}
        step_joints_local = {}
        step_joints_world = {}
        step_targets_world = {}
        step_phases = {}
        step_stats = {}

        for leg_name in LEG_ORDER:
            leg_traj = trajectory_data["trajectories"][leg_name]
            target_local = leg_traj["local_targets"][step_idx]
            target_world = leg_traj["world_targets"][step_idx]
            phase_id = int(leg_traj["phase_ids"][step_idx])

            final_angles, _angle_history, _ee_history, converged, iterations, final_error = solve_ik_to_target(
                current_angles[leg_name],
                target_local,
                alpha=alpha,
                tol=tol,
                max_iters=max_iters,
                damping=damping,
            )
            current_angles[leg_name] = final_angles

            joints_local = fk_joint_positions(*final_angles)
            joints_world = joints_local + trajectory_data["base_offsets"][leg_name]

            step_angles[leg_name] = final_angles
            step_joints_local[leg_name] = joints_local
            step_joints_world[leg_name] = joints_world
            step_targets_world[leg_name] = target_world
            step_phases[leg_name] = phase_id
            step_stats[leg_name] = (converged, iterations, final_error)

            if verbose:
                phase_name = "stance" if phase_id == 0 else "swing"
                print(
                    f"step={step_idx + 1}/{trajectory_data['num_steps']} "
                    f"leg={leg_name} phase={phase_name} "
                    f"converged={converged} iters={iterations} err={final_error:.4f}"
                )

        angle_frames.append(step_angles)
        joint_frames_local.append(step_joints_local)
        joint_frames_world.append(step_joints_world)
        foot_target_frames_world.append(step_targets_world)
        phase_frames.append(step_phases)
        solve_stats.append(step_stats)

    return {
        "leg_order": LEG_ORDER,
        "groups": trajectory_data["groups"],
        "base_offsets": trajectory_data["base_offsets"],
        "start_angles": trajectory_data["start_angles"],
        "num_steps": trajectory_data["num_steps"],
        "trajectory_data": trajectory_data,
        "angle_frames": angle_frames,
        "joint_frames_local": joint_frames_local,
        "joint_frames_world": joint_frames_world,
        "foot_target_frames_world": foot_target_frames_world,
        "phase_frames": phase_frames,
        "solve_stats": solve_stats,
    }


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


def run_tripod_gait_demo(cycles=2, **kwargs):
    base_solution = solve_tripod_cycle_ik(**kwargs)
    if cycles <= 1:
        base_solution["joint_commands"] = generate_joint_commands(base_solution)
        return base_solution

    repeated_solution = dict(base_solution)
    repeated_solution["angle_frames"] = base_solution["angle_frames"] * cycles
    repeated_solution["joint_frames_local"] = base_solution["joint_frames_local"] * cycles
    repeated_solution["joint_frames_world"] = base_solution["joint_frames_world"] * cycles
    repeated_solution["foot_target_frames_world"] = base_solution["foot_target_frames_world"] * cycles
    repeated_solution["phase_frames"] = base_solution["phase_frames"] * cycles
    repeated_solution["solve_stats"] = base_solution["solve_stats"] * cycles
    repeated_solution["num_steps"] = base_solution["num_steps"] * cycles
    repeated_solution["joint_commands"] = generate_joint_commands(repeated_solution)
    return repeated_solution
