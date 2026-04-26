import numpy as np

try:
    from .inverse_kinematics import (
        START_T1_DEG,
        START_T2_DEG,
        START_T3_DEG,
        fk_and_jacobian,
        fk_joint_positions,
        solve_ik_to_target,
    )
except ImportError:
    try:
        from kinematics.inverse_kinematics import (
            START_T1_DEG,
            START_T2_DEG,
            START_T3_DEG,
            fk_and_jacobian,
            fk_joint_positions,
            solve_ik_to_target,
        )
    except ImportError:
        from inverse_kinematics import (
            START_T1_DEG,
            START_T2_DEG,
            START_T3_DEG,
            fk_and_jacobian,
            fk_joint_positions,
            solve_ik_to_target,
        )


def foot_trajectory(
    start_foot_pos,
    step_length=3.0,
    step_height=1.0,
    stance_targets=16,
    swing_targets=20,
    lateral_shift=0.0,
    lateral_swing_amp=0.2,
):
    """
    Plan one simple gait cycle (stance then swing) as discrete Cartesian targets.

    The goal of this function is to be easy to tune:
    - Increase `step_length` for longer steps.
    - Increase `step_height` for higher toe clearance during swing.
    - Increase `stance_targets` / `swing_targets` for smoother motion.
    - Use `lateral_shift` to bias the whole trajectory sideways in +Y / -Y.
    - Use `lateral_swing_amp` to add side-to-side sweep during swing
      (helps visualize base-servo XY behavior).

    Inputs:
    - start_foot_pos: numpy array [x, y, z] from FK for the current starting pose.
      This point is treated as the *front ground-contact* location.
    - step_length: total fore-aft travel distance for one cycle.
    - step_height: max lift above the ground-contact plane during swing.
    - stance_targets: number of solved IK targets in the stance phase.
    - swing_targets: number of solved IK targets in the swing phase.
    - lateral_shift: constant sideways offset applied to the whole trajectory.
    - lateral_swing_amp: half-amplitude of Y sweep during swing.

    Output:
    - targets: (N, 3) array of Cartesian targets to solve sequentially.
    - phase_ids: (N,) array with 0=stance and 1=swing for plotting/debug.
    """
    # Make a local copy so the caller's array is never modified in-place.
    start = np.array(start_foot_pos, dtype=float).copy()

    # Optional constant lateral offset (positive is +Y, negative is -Y).
    start[1] += lateral_shift

    # The ground/contact plane is defined by the starting Z level.
    ground_z = start[2]

    # Define contact points for one cycle:
    # - front_contact: where the cycle starts and ends.
    # - back_contact: stance ends here before lifting for swing.
    front_contact = start
    back_contact = start + np.array([-step_length, 0.0, 0.0])

    # ---------------------------
    # 1) STANCE PHASE TARGETS
    # ---------------------------
    # During stance, the foot stays on the ground while the body passes over it.
    # Here, in world coordinates, we model that as the foot moving from front to back.
    # `endpoint=False` avoids duplicating the exact back_contact target when swing starts.
    stance_s = np.linspace(0.0, 1.0, max(2, stance_targets), endpoint=False)
    stance_xyz = np.column_stack([
        front_contact[0] + (back_contact[0] - front_contact[0]) * stance_s,
        np.full_like(stance_s, start[1]),
        np.full_like(stance_s, ground_z),
    ])

    # ---------------------------
    # 2) SWING PHASE TARGETS
    # ---------------------------
    # During swing, the foot travels from back to front with vertical clearance.
    # X/Y are linear in phase; Z uses a sine bump so lift-off and touch-down are smooth:
    # - phase = 0 -> z lift 0
    # - phase = 0.5 -> z lift step_height (peak)
    # - phase = 1 -> z lift 0
    swing_s = np.linspace(0.0, 1.0, max(2, swing_targets))
    swing_x = back_contact[0] + (front_contact[0] - back_contact[0]) * swing_s

    # Add optional side-to-side sweep in Y during swing.
    # This is centered around `start[1]`, peaking mid-swing.
    swing_y = start[1] + lateral_swing_amp * np.sin(np.pi * swing_s)
    swing_z = ground_z + step_height * np.sin(np.pi * swing_s)
    swing_xyz = np.column_stack([swing_x, swing_y, swing_z])

    # Stack full cycle targets: stance first, then swing.
    targets = np.vstack([stance_xyz, swing_xyz])
    phase_ids = np.concatenate([
        np.zeros(len(stance_xyz), dtype=int),
        np.ones(len(swing_xyz), dtype=int),
    ])
    return targets, phase_ids


def solve_trajectory_targets(
    start_angles=None,
    targets=None,
    phase_ids=None,
    alpha=0.02,
    tol=0.05,
    max_iters=6000,
    damping=0.1,
    pause_frames=16,
    verbose=True,
):
    """
    Solve IK target-by-target and build frame histories for downstream animation.
    """
    if start_angles is None:
        start_angles = (START_T1_DEG, START_T2_DEG, START_T3_DEG)

    if targets is None:
        start_foot_pos, _ = fk_and_jacobian(*start_angles)
        targets, phase_ids = foot_trajectory(start_foot_pos=start_foot_pos)
    elif phase_ids is None:
        phase_ids = np.zeros(len(targets), dtype=int)

    all_origin_frames = []
    all_path_frames = []
    all_target_frames = []
    all_target_index_frames = []
    solve_stats = []

    start_foot_pos, _ = fk_and_jacobian(*start_angles)
    cycle_path = [start_foot_pos.copy()]
    t1, t2, t3 = start_angles

    for target_idx, target in enumerate(targets):
        final_angles, angle_history, ee_history, converged, iterations, final_error_norm = solve_ik_to_target(
            (t1, t2, t3),
            target,
            alpha=alpha,
            tol=tol,
            max_iters=max_iters,
            damping=damping,
        )
        t1, t2, t3 = final_angles
        solve_stats.append((converged, iterations, final_error_norm))

        if verbose:
            print(
                f"Target {target_idx + 1}/{len(targets)}: target={target}, "
                f"phase={'stance' if phase_ids[target_idx] == 0 else 'swing'}, "
                f"converged={converged}, iterations={iterations}, final_error={final_error_norm:.4f}"
            )

        origin_history = np.array([fk_joint_positions(a1, a2, a3) for (a1, a2, a3) in angle_history])
        ee_history = np.array(ee_history)

        for frame_idx in range(len(origin_history)):
            all_origin_frames.append(origin_history[frame_idx])
            all_path_frames.append(np.vstack([cycle_path, ee_history[:frame_idx + 1]]))
            all_target_frames.append(target)
            all_target_index_frames.append(target_idx)

        cycle_path.extend(ee_history[1:].tolist())

        for _ in range(max(0, pause_frames)):
            all_origin_frames.append(origin_history[-1])
            all_path_frames.append(np.array(cycle_path))
            all_target_frames.append(target)
            all_target_index_frames.append(target_idx)

    converged_flags = [stat[0] for stat in solve_stats]
    all_converged = all(converged_flags)
    failed_target_indices = [idx for idx, ok in enumerate(converged_flags) if not ok]
    print(f"All targets converged: {all_converged}")
    if not all_converged:
        print(f"Non-converged target indices: {failed_target_indices}")

    return {
        "targets": np.array(targets),
        "phase_ids": np.array(phase_ids),
        "all_origin_frames": np.array(all_origin_frames),
        "all_path_frames": all_path_frames,
        "all_target_frames": np.array(all_target_frames),
        "all_target_index_frames": np.array(all_target_index_frames),
        "solve_stats": solve_stats,
        "start_angles": start_angles,
        "final_angles": (t1, t2, t3),
    }


if __name__ == "__main__":
    print("Running swing_position.solve_trajectory_targets()...")
    solve_trajectory_targets(
        verbose=True,
        pause_frames=0,
    )
