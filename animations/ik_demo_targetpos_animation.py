import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kinematics.inverse_kinematics import (
    START_T1_DEG,
    START_T2_DEG,
    START_T3_DEG,
    fk_joint_positions,
    random_reachable_target,
    set_axes_equal,
    solve_ik_to_target,
)


rng = np.random.default_rng()
num_targets = 10
frame_interval_ms = 60
pause_ms = 3000
pause_frames = max(1, pause_ms // frame_interval_ms)

all_origin_frames = []
all_path_frames = []
all_target_frames = []
all_segment_frames = []
all_targets = []

t1 = START_T1_DEG
t2 = START_T2_DEG
t3 = START_T3_DEG

for segment_idx in range(num_targets):
    target = random_reachable_target(rng)
    all_targets.append(target.copy())

    final_angles, angle_history, ee_history, converged, iterations, final_error_norm = solve_ik_to_target(
        (t1, t2, t3), target
    )
    t1, t2, t3 = final_angles

    print(
        f"Target {segment_idx + 1}/{num_targets}: target={target}, "
        f"converged={converged}, iterations={iterations}, final_error={final_error_norm:.4f}"
    )

    origin_history = np.array([fk_joint_positions(a1, a2, a3) for (a1, a2, a3) in angle_history])
    ee_history = np.array(ee_history)

    for frame_idx in range(len(origin_history)):
        all_origin_frames.append(origin_history[frame_idx])
        all_path_frames.append(ee_history[:frame_idx + 1])
        all_target_frames.append(target)
        all_segment_frames.append(segment_idx + 1)

    for _ in range(pause_frames):
        all_origin_frames.append(origin_history[-1])
        all_path_frames.append(ee_history)
        all_target_frames.append(target)
        all_segment_frames.append(segment_idx + 1)

all_origin_frames = np.array(all_origin_frames)
all_target_frames = np.array(all_target_frames)
all_targets = np.array(all_targets)

all_points = np.vstack([
    all_origin_frames.reshape(-1, 3),
    all_targets.reshape(-1, 3)
])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1, 1, 1))
ax.set_proj_type("ortho")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3-DOF IK Motion Across 10 Random Targets")
set_axes_equal(ax, all_points)

leg_line, = ax.plot([], [], [], "o-", lw=3, color="tab:blue", label="Leg")
path_line, = ax.plot([], [], [], "-", lw=2, color="tab:green", label="End-effector path")
tip_marker, = ax.plot([], [], [], "o", color="tab:orange", ms=8, label="End-effector")
target_marker, = ax.plot([], [], [], "o", color="red", ms=8, label="Target")
ax.legend(loc="upper left")


def init_anim():
    leg_line.set_data([], [])
    leg_line.set_3d_properties([])
    path_line.set_data([], [])
    path_line.set_3d_properties([])
    tip_marker.set_data([], [])
    tip_marker.set_3d_properties([])
    target_marker.set_data([], [])
    target_marker.set_3d_properties([])
    return leg_line, path_line, tip_marker, target_marker


def update_anim(frame):
    points = all_origin_frames[frame]
    leg_line.set_data(points[:, 0], points[:, 1])
    leg_line.set_3d_properties(points[:, 2])

    path = all_path_frames[frame]
    path_line.set_data(path[:, 0], path[:, 1])
    path_line.set_3d_properties(path[:, 2])

    tip = points[3]
    tip_marker.set_data([tip[0]], [tip[1]])
    tip_marker.set_3d_properties([tip[2]])

    target = all_target_frames[frame]
    target_marker.set_data([target[0]], [target[1]])
    target_marker.set_3d_properties([target[2]])

    ax.set_title(
        f"3-DOF IK Motion - Target {all_segment_frames[frame]}/{num_targets} "
        f"(frame {frame + 1}/{len(all_origin_frames)})"
    )
    return leg_line, path_line, tip_marker, target_marker


anim = FuncAnimation(
    fig,
    update_anim,
    init_func=init_anim,
    frames=len(all_origin_frames),
    interval=frame_interval_ms,
    repeat=False,
    blit=False,
)

_ = anim
plt.show()
