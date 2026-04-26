import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from kinematics.inverse_kinematics import set_axes_equal
from kinematics.tripod_gait import LEG_ORDER, run_tripod_gait_demo


FRAME_INTERVAL_MS = 60
TRIPOD_A_COLOR = "tab:blue"
TRIPOD_B_COLOR = "tab:orange"


def _body_rectangle_points(base_offsets):
    left_front = base_offsets["left_front"]
    right_front = base_offsets["right_front"]
    right_rear = base_offsets["right_rear"]
    left_rear = base_offsets["left_rear"]
    return np.vstack([left_front, right_front, right_rear, left_rear, left_front])


def main():
    gait = run_tripod_gait_demo(
        cycles=2,
        verbose=False,
    )

    groups = gait["groups"]
    base_offsets = gait["base_offsets"]
    frames = gait["joint_frames_world"]
    target_frames = gait["foot_target_frames_world"]
    num_frames = len(frames)

    body_rect = _body_rectangle_points(base_offsets)
    all_leg_points = np.vstack([
        np.vstack([frames[f][leg] for leg in LEG_ORDER])
        for f in range(num_frames)
    ])
    all_target_points = np.vstack([
        np.vstack([target_frames[f][leg] for leg in LEG_ORDER])
        for f in range(num_frames)
    ])
    all_points = np.vstack([all_leg_points, all_target_points, body_rect])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax, all_points)

    body_line, = ax.plot(
        body_rect[:, 0],
        body_rect[:, 1],
        body_rect[:, 2],
        "-",
        lw=2,
        color="black",
        label="Body rectangle",
    )

    hip_points = np.vstack([base_offsets[leg] for leg in LEG_ORDER])
    hip_markers = ax.scatter(
        hip_points[:, 0], hip_points[:, 1], hip_points[:, 2], c="black", s=20, label="Hip joints"
    )

    leg_lines = {}
    foot_markers = {}
    target_markers = {}
    for leg in LEG_ORDER:
        color = TRIPOD_A_COLOR if leg in groups["tripod_a"] else TRIPOD_B_COLOR
        leg_line, = ax.plot([], [], [], "o-", lw=2, color=color)
        foot_marker, = ax.plot([], [], [], "o", color=color, ms=6)
        target_marker, = ax.plot([], [], [], "x", color=color, ms=6)
        leg_lines[leg] = leg_line
        foot_markers[leg] = foot_marker
        target_markers[leg] = target_marker

    legend_items = [
        body_line,
        plt.Line2D([0], [0], color=TRIPOD_A_COLOR, lw=2, label="Tripod A"),
        plt.Line2D([0], [0], color=TRIPOD_B_COLOR, lw=2, label="Tripod B"),
        hip_markers,
    ]
    ax.legend(handles=legend_items, loc="upper left")

    def init_anim():
        for leg in LEG_ORDER:
            leg_lines[leg].set_data([], [])
            leg_lines[leg].set_3d_properties([])
            foot_markers[leg].set_data([], [])
            foot_markers[leg].set_3d_properties([])
            target_markers[leg].set_data([], [])
            target_markers[leg].set_3d_properties([])
        return tuple(list(leg_lines.values()) + list(foot_markers.values()) + list(target_markers.values()))

    def update_anim(frame_idx):
        for leg in LEG_ORDER:
            points = frames[frame_idx][leg]
            leg_lines[leg].set_data(points[:, 0], points[:, 1])
            leg_lines[leg].set_3d_properties(points[:, 2])

            foot = points[3]
            foot_markers[leg].set_data([foot[0]], [foot[1]])
            foot_markers[leg].set_3d_properties([foot[2]])

            target = target_frames[frame_idx][leg]
            target_markers[leg].set_data([target[0]], [target[1]])
            target_markers[leg].set_3d_properties([target[2]])

        ax.set_title(f"Tripod Gait Demo (frame {frame_idx + 1}/{num_frames})")
        return tuple(list(leg_lines.values()) + list(foot_markers.values()) + list(target_markers.values()))

    anim = FuncAnimation(
        fig,
        update_anim,
        init_func=init_anim,
        frames=num_frames,
        interval=FRAME_INTERVAL_MS,
        repeat=True,
        blit=False,
    )

    _ = anim
    plt.show()


if __name__ == "__main__":
    main()
