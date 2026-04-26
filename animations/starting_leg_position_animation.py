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
    set_axes_equal,
)


frame_interval_ms = 60

t1 = START_T1_DEG
t2 = START_T2_DEG
t3 = START_T3_DEG
points = fk_joint_positions(t1, t2, t3)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect((1, 1, 1))
ax.set_proj_type("ortho")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3-DOF Leg Starting Pose (Static)")

set_axes_equal(ax, np.vstack([np.array([[0.0, 0.0, 0.0]]), points]))

leg_line, = ax.plot([], [], [], "o-", lw=3, color="tab:blue", label="Leg")
tip_marker, = ax.plot([], [], [], "o", color="tab:orange", ms=8, label="End-effector")
ax.legend(loc="upper left")


def init_anim():
    leg_line.set_data([], [])
    leg_line.set_3d_properties([])
    tip_marker.set_data([], [])
    tip_marker.set_3d_properties([])
    return leg_line, tip_marker


def update_anim(_frame):
    leg_line.set_data(points[:, 0], points[:, 1])
    leg_line.set_3d_properties(points[:, 2])

    tip = points[3]
    tip_marker.set_data([tip[0]], [tip[1]])
    tip_marker.set_3d_properties([tip[2]])
    return leg_line, tip_marker


anim = FuncAnimation(
    fig,
    update_anim,
    init_func=init_anim,
    frames=120,
    interval=frame_interval_ms,
    repeat=True,
    blit=False,
)

_ = anim
plt.show()
