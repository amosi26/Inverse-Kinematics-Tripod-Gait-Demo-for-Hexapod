import numpy as np

# Shared default starting pose (degrees). Update these to change all demos that import them.
START_T1_DEG = 0.0
START_T2_DEG = 45.0
START_T3_DEG = -120.0


def wrap_angle_deg(angle_deg):
    # Wrap any angle to the range [-180, 180]
    return ((angle_deg + 180.0) % 360.0) - 180.0


def fk_and_jacobian(t1, t2, t3):
    # Denavite-Hartenberg parameters for the 3-DOF robotic arm
    l1x = 2.14246204
    l2x = 3.01904217
    l3x = 3.23556539

    l1z = 0.0
    l2z = 0.0
    l3z = 0.0

    a1 = 90
    a2 = 0 
    a3 = 0

    #Denavite-Hartenberg parameter table
    dh_table = np.array([[np.deg2rad(t1), np.deg2rad(a1), l1x, l1z],
                        [np.deg2rad(t2), np.deg2rad(a2), l2x, l2z],
                        [np.deg2rad(t3), np.deg2rad(a3), l3x, l3z]])

    #Homogeneous transformation matrix from frame i to frame i+1
    i = 0
    htm_0_1 = np.array([
        [np.cos(dh_table[i,0]), -np.sin(dh_table[i,0]) * np.cos(dh_table[i,1]),  np.sin(dh_table[i,0]) * np.sin(dh_table[i,1]), dh_table[i,2] * np.cos(dh_table[i,0])],
        [np.sin(dh_table[i,0]),  np.cos(dh_table[i,0]) * np.cos(dh_table[i,1]), -np.cos(dh_table[i,0]) * np.sin(dh_table[i,1]), dh_table[i,2] * np.sin(dh_table[i,0])],
        [0,                      np.sin(dh_table[i,1]),                          np.cos(dh_table[i,1]),                         dh_table[i,3]],
        [0, 0, 0, 1]
    ])

    #Homogeneous transformation matrix from frame 1 to frame 2
    i = 1
    htm_1_2 = np.array([
        [np.cos(dh_table[i,0]), -np.sin(dh_table[i,0]) * np.cos(dh_table[i,1]),  np.sin(dh_table[i,0]) * np.sin(dh_table[i,1]), dh_table[i,2] * np.cos(dh_table[i,0])],
        [np.sin(dh_table[i,0]),  np.cos(dh_table[i,0]) * np.cos(dh_table[i,1]), -np.cos(dh_table[i,0]) * np.sin(dh_table[i,1]), dh_table[i,2] * np.sin(dh_table[i,0])],
        [0,                      np.sin(dh_table[i,1]),                          np.cos(dh_table[i,1]),                         dh_table[i,3]],
        [0, 0, 0, 1]
    ])

    #Homogeneous transformation matrix from frame 2 to frame 3
    i = 2
    htm_2_3 = np.array([
        [np.cos(dh_table[i,0]), -np.sin(dh_table[i,0]) * np.cos(dh_table[i,1]),  np.sin(dh_table[i,0]) * np.sin(dh_table[i,1]), dh_table[i,2] * np.cos(dh_table[i,0])],
        [np.sin(dh_table[i,0]),  np.cos(dh_table[i,0]) * np.cos(dh_table[i,1]), -np.cos(dh_table[i,0]) * np.sin(dh_table[i,1]), dh_table[i,2] * np.sin(dh_table[i,0])],
        [0,                      np.sin(dh_table[i,1]),                          np.cos(dh_table[i,1]),                         dh_table[i,3]],
        [0, 0, 0, 1]
    ])


    htm_0_3 = htm_0_1 @ htm_1_2 @ htm_2_3

    r_0_0 = np.eye(3)  # Rotation matrix from frame 0 to frame 0 (identity)
    r_0_1 = htm_0_1[:3, :3]  # Rotation matrix from frame 0 to frame 1
    r_0_2 = htm_0_1 @ htm_1_2
    r_0_2 = r_0_2[:3, :3]  # Rotation matrix from frame 0 to frame 2
    z = np.array([0, 0, 1])  # Z-axis unit vector


    #displacement from center of frame 0 to center of frame 3
    d3_0 = htm_0_3[:3, 3]

    #displacement from center of frame 0 to center of frame 2
    htm_0_2 = htm_0_1 @ htm_1_2
    d2_0 = htm_0_2[:3, 3]

    #displacment from center of frame 0 to center of frame 1
    d1_0 = htm_0_1[:3, 3]

    #displacement from center of frame 0 to center of frame 0 
    d0_0 = np.array([0, 0, 0])




    #computing the jacobian matrix 
    jt1 = np.cross(r_0_0 @ z, d3_0 - d0_0) #joint 1 contribution to the Jacobian
    jt2 = np.cross(r_0_1 @ z, d3_0 - d1_0) #joint 2 contribution to the Jacobian
    jt3 = np.cross(r_0_2 @ z, d3_0 - d2_0) #joint 3 contribution to the Jacobian

    jr1 = r_0_0 @ z #joint 1 rotation contribution to the Jacobian
    jr2 = r_0_1 @ z #joint 2 rotation contribution to the Jacobian
    jr3 = r_0_2 @ z #joint 3 rotation contribution to the Jacob

    Jacobian = np.column_stack([
        np.concatenate((jt1, jr1)),
        np.concatenate((jt2, jr2)),
        np.concatenate((jt3, jr3))
    ])

    return d3_0, Jacobian


def fk_joint_positions(t1, t2, t3):
    # Returns joint origin points [p0, p1, p2, p3] for plotting.
    l1x = 2.14246204
    l2x = 3.01904217
    l3x = 3.23556539

    l1z = 0.0
    l2z = 0.0
    l3z = 0.0

    a1 = 90
    a2 = 0
    a3 = 0

    dh_table = np.array([
        [np.deg2rad(t1), np.deg2rad(a1), l1x, l1z],
        [np.deg2rad(t2), np.deg2rad(a2), l2x, l2z],
        [np.deg2rad(t3), np.deg2rad(a3), l3x, l3z]
    ])

    i = 0
    htm_0_1 = np.array([
        [np.cos(dh_table[i, 0]), -np.sin(dh_table[i, 0]) * np.cos(dh_table[i, 1]),  np.sin(dh_table[i, 0]) * np.sin(dh_table[i, 1]), dh_table[i, 2] * np.cos(dh_table[i, 0])],
        [np.sin(dh_table[i, 0]),  np.cos(dh_table[i, 0]) * np.cos(dh_table[i, 1]), -np.cos(dh_table[i, 0]) * np.sin(dh_table[i, 1]), dh_table[i, 2] * np.sin(dh_table[i, 0])],
        [0,                       np.sin(dh_table[i, 1]),                           np.cos(dh_table[i, 1]),                          dh_table[i, 3]],
        [0, 0, 0, 1]
    ])

    i = 1
    htm_1_2 = np.array([
        [np.cos(dh_table[i, 0]), -np.sin(dh_table[i, 0]) * np.cos(dh_table[i, 1]),  np.sin(dh_table[i, 0]) * np.sin(dh_table[i, 1]), dh_table[i, 2] * np.cos(dh_table[i, 0])],
        [np.sin(dh_table[i, 0]),  np.cos(dh_table[i, 0]) * np.cos(dh_table[i, 1]), -np.cos(dh_table[i, 0]) * np.sin(dh_table[i, 1]), dh_table[i, 2] * np.sin(dh_table[i, 0])],
        [0,                       np.sin(dh_table[i, 1]),                           np.cos(dh_table[i, 1]),                          dh_table[i, 3]],
        [0, 0, 0, 1]
    ])

    i = 2
    htm_2_3 = np.array([
        [np.cos(dh_table[i, 0]), -np.sin(dh_table[i, 0]) * np.cos(dh_table[i, 1]),  np.sin(dh_table[i, 0]) * np.sin(dh_table[i, 1]), dh_table[i, 2] * np.cos(dh_table[i, 0])],
        [np.sin(dh_table[i, 0]),  np.cos(dh_table[i, 0]) * np.cos(dh_table[i, 1]), -np.cos(dh_table[i, 0]) * np.sin(dh_table[i, 1]), dh_table[i, 2] * np.sin(dh_table[i, 0])],
        [0,                       np.sin(dh_table[i, 1]),                           np.cos(dh_table[i, 1]),                          dh_table[i, 3]],
        [0, 0, 0, 1]
    ])

    htm_0_2 = htm_0_1 @ htm_1_2
    htm_0_3 = htm_0_2 @ htm_2_3

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = htm_0_1[:3, 3]
    p2 = htm_0_2[:3, 3]
    p3 = htm_0_3[:3, 3]

    return np.vstack([p0, p1, p2, p3])


def set_axes_equal(ax, points):
    # Keeps 3D axis roughly equal so geometry does not look distorted.
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.max(maxs - mins)
    if span < 1e-6:
        span = 1.0
    half = 0.6 * span
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)



def random_reachable_target(rng):
    # Sample random joint angles and use FK so the resulting point is guaranteed reachable.
    t1 = rng.uniform(-180.0, 180.0)
    t2 = rng.uniform(-180.0, 180.0)
    t3 = rng.uniform(-180.0, 180.0)
    position, _ = fk_and_jacobian(t1, t2, t3)
    return position


def solve_ik_to_target(start_angles, target, alpha=0.01, tol=0.1, max_iters=10000, damping=0.1):
    t1, t2, t3 = start_angles
    angle_history = [(t1, t2, t3)]

    current, _ = fk_and_jacobian(t1, t2, t3)
    ee_history = [current.copy()]
    converged = False
    k = 0

    for k in range(max_iters):
        current, Jacobian = fk_and_jacobian(t1, t2, t3)
        error = target - current

        if np.linalg.norm(error) < tol:
            converged = True
            break

        Jv = Jacobian[:3, :]  # top 3 rows only

        # IK step using a pseudoinverse form.
        # - If damping > 0: damped pseudoinverse (more numerically stable near singularities).
        # - If damping == 0: standard Moore-Penrose pseudoinverse.
        if damping > 0.0:
            lam2 = damping ** 2
            identity_3 = np.eye(3)
            Jv_pinv = Jv.T @ np.linalg.pinv(Jv @ Jv.T + lam2 * identity_3)
        else:
            Jv_pinv = np.linalg.pinv(Jv)

        dq = Jv_pinv @ (alpha * error)

        t1 += np.rad2deg(dq[0])
        t2 += np.rad2deg(dq[1])
        t3 += np.rad2deg(dq[2])

        # Keep external joint-angle state in [-180, 180]
        t1 = wrap_angle_deg(t1)
        t2 = wrap_angle_deg(t2)
        t3 = wrap_angle_deg(t3)

        angle_history.append((t1, t2, t3))
        current, _ = fk_and_jacobian(t1, t2, t3)
        ee_history.append(current.copy())

    final_angles = (t1, t2, t3)
    final_error_norm = np.linalg.norm(target - current)
    return final_angles, angle_history, ee_history, converged, k + 1, final_error_norm
