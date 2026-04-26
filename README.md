# Hexapod Leg Inverse Kinematics Proof of Concept

This project demonstrates a proof of concept for solving the inverse kinematics (IK) of a single hexapod leg using the Jacobian pseudoinverse method. The code models a 3-DOF (degrees of freedom) robotic leg, with kinematic parameters derived from the CAD model of the hexapod and later optimized for a simpler Denavit-Hartenberg (DH) table.

## Project Purpose
- **Goal:** Provide a working example of iterative inverse kinematics for a hexapod leg, using the Jacobian pseudoinverse (with optional damping for stability).
- **Scope:** This is a prototype for algorithm development and testing. The code is not final and will be updated or replaced as the project evolves.

## Mathematical Background
- **Forward Kinematics:** Uses DH parameters to compute the end-effector (foot) position from joint angles.
- **Jacobian Matrix:** The Jacobian relates joint velocities to end-effector velocities. For a 3-DOF leg, a 6x3 Jacobian is computed, but only the linear (top 3 rows) part is used for position IK.
- **Pseudoinverse IK:** The solver iteratively updates joint angles to minimize the error between the current and target foot position. The update step uses the (damped) pseudoinverse of the Jacobian:
  
  $$
  \Delta \theta = J_v^+ (\alpha (x_{target} - x_{current}))
  $$
  where $J_v^+$ is the (damped) pseudoinverse of the linear Jacobian, $\alpha$ is a step size, and $x$ are positions.
- **Damping:** Damping (Levenberg-Marquardt style) is used to improve stability near singularities.

## Kinematic Parameters
- **Measurements:** Link lengths and offsets were initially measured from the hexapod's CAD model. These were later adjusted to simplify the DH table and make the math more tractable for rapid prototyping.

## Project Status & Future Work
- **Constraints:** The current solver does not enforce joint or workspace constraints. These will be added in future iterations.
- **CAD Updates:** When the final CAD is complete and the preferred IK method is established, this code will be updated or replaced to match the new design.
- **Experimental:** This code is for experimentation and may be scrapped or heavily modified as the project progresses.

## Files
- `kinematics/inverse_kinematics.py`: Core kinematics and IK solver.
- `kinematics/swing_position.py`: Example trajectory planning and IK solving for a leg swing/stance cycle.
- `kinematics/tripod_gait.py`: Tripod gait planner/solver using swing trajectory + IK.
- `animations/`: Demo animation scripts.

---

**Author:** Andrei Mosincat
**Branch:** Feature/IK-leg (HexapodProject)
