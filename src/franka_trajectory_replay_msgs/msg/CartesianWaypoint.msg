# One point of a Cartesian replay trajectory.

# O_T_EE target: the end effector frame in the controller's base_frame (fr3_link0).
geometry_msgs/Pose pose

# Base-frame linear and angular velocity of the reference at this point. The linear part
# selects cubic Hermite interpolation of the position between points; the angular part is
# informative only (orientation is slerped). When every point's twist is exactly zero the
# trajectory carries no velocity information and positions are interpolated linearly.
geometry_msgs/Twist twist

# Joint configuration the nullspace term pulls toward at this point: exactly 7 values, or
# empty to keep the previous point's (or, for the first point, the controller's current) target.
float64[] nullspace_positions

builtin_interfaces/Duration time_from_start
