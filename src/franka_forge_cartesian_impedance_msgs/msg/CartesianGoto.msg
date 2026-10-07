# Quintic ramp of the Cartesian reference (and the nullspace configuration) to a target.

# O_T_EE target in the controller's base_frame.
geometry_msgs/Pose pose

# Exactly 7 values, or empty to keep the current nullspace target.
float64[] nullspace_positions

# Requested duration in seconds. 0 derives it from goto_min_duration and the goto velocity
# limits; a positive value is stretched to those limits as well, never shortened below them.
float64 duration
