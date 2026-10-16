# Per-cycle state of the Cartesian impedance replay controller.
std_msgs/Header header

# Controller phase (0 idle, 1 goto, 2 trajectory, 3 stopping, 4 policy) and the phase clock
# in seconds.
int32 phase
float64 trajectory_time

# Sampler output before the reference filter. Under a PolicyGoal this is the goal clipped
# against the measured pose this cycle.
geometry_msgs/Pose target
# Reference after the example's first-order filter; the pose the law was applied to.
geometry_msgs/Pose reference
# Measured pose of the controlled frame (the flange, moved by tool_in_flange).
geometry_msgs/Pose measured

# The controlled frame on the flange, [x y z qx qy qz qw]: the static tool offset, or the
# frame the last PolicyGoal set.
float64[7] tool_in_flange
# The last PolicyGoal's preclipped goal (identity until one has arrived), and whether its
# clip bound this cycle.
geometry_msgs/Pose goal
bool position_clipped
bool orientation_clipped

# The law's error vector [p - p_d ; -R * vec(q_c^-1 q_d)], base frame.
float64[6] error

float64[7] nullspace_reference
float64[7] tau_task
float64[7] tau_nullspace
float64[7] tau_coriolis
float64[7] tau_command
