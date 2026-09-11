# Per-cycle state of the Cartesian impedance replay controller.
std_msgs/Header header

# Controller phase (0 idle, 1 goto, 2 trajectory, 3 stopping) and the phase clock in seconds.
int32 phase
float64 trajectory_time

# Sampler output before the reference filter.
geometry_msgs/Pose target
# Reference after the example's first-order filter; the pose the law was applied to.
geometry_msgs/Pose reference
# Measured O_T_EE.
geometry_msgs/Pose measured

# The law's error vector [p - p_d ; -R * vec(q_c^-1 q_d)], base frame.
float64[6] error

float64[7] nullspace_reference
float64[7] tau_task
float64[7] tau_nullspace
float64[7] tau_coriolis
float64[7] tau_command
