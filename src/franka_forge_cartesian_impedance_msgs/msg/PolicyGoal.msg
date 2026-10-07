# A live policy goal for the Cartesian impedance controller, decoded the way ForgeUltra's
# `_apply_action` decodes it: the controller clips this goal against the measured grasp pose
# on every control cycle (policy_clip_position_m / policy_clip_orientation_rad, in the frame
# policy_clip_frame_yaw selects) and runs the impedance law at the frame `tool_in_flange`
# names. The policy process only decodes `bolt_tip + action` once per tick.

# The preclipped grasp-frame goal in the controller's base_frame.
geometry_msgs/Pose goal

# Where the grasp frame currently is on the flange: [x y z qx qy qz qw]. The controller
# measures its pose and differentiates its law at this point from now on, so the impedance
# follows the live fingertip midpoint. Empty keeps the controlled frame the controller has
# (the static tool offset until a goal has set one).
float64[] tool_in_flange

# Exactly 7 values, or empty to keep the current nullspace target.
float64[] nullspace_positions
