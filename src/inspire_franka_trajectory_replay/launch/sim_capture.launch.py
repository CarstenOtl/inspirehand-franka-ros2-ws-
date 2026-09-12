"""The hand-guided capture stack against MuJoCo instead of the FCI.

    ros2 launch inspire_franka_trajectory_replay sim_capture.launch.py

Then, in a second sourced shell, the capture tool exactly as on hardware, with
the profile that knows MuJoCo has no FrankaRobotState:

    ros2 run inspire_franka_trajectory_replay capture_demo --profile sim \\
        --note "rehearsing the pinch"

This exists because neither of the two launches next to it is right for a
capture. ``sim.launch.py`` claims the arm with a trajectory controller and never
starts the hand driver, so there is no ``/inspire_hand/command`` to press a key
into. ``sim_replay.launch.py`` starts the driver and the bridge, but it also
spawns the replay controller, which claims the arm and drives it to a setpoint --
the opposite of an arm you can push around.

What this launch assembles instead
----------------------------------
*The arm floats.* Nothing claims it (``arm_command_interface:=none``) and the
scene is the gravity-free one, which is what the torque replay path already uses
because libfranka compensates gravity underneath a torque controller on the real
arm. Together those are the simulator's equivalent of
``gravity_compensation_example_controller``: the arm holds its pose and can be
pushed. With the viewer up, ctrl-drag a link to guide it by hand.

*The hand is the real driver.* ``inspire_hand_driver`` runs in ``mock`` mode and
``inspire_hand_sim_bridge`` forwards its radians into MuJoCo, so every key press
goes through the driver's unit conversion, range rejection, register
quantisation and thumb-abduction overlay exactly as it would on the bench. The
hand control is genuinely under test here; the arm is scenery.

What it cannot rehearse
-----------------------
``franka_msgs/FrankaRobotState`` is libfranka's own message. There is no
measured torque, no external wrench, no ``O_T_EE``, no collision indicator and
no load model in simulation, and none of them are faked: a session recorded here
carries ``/joint_states`` and nothing more from the arm, its TCP is forward
kinematics rather than the robot's own pose, and both the session manifest and
the extracted artifact are marked ``hand_guided_sim``. It rehearses the capture
and extraction path. It does not produce training data.

The mock transport also slews toward its target at a fixed rate rather than
modelling the hand's closed-loop response, and there is no RS485 latency or bus
contention. Timing conclusions belong on hardware.
The description itself lives in
:mod:`inspire_franka_trajectory_replay.launch_files.sim_capture`, so that its
choices can be imported and tested; this file is the launch-system entry point.
"""

from inspire_franka_trajectory_replay.launch_files.sim_capture import (
    generate_launch_description,
)

__all__ = ["generate_launch_description"]
