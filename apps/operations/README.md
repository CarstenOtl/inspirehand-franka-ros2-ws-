# Robot operations

Small one-shot commands for an FR3 and Inspire RH56 that are already running
through `inspire_franka_bringup`. Run them from a second sourced shell inside
the container.

## Zero/open the hand

```bash
./apps/operations/zero_hand.py
```

This publishes all six channels to `/inspire_hand/command` until feedback on
`/inspire_hand/state` confirms the pose. The driver's command unit is an
**open ratio**, so the script sends `1.0`: that is fully open and corresponds
to zero radians in the hand URDF. Sending ratio `0.0` would fully close it.

Different hand namespace:

```bash
./apps/operations/zero_hand.py \
  --command-topic /right_hand/command --state-topic /right_hand/state
```

## Activate the policy Cartesian impedance controller

```bash
./apps/operations/activate_policy_controller.py
```

Puts the arm under the controller the policy rollout uses, and holds. This is
the bench step between "the law is right in simulation" and "run the policy on
hardware": it exercises the controller with no policy in the loop.

It refuses to run unless the controller is on ForgeUltra's law -- `axis_angle`
rotation error and `nullspace_damping_lambda: 0.0`, which is what
`controllers_policy.yaml` sets. The franka example's variant leaks the
nullspace joint spring onto the tool and halves the effective rotational
stiffness, so a session that silently ran it would read as a test of this one.
`--allow-example-law` characterises that law on purpose.

Activation commands no motion: the controller snapshots the measured pose,
orientation and joint configuration on its first update and holds them. Expect
the arm to be compliant about that pose at 565 N/m, and the elbow to settle
further than it does under the example law, because the exact projection
leaves the redundant direction genuinely free.

Bring the cell up first with the profile that loads the controller, then home
the arm, then run this from a second sourced shell:

```bash
ros2 launch inspire_franka_trajectory_replay replay.launch.py \
  arm_controller:=policy robot_ip:=172.16.0.2 hand_port:=/dev/ttyUSB0
./apps/operations/home_arm.py
./apps/operations/activate_policy_controller.py
```

One bounded step from the held pose, to excite the law and read back the
tracking error:

```bash
./apps/operations/activate_policy_controller.py --nudge 0 0 -0.01 --nudge-yaw -5
```

`--nudge` is in metres in the base frame, relative to the pose the controller
is holding, capped at 0.05 m per axis and 10 deg of yaw; the controller clamps
again on its own side. `--hold SECONDS` ends the session on its own instead of
waiting for Ctrl-C, `--report-hz` sets how often the tracking status prints,
and `--keep-active` leaves the controller in charge on exit.

On Ctrl-C, timeout, or an error, it hands the arm back to whichever controller
held the command interfaces before, through one atomic `switch_controller` so
the arm is never left unclaimed. `--dry-run` prints the plan without
connecting to ROS.

## Home the arm

```bash
./apps/operations/home_arm.py
```

After a confirmation prompt, this loads Franka's
`move_to_start_example_controller`, deactivates any controller that currently
claims an FR3 joint, moves from the measured pose to Franka's standard ready
configuration, and verifies `/joint_states` feedback. On Ctrl-C, timeout, or an
error after motion starts, it makes a best-effort controller deactivation.
The controller generates a smooth joint-space move but does not know about
objects in the cell, so the operator must clear the swept path before accepting
the prompt.

The default target is:

```text
[0, -pi/4, 0, -3*pi/4, 0, pi/2, pi/4] rad
```

Seven literal zeros are not a valid FR3 pose: joint 4's allowed range is
entirely negative, and joint 6's is entirely positive. A different valid pose
can be supplied explicitly:

```bash
./apps/operations/home_arm.py --target Q1 Q2 Q3 Q4 Q5 Q6 Q7
```

For automation, use `--yes`; use `--dry-run` to validate and print a target
without connecting to ROS. Namespaced arms use `--namespace`, with
`--arm-prefix` when the bring-up used one.

The arm utility is intended for the normal launch:

```bash
ros2 launch inspire_franka_bringup inspire_franka.launch.py
```

It intentionally does not launch hardware itself, so a typo cannot start a
second controller manager or a second serial driver alongside a live session.
