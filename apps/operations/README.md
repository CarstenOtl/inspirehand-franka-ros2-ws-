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
