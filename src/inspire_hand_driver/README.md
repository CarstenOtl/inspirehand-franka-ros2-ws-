# inspire_hand_driver

ROS 2 driver for the Inspire Robotics RH56 dexterous hand over RS485.

Full bring-up instructions, the cable pinout, the interface reference and the
open-ratio/radian conventions are in [`docs/hand.md`](../../docs/hand.md). This
file is the short version.

```bash
# read-only; cannot move the hand
ros2 run inspire_hand_driver inspire_hand_probe /dev/ttyUSB0

ros2 launch inspire_hand_driver inspire_hand.launch.py port:=/dev/ttyUSB0
ros2 launch inspire_hand_driver inspire_hand.launch.py mock:=true \
    publish_description:=true start_rviz:=true
```

## Shape of the code

| | |
|---|---|
| `protocol.py` | the serial transport and the register map. Speaks both wire formats the RH56 family ships with (Modbus RTU, and Inspire's legacy `EB 90` framing), plus a mock that slews to its targets so the whole pipeline runs with no hardware. |
| `kinematics.py` | the six register channels ↔ the URDF's twelve joints, including the four-bar coupling that makes six of them followers. |
| `driver_node.py` | the ROS node: polls state, publishes it in both radians and open ratios, and accepts commands in either. |
| `probe.py` | scans protocols, baud rates and hand IDs, and prints the launch line for whatever answers. |

`protocol.py` and `kinematics.py` are deliberately free of `rclpy`, which is why
the tests need neither a built workspace nor hardware:

```bash
python3 -m pytest test -q
```

## The thing most likely to confuse

The hand has **six actuators and twelve joints**. Commands address six; state
reports twelve. A follower cannot be commanded independently of the joint that
drives it, so naming one in a command is rejected rather than silently
redirected.
