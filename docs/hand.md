# The Inspire hand: wiring, bring-up, and what the driver does

The RH56 is a six-actuator, twelve-joint dexterous hand on an RS485 bus. This
page covers getting it to answer at all, which is where nearly all the time
goes; once it answers, the driver is undramatic.

## Bring-up order

**1. Power it.** The USB-RS485 adapter carries *data only*. The hand needs its
own 24 V supply and is completely silent without one. This is by far the most
common bring-up failure.

**2. Wire A/B.** The RH56's Lemo FGG-1B-8P cable carries four signal pairs, and
**only one of them is RS485**:

| Wire | Function | |
|---|---|---|
| **Yellow** | **485_A / CAN_H** | → adapter terminal **A** |
| **Green** | **485_B / CAN_L** | → adapter terminal **B** |
| Red (thick) | VCC 24 V | |
| Black (thick) | GND | → adapter GND, if it has one |
| White | TX+ (Ethernet) | |
| Blue | TX− (Ethernet) | |
| Red (thin) | RX+ (Ethernet) | |
| Black (thin) | RX− (Ethernet) | |

Landing white+blue on the A/B terminals is an easy mistake and looks exactly
like a dead bus.

**3. Probe.** Read-only; it cannot move the hand:

```bash
ros2 run inspire_hand_driver inspire_hand_probe /dev/ttyUSB0
```

It scans both wire protocols across the plausible baud rates and hand IDs, then
prints the exact launch line for whatever answered. Factory defaults are
**ID 1, 115200 8N1, Modbus RTU**; the baud register accepts only
115200 / 57600 / 19200 / 921600.

**4. Launch.**

```bash
ros2 launch inspire_franka_bringup hand.launch.py port:=/dev/ttyUSB0
```

No hardware? `hand.launch.py mock:=true` runs a simulated hand that slews to its
targets at a finite rate, which exercises the whole pipeline.

### If it stays silent

Work down the physical layer, most likely first:

1. Wrong wires — see the table above.
2. A and B swapped. Harmless to try, and worth trying early; a reversed pair
   idles low and shows up as a solid RX LED plus stray `0x00` bytes.
3. No common ground between the adapter and the hand's 24 V supply. RS485 is
   differential but still needs a shared reference.
4. A wire loose in the screw terminal.
5. The unit is a **CAN variant**. The RH56 series ships as RS485 *or* CAN, and
   yellow/green carry CAN_H/CAN_L on those units. No RS485 adapter will ever
   reach one.

### Inside the container

`docker-compose.yml` maps `/dev` in and adds the container to `dialout`, so
`/dev/ttyUSB0` is usable without `privileged: true`. If the port opens on the
host but not in the container, check that the host user is in `dialout` too.

## Two wire protocols

The RH56 family ships with one of two incompatible framings depending on
firmware vintage. Both address the same registers, so only the framing differs
and everything above the transport is protocol-agnostic:

- `modbus` — standard Modbus RTU, FC 0x03 read / 0x10 write, CRC16. Newer
  firmware.
- `legacy` — Inspire's `EB 90` framing with an 8-bit additive checksum. Older
  units, and what the vendor's ROS 1 package speaks.

The probe tries both.

## Units: open ratios and radians

The hand's `ANGLE` registers run **0 = fully closed, 1000 = fully open**. The
driver calls that normalised form an *open ratio*: `1.0` open, `0.0` closed.

The URDF works in radians, and every driven joint has its **lower** limit at the
open pose. So:

```
angle_rad = lower + (1 - open_ratio) * (upper - lower)
```

Both forms are published, because both are useful: open ratios are the natural
way to script a grasp, and radians are what `robot_state_publisher` needs to
produce TF.

## Six actuators, twelve joints

Each finger's `*_intermediate` joint is driven off its `*_proximal` joint by a
four-bar linkage, and the thumb has two such followers. The firmware exposes
only the six driven DOF; the rest are mechanical.

That means **commands address six joints and state reports twelve**. Publishing
only the driven six would leave `robot_state_publisher` unable to place any
fingertip, so the driver computes the followers and publishes all twelve. The
coupling constants and their derivation are in
`src/inspire_hand_description/MODEL_PROVENANCE.md`.

Commanding a follower is rejected rather than silently redirected — the hardware
cannot do it.

## Interface

Everything is relative to the node's name (default `inspire_hand`), so two hands
coexist as two nodes with different names and different ports.

| Topic | Type | |
|---|---|---|
| `~/joint_states` | `sensor_msgs/JointState` | all twelve joints, radians — feeds `robot_state_publisher` |
| `~/state` | `sensor_msgs/JointState` | channels `"1".."6"`, position as open ratio, effort as raw current |
| `~/grip_force` | `sensor_msgs/JointState` | measured grip force per channel |
| `~/command` | `sensor_msgs/JointState` | subscribed; see below |

| Service | Type |
|---|---|
| `~/set_angles` | `inspire_hand_msgs/srv/SetAngles` |
| `~/set_speed` | `inspire_hand_msgs/srv/SetSpeed` |
| `~/set_force` | `inspire_hand_msgs/srv/SetForce` |

`~/command` and `set_angles` accept **either** channel ids (`"1".."6"`, values
read as open ratios) **or** driven joint names (values read as radians). The two
name sets are disjoint, so a message says which it means; mixing them in one
message is rejected rather than guessed at.

Any subset may be addressed. A DOF that is not named holds its previous target
rather than snapping to a default — which is what makes a partial command safe.

Channel order is the hand's own register order, and is used consistently
everywhere including the simulation's controller config:

| channel | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| | little | ring | middle | index | thumb bend | thumb rotation |

### Examples

```bash
# Curl the four fingers, thumb untouched (open ratios)
ros2 service call /inspire_hand/set_angles inspire_hand_msgs/srv/SetAngles \
  "{name: ['1','2','3','4'], open_ratio: [0.2, 0.2, 0.2, 0.2]}"

# The same thing in radians, by joint name
ros2 topic pub -1 /inspire_hand/command sensor_msgs/msg/JointState \
  "{name: [index_proximal_joint, middle_proximal_joint], position: [1.2, 1.2]}"

# Open everything
ros2 service call /inspire_hand/set_angles inspire_hand_msgs/srv/SetAngles \
  "{name: ['1','2','3','4','5','6'], open_ratio: [1,1,1,1,1,1]}"

# Slow it down, and cap grip force
ros2 service call /inspire_hand/set_speed inspire_hand_msgs/srv/SetSpeed \
  "{name: ['1','2','3','4','5','6'], speed: [300,300,300,300,300,300]}"
```

Speed and force live in volatile registers. The driver never commits them to
flash, so they are forgotten on power cycle — deliberately, because a bad value
written to flash is awkward to undo.

## Rates

The driver polls at 50 Hz by default (`publish_rate_hz`). Each cycle is three
register reads on a half-duplex bus, so this is not a control loop and should
not be treated as one. RS485 drops the odd frame under EMI; the driver tolerates
`max_read_failures` consecutive misses (5) before it reports the hand as lost,
and says so again when it comes back.
