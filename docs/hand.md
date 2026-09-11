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
angle_rad   = lower + (1 - open_ratio) * (upper - lower)
open_ratio  = 1 - (angle_rad - lower) / (upper - lower)
```

Both forms are published, because both are useful: open ratios are the natural
way to script a grasp, and radians are what `robot_state_publisher` needs to
produce TF.

The limits below are read straight out of
`inspire_hand_description`'s URDF, and
`inspire_hand_description/test/test_mimic_matches_driver.py` fails if the two
ever disagree, so this table cannot drift from the description:

| DOF            | ch | driven URDF joint          | ratio 1.0 (open) | ratio 0.0 (closed) | rad per register step |
|----------------|----|----------------------------|------------------|--------------------|-----------------------|
| pinky          | 1  | pinky_proximal_joint       | 0.000 rad        | 1.470 rad          | 0.00147               |
| ring           | 2  | ring_proximal_joint        | 0.000 rad        | 1.470 rad          | 0.00147               |
| middle         | 3  | middle_proximal_joint      | 0.000 rad        | 1.470 rad          | 0.00147               |
| index          | 4  | index_proximal_joint       | 0.000 rad        | 1.470 rad          | 0.00147               |
| thumb_bend     | 5  | thumb_proximal_pitch_joint | 0.000 rad        | 0.600 rad          | 0.00060               |
| thumb_rotation | 6  | thumb_proximal_yaw_joint   | 0.000 rad        | 1.308 rad          | 0.00131               |

Two consequences worth keeping in mind. The registers are integers 0..1000, so
the last column is the finest step the hand can be commanded to take — a
trajectory whose fingers move less than that per sample is being quantised, not
tracked. And because *every* driven joint has its open pose at 0 rad, the two
conventions run in opposite directions: **a rising joint_states value means a
closing hand, and a rising commanded ratio means an opening one.**

Code should not restate these numbers.
`inspire_hand_driver.kinematics.rad_to_open_ratio` and its inverse are the
conversion, and `inspire_franka_trajectory_replay` derives its own limit checks
from `kinematics.DOFS` for exactly this reason: a copied constant that drifted
would not fail a comparison, it would silently mis-scale every command.

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

`~/command` and `set_angles` both take **open ratios: `1.0` fully open, `0.0`
fully closed.** Names say only *which* DOF to address — channel ids
(`"1".."6"`) or driven joint names, mixed freely — and never what the unit is.

**Commands and `joint_states` run opposite ways, deliberately.** `joint_states`
is in radians because that is what the URDF and `robot_state_publisher` need,
and there `0.0` is the *open* pose; a rising `joint_states` value means a
closing hand, while a rising commanded ratio means an opening one. Convert at
the boundary with `kinematics.rad_to_open_ratio` if you are holding radians —
which is what `inspire_franka_trajectory_replay` does, keeping its trajectories
and homing YAMLs in radians because those files also carry the FR3's joints.

A target outside `[0.0, 1.0]` **rejects the whole message or request** and says
which entries offended. It is not clamped. The unit used to be inferred from
the naming, so `1.5` as a channel id clamped to a fully open hand and the same
`1.5` as a joint name clamped to a fully closed one — one number, opposite ends
of travel, nothing logged either way.

Any subset may be addressed. A DOF that is not named holds its previous target
rather than snapping to a default — which is what makes a partial command safe.

As a final calibration overlay, thumb abduction (`thumb_proximal_yaw_joint`,
channel `"6"`) is rescaled from the commanded `[0.0, 1.0]` onto the open-ratio
range `[0.25, 1.0]`, because at `0.0` the thumb swings past the palm plane:

```
physical = 0.25 + 0.75 * commanded
```

The rescale happens after normal range validation; every other DOF keeps the
unmodified `[0.0, 1.0]` command range unchanged. This is enforced at the
driver's shared command boundary, so it applies equally to topic commands,
`set_angles` service calls, trajectory replay, and future command publishers.

Because it is a rescale rather than a floor, the map stays monotonic and no two
commands collapse onto the same pose — but the whole range contracts, so a
commanded `0.5` now reaches `0.625` rather than `0.5`.

For example, this deliberately publishes the raw value `0.0`; the driver then
writes `0.25` (register value `250`) for thumb abduction:

```bash
ros2 topic pub -1 /inspire_hand/command sensor_msgs/msg/JointState \
  "{name: [thumb_proximal_yaw_joint], position: [0.0]}"
```

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

# The same thing on the topic, by joint name - still open ratios
ros2 topic pub -1 /inspire_hand/command sensor_msgs/msg/JointState \
  "{name: [index_proximal_joint, middle_proximal_joint], position: [0.2, 0.2]}"

# Rejected: 1.2 is outside [0, 1]. Previously this clamped to fully open.
ros2 topic pub -1 /inspire_hand/command sensor_msgs/msg/JointState \
  "{name: [index_proximal_joint], position: [1.2]}"

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

### How fast can the hand be commanded?

Four ceilings, and the lowest wins.

**The wire.** RS485 is half-duplex, so a request and its reply add rather than
overlap. At 115200 baud, 8N1, a six-register block costs 25 bytes to read and 29
to write — 2.17 ms and 2.52 ms of pure byte time. The Modbus spec also asks for
a 1.75 ms silence either side of every frame at this baud rate; whether the hand
and the USB-serial adapter actually impose it is not something a datasheet
answers, and it more than doubles the cost if they do. So the wire alone allows
somewhere between 170 and 400 transactions per second.

**The device turnaround.** The gap between the end of a request and the start of
the reply is the hand's own, and is not published anywhere.

**The bus budget.** Commands share the line with the driver's state polling, and
polling three blocks at 50 Hz costs 33–85 % of the bus before a single target is
sent. That is why the driver has a `state_extras_divisor`: only the angles are
needed to publish joint states, so current and force can be fetched once per N
publishes and held in between. The replay launch sets it to 5, which drops
polling to roughly 14–36 % of the bus. It defaults to 1 everywhere else, so
ordinary bring-up is unchanged.

**The hand.** Bench replay measured about 0.17 s of lag between a commanded step
and the joint arriving — the actuator's own closed-loop response, and slower
than everything above by two orders of magnitude.

Both the model and the measurement are in one tool, which needs no hardware for
the first half:

```bash
ros2 run inspire_hand_driver inspire_hand_benchmark /dev/ttyUSB0
```

It re-commands the pose the hand is already holding, so it does not move it;
`--move` is required before it will command anything else.

**In practice: stream at 50 Hz.** That is what the replay runner defaults to and
what the bench run was verified at. The ceiling is not the reason — 0.17 s of
actuator lag means the hand cannot use a faster stream — so raise the rate only
if a trajectory's own smoothness demands it, and re-run the benchmark first.
