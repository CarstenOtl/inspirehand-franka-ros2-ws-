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
| `~/diagnostics` | `diagnostic_msgs/DiagnosticArray` | per-DOF firmware status, error bits, temperature; see [Force threshold and stall guard](#force-threshold-and-stall-guard) |
| `~/command` | `sensor_msgs/JointState` | subscribed; see below |

| Service | Type |
|---|---|
| `~/set_angles` | `inspire_hand_msgs/srv/SetAngles` |
| `~/set_speed` | `inspire_hand_msgs/srv/SetSpeed` |
| `~/set_force` | `inspire_hand_msgs/srv/SetForce` |
| `~/clear_errors` | `std_srvs/srv/Trigger` |
| `~/set_compliance` | `std_srvs/srv/SetBool` | enter/leave [compliant mode](#compliant-mode) |
| `~/tare_force` | `std_srvs/srv/Trigger` | take the current fingertip readings as "nothing is touching me" |
| `~/calibrate_force` | `std_srvs/srv/Trigger` | run the hand's own force-sensor calibration; **the hand moves by itself for six seconds, and jammed doing it** — refused unless `calibration_mode` says otherwise |

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
written to flash is awkward to undo. The driver re-applies its own force
threshold when it notices the hand has lost it; see the next section.

## Force threshold and stall guard

The hand is position controlled, and a finger that cannot reach its target does
not wait politely. It pushes until the actuator's protection trips, the
firmware latches a locked-rotor or over-current error for that DOF, and from
then on the DOF ignores every target until `CLEAR_ERROR` is written or the hand
is power-cycled. That is the "finger died, reboot fixed it" failure. The driver
now does two things about it, both on by default.

**The force threshold** is the firmware's own mechanism (`FORCE_SET`, the
register `~/set_force` writes). A DOF whose fingertip force reaches it stops
there, reports "stopped at force threshold", and raises no error. The driver
applies `startup_force` (default `500`, on the hand's 0–1000 g scale) to all six
DOF at startup, re-applies it when the hand comes back after going silent, and
reads `FORCE_SET` back every `limits_check_interval_sec` (2 s) so a hand that
rebooted between two reads gets its threshold back too. `~/set_force` still
overrides it per DOF — the capture presets ask for 150–200 g pinches and get
them — and what a service call set is what the readback then expects. Pass
`startup_force:=0` to leave the hand's power-on value alone, which is the old
behaviour.

The threshold only sees force at the fingertip sensor. Contact elsewhere on the
finger, a thumb rotation jammed against the palm, or a stall that happens before
the threshold was applied still ends in a latched error, which is where the
second mechanism comes in.

**The stall guard** (`stall_guard`) reads the `STATUS` and `ERROR` blocks every
extras cycle. When the firmware reports a DOF stopped on a fault (status 5, 6 or
7, or a clearable error bit), the driver:

1. backs that DOF's target off to where it actually is plus `stall_backoff`
   counts towards open (default 30, i.e. 3 % of travel), leaving the other DOF
   where they were;
2. writes `CLEAR_ERROR`, at most once per `clear_error_interval_sec`;
3. for `stall_holdoff_sec` (1 s) clamps any command that would take that DOF
   back past the backed-off angle, and logs that it is doing so.

Backing off comes before clearing so the cleared finger is not driven straight
back into the same obstacle. The hold-off is what makes a 50 Hz replay stream
safe: re-sending the same unreachable target stalls the finger at most once per
second rather than continuously. After the hold-off, commands pass again, and
if the obstacle is still there the cycle repeats — which is the intended
outcome, since the alternative is a dead finger.

Every stall, backoff, clear and clamp is logged with the finger's name and the
firmware's own status and error words. `~/diagnostics` carries the same per DOF
(level `ERROR` while stalled, `WARN` when parked at the force threshold), so
`ros2 topic echo /inspire_hand/diagnostics` is the first thing to look at when a
finger stops. `~/clear_errors` writes `CLEAR_ERROR` on request, for when the
guard is off or for checking whether a stopped finger answers without a power
cycle:

```bash
ros2 service call /inspire_hand/clear_errors std_srvs/srv/Trigger
```

`CLEAR_ERROR` (address 1004) shares a Modbus register with `SAVE` (1005), which
commits every volatile parameter to flash. The driver writes exactly `1` there
and nowhere else, and the mock counts flash saves so the tests can assert it
stays at zero. Over-temperature is the one error the write does not clear; the
manual says it clears itself once the actuator cools.

Neither mechanism has been exercised on the real hand yet: the mock models a
finger that stops at an obstacle, and the tests in
`inspire_hand_driver/test/test_stall_guard.py` run against that. Two things
worth watching on first hardware contact are whether the hand accepts the
nine-register health read spanning `ERROR`, `STATUS` and `TEMP` (the driver
falls back to three reads if it refuses), and whether the byte order of those
byte-per-DOF blocks matches the vendor example the driver follows (lower
address in the low byte). `ros2 run inspire_hand_driver inspire_hand_probe`
prints the raw words.

## Compliant mode

The nearest thing this hand has to the arm's gravity-compensation controller,
and it is worth being clear about how far that is. The FR3 floats because its
joints are backdrivable and the controller commands zero torque: the compliance
is physical and the controller merely stops fighting it. The RH56 takes a
position setpoint and nothing else, so the give has to be manufactured — the
driver reads fingertip force every cycle and retreats that finger's target
towards open in proportion to it. Push a fingertip and the finger opens; let go
and it closes back onto the commanded grasp.

```bash
ros2 service call /inspire_hand/set_compliance std_srvs/srv/SetBool "{data: true}"
# ... adjust the grip by hand ...
ros2 service call /inspire_hand/set_compliance std_srvs/srv/SetBool "{data: false}"
```

or `compliance:=true` at launch. Per DOF, in register counts:

```
yield = clamp((force - compliance_deadband) * compliance_counts_per_gram, 0, compliance_max_yield)
```

slew-limited to `compliance_yield_rate` opening and `compliance_return_rate`
closing back. Every one of those is a dynamic parameter, so `ros2 param set`
retunes the spring between one cycle and the next — which is the only workable
way to tune something whose test is pushing on it with a finger. Values the law
cannot use are refused rather than clamped.

```bash
ros2 param set /inspire_hand compliance_counts_per_gram 0.8
```

The shipped gain is **0.6 counts/g**. With the 80 g default deadband that puts
a 400 g fingertip load at 192 counts — 19 % of travel — and reaches
`compliance_max_yield` (300) at 580 g. Past that the finger has given all it
is going to: raise `compliance_max_yield` if you want a hard shove to keep
opening, and remember that the earlier bench pushes measured **1400–2500 g**,
so saturation is the normal case for a deliberate push rather than an edge
one. It is the light bump this gain is tuned for.

**The yield is an offset, never a command.** `~/command` and `~/set_angles`
still set the rest position; the offset is added on the way to the registers. A
partial command sent while someone is holding a fingertip merges onto the
commanded pose, not onto the pushed-open one, so a grasp held through a
compliant episode returns to exactly the grasp that was asked for. `~/state`
still reports where the fingers physically are, and `~/diagnostics` carries the
applied yield per DOF.

### The force reading is signed

`FORCE_ACT` is a signed 16-bit value, which a register map otherwise full of
0..1000 quantities does not advertise. On this rig the six fingertips read
**−2, −12, −26, −11, +1 and −90 g with nothing touching them** — the sensors'
own zero offset. Read unsigned those are 65534, 65524, 65510, 65525, 1 and
65446, and any threshold comparison then sees a hand under enormous load while
it holds nothing; compliant mode would drive every finger to `max_yield` the
moment it was switched on. `HandTransport.read_forces` signs them
(`to_signed16`); `FORCE_SET` is a commanded threshold and stays unsigned.

The practical consequence for tuning: a deadband only has to clear zero to
cover the resting offset, but it also has to cover the preload of whatever the
hand is already gripping, which is the larger number.

### What it cannot feel

One direction only. `FORCE_ACT` measures compression of the fingertip **pad**,
so pushing into the pad is the only input the law has: a finger gives and never
closes on its own. Push a finger anywhere else — the middle phalanx, the side
of the tip — and this reads nothing at all; that contact ends at the stall
guard instead, which is the right place for it. Thumb rotation (channel 6) has
no pad and is left out of `compliance_channels` by default.

`FORCE_SET` is **not** a ceiling on what can be sensed, though this page said it
was. The threshold governs the closing motion — a finger driving shut gives up
when it reaches the threshold — and has no bearing on what the sensor reports
when you load a finger that is already holding station. Measured with
`FORCE_SET` at its 500 g default, a firm push on a held fingertip reads
**1400–1900 g**. The full range is available to compliant mode, and a gentle
pinch preset does not starve it.

### Zeroing the fingertips: two different things

The fingertip sensors do not sit at zero, and they do not stay where they sit.
Measured on this rig: the index pad rested at −11 g, was pushed to 2511 g, and
then sat at **+219 g — fully open, motor off, touching nothing** — and stayed
there, with no decay over a minute, while its four neighbours held −6 to −11 g.
Against a fixed deadband that is a standing 183 g of phantom push, so the
finger holds a permanent partial yield and never comes home. That is what the
bench check reports as "gave, but did not come back".

There are two answers, and they are not alternatives to each other.

| | `~/tare_force` | `~/calibrate_force` |
|---|---|---|
| where | in the driver: a baseline is captured and subtracted | in the hand: `GESTURE_FORCE_CLB`, register 1009 |
| what changes | what compliant mode compares against | what `FORCE_ACT` reports, to every reader |
| motion | none | the hand drives its own DOF, by itself |
| duration | one cycle | 3 s for the fingers, 6 s including the thumb |
| undo | take another | none; it replaces the previous calibration |

**The tare** is what compliant mode uses. Entering the mode takes one
automatically — so keep your hands off the fingertips at that moment — and
`~/tare_force` takes another whenever the zero has walked since. The per-channel
zero currently in force is published as `force_zero` in `~/diagnostics`.

**The calibration** is the button the Inspire desktop app has, and it fixes the
reference at the source rather than downstream of it:

```bash
# refused unless a mode was chosen -- see below, and read it before you do
ros2 param set /inspire_hand calibration_mode full
ros2 service call /inspire_hand/set_compliance std_srvs/srv/SetBool "{data: false}"
ros2 service call /inspire_hand/calibrate_force std_srvs/srv/Trigger
```

The manual (§2.4.6) describes the routine: hold five fingers fully open; bend
the four fingers; hold the four open and bend the thumb; extend the thumb. It
is emphatic that the hand must be in a no-load state throughout — nothing
touching any finger, which includes anything it is holding.

#### It jammed this hand, so it is off by default

`calibration_mode` defaults to **`none`**, and `~/calibrate_force` refuses. On
the rig this was written against, the routine closed the whole hand and jammed
it. That is not a service to leave open on a hand that drives itself for six
seconds with the stall guard stood down.

The three modes:

| `calibration_mode` | what happens |
|---|---|
| `none` (default) | the service refuses and says why |
| `fingers` | start the routine, write `GESTURE_FORCE_CLB` back to `0` at `calibration_finger_sec` (3 s of the 6), before the thumb steps — **this is the one that jammed** |
| `full` | all four steps, thumb included; what the Inspire desktop app does |

(`none`, not `off`: YAML reads `off` as the boolean `false`, so
`calibration_mode:=off` would never arrive as a string at all.)

Why there is no fourth, better option: `GESTURE_FORCE_CLB` is **one register
write and one fixed sequence**, with no per-DOF variant and no way to say which
channels to touch. It cannot be asked to calibrate the fingers and leave the
thumb alone. The only lever a driver has is *when to stop it*, and the sequence
does run in a useful order — open all five, bend the four fingers, **then**
bend the thumb, then extend it — so `fingers` mode cuts it off after the finger
half. Two things about that were never documented, and the hardware has now
answered one of them badly:

**Does writing 0 stop it?** The manual gives `1` as start and says nothing
about `0` mid-sequence. The driver writes it and then watches for 1.5 s whether
the thumb settles onto the pose *it* commanded; if not, something else is still
driving, and the log says so:

```
the hand did NOT stop its calibration routine when asked: thumb_bend is at 0.412
against a commanded 1.000. ...
```

On this rig the observed outcome was a jammed hand, which is consistent with
the firmware not letting go and the two writers fighting over the same
actuators. Treat `fingers` as unproven and be ready to power-cycle.

**Does a routine cut short commit anything?** Also undocumented — the firmware
may only write its new references at the end of the full sequence, in which
case `fingers` mode buys nothing even when it does not jam. Compare resting
`~/grip_force` before and after to find out.

**The thumb extends either way.** The routine's own first step is *hold five
fingers fully open*, and five includes the thumb. Nothing can prevent that
while the routine runs at all. What `fingers` mode keeps the thumb out of is
the **bending**, three steps later, which is where it meets the fingers.

Before writing the register the driver opens the hand into a known pose and
waits for it to arrive rather than assuming. Which DOF get posed follows the
mode: `full` opens all six, which swings thumb rotation clear of the fingers'
sweep; `fingers` opens only the four, because otherwise the driver would be the
only reason the thumb moves at all. A DOF that has not arrived within
`calibration_clearance_sec` (3 s) **abandons the calibration** rather than
starting it anyway, and the log says which DOF and where it got to. Set
`calibration_clearance:=false` to skip the staging entirely.

**If it jams:**

```bash
ros2 service call /inspire_hand/clear_errors std_srvs/srv/Trigger
ros2 service call /inspire_hand/set_angles inspire_hand_msgs/srv/SetAngles \
    "{name: ['1','2','3','4','5','6'], open_ratio: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}"
```

and power-cycle the hand if that does not shift it.

Whether the result survives a power cycle is not documented: the manual lists
`SAVE` (register 1005) as the way to commit parameters to flash but does not
say whether calibration data goes through it, and the driver does not write
`SAVE` — it shares a register with `CLEAR_ERROR`, and committing to flash is
not something to do as a side effect. If the zero is wrong again after a power
cycle, run the calibration again.

Calibrating is not a cure for the walk, either. It re-establishes the reference
at that moment; it does not stop the pad from stepping again the next time you
lean on it. Expect to calibrate occasionally and tare often.

### Cost and honest expectations

Force normally rides `state_extras_divisor`, which under the replay launch's
divisor of 5 samples it at 10 Hz. Compliant mode reads it every cycle
regardless, so a cycle becomes angles + force + a write rather than angles
alone — roughly 7–21 ms of a 20 ms slot at 50 Hz (see [Rates](#rates)). A hand
being streamed targets at the same time wants a lower `publish_rate_hz` while
it is compliant.

And the loop is slow. The actuator's own lag is about 0.17 s against the arm's
~1 ms, so this is two orders of magnitude off the FR3 and a stiff gain on top
of that lag gives a finger that buzzes against your hand rather than yielding
to it. The slew limits are what keep the loop's dynamics slower than the
plant's; expect something that gives over a few tenths of a second, not
something that floats. Raise `compliance_counts_per_gram` until it feels
responsive and stop well before it feels alive.

**None of this has been on hardware.** The law is tested against exact time
steps and the driver's half against the mock's new `external_force` hook, which
is a number standing in for a thumb. The defaults were picked to be too soft
rather than too stiff, and what they are worth depends on numbers only a real
fingertip can give: what `FORCE_ACT` reads at rest, its noise floor, its counts
per newton, and how much of it survives to the driver at 50 Hz. Watch
`ros2 topic echo /inspire_hand/grip_force` while pushing each finger before
trusting any of the gains here.

## Rates

The driver polls at 50 Hz by default (`publish_rate_hz`). Each cycle is up to
four register reads on a half-duplex bus (angles, current, force, and the
status/error block), so this is not a control loop and should not be treated as
one. RS485 drops the odd frame under EMI; the driver tolerates
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
sent (the health block the stall guard reads is a fourth). That is why the
driver has a `state_extras_divisor`: only the angles are needed to publish joint
states, so current, force and health can be fetched once per N publishes and
held in between. The replay launch sets it to 5, which drops
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
