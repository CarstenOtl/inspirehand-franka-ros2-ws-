"""Record a hand-guided FR3 + Inspire RH56 demonstration.

    ros2 run inspire_franka_trajectory_replay capture_demo --note "pick the red block"

Run it beside, not instead of, the ordinary bringup::

    ros2 launch inspire_franka_bringup inspire_franka.launch.py \
        gravity_compensation:=true

In that mode ``gravity_compensation_example_controller`` holds the arm's effort
interfaces and commands zero torque, so the arm floats and is moved by hand.
**This tool never commands the arm.** It publishes to the hand and it records;
there is no code path here that writes an arm command interface, which is why
it is safe to run against a floating arm.

What is recorded
----------------
One rosbag per session. By default it is *lean*: the arm comes from
``/franka/joint_states`` at the controller's own 1 kHz, which is every number
the 15 Hz trajectory artifact is built from, and the session is ~60 MB for
three minutes instead of ~1.4 GB.

``--full-state`` adds ``franka_msgs/FrankaRobotState`` at 1 kHz on top: measured
and desired joint state, motor-side state, external and filtered torques, both
external wrenches, ``O_T_EE``/``F_T_EE``/``EE_T_K``, the elbow, the collision
and contact indicators, the robot mode, the error flags and the load model.
None of that is recoverable after the fact, so a take meant as training
material wants it -- but it is also ~90% of what a later ``extract_demo``
spends its time on, which is why exploratory takes do not pay for it. See
:data:`RECORDED_TOPICS` and :data:`FULL_STATE_ONLY_TOPICS`.

The intended loop is: capture lean, pull the waypoints out with
``extract_waypoints`` (no bag reading at all), confirm they replay, and only
then re-capture the motion with ``--full-state`` to augment into training.

Recording goes through :class:`franka_trajectory_replay.recording.BagRecorder`,
i.e. the ``ros2 bag record`` CLI. That is not incidental: an rclpy subscriber
drops messages at 1 kHz, which would silently thin exactly the channel this
session exists to produce. Do not replace it with a Python subscriber.

Before anything starts, every topic is checked and the session refuses to begin
if one is missing, with a per-topic report (the same fail-closed shape as
``apps/traj_replay/tests/system_check.py``).

Keyboard
--------
Hand presets come from ``config/hand_presets.yaml`` and are bound to their own
keys -- nothing about a posture is written here. Pressing a preset key both
issues the command and writes a timestamped event into the session, so the
demonstration's hand action channel is recorded ground truth rather than
something inferred afterwards from measured finger angles.

    <preset keys>  command that posture (see the banner the tool prints)
    <jog keys>     nudge one hand DOF by one step
    c              mark this instant and record every joint state
    s              mark a segment boundary
    ?              reprint the key map
    q              end the session and close the bag

Output
------
``logs/demo_capture/<UTC stamp>/`` with the raw ``bag/``, the ``events.jsonl``
command and marker log, and ``manifest.json``.

Turn a session's waypoints into a replayable artifact with ``extract_waypoints``
(events only, no bag), or the whole demonstrated motion with ``extract_demo``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from franka_trajectory_replay import limits
from franka_trajectory_replay.recording import BagRecorder
from inspire_hand_driver import kinematics as kin

from .hand_presets import PresetTable, load_presets, open_ratio_to_radians


SCHEMA_VERSION = 1

#: Where a session lands, relative to the workspace root.
DEFAULT_OUTPUT_ROOT = "logs/demo_capture"

#: Reserved keys, which a preset may not claim. Mirrored in hand_presets.yaml,
#: where the loader enforces it.
KEY_SEGMENT = "s"
KEY_CAPTURE = "c"
KEY_HELP = "?"
KEY_QUIT = "q"

#: Every key the tool keeps for itself. A preset or a jog control that claims
#: one of these is a load error in hand_presets.py, not a shadowed binding.
#:
#: There is deliberately no second, lighter marker key. An earlier version had
#: one that wrote a bare timestamp next to this one's full snapshot; it recorded
#: strictly less and meant deciding, mid-demonstration, which kind of mark this
#: moment deserved. One key that stores everything is the whole interface.
RESERVED_KEYS = (KEY_SEGMENT, KEY_CAPTURE, KEY_HELP, KEY_QUIT)

#: Where the pose snapshot reads the arm from. Deliberately the 30 Hz merged
#: view and not the 1 kHz robot_state: this is a Python callback holding a
#: latest value for a keypress, and subscribing it to a 1 kHz topic would burn
#: the executor for no gain. The 1 kHz channel is recorded by the bag, and the
#: snapshot's own timestamp is what locates it there.
SNAPSHOT_ARM_TOPIC = "/joint_states"


@dataclass(frozen=True)
class RecordedTopic:
    """One topic in the session bag, and how its presence is proven."""

    topic: str
    #: ``live`` waits for a message, ``subscriber``/``publisher`` only require
    #: the other end to exist. A command channel has no traffic until we make
    #: some, so waiting on one would deadlock the preflight; what matters there
    #: is that the driver is listening.
    check: str
    why: str
    #: Message type, declared rather than looked up. Resolving it from the graph
    #: instead loses a race: a node that has just been created has not finished
    #: DDS discovery, so every topic on a perfectly healthy system reads back as
    #: "not advertised" and the session is refused for no reason. Declaring the
    #: type means the subscription can be made immediately and the only question
    #: left is whether a message arrives.
    msg_type: str = ""


#: Everything the session records. The arm's full state is the point of the
#: exercise, so it leads; ``joint_states`` is kept as well because it is the
#: 30 Hz view every other tool in this workspace reads, and having both in one
#: bag is what lets an extraction be checked against them.
RECORDED_TOPICS: Sequence[RecordedTopic] = (
    RecordedTopic(
        "/franka_robot_state_broadcaster/robot_state",
        "live",
        "the complete FrankaRobotState at the controller's 1000 Hz update rate: "
        "q, dq, tau_J, q_d, dq_d, tau_J_d, ddq_d, theta, dtheta, dtau_J, tau_ext, "
        "O_F_ext, K_F_ext, O_T_EE, F_T_EE, EE_T_K, elbow, collision/contact "
        "indicators, robot_mode, errors and the load model",
        "franka_msgs/msg/FrankaRobotState",
    ),
    RecordedTopic(
        "/franka/joint_states",
        "live",
        "joint_state_broadcaster's own 1 kHz q/dq/effort, independent of the "
        "FrankaRobotState path",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/joint_states",
        "live",
        "the 30 Hz merged view the rest of this workspace reads (TF, RViz, the "
        "MuJoCo passive viewer)",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/joint_states",
        "live",
        "all twelve hand joints in radians, followers included - the measured "
        "hand pose extraction reads",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/state",
        "live",
        "the same reading in the hand's own open-ratio units, i.e. directly "
        "comparable with what was commanded",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/grip_force",
        "live",
        "measured grip force per channel; the only evidence of contact the hand has",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/command",
        "subscriber",
        "the hand action channel. Recorded so the bag alone shows what was "
        "commanded and when, independently of events.jsonl",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/tf",
        "publisher",
        "frames over time, so a later camera or object-pose channel can be "
        "aligned to this session without re-deriving kinematics",
        "tf2_msgs/msg/TFMessage",
    ),
    RecordedTopic(
        "/tf_static",
        "publisher",
        "the fixed frames, including the hand's flange mount",
        "tf2_msgs/msg/TFMessage",
    ),
)

#: The same session against MuJoCo instead of the FCI.
#:
#: Two things are simply not available in simulation and are not faked here.
#: ``FrankaRobotState`` is libfranka's own message -- there is no tau_ext, no
#: O_T_EE, no collision indicator and no load model without a real controller
#: box, so a simulated session records none of them and its artifact says so.
#: And ``joint_state_broadcaster`` publishes one ``/joint_states`` carrying all
#: nineteen joints rather than the arm's ``/franka/joint_states``.
#:
#: The hand, by contrast, is the real thing: ``sim_replay.launch.py`` runs the
#: actual driver in mock mode with ``inspire_hand_sim_bridge`` feeding MuJoCo,
#: so every command this tool sends goes through the driver's unit conversion,
#: range rejection and abduction overlay exactly as it would on the bench. That
#: is what makes a simulated session worth running: the hand control is under
#: test, the arm is scenery.
SIM_TOPICS: Sequence[RecordedTopic] = (
    RecordedTopic(
        "/joint_states",
        "live",
        "all nineteen simulated joints - arm and hand, driven and follower - from "
        "joint_state_broadcaster. In simulation this is the whole arm state there is",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/joint_states",
        "live",
        "the real driver's twelve joints in radians, mock transport",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/state",
        "live",
        "the real driver's open-ratio readback",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/grip_force",
        "live",
        "the mock transport's grip force; not a contact measurement",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/inspire_hand/command",
        "subscriber",
        "the hand action channel, identical to the hardware one",
        "sensor_msgs/msg/JointState",
    ),
    RecordedTopic(
        "/clock",
        "live",
        "simulation time. The session runs on it, so that the event log and the "
        "bag share one clock",
        "rosgraph_msgs/msg/Clock",
    ),
    RecordedTopic("/tf", "publisher", "frames over time", "tf2_msgs/msg/TFMessage"),
    RecordedTopic("/tf_static", "publisher", "the fixed frames", "tf2_msgs/msg/TFMessage"),
)

#: Profile name -> what that profile records. ``hardware`` is the default and
#: the only one whose output is a demonstration; ``sim`` is a rehearsal.
TOPIC_PROFILES = {"hardware": RECORDED_TOPICS, "sim": SIM_TOPICS}

#: Topics recorded only when ``--full-state`` is given.
#:
#: ``FrankaRobotState`` is 3.7 kB per message at 1 kHz -- 623 MB of a 1.4 GB
#: three-minute session, and about 90% of what a later extraction spends its
#: time on, because every one of those messages has to go through
#: ``rclpy.deserialize_message`` to produce an artifact that is then low-passed
#: and decimated to 15 Hz. Measured on a 176 s session: ~405 s to deserialize
#: the robot_state alone, against ~7 s for the whole lean set.
#:
#: So it is not recorded by default. A lean session still carries
#: ``/franka/joint_states`` at the same 1 kHz, which is every number the
#: trajectory artifact actually contains; what a lean bag gives up is the rest
#: of the FCI's field set -- tau_ext, both external wrenches, O_T_EE and the
#: other poses, the elbow, the collision and contact indicators, the error
#: flags and the load model. Those matter for training augmentation and for
#: after-the-fact contact analysis, and nothing recovers them later. Pass
#: ``--full-state`` for a take you already believe is worth keeping.
FULL_STATE_ONLY_TOPICS = frozenset({"/franka_robot_state_broadcaster/robot_state"})


def profile_topics(profile: str, full_state: bool) -> List[RecordedTopic]:
    """The topics a session records, given its profile and state detail.

    ``sim`` has no ``FrankaRobotState`` to drop, so ``full_state`` does nothing
    there rather than meaning something different -- a rehearsal records what a
    rehearsal can record either way.
    """
    entries = list(TOPIC_PROFILES[profile])
    if full_state:
        return entries
    return [entry for entry in entries if entry.topic not in FULL_STATE_ONLY_TOPICS]


#: The controller that has to be running for the arm to be guidable by hand.
GRAVITY_COMPENSATION_CONTROLLER = "gravity_compensation_example_controller"

#: Controllers that must NOT be active: anything holding the arm's command
#: interfaces alongside gravity compensation would be commanding the arm while
#: an operator has their hands on it.
_ARM_COMMANDING_SUFFIXES = (
    "trajectory_replay_controller",
    "cartesian_trajectory_replay_controller",
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _atomic_json(path: Path, document) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EventLog:
    """Append-only JSONL of hand commands and markers, flushed on every write.

    Flushed and fsynced per line rather than buffered: a session that ends on a
    power cut or a Ctrl-C should still describe every command it issued up to
    that moment, and these lines are the only record of operator intent.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self.counts: Dict[str, int] = {}

    def write(self, event: str, clock_ns: int, **fields) -> Dict[str, object]:
        record = {
            "event": event,
            "t_ros_ns": int(clock_ns),
            "t_wall_utc": datetime.now(timezone.utc).isoformat(),
        }
        record.update(fields)
        with self._lock:
            self.counts[event] = self.counts.get(event, 0) + 1
            json.dump(record, self._stream, allow_nan=False)
            self._stream.write("\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        return record

    def close(self) -> None:
        with self._lock:
            if not self._stream.closed:
                self._stream.close()


class CaptureNode(Node):
    """Publishes hand commands, calls the hand's speed/force services, and checks topics.

    It holds no arm publisher, client or command interface of any kind. That is
    the property which makes this tool safe to run against an arm somebody has
    their hands on, so keep it that way.
    """

    def __init__(
        self, command_topic: str, hand_namespace: str, use_sim_time: bool = False
    ) -> None:
        super().__init__(
            "demo_capture",
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
        )
        self._command_publisher = self.create_publisher(JointState, command_topic, 10)
        self._hand_namespace = hand_namespace
        self._speed_client = None
        self._force_client = None
        try:
            from inspire_hand_msgs.srv import SetForce, SetSpeed

            self._speed_client = self.create_client(SetSpeed, f"{hand_namespace}/set_speed")
            self._force_client = self.create_client(SetForce, f"{hand_namespace}/set_force")
            self._speed_type = SetSpeed
            self._force_type = SetForce
        except ImportError:  # pragma: no cover - inspire_hand_msgs is a hard dependency
            self.get_logger().warn("inspire_hand_msgs unavailable; speed/force will not be set")

        # Latest-value subscriptions, so a keypress can record where everything
        # is without going back to the bag. Read-only on the arm's side.
        self._latest: Dict[str, JointState] = {}
        for role, topic in (
            ("arm", SNAPSHOT_ARM_TOPIC),
            ("hand_rad", f"{hand_namespace}/joint_states"),
            ("hand_ratio", f"{hand_namespace}/state"),
            ("hand_force", f"{hand_namespace}/grip_force"),
        ):
            self.create_subscription(
                JointState,
                topic,
                (lambda message, role=role: self._latest.__setitem__(role, message)),
                qos_profile_sensor_data,
            )

    def clock_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def publish_hand(self, joint_names, open_ratio) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = list(joint_names)
        message.position = [float(value) for value in open_ratio]
        self._command_publisher.publish(message)

    @staticmethod
    def _as_dict(message):
        if message is None:
            return None
        return {
            "stamp_ns": int(message.header.stamp.sec) * 1000000000
            + int(message.header.stamp.nanosec),
            "name": list(message.name),
            "position": [float(value) for value in message.position],
            "velocity": [float(value) for value in message.velocity],
            "effort": [float(value) for value in message.effort],
        }

    def measured_open_ratios(self):
        """The hand's six driven DOF as open ratios, or ``None`` if unheard.

        Read from ``~/state`` rather than converted back from ``~/joint_states``
        radians: the driver publishes the ratio directly, and round-tripping
        through radians would have to undo the thumb abduction overlay to be
        correct.
        """
        message = self._latest.get("hand_ratio")
        if message is None:
            return None
        by_name = dict(zip(message.name, message.position))
        channels = [by_name.get(channel) for channel in ("1", "2", "3", "4", "5", "6")]
        if any(value is None for value in channels):
            return None
        return [float(value) for value in channels]

    def arm_positions(self) -> Optional[List[float]]:
        """The seven FR3 joints in fr3_joint1..7 order, or ``None`` until the arm is heard.

        Cheap enough to call on every idle pass of the key loop, which is why the
        braking-zone watch reads this rather than building a whole ``snapshot()``.
        """
        message = self._latest.get("arm")
        if message is None:
            return None
        by_name = dict(zip(message.name, message.position))
        try:
            return [float(by_name[f"fr3_joint{index}"]) for index in range(1, 8)]
        except KeyError:
            return None

    def snapshot(self) -> Dict[str, object]:
        """Everything this node can currently see, for a pose capture event."""
        return {
            "arm_joint_states": self._as_dict(self._latest.get("arm")),
            "hand_joint_states": self._as_dict(self._latest.get("hand_rad")),
            "hand_state_open_ratio": self._as_dict(self._latest.get("hand_ratio")),
            "hand_grip_force": self._as_dict(self._latest.get("hand_force")),
        }

    def _call(self, client, request, timeout: float):
        if client is None:
            return None, "client unavailable"
        if not client.wait_for_service(timeout_sec=timeout):
            return None, "service not available"
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not future.done():
            time.sleep(0.01)
        if not future.done():
            return None, "service call timed out"
        response = future.result()
        return bool(response.accepted), str(response.message)

    def set_speed(self, joint_names, speed: int, timeout: float = 2.0):
        if self._speed_client is None:
            return None, "client unavailable"
        request = self._speed_type.Request()
        request.name = list(joint_names)
        request.speed = [int(speed)] * len(joint_names)
        return self._call(self._speed_client, request, timeout)

    def set_force(self, joint_names, force: int, timeout: float = 2.0):
        if self._force_client is None:
            return None, "client unavailable"
        request = self._force_type.Request()
        request.name = list(joint_names)
        request.force = [int(force)] * len(joint_names)
        return self._call(self._force_client, request, timeout)


@dataclass
class TopicStatus:
    """Preflight result for one topic."""

    topic: str
    check: str
    ok: bool = False
    detail: str = "not checked"


def check_topics(
    node: Node,
    topics: Sequence[RecordedTopic],
    timeout: float,
    settle: float = 2.0,
) -> List[TopicStatus]:
    """Prove every topic is really there before a single sample is recorded.

    ``live`` topics have to deliver a message; a topic that merely exists in the
    graph can still be a driver that has stopped publishing, and a session which
    discovers that afterwards is a wasted demonstration. Command channels are
    checked from the other side -- somebody has to be subscribed - because no
    traffic exists on them until this tool creates it.

    ``settle`` is DDS discovery time. A node that has only just been created
    knows about nothing yet, so the publisher and subscriber counts are read
    after it, never before: reading them immediately reports a healthy system as
    an empty graph and refuses a session that should have run.
    """
    from rosidl_runtime_py.utilities import get_message

    statuses = [TopicStatus(entry.topic, entry.check) for entry in topics]
    by_topic = {status.topic: status for status in statuses}
    seen: Dict[str, threading.Event] = {}
    subscriptions = []

    deadline = time.monotonic() + settle
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    graph = dict(node.get_topic_names_and_types())

    for entry in topics:
        if entry.check != "live":
            continue
        kind = entry.msg_type or (graph.get(entry.topic) or [""])[0]
        if not kind:
            by_topic[entry.topic].detail = "not advertised, and no declared type to subscribe with"
            continue
        flag = threading.Event()
        seen[entry.topic] = flag
        subscriptions.append(
            node.create_subscription(
                get_message(kind),
                entry.topic,
                (lambda _message, flag=flag: flag.set()),
                qos_profile_sensor_data,
            )
        )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not all(flag.is_set() for flag in seen.values()):
        rclpy.spin_once(node, timeout_sec=0.1)

    for entry in topics:
        status = by_topic[entry.topic]
        if entry.check == "live":
            flag = seen.get(entry.topic)
            if flag is None:
                continue
            if flag.is_set():
                status.ok = True
                status.detail = f"live ({entry.msg_type or graph[entry.topic][0]})"
            elif entry.topic not in graph:
                status.detail = "nothing is publishing it"
            else:
                status.detail = f"advertised but silent for {timeout:g} s"
        elif entry.check == "subscriber":
            count = node.count_subscribers(entry.topic)
            status.ok = count > 0
            status.detail = (
                f"{count} subscriber(s)" if count
                else "nobody is listening; is the hand driver up?"
            )
        elif entry.check == "publisher":
            count = node.count_publishers(entry.topic)
            status.ok = count > 0
            status.detail = f"{count} publisher(s)" if count else "no publisher"
        else:  # pragma: no cover - RecordedTopic.check is a closed set
            status.detail = f"unknown check {entry.check!r}"

    for subscription in subscriptions:
        node.destroy_subscription(subscription)
    return statuses


def list_controllers(node: Node, timeout: float = 5.0):
    """Active controllers, or ``None`` if the controller manager did not answer."""
    try:
        from controller_manager_msgs.srv import ListControllers
    except ImportError:  # pragma: no cover
        return None
    client = node.create_client(ListControllers, "/controller_manager/list_controllers")
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            return None
        future = client.call_async(ListControllers.Request())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not future.done():
            rclpy.spin_once(node, timeout_sec=0.05)
        if not future.done():
            return None
        return [
            {
                "name": controller.name,
                "type": controller.type,
                "state": controller.state,
                "claimed_interfaces": list(controller.claimed_interfaces),
            }
            for controller in future.result().controller
        ]
    finally:
        node.destroy_client(client)


#: The simulation's stand-in for gravity compensation: a forward command
#: controller on the arm's effort interface, fed zeros by sim_capture.launch.py.
#:
#: It has to be *active*, not absent. An unclaimed arm is not a free one -- the
#: MuJoCo hardware holds unclaimed joints on their last desired position, so an
#: arm nobody has claimed comes up rigid. Zero commanded torque in the
#: gravity-free scene is what actually floats it.
SIM_ZERO_EFFORT_CONTROLLER = "fr3_effort_forward_command_controller"

#: Arm controllers that drive the simulated arm to a setpoint. Any of these
#: active means the arm is being held somewhere and cannot be guided.
_SIM_ARM_SETPOINT_CONTROLLERS = (
    "fr3_joint_trajectory_controller",
    "fr3_position_forward_command_controller",
)


def gravity_compensation_problem(controllers, profile: str = "hardware") -> Optional[str]:
    """Why this controller set is not safe to hand-guide, or ``None`` if it is.

    On hardware the question is whether libfranka is holding the arm at zero
    torque, and ``gravity_compensation_example_controller`` is zero by
    construction. In simulation the equivalent is a forward command controller
    on the effort interface with zeros published into it, so the controller
    being active is necessary but -- unlike the hardware case -- not by itself
    sufficient: nothing here can see what is on its command topic. That is why
    the launch publishes the zeros rather than leaving it to the operator.
    """
    if controllers is None:
        return "the controller manager did not answer /controller_manager/list_controllers"
    active = {entry["name"] for entry in controllers if entry["state"] == "active"}

    if profile == "sim":
        # A position or trajectory controller holds the arm at a setpoint, and
        # so does the replay controller -- which sim_replay.launch.py spawns,
        # and which is why that is not the launch to capture against.
        holding = sorted(
            name
            for name in active
            if name in _SIM_ARM_SETPOINT_CONTROLLERS
            or name.endswith(_ARM_COMMANDING_SUFFIXES)
        )
        if holding:
            return (
                f"{holding} is active and is driving the simulated arm to a setpoint. "
                "Launch inspire_franka_trajectory_replay sim_capture.launch.py instead"
            )
        if SIM_ZERO_EFFORT_CONTROLLER not in active:
            # Deliberately not "no controller is claiming it, so it must be
            # free". Unclaimed joints are held on their last desired position by
            # the MuJoCo hardware, so a bare arm is rigid, not floating.
            return (
                f"{SIM_ZERO_EFFORT_CONTROLLER} is not active "
                f"(active: {sorted(active) or 'none'}), so the simulated arm is held "
                "rather than floating. Launch inspire_franka_trajectory_replay "
                "sim_capture.launch.py"
            )
        return None

    if GRAVITY_COMPENSATION_CONTROLLER not in active:
        return (
            f"{GRAVITY_COMPENSATION_CONTROLLER} is not active "
            f"(active: {sorted(active) or 'none'}). "
            "Relaunch the bringup with gravity_compensation:=true"
        )
    commanding = sorted(
        name for name in active if name.endswith(_ARM_COMMANDING_SUFFIXES)
    )
    if commanding:
        return f"{commanding} is active and would command the arm while it is being guided"
    return None


def format_report(statuses: Sequence[TopicStatus]) -> str:
    width = max(len(status.topic) for status in statuses)
    lines = ["", "Session preflight", "-----------------"]
    for status in statuses:
        lines.append(
            f"{status.topic:<{width}}  {'PASS' if status.ok else 'FAIL':<4}  "
            f"[{status.check}] {status.detail}"
        )
    lines.append(f"{'Overall':<{width}}  {'PASS' if all(s.ok for s in statuses) else 'FAIL'}")
    return "\n".join(lines)


def format_capture(index: int, snapshot: Dict[str, object]) -> str:
    """One captured pose, printed so a good grasp can become a preset.

    The hand block is emitted in the shape ``hand_presets.yaml`` wants, because
    the reason to jog to a grasp and then capture it is almost always that you
    want to keep it.
    """
    lines = [f"pose capture {index}"]

    arm = snapshot.get("arm_joint_states")
    if arm:
        joints = [
            (name, value)
            for name, value in zip(arm["name"], arm["position"])
            if name.startswith("fr3_joint")
        ]
        lines.append(
            "  arm  " + "  ".join(f"{name[-1]}:{value:+.4f}" for name, value in joints)
        )
    else:
        lines.append(f"  arm  no {SNAPSHOT_ARM_TOPIC} received yet")

    ratios = snapshot.get("hand_state_open_ratio")
    if ratios:
        by_channel = dict(zip(ratios["name"], ratios["position"]))
        lines.append("  hand measured open_ratio (paste into hand_presets.yaml):")
        for dof, joint in zip(kin.DOFS, kin.DRIVEN_JOINTS):
            value = by_channel.get(dof.channel)
            if value is not None:
                lines.append(f"      {joint}: {float(value):.4f}")
    else:
        lines.append("  hand  no ~/state received yet")
    return "\n".join(lines)


#: Remaining velocity headroom, in rad/s, below which a joint is called out during capture.
#:
#: The FR3's velocity limit is not a constant. It falls toward zero as a joint nears its
#: position limit, because the arm must still be able to brake before reaching it, so a pose
#: recorded inside that falloff cannot be replayed *at any speed* - slowing down, filtering
#: and --time-scale all leave the joint angle exactly where it is. There is no way to repair
#: such a recording afterwards, which makes the demonstration itself the only place to catch
#: it, while the operator still has a hand on the arm. Hand-guided motion peaks around
#: 0.7 rad/s, so a joint with less headroom than this is already in territory a replay may
#: refuse, and the margin leaves time to rotate it back before the take is spoiled.
BRAKING_WARN_HEADROOM = 1.0


def braking_headroom(positions, headroom: float = BRAKING_WARN_HEADROOM):
    """Arm joints whose remaining velocity headroom has fallen below ``headroom``.

    Returns ``[(name, angle, allowed, blocked)]``: ``allowed`` is the tighter of the two
    directions at this configuration, and ``blocked`` marks a joint already past the point
    where the replay guard refuses outright rather than merely getting close.

    The numbers come from ``franka_trajectory_replay.limits`` - the same table the replay
    guard checks against - so this warning cannot drift away from what will later reject
    the recording.
    """
    upper = limits.upper_velocity_limits(positions)
    lower = limits.lower_velocity_limits(positions)
    tight = []
    for index in range(7):
        allowed = min(float(upper[index]), float(-lower[index]))
        if allowed < headroom:
            tight.append(
                (f"fr3_joint{index + 1}", float(positions[index]), allowed, allowed <= 0.0)
            )
    return tight


def format_braking_warning(tight) -> str:
    """The operator-facing line for :func:`braking_headroom`, or ``''`` for nothing to say."""
    lines = []
    for name, angle, allowed, blocked in tight:
        if blocked:
            lines.append(
                f"  !! {name} at {angle:+.4f} rad has run out of travel. A replay of this "
                "pose will be refused and no slow-down can rescue it - rotate the joint "
                "back before continuing."
            )
        else:
            lines.append(
                f"  !  {name} at {angle:+.4f} rad is near its limit: {allowed:.2f} rad/s of "
                "headroom left before a replay starts being refused."
            )
    return "\n".join(lines)


def key_map_banner(presets: PresetTable) -> str:
    names = [preset.name for preset in presets]
    if presets.jog is not None:
        names += [control.label for control in presets.jog.controls]
    width = max([len(name) for name in names] + [8])

    lines = ["", "Keys", "----"]
    for preset in presets:
        # First sentence only. The full rationale belongs in the YAML, where
        # there is room for it; a key map that wraps over four terminal lines
        # per entry is one nobody reads.
        summary = preset.description.split(". ")[0].rstrip(".")
        lines.append(f"  {preset.key}  {preset.name:<{width}}  {summary}")

    if presets.jog is not None:
        lines.append("")
        for control in presets.jog.controls:
            lines.append(
                f"  {control.close_key}  {control.label:<{width}}  "
                f"close by {presets.jog.step:g} open ratio (more flexion)"
            )
            lines.append(
                f"  {control.open_key}  {control.label:<{width}}  "
                f"open by {presets.jog.step:g} open ratio (less flexion)"
            )

    lines.append("")
    lines += [
        f"  {KEY_CAPTURE}  {'capture':<{width}}  mark this instant and record every joint state",
        f"  {KEY_SEGMENT}  {'segment':<{width}}  close the current segment and start the next",
        f"  {KEY_HELP}  {'help':<{width}}  reprint this map",
        f"  {KEY_QUIT}  {'quit':<{width}}  end the session and close the bag",
    ]

    pinned = presets.fixed_names()
    if pinned:
        lines.append("")
        for joint, value in pinned.items():
            lines.append(f"  held at open ratio {value:g} for the whole session: {joint}")

    lines += ["", "The arm is never commanded from here. Move it by hand."]
    return "\n".join(lines)


class KeyboardSession:
    """Raw-mode keyboard on the recorder's own terminal.

    The recorder is started without stdin precisely so that this loop owns the
    terminal (see BagRecorder), which is what lets the keys be pressed in the
    same shell the session is running in.
    """

    def __init__(self) -> None:
        self.fd = None
        self.settings = None

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("capture_demo needs a terminal on stdin for its key map")
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read_key(self, timeout: float = 0.2) -> Optional[str]:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if readable else None

    def __exit__(self, _type, _value, _traceback):
        if self.settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


class HandController:
    """The hand command in force, and the two ways the operator changes it.

    A preset replaces the whole posture; a jog key nudges one DOF. Both go out
    as a complete six-DOF command and both write the same ``hand_command``
    event, so a jogged grasp is recorded ground truth exactly as a preset is and
    extraction needs to know nothing about the difference.

    Speed and force are resent only when they change. Every service call is time
    a half-duplex RS485 bus cannot be carrying a target, and jogging is a key
    held down: resending them per press would starve the thing being jogged.
    """

    def __init__(self, node, presets: PresetTable, events: EventLog) -> None:
        self.node = node
        self.presets = presets
        self.events = events
        #: ``None`` until the first command; the hand holds whatever it held.
        self.open_ratio: Optional[Tuple[float, ...]] = None
        self.speed: Optional[int] = None
        self.force: Optional[int] = None

    def _send(self, open_ratio, speed: int, force: int, **fields) -> Tuple[float, ...]:
        # Pinned DOF are reapplied on the way out, so nothing that reaches the
        # hand can have moved one, whatever built the vector.
        open_ratio = self.presets.apply_fixed(open_ratio)

        speed_ok, speed_detail = (None, "unchanged")
        force_ok, force_detail = (None, "unchanged")
        if speed != self.speed:
            speed_ok, speed_detail = self.node.set_speed(kin.DRIVEN_JOINTS, speed)
            if speed_ok is not False:
                self.speed = speed
        if force != self.force:
            force_ok, force_detail = self.node.set_force(kin.DRIVEN_JOINTS, force)
            if force_ok is not False:
                self.force = force

        self.node.publish_hand(kin.DRIVEN_JOINTS, open_ratio)
        self.open_ratio = open_ratio
        self.events.write(
            "hand_command",
            self.node.clock_ns(),
            speed=int(speed),
            force=int(force),
            speed_accepted=speed_ok,
            speed_detail=speed_detail,
            force_accepted=force_ok,
            force_detail=force_detail,
            open_ratio=dict(zip(kin.DRIVEN_JOINTS, open_ratio)),
            open_ratio_rad=dict(
                zip(kin.DRIVEN_JOINTS, open_ratio_to_radians(open_ratio))
            ),
            fixed=self.presets.fixed_names(),
            **fields,
        )
        return open_ratio

    def apply_preset(self, preset) -> str:
        self._send(
            preset.open_ratio,
            preset.speed,
            preset.force,
            source="preset",
            preset=preset.name,
            key=preset.key,
        )
        return f"hand -> {preset.name}"

    def jog(self, control, delta: float) -> str:
        """Nudge one DOF, seeding from the measured pose if nothing was commanded yet."""
        jog = self.presets.jog
        seeded_from = "command"
        current = self.open_ratio
        if current is None:
            # Nothing has been commanded, so the only honest starting point is
            # where the hand actually is. Recorded as such: the first jogged
            # command is the one place the commanded and measured conventions
            # touch, and a reader should be able to see that.
            measured = self.node.measured_open_ratios()
            if measured is None:
                return "no hand state yet; press a preset key first"
            current = tuple(measured)
            seeded_from = "measured"

        target = list(current)
        before = target[control.dof]
        target[control.dof] = min(1.0, max(0.0, before + delta))
        applied = self._send(
            target,
            jog.speed if self.speed is None else self.speed,
            jog.force if self.force is None else self.force,
            source="jog",
            joint=control.joint,
            key=control.close_key if delta < 0 else control.open_key,
            step=float(delta),
            seeded_from=seeded_from,
        )
        after = applied[control.dof]
        if 0.0 < after < 1.0:
            edge = ""
        else:
            edge = "  (fully closed)" if after <= 0.0 else "  (fully open)"
        arrow = "close" if delta < 0 else "open"
        return f"{control.label}: {arrow} {before:.2f} -> {after:.2f}{edge}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"directory the timestamped session lands in (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--name", default="", help="short label folded into the session directory")
    parser.add_argument(
        "--note",
        default="",
        help="operator note recorded in the manifest: what this demonstration is",
    )
    parser.add_argument(
        "--bringup-command",
        default="",
        help="the bringup command line this session ran beside, recorded verbatim in "
             "the manifest. The active controller list is recorded regardless and is "
             "the machine-checked half of the same question.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(TOPIC_PROFILES),
        default="hardware",
        help="hardware: the FR3 over the FCI, recording the complete FrankaRobotState "
             "at 1 kHz - the only profile whose output is a demonstration. "
             "sim: MuJoCo through sim_replay.launch.py, which has no FrankaRobotState "
             "at all; the session and everything extracted from it are marked as a "
             "rehearsal so they cannot be mistaken for hardware data.",
    )
    parser.add_argument(
        "--full-state",
        action="store_true",
        help="also record the complete FrankaRobotState at 1 kHz. Off by default: it is "
             "~3.7 kB/message, it makes the bag roughly twenty times larger, and it is "
             "about 90% of a later extraction's runtime, while the trajectory artifact "
             "is built from /franka/joint_states either way. Turn it on for a take you "
             "intend to keep as training data - tau_ext, the external wrenches, O_T_EE, "
             "the elbow, the collision indicators and the load model exist only in a "
             "full-state bag and cannot be recovered afterwards.",
    )
    parser.add_argument(
        "--presets", default=None, help="hand preset YAML (default: the packaged one)"
    )
    parser.add_argument("--hand-namespace", default="/inspire_hand")
    parser.add_argument("--storage-id", default="sqlite3", help="rosbag2 storage plugin")
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=10.0,
        help="how long a live topic has to produce a message before the session is refused",
    )
    parser.add_argument(
        "--extra-topic",
        action="append",
        default=[],
        metavar="TOPIC",
        help="also record this topic; repeatable. Its presence is required like any other.",
    )
    parser.add_argument(
        "--allow-without-gravity-compensation",
        action="store_true",
        help="record even though the arm is not floating. For a mock or fake-hardware "
             "rehearsal of the tool itself; never for a real demonstration.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        argv = argv[: argv.index("--ros-args")]
    args = _parser().parse_args(argv)
    if args.preflight_timeout <= 0:
        _parser().error("--preflight-timeout must be positive")

    try:
        presets = load_presets(args.presets)
    except (OSError, ValueError) as exc:
        print(f"preset error: {exc}", file=sys.stderr)
        return 2

    topics = profile_topics(args.profile, args.full_state) + [
        RecordedTopic(topic, "live", "requested with --extra-topic")
        for topic in args.extra_topic
    ]
    simulated = args.profile == "sim"
    command_topic = f"{args.hand_namespace}/command"

    stamp = _utc_stamp()
    name = "".join(c for c in args.name if c.isalnum() or c in "-_") or "session"
    session_dir = Path(args.output_root).expanduser() / f"{stamp}_{name}"
    session_dir.mkdir(parents=True, exist_ok=False)
    bag_dir = session_dir / "bag"

    rclpy.init(args=None)
    try:
        # In simulation the bag's header stamps are simulation time, so the
        # event log has to be on the same clock or every marker lands in the
        # wrong place. On hardware there is no /clock and this is a no-op.
        node = CaptureNode(command_topic, args.hand_namespace, use_sim_time=simulated)
    except Exception:
        # Nothing has been recorded and nothing is open, but rclpy is up; leave
        # the graph the way this process found it.
        rclpy.shutdown()
        raise
    recorder = None
    events = None
    controllers = None
    statuses = []
    exit_code = 0
    stopped_reason = "unknown"
    try:
        statuses = check_topics(node, topics, args.preflight_timeout)
        print(format_report(statuses), flush=True)
        controllers = list_controllers(node)
        problem = gravity_compensation_problem(controllers, args.profile)
        if problem:
            print(f"\ngravity compensation: {problem}", flush=True)
        elif simulated:
            print("\narm: nothing claims the simulated arm", flush=True)
        else:
            print(
                f"\ngravity compensation: {GRAVITY_COMPENSATION_CONTROLLER} is active",
                flush=True,
            )

        if not all(status.ok for status in statuses):
            print("\nrefusing to record: not every channel is live (see the report above).",
                  file=sys.stderr)
            return 1
        if problem and not args.allow_without_gravity_compensation:
            print("\nrefusing to record: the arm is not hand-guidable.", file=sys.stderr)
            return 1

        events = EventLog(session_dir / "events.jsonl")
        recorder = BagRecorder(bag_dir, [entry.topic for entry in topics], args.storage_id)
        recorder.start()
        events.write("session_start", node.clock_ns(), bag_dir=str(bag_dir))

        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True)
        spin.start()

        print(key_map_banner(presets), flush=True)
        if simulated:
            print(
                "\nSIMULATION REHEARSAL: there is no FrankaRobotState in MuJoCo, so this"
                "\nsession records /joint_states instead and carries none of the FCI's"
                "\ntorque, wrench, pose or collision channels. The artifact is marked"
                "\n'hand_guided_sim' and is not training data.",
                flush=True,
            )
        if not simulated and not args.full_state:
            print(
                "\nLEAN BAG: recording /franka/joint_states at 1 kHz, not the full"
                "\nFrankaRobotState. Waypoints and the 15 Hz trajectory come out of this"
                "\nunchanged; tau_ext, the wrenches, O_T_EE, the elbow, the collision"
                "\nindicators and the load model are NOT recorded and cannot be recovered"
                "\nlater. Re-run with --full-state for a take meant as training data.",
                flush=True,
            )
        print(f"recording into {session_dir}", flush=True)
        print("The hand holds its current pose until a preset key is pressed.", flush=True)
        print(
            f"Watching the arm's joint limits: a joint with under {BRAKING_WARN_HEADROOM:.1f} "
            "rad/s of headroom left is called out here, because a pose recorded against a "
            "limit cannot be replayed at any speed.\n",
            flush=True,
        )

        hand = HandController(node, presets, events)
        segments = 0
        captures = 0
        # Edge-triggered, on the idle pass of the key loop: the state is printed when a
        # joint enters or leaves the warning band, never repeatedly while it sits there.
        # A demonstration nobody can replay is worth interrupting; a message every 200 ms
        # is not, and would push the pose captures the operator is reading off the screen.
        braking_state: Tuple[Tuple[str, bool], ...] = ()
        with KeyboardSession() as keyboard:
            while True:
                key = keyboard.read_key()
                if key is None:
                    positions = node.arm_positions()
                    if positions is not None:
                        tight = braking_headroom(positions)
                        state = tuple((name, blocked) for name, _, _, blocked in tight)
                        if state != braking_state:
                            if tight:
                                print(format_braking_warning(tight), flush=True)
                            elif braking_state:
                                print("  .. joints clear of their limits again", flush=True)
                            events.write(
                                "braking_zone",
                                node.clock_ns(),
                                joints=[
                                    {
                                        "joint": name,
                                        "position": angle,
                                        "allowed_velocity": allowed,
                                        "blocked": blocked,
                                    }
                                    for name, angle, allowed, blocked in tight
                                ],
                            )
                            braking_state = state
                    continue
                if key == KEY_QUIT:
                    stopped_reason = "operator pressed q"
                    break
                if key == KEY_HELP:
                    print(key_map_banner(presets), flush=True)
                    continue
                if key == KEY_SEGMENT:
                    segments += 1
                    events.write("segment", node.clock_ns(), index=segments)
                    print(f"segment boundary {segments}", flush=True)
                    continue
                if key == KEY_CAPTURE:
                    captures += 1
                    snapshot = node.snapshot()
                    events.write(
                        "pose_capture",
                        node.clock_ns(),
                        index=captures,
                        commanded_open_ratio=(
                            dict(zip(kin.DRIVEN_JOINTS, hand.open_ratio))
                            if hand.open_ratio is not None
                            else None
                        ),
                        **snapshot,
                    )
                    print(format_capture(captures, snapshot), flush=True)
                    continue
                jogged = presets.jog.control_for(key) if presets.jog is not None else None
                if jogged is not None:
                    print(hand.jog(*jogged), flush=True)
                    continue
                preset = presets.by_key(key)
                if preset is None:
                    continue
                print(hand.apply_preset(preset), flush=True)
    except KeyboardInterrupt:
        stopped_reason = "interrupted"
        print("\ninterrupted; closing the bag", flush=True)
    except (OSError, RuntimeError, TimeoutError) as exc:
        stopped_reason = f"error: {exc}"
        print(f"capture error: {exc}", file=sys.stderr)
        exit_code = 2
    finally:
        # The bag is closed on every path out of here, including a preflight
        # failure that never opened one: a half-written session is still worth
        # more than a truncated file nobody can read.
        if recorder is not None:
            recorder.stop()
        if events is not None:
            events.write("session_end", node.clock_ns(), reason=stopped_reason)
            events.close()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "hand_guided_capture_sim" if simulated else "hand_guided_capture",
            "profile": args.profile,
            # Read by extract_demo to know whether the FCI field set is there to
            # extract, and by anything downstream asking whether this session can
            # still be augmented into training data.
            "full_state": bool(args.full_state),
            "simulated": simulated,
            "session_dir": str(session_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stopped_reason": stopped_reason,
            "bag_dir": str(bag_dir),
            "bag_storage_id": args.storage_id,
            "event_log": "events.jsonl",
            "event_counts": dict(events.counts) if events is not None else {},
            "note": args.note,
            "bringup_command": args.bringup_command,
            "gravity_compensation": {
                "controller": None if simulated else GRAVITY_COMPENSATION_CONTROLLER,
                "verified": gravity_compensation_problem(controllers, args.profile) is None,
                "overridden": bool(args.allow_without_gravity_compensation),
            },
            "active_controllers": controllers,
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "topics": [
                {"topic": entry.topic, "check": entry.check, "why": entry.why} for entry in topics
            ],
            "preflight": [
                {"topic": s.topic, "check": s.check, "ok": s.ok, "detail": s.detail}
                for s in statuses
            ],
            "hand_presets": {
                "path": str(presets.source),
                "sha256": _sha256(presets.source),
                "presets": [preset.as_event() for preset in presets],
            },
            "hand_namespace": args.hand_namespace,
            "command_topic": command_topic,
        }
        _atomic_json(session_dir / "manifest.json", manifest)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f"\nsession written to {session_dir}", flush=True)
        if exit_code == 0 and recorder is not None:
            print(
                "extract it with:\n"
                f"  ros2 run inspire_franka_trajectory_replay extract_demo {session_dir}",
                flush=True,
            )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
