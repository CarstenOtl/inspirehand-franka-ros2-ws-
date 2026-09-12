"""The capture tool's fail-closed checks and its session bookkeeping."""

import json

import pytest

from inspire_hand_driver import kinematics as kin

from inspire_franka_trajectory_replay import capture
from inspire_franka_trajectory_replay.capture import (
    GRAVITY_COMPENSATION_CONTROLLER,
    RECORDED_TOPICS,
    EventLog,
    TopicStatus,
    format_report,
    gravity_compensation_problem,
    key_map_banner,
)
from inspire_franka_trajectory_replay.hand_presets import load_presets


def _controller(name, state="active", claimed=()):
    return {"name": name, "type": "t", "state": state, "claimed_interfaces": list(claimed)}


# -- what gets recorded ------------------------------------------------------


def test_the_full_franka_robot_state_is_recorded_not_just_joint_states():
    topics = {entry.topic for entry in RECORDED_TOPICS}
    assert "/franka_robot_state_broadcaster/robot_state" in topics
    assert "/joint_states" in topics


def test_every_hand_channel_including_the_command_one_is_recorded():
    topics = {entry.topic for entry in RECORDED_TOPICS}
    assert {
        "/inspire_hand/joint_states",
        "/inspire_hand/state",
        "/inspire_hand/grip_force",
        "/inspire_hand/command",
    } <= topics


def test_every_recorded_topic_says_why_it_is_there_and_how_it_is_proven():
    for entry in RECORDED_TOPICS:
        assert entry.check in ("live", "subscriber", "publisher")
        assert entry.why.strip()


def test_every_recorded_topic_declares_its_message_type():
    """Resolving the type from the graph instead loses a discovery race.

    A node that has only just been created has not finished DDS discovery, so
    every topic on a healthy system reads back as unadvertised and the session
    is refused for no reason. Declaring the type removes the lookup.
    """
    for entry in RECORDED_TOPICS:
        assert entry.msg_type.count("/") == 2, entry.topic


def test_the_command_channel_is_checked_by_its_subscriber_not_by_its_traffic():
    """Nothing publishes to it until a key is pressed, so waiting would deadlock."""
    command = next(e for e in RECORDED_TOPICS if e.topic.endswith("/command"))
    assert command.check == "subscriber"


def test_the_state_channels_must_actually_be_publishing():
    for topic in ("/franka_robot_state_broadcaster/robot_state", "/inspire_hand/joint_states"):
        entry = next(e for e in RECORDED_TOPICS if e.topic == topic)
        assert entry.check == "live"


# -- the hand-guidable check -------------------------------------------------


def test_gravity_compensation_active_is_the_only_accepted_state():
    assert gravity_compensation_problem([_controller(GRAVITY_COMPENSATION_CONTROLLER)]) is None


def test_an_inactive_gravity_compensation_controller_is_refused():
    problem = gravity_compensation_problem(
        [_controller(GRAVITY_COMPENSATION_CONTROLLER, state="inactive")]
    )
    assert "gravity_compensation:=true" in problem


def test_a_controller_that_would_command_the_arm_is_refused():
    problem = gravity_compensation_problem(
        [
            _controller(GRAVITY_COMPENSATION_CONTROLLER),
            _controller("trajectory_replay_controller"),
        ]
    )
    assert "would command the arm" in problem


def test_a_silent_controller_manager_is_refused_rather_than_assumed_safe():
    assert "did not answer" in gravity_compensation_problem(None)


# -- reporting ---------------------------------------------------------------


def test_the_preflight_report_names_every_topic_and_its_verdict():
    text = format_report(
        [
            TopicStatus("/a", "live", True, "live"),
            TopicStatus("/b", "subscriber", False, "nobody is listening"),
        ]
    )
    assert "/a" in text and "/b" in text
    assert "PASS" in text and "FAIL" in text
    assert text.strip().endswith("FAIL")


def test_the_key_map_lists_every_preset_jog_key_and_reserved_key():
    presets = load_presets()
    banner = key_map_banner(presets)
    for preset in presets:
        assert f"  {preset.key}  " in banner
    for key in capture.RESERVED_KEYS:
        assert f"  {key}  " in banner
    for control in presets.jog.controls:
        assert f"  {control.close_key}  " in banner
        assert f"  {control.open_key}  " in banner
    assert "never commanded" in banner


def test_the_key_map_says_which_dof_is_pinned_for_the_session():
    banner = key_map_banner(load_presets())
    assert "thumb_proximal_yaw_joint" in banner
    assert "held at open ratio" in banner


def test_the_capture_key_is_reserved_and_advertised():
    assert capture.KEY_CAPTURE in capture.RESERVED_KEYS
    assert "record every joint state" in key_map_banner(load_presets())


def test_the_pose_snapshot_reads_the_arm_at_30_hz_not_at_1_khz():
    """A Python callback on the 1 kHz robot_state would burn the executor.

    The snapshot only needs a latest value for a keypress; the 1 kHz channel is
    what the bag is for, and the snapshot's timestamp locates it there.
    """
    assert capture.SNAPSHOT_ARM_TOPIC == "/joint_states"


def test_a_captured_pose_prints_the_hand_in_the_shape_the_preset_file_wants():
    snapshot = {
        "arm_joint_states": {
            "name": [f"fr3_joint{i}" for i in range(1, 8)],
            "position": [0.1] * 7,
        },
        "hand_state_open_ratio": {
            "name": ["1", "2", "3", "4", "5", "6"],
            "position": [0.25, 0.25, 0.25, 0.4, 0.3, 0.0],
        },
    }

    text = capture.format_capture(3, snapshot)

    assert "pose capture 3" in text
    assert "index_proximal_joint: 0.4000" in text
    assert "thumb_proximal_yaw_joint: 0.0000" in text


def test_a_captured_pose_says_so_when_a_channel_has_not_arrived():
    text = capture.format_capture(1, {"arm_joint_states": None, "hand_state_open_ratio": None})
    assert "no /joint_states received yet" in text
    assert "no ~/state received yet" in text


# -- the event log -----------------------------------------------------------


def test_every_event_carries_the_ros_clock_so_it_aligns_with_the_bag(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.write("pose_capture", 1234, index=1)
    log.close()

    record = json.loads((tmp_path / "events.jsonl").read_text().strip())
    assert record["event"] == "pose_capture"
    assert record["t_ros_ns"] == 1234
    assert record["index"] == 1
    assert "t_wall_utc" in record


def test_the_event_log_is_readable_after_every_write_not_only_at_close(tmp_path):
    """A session that dies mid-demonstration still has to describe what it did."""
    log = EventLog(tmp_path / "events.jsonl")
    log.write("hand_command", 1, preset="open")

    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["preset"] == "open"
    log.close()


def test_the_event_log_counts_what_it_wrote_for_the_manifest(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.write("pose_capture", 1, index=1)
    log.write("pose_capture", 2, index=2)
    log.write("segment", 3, index=1)
    log.close()

    assert log.counts == {"pose_capture": 2, "segment": 1}


def test_a_non_finite_field_is_refused_rather_than_written_as_nan(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    with pytest.raises(ValueError):
        log.write("hand_command", 1, value=float("nan"))
    log.close()


# -- the hand command in force -----------------------------------------------


class _StubNode:
    """Enough of CaptureNode for HandController, with no ROS in the way."""

    def __init__(self, measured=None):
        self.measured = measured
        self.published = []
        self.speeds = []
        self.forces = []
        self._clock = 0

    def clock_ns(self):
        self._clock += 1
        return self._clock

    def publish_hand(self, joint_names, open_ratio):
        self.published.append(tuple(float(value) for value in open_ratio))

    def set_speed(self, joint_names, speed, timeout=2.0):
        self.speeds.append(int(speed))
        return True, "ok"

    def set_force(self, joint_names, force, timeout=2.0):
        self.forces.append(int(force))
        return True, "ok"

    def measured_open_ratios(self):
        return self.measured


def _hand_controller(tmp_path, measured=None):
    from inspire_franka_trajectory_replay.capture import EventLog, HandController

    node = _StubNode(measured)
    log = EventLog(tmp_path / "events.jsonl")
    return node, log, HandController(node, load_presets(), log)


def _events(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_a_preset_publishes_its_whole_posture_and_its_speed_and_force(tmp_path):
    node, log, hand = _hand_controller(tmp_path)

    hand.apply_preset(load_presets().by_name("preshape"))
    log.close()

    assert node.published == [load_presets().by_name("preshape").open_ratio]
    assert node.speeds == [500] and node.forces == [200]


def test_speed_and_force_are_not_resent_when_they_have_not_changed(tmp_path):
    """Every service call is time a half-duplex bus cannot carry a target."""
    node, log, hand = _hand_controller(tmp_path)
    preset = load_presets().by_name("preshape")

    hand.apply_preset(preset)
    hand.apply_preset(preset)
    log.close()

    assert node.speeds == [500] and node.forces == [200]
    assert len(node.published) == 2


def test_a_jog_key_moves_one_dof_by_one_step_and_leaves_the_rest_alone(tmp_path):
    node, log, hand = _hand_controller(tmp_path)
    presets = load_presets()
    start = presets.by_name("preshape")
    hand.apply_preset(start)
    control, delta = presets.jog.control_for("[")

    hand.jog(control, delta)
    log.close()

    before, after = node.published[0], node.published[1]
    assert after[control.dof] == pytest.approx(before[control.dof] - presets.jog.step)
    for index in range(6):
        if index != control.dof:
            assert after[index] == before[index]


def test_jogging_clamps_at_the_ends_of_the_commandable_range(tmp_path):
    node, log, hand = _hand_controller(tmp_path)
    presets = load_presets()
    hand.apply_preset(presets.by_name("open"))
    control, delta = presets.jog.control_for("]")  # already fully open

    message = hand.jog(control, delta)
    log.close()

    assert node.published[-1][control.dof] == 1.0
    assert "fully open" in message


def test_jogging_never_moves_the_pinned_thumb_abduction(tmp_path):
    node, log, hand = _hand_controller(tmp_path)
    presets = load_presets()
    pinned = kin.dof_index("thumb_proximal_yaw_joint")
    hand.apply_preset(presets.by_name("preshape"))
    for key in ("-", "=", "[", "]"):
        hand.jog(*presets.jog.control_for(key))
    log.close()

    assert all(command[pinned] == 0.0 for command in node.published)


def test_jogging_before_any_preset_seeds_from_the_measured_pose(tmp_path):
    measured = [0.3, 0.3, 0.3, 0.8, 0.7, 0.9]
    node, log, hand = _hand_controller(tmp_path, measured=measured)
    presets = load_presets()
    control, delta = presets.jog.control_for("[")

    hand.jog(control, delta)
    log.close()

    published = node.published[0]
    assert published[control.dof] == pytest.approx(measured[control.dof] - presets.jog.step)
    # The pinned DOF is still forced, even coming from a measured seed.
    assert published[kin.dof_index("thumb_proximal_yaw_joint")] == 0.0
    assert _events(tmp_path)[0]["seeded_from"] == "measured"


def test_jogging_with_no_hand_state_at_all_refuses_rather_than_guessing(tmp_path):
    node, log, hand = _hand_controller(tmp_path, measured=None)
    control, delta = load_presets().jog.control_for("[")

    message = hand.jog(control, delta)
    log.close()

    assert node.published == []
    assert "press a preset key first" in message


def test_a_jogged_command_is_logged_the_same_way_a_preset_is(tmp_path):
    """Extraction reads the action channel by event type, not by how it was made."""
    node, log, hand = _hand_controller(tmp_path)
    presets = load_presets()
    hand.apply_preset(presets.by_name("preshape"))
    hand.jog(*presets.jog.control_for("["))
    log.close()

    records = _events(tmp_path)
    assert [record["event"] for record in records] == ["hand_command", "hand_command"]
    assert [record["source"] for record in records] == ["preset", "jog"]
    for record in records:
        assert set(record["open_ratio_rad"]) == set(kin.DRIVEN_JOINTS)


# -- the simulation profile --------------------------------------------------


def test_the_two_profiles_are_hardware_and_sim():
    assert set(capture.TOPIC_PROFILES) == {"hardware", "sim"}
    assert capture.TOPIC_PROFILES["hardware"] is RECORDED_TOPICS


def test_the_sim_profile_does_not_pretend_to_have_the_fci_state():
    """FrankaRobotState is libfranka's. Faking it is the one thing not to do."""
    topics = {entry.topic for entry in capture.SIM_TOPICS}
    assert "/franka_robot_state_broadcaster/robot_state" not in topics
    assert "/franka/joint_states" not in topics
    assert "/joint_states" in topics


def test_the_sim_profile_still_records_every_hand_channel():
    """The hand is the real driver in mock mode, so it is genuinely under test."""
    topics = {entry.topic for entry in capture.SIM_TOPICS}
    assert {
        "/inspire_hand/joint_states",
        "/inspire_hand/state",
        "/inspire_hand/grip_force",
        "/inspire_hand/command",
    } <= topics


def test_the_sim_profile_records_the_simulation_clock():
    clock = next(e for e in capture.SIM_TOPICS if e.topic == "/clock")
    assert clock.check == "live"
    assert clock.msg_type == "rosgraph_msgs/msg/Clock"


def test_every_sim_topic_declares_its_type_and_a_reason():
    for entry in capture.SIM_TOPICS:
        assert entry.msg_type.count("/") == 2, entry.topic
        assert entry.why.strip()


def test_in_simulation_a_guidable_arm_is_one_under_zero_effort_control():
    assert gravity_compensation_problem(
        [_controller(capture.SIM_ZERO_EFFORT_CONTROLLER)], "sim"
    ) is None


def test_an_unclaimed_simulated_arm_is_refused_because_it_is_held_not_free():
    """The MuJoCo hardware holds unclaimed joints on their last desired position.

    This is the bug the first version of the sim profile had: it treated "no
    controller" as "floating", and the arm came up rigid.
    """
    problem = gravity_compensation_problem([_controller("joint_state_broadcaster")], "sim")
    assert "held rather than floating" in problem
    assert capture.SIM_ZERO_EFFORT_CONTROLLER in problem


def test_an_empty_controller_list_is_refused_in_simulation():
    assert gravity_compensation_problem([], "sim") is not None


def test_a_stock_sim_controller_holding_the_arm_is_refused():
    problem = gravity_compensation_problem(
        [
            _controller("fr3_joint_trajectory_controller"),
            _controller(capture.SIM_ZERO_EFFORT_CONTROLLER),
        ],
        "sim",
    )
    assert "driving the simulated arm to a setpoint" in problem
    assert "sim_capture.launch.py" in problem


def test_the_replay_controller_holding_the_simulated_arm_is_refused():
    """sim_replay.launch.py spawns it, which is why that is the wrong launch."""
    problem = gravity_compensation_problem(
        [_controller("trajectory_replay_controller")], "sim"
    )
    assert "driving the simulated arm to a setpoint" in problem


def test_simulation_does_not_require_the_hardware_gravity_compensation_controller():
    """There is no such controller in MuJoCo; the question is asked differently."""
    controllers = [_controller(capture.SIM_ZERO_EFFORT_CONTROLLER)]
    assert gravity_compensation_problem(controllers, "sim") is None
    assert gravity_compensation_problem(controllers, "hardware") is not None



def test_there_is_exactly_one_marker_key_and_it_records_everything():
    """One button. Deciding mid-demonstration which kind of mark to leave was
    the thing the second, lighter marker key cost, and it recorded strictly less."""
    assert capture.KEY_CAPTURE in capture.RESERVED_KEYS
    assert not hasattr(capture, "KEY_WAYPOINT")
    assert set(capture.RESERVED_KEYS) == {
        capture.KEY_SEGMENT,
        capture.KEY_CAPTURE,
        capture.KEY_HELP,
        capture.KEY_QUIT,
    }


# -- lean versus full-state recording ----------------------------------------


def test_a_lean_hardware_session_drops_only_the_robot_state():
    """The 1 kHz arm angles must survive; they are what the artifact is built from."""
    lean = {entry.topic for entry in capture.profile_topics("hardware", full_state=False)}
    assert "/franka_robot_state_broadcaster/robot_state" not in lean
    assert "/franka/joint_states" in lean
    assert "/inspire_hand/joint_states" in lean


def test_full_state_records_exactly_what_the_hardware_profile_declares():
    full = capture.profile_topics("hardware", full_state=True)
    assert [entry.topic for entry in full] == [entry.topic for entry in RECORDED_TOPICS]


def test_lean_is_the_only_difference_between_the_two():
    lean = {entry.topic for entry in capture.profile_topics("hardware", full_state=False)}
    full = {entry.topic for entry in capture.profile_topics("hardware", full_state=True)}
    assert full - lean == capture.FULL_STATE_ONLY_TOPICS


def test_full_state_is_meaningless_in_simulation_rather_than_different():
    """MuJoCo has no FrankaRobotState to drop, so the flag must not change sim."""
    lean = [entry.topic for entry in capture.profile_topics("sim", full_state=False)]
    full = [entry.topic for entry in capture.profile_topics("sim", full_state=True)]
    assert lean == full == [entry.topic for entry in capture.SIM_TOPICS]


def test_the_default_is_lean():
    args = capture._parser().parse_args([])
    assert args.full_state is False


# -- the joint-limit watch ---------------------------------------------------
#
# A demonstration performed with a joint against its position limit cannot be
# replayed at any speed, and nothing about a floating arm tells the operator it
# has happened. These cover the only moment it can still be corrected.


READY_POSE = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def test_a_pose_in_open_space_is_not_warned_about():
    assert capture.braking_headroom(READY_POSE) == []


def test_a_joint_approaching_its_limit_is_called_out_before_it_is_stuck():
    pose = list(READY_POSE)
    pose[4] = 2.84  # joint 5, just inside the braking falloff
    tight = capture.braking_headroom(pose)
    assert [name for name, _, _, _ in tight] == ["fr3_joint5"]
    name, angle, allowed, blocked = tight[0]
    assert angle == pytest.approx(2.84)
    assert 0.0 < allowed < capture.BRAKING_WARN_HEADROOM
    assert not blocked


def test_a_joint_out_of_travel_is_marked_blocked_not_merely_close():
    pose = list(READY_POSE)
    pose[4] = 2.8738  # what a real hand-guided session reached: against the stop
    tight = capture.braking_headroom(pose)
    assert [(name, blocked) for name, _, _, blocked in tight] == [("fr3_joint5", True)]


def test_the_warning_says_that_slowing_down_will_not_rescue_a_blocked_joint():
    pose = list(READY_POSE)
    pose[4] = 2.8738
    text = capture.format_braking_warning(capture.braking_headroom(pose))
    assert "fr3_joint5" in text
    assert "no slow-down can rescue it" in text


def test_the_warning_threshold_clears_the_speeds_hand_guiding_actually_reaches():
    # Measured peak on hand-guided sessions is ~0.7 rad/s; warning below that would
    # only fire once the recording was already spoiled.
    assert capture.BRAKING_WARN_HEADROOM > 0.7


def test_the_watch_uses_the_same_limit_table_as_the_replay_guard():
    # Not a copied constant: a drifted duplicate would warn about the wrong angle.
    from franka_trajectory_replay import limits

    pose = list(READY_POSE)
    pose[4] = 2.84
    allowed = capture.braking_headroom(pose)[0][2]
    assert allowed == pytest.approx(
        min(
            float(limits.upper_velocity_limits(pose)[4]),
            float(-limits.lower_velocity_limits(pose)[4]),
        )
    )
