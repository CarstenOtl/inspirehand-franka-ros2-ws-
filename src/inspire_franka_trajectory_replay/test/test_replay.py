import threading
import time

import numpy as np
import pytest

from inspire_franka_trajectory_replay.replay import _stream_hand


class RecordingNode:
    def __init__(self):
        self.commands = []

    def command_hand(self, positions):
        self.commands.append(np.asarray(positions))

    def get_clock(self):
        return self

    def now(self):
        return self

    @property
    def nanoseconds(self):
        return time.monotonic_ns()


def test_hand_stream_starts_only_after_arm_acceptance():
    node = RecordingNode()
    started = threading.Event()
    stopped = threading.Event()
    errors = []
    thread = threading.Thread(
        target=_stream_hand,
        args=(node, started, stopped, np.array([0.0]), np.zeros((1, 6)), errors),
    )
    thread.start()
    assert not node.commands

    started.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert len(node.commands) == 1
    assert not errors


def test_hand_stream_stops_while_waiting_for_arm_acceptance():
    node = RecordingNode()
    started = threading.Event()
    stopped = threading.Event()
    stopped.set()
    errors = []

    _stream_hand(node, started, stopped, np.array([0.0]), np.zeros((1, 6)), errors)

    assert not node.commands
    assert not errors


def test_command_hand_converts_radians_to_open_ratios():
    """The one place the runner's radians meet the driver's ratios.

    Everything above ``command_hand`` is in radians: the Forge trajectories,
    the homing YAMLs (which also hold the FR3's seven joints, so they cannot be
    ratios), and the tracking comparison against ``joint_states``. The driver
    commands in open ratios, running the opposite way. Getting this backwards
    would command a fully open hand where the recording wanted a closed one, so
    it is pinned at both ends of travel and in between.
    """
    from inspire_franka_trajectory_replay.replay import (
        HAND_LOWER,
        HAND_UPPER,
        CoordinatedReplayClient,
    )
    from inspire_franka_trajectory_replay.trajectory import HAND_JOINTS

    published = []

    class FakePublisher:
        def publish(self, message):
            published.append(message)

    from builtin_interfaces.msg import Time

    class FakeClock:
        def now(self):
            return self

        def to_msg(self):
            return Time()

    node = CoordinatedReplayClient.__new__(CoordinatedReplayClient)
    node._hand_publisher = FakePublisher()
    node.get_clock = FakeClock

    # The open pose is 0 rad and ratio 1.0; the closed pose is each joint's
    # upper limit and ratio 0.0. Opposite directions, which is the point.
    node.command_hand(HAND_LOWER)
    assert list(published[-1].position) == [1.0] * len(HAND_JOINTS)

    node.command_hand(HAND_UPPER)
    assert list(published[-1].position) == [0.0] * len(HAND_JOINTS)

    node.command_hand(0.5 * HAND_UPPER)
    assert list(published[-1].position) == pytest.approx([0.5] * len(HAND_JOINTS))

    assert list(published[-1].name) == list(HAND_JOINTS)


def test_command_hand_output_is_inside_the_drivers_accepted_range():
    """Every sample of the checked-in recording must survive the range check.

    The driver rejects a ratio outside [0, 1] instead of clamping it, so a
    conversion that overshot even slightly would now abort a replay mid-stream
    rather than quietly saturating.
    """
    from inspire_franka_trajectory_replay.replay import HAND_DOF
    from inspire_hand_driver import kinematics as kin

    for index in HAND_DOF:
        dof = kin.DOFS[index]
        for radians in np.linspace(dof.lower, dof.upper, 50):
            ratio = kin.rad_to_open_ratio(index, radians)
            assert 0.0 <= ratio <= 1.0
