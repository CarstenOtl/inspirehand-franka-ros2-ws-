"""isaac_hand_relay.py's RH56 mode: the hand's register interface over the linkage."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("rclpy")
linkage = pytest.importorskip("inspire_hand_driver.rh56_linkage")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "isaac_hand_relay.py"


@pytest.fixture(scope="module")
def relay():
    spec = importlib.util.spec_from_file_location("isaac_hand_relay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registers_and_leader_angles_round_trip_along_the_linkage(relay):
    rh56 = relay.Rh56Registers()
    registers = [100.0, 100.0, 100.0, 862.0, 937.0, 94.0]
    leaders = rh56.leader_angles(registers)
    assert rh56.registers_of(leaders) == pytest.approx(registers, abs=1e-3)
    # The driven joints in register order are the table's leaders: index open at 1000.
    assert rh56.leader_angles([1000.0] * 6) == pytest.approx([0.0] * 6, abs=1e-6)
    # The map the runner uses agrees with the relay's table.
    hand_map = linkage.load_hand_map()
    assert leaders[3] == pytest.approx(float(hand_map.rad_from_register("index", 862.0)), abs=1e-6)


def _pose(rh56, registers):
    """Every simulated hand joint (workspace names) where the linkage puts it at ``registers``."""

    positions = {}
    for channel, register in zip(rh56.channels, registers):
        for joint, q in rh56.table.joints_at(channel, register).items():
            positions[linkage.WORKSPACE_JOINT[joint]] = q
    return positions


def _run(rh56, seconds, pose, dt=1.0e-3):
    for _ in range(int(round(seconds / dt))):
        rh56.step(dt, pose(rh56), {}, linkage.force_filter_alpha(dt))


def test_an_unloaded_rod_slews_at_the_rod_speed_without_stalling(relay):
    rh56 = relay.Rh56Registers()
    table = linkage.load_joint_table()
    rh56.registers = [1000.0] * 6
    rh56.targets = [0.0] * 6
    # The hand follows its targets exactly: the drives carry no load.
    _run(rh56, 0.1, lambda r: _pose(r, r.registers))
    for k, channel in enumerate(rh56.channels):
        rate = 1000.0 * linkage.NOMINAL_ROD_SPEED_MM_S / table.stroke_mm[channel]
        assert rh56.registers[k] == pytest.approx(1000.0 - 0.1 * rate, abs=rate * 1e-3)
    assert not any(rh56.stalled)
    # A full stroke takes stroke / speed (about 0.45 s), without overshooting.
    _run(rh56, 1.0, lambda r: _pose(r, r.registers))
    assert rh56.registers == pytest.approx([0.0] * 6)


def _blocked_index(relay, dt, alpha, seconds=0.5):
    """The index rigidly blocked at register 650 while commanded closed; returns the drive."""

    rh56 = relay.Rh56Registers(force_stop_n=50.0)
    rh56.registers = [1000.0, 1000.0, 1000.0, 700.0, 1000.0, 1000.0]
    rh56.targets = [1000.0, 1000.0, 1000.0, 0.0, 1000.0, 1000.0]
    blocked = _pose(rh56, [1000.0, 1000.0, 1000.0, 650.0, 1000.0, 1000.0])
    for _ in range(int(round(seconds / dt))):
        rh56.step(dt, blocked, {}, alpha)
    return rh56, blocked


def test_a_blocked_finger_stalls_where_the_training_drive_does(relay):
    """The 1 kHz relay against ForgeUltra's own step (1/120 s, alpha 0.2) on the same block.

    Against a rigid block the stiff drives load the rod far faster than the 37 ms
    low pass follows, so both overshoot the 50 N stop by the same ~40 register
    counts (~700 N filtered) before stalling; what matters is that they agree.
    """

    index = relay.Rh56Registers().channels.index("index")
    ours, blocked = _blocked_index(relay, 1.0e-3, linkage.force_filter_alpha(1.0e-3))
    training, _ = _blocked_index(relay, 1.0 / 120.0, 0.2)
    assert ours.stalled[index] and training.stalled[index]
    assert ours.registers[index] == pytest.approx(training.registers[index], abs=2.0)
    assert 600.0 < ours.registers[index] < 650.0
    # It holds there: the stall is the stop, not a slow approach.
    stalled_at = ours.registers[index]
    _run(ours, 0.5, lambda r: blocked)
    assert ours.registers[index] == pytest.approx(stalled_at, abs=1.0)
    # Without the stop the drive would keep closing to register 0 into the block.
    q = blocked["index_proximal_joint"]
    held = ours.table.joints_at("index", stalled_at)["index_joint_0"]
    unstopped = ours.table.joints_at("index", 0.0)["index_joint_0"]
    assert (held - q) < 0.1 * (unstopped - q)
    # The obstacle gives way: the load falls under 90 % of the stop and the rod closes on.
    _run(ours, 0.5, lambda r: _pose(r, r.registers))
    assert not ours.stalled[index]
    assert ours.registers[index] < stalled_at - 100.0


def test_an_opening_rod_never_stalls(relay):
    rh56 = relay.Rh56Registers(force_stop_n=50.0)
    rh56.registers = [100.0] * 6
    rh56.targets = [1000.0] * 6
    held = _pose(rh56, [100.0] * 6)
    _run(rh56, 0.05, lambda r: held)
    assert not any(rh56.stalled)
    assert all(r > 100.0 for r in rh56.registers)


def test_the_relay_drives_match_the_rh56_hand_mjcf():
    import sys

    sys.path.insert(0, str(SCRIPT.parent))
    import make_hand_mjcf as hand

    table = linkage.load_joint_table()
    for channel in table.channels:
        gains = linkage.DRIVE_GAINS[table.dof_type[channel]]
        for joint in table.joints_at(channel, 500.0):
            kp, kd, effort = hand.RH56_DRIVES[linkage.WORKSPACE_JOINT[joint]]
            assert (kp, kd) == gains
            assert effort == linkage.DRIVE_EFFORT_LIMIT_NM
