"""Build a replayable trajectory of N consecutive cycles that ends at the home pose.

    ros2 run inspire_franka_trajectory_replay make_cycle_trajectory \
        apps/traj_replay/demo_trajs/traj_1 --cycles 5 \
        --output apps/traj_replay/demo_trajs/threading_5x

Why this exists rather than ``--cycle N`` five times
----------------------------------------------------
A Forge threading capture already holds its cycles back to back, and the seams
between them are continuous -- every step in the file is one the FR3 could
make, which is not true of the pickplace captures. So five cycles are simply
five consecutive slices of the recording, taken whole. The runner's ``--cycle``
selects exactly one, which is the right unit to *validate*, but not the run you
want on hardware.

What the recording does not do is come back. Each cycle ends in a
``return_to_reset`` phase that stops about 0.11 rad from the homing pose and
begins the next rollout from there, so a capture played to its end leaves the
arm short of where it started. This appends the missing piece: a cubic Hermite
ramp onto the exact homing pose.

It matches the recording's arrival velocity rather than starting from rest.
That matters -- the last recorded sample is mid-``return_to_reset`` and still
moving at about 0.2 rad/s, already in the direction of home, so a ramp that
began at rest would put a 0.2 rad/s velocity step in the middle of the stream
for the spline downstream to absorb. Matching it makes the seam C1 and lets the
motion simply carry on. The ramp ends at rest, which is what the preparation's
lead-out wants to brake from and what leaves the arm parked at home.

The result is written in the coordinated NPZ form rather than the Forge one, so
it is one continuous run by construction and the reader needs neither
``--cycle`` nor ``--segment`` to replay it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .trajectory import (
    ARM_JOINTS,
    FORGE_HAND_JOINTS,
    HAND_JOINTS,
    _reset_steps,
    resolve_trajectory,
)


def load_home(path):
    """The arm and hand blocks of a homing YAML, in radians."""
    pose = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    names = [str(name) for name in pose.get("joint_names", ())]
    positions = np.asarray(pose.get("positions", ()), dtype=float)
    if len(names) != len(positions):
        raise ValueError("homing YAML joint_names and positions have different lengths")
    columns = {name: index for index, name in enumerate(names)}
    missing = [name for name in ARM_JOINTS + HAND_JOINTS if name not in columns]
    if missing:
        raise ValueError(f"homing YAML is missing commandable joints: {missing}")
    return (
        positions[[columns[name] for name in ARM_JOINTS]],
        positions[[columns[name] for name in HAND_JOINTS]],
    )


def select_cycles(cycle_field, first, count):
    """Rows of ``count`` consecutive cycles starting at ``first``, as one block.

    The cycles must be contiguous in the file -- they are, in every capture
    this reads -- because the point is to keep the recorded transition between
    them rather than to stitch one.
    """
    available = [int(value) for value in np.unique(cycle_field)]
    wanted = list(range(first, first + count))
    unknown = [value for value in wanted if value not in available]
    if unknown:
        raise ValueError(f"cycles {unknown} are not in this recording; it has {available}")
    rows = np.flatnonzero(np.isin(cycle_field, wanted))
    if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
        raise ValueError(f"cycles {wanted} are not contiguous rows in this recording")
    return rows


def _hermite(start, velocity, end, seconds, count):
    """Cubic Hermite from ``start`` at ``velocity`` to ``end`` at rest.

    ``p(s) = h00 p0 + h10 T v0 + h01 p1``, the ``h11 T v1`` term dropping out
    because the ramp arrives stationary. Sampled over ``s`` in ``(0, 1]``: the
    ``s = 0`` sample is omitted because it would repeat the last recorded one.
    """
    s = (np.arange(1, count + 1, dtype=float) / count)[:, None]
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    return h00 * start + h10 * seconds * velocity + h01 * end


def return_to_home(arm, hand, home_arm, home_hand, seconds, dt):
    """Ramp the arm and the hand from where the recording leaves them onto home.

    The arrival velocity is a one-sided difference over the last two recorded
    samples, which is the same estimate the preparation's own lead-out uses.
    """
    count = int(round(seconds / dt))
    if count < 2:
        raise ValueError(f"--return-seconds {seconds} is under two samples at {1 / dt:.0f} Hz")
    arm_velocity = (arm[-1] - arm[-2]) / dt
    hand_velocity = (hand[-1] - hand[-2]) / dt
    return (
        _hermite(arm[-1], arm_velocity, home_arm, seconds, count),
        _hermite(hand[-1], hand_velocity, home_hand, seconds, count),
    )


def build(source, home_path, first, count, seconds, environment, joint5_cap=None):
    """Assemble the cycles and the ramp; returns time, arm, hand and a report."""
    npz_path = resolve_trajectory(str(source))
    metadata = json.loads((npz_path.parent / "metadata.json").read_text(encoding="utf-8"))
    names = [str(name) for name in metadata.get("joint_names", ())]
    home_arm, home_hand = load_home(home_path)

    with np.load(npz_path, allow_pickle=False) as data:
        if "joint_pos" not in data or "cycle" not in data:
            raise ValueError(
                "this needs a Forge capture with joint_pos and a cycle field; the "
                "pickplace captures have neither and their episodes are separated "
                "by a reset teleport, so they cannot be concatenated"
            )
        positions = np.asarray(data["joint_pos"], dtype=float)
        if not 0 <= environment < positions.shape[1]:
            raise ValueError(
                f"environment {environment} is outside [0, {positions.shape[1] - 1}]"
            )
        cycle_field = np.asarray(data["cycle"]).ravel()
        rows = select_cycles(cycle_field, first, count)
        selected = positions[rows, environment, :]
        time = np.asarray(data["sample_time_s"], dtype=float)[rows]

    arm = selected[:, [names.index(name) for name in ARM_JOINTS]]
    hand = selected[:, [names.index(name) for name in FORGE_HAND_JOINTS]]

    cap_report = {}
    if joint5_cap is not None:
        joint5_cap = float(joint5_cap)
        if not np.isfinite(joint5_cap):
            raise ValueError("--joint5-cap must be finite")
        recorded = arm[:, 4].copy()
        arm[:, 4] = np.minimum(recorded, joint5_cap)
        changed = np.flatnonzero(arm[:, 4] != recorded)
        cap_report = {
            "joint5_cap_rad": joint5_cap,
            "joint5_recorded_max_rad": float(recorded.max()),
            "joint5_capped_samples": int(len(changed)),
            "joint5_maximum_change_rad": float(np.abs(arm[:, 4] - recorded).max()),
        }

    # The recorded cycles are only worth concatenating if their seams really are
    # continuous. Checked here rather than assumed, against the same FR3
    # velocity bound the reader uses.
    breaks = _reset_steps(arm, time)
    if len(breaks):
        raise ValueError(
            f"cycles {first}..{first + count - 1} contain {len(breaks)} steps the FR3 "
            f"cannot follow, at rows {(rows[0] + breaks).tolist()}; they are not one "
            f"continuous run"
        )

    dt = float(np.median(np.diff(time)))
    ramp_arm, ramp_hand = return_to_home(arm, hand, home_arm, home_hand, seconds, dt)
    arm = np.vstack([arm, ramp_arm])
    hand = np.vstack([hand, ramp_hand])
    time = np.arange(len(arm), dtype=float) * dt

    report = {
        "cycles": list(range(first, first + count)),
        "cycle_samples": int(len(rows)),
        "ramp_samples": int(len(ramp_arm)),
        "rate_hz": 1.0 / dt,
        "start_from_home": float(np.abs(arm[0] - home_arm).max()),
        "recorded_end_from_home": float(np.abs(arm[len(rows) - 1] - home_arm).max()),
        "final_from_home": float(np.abs(arm[-1] - home_arm).max()),
        "source": str(npz_path),
        "home": str(home_path),
        "environment": environment,
        **cap_report,
    }
    return time, arm, hand, report


def write(output, time, arm, hand, report):
    """Write the coordinated NPZ and a metadata.json recording where it came from."""
    directory = Path(output).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "replay_data.npz",
        joint_pos_arm=arm,
        joint_pos_hand=hand,
        arm_joint_names=np.array(ARM_JOINTS),
        hand_joint_names=np.array(HAND_JOINTS),
        sample_time_s=time,
    )
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "data_file": "replay_data.npz",
                "generated_by": "inspire_franka_trajectory_replay.make_cycles",
                "description": (
                    f"Cycles {report['cycles'][0]}-{report['cycles'][-1]} of the source "
                    f"capture, concatenated as recorded, followed by a cubic Hermite ramp onto "
                    f"the homing pose. One continuous run: no --cycle or --segment needed."
                ),
                "units": "radians",
                "recording_frequency_hz": report["rate_hz"],
                "sample_count": int(len(time)),
                "arm_joint_names": list(ARM_JOINTS),
                "hand_joint_names": list(HAND_JOINTS),
                **report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="Forge capture directory or its replay_data.npz")
    parser.add_argument("--output", required=True, help="directory to write the result into")
    parser.add_argument("--home", default=None, help="homing YAML (default: alongside the source)")
    parser.add_argument("--cycles", type=int, default=5, help="how many cycles to run")
    parser.add_argument("--first", type=int, default=1, help="first cycle to take")
    parser.add_argument("--env", type=int, default=0, help="environment in a batched capture")
    parser.add_argument(
        "--joint5-cap",
        type=float,
        default=None,
        help="cap only fr3_joint5 waypoints at this upper value in radians",
    )
    parser.add_argument(
        "--return-seconds",
        type=float,
        default=2.0,
        help="duration of the closing ramp onto the homing pose",
    )
    args = parser.parse_args(argv)

    if args.cycles < 1:
        parser.error("--cycles must be at least 1")
    if args.return_seconds <= 0:
        parser.error("--return-seconds must be positive")
    home_path = args.home or (
        Path(args.trajectory).expanduser().resolve().parent / "homing" / "threading.yaml"
    )

    try:
        time, arm, hand, report = build(
            args.trajectory, home_path, args.first, args.cycles,
            args.return_seconds, args.env, args.joint5_cap,
        )
        directory = write(args.output, time, arm, hand, report)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}")
        return 2

    print(f"source: {report['source']}")
    print(f"home:   {report['home']}")
    print(
        f"cycles {report['cycles'][0]}-{report['cycles'][-1]}: {report['cycle_samples']} samples "
        f"at {report['rate_hz']:.0f} Hz, plus a {report['ramp_samples']}-sample return ramp"
    )
    print(
        f"starts {report['start_from_home']:.4f} rad from home; the recording ends "
        f"{report['recorded_end_from_home']:.4f} rad short, and the ramp closes that "
        f"to {report['final_from_home']:.2e} rad"
    )
    if "joint5_cap_rad" in report:
        print(
            f"joint 5: capped {report['joint5_capped_samples']} samples at "
            f"{report['joint5_cap_rad']:.4f} rad (largest change "
            f"{report['joint5_maximum_change_rad']:.6f} rad)"
        )
    print(f"{len(time)} samples over {time[-1]:.2f} s -> {directory}")
    print(
        "\nValidate it against the FR3's limits before running it:\n"
        f"  ros2 run inspire_franka_trajectory_replay replay_trajectory {directory} \\\n"
        f"    --home {report['home']} --dry-run"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
