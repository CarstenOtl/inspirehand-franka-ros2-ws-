"""Where each cycle lets go: the release-phase flag a trajectory carries.

A threading rollout is three things in a row, every cycle: the policy drives
the nut onto the bolt (``policy``), then a scripted phase opens the hand and
retreats (``follow_waypoints``), then the arm goes back to the reset pose
(``return_to_reset``). The middle one is the release -- in ``traj_3_multi`` the
index finger goes from 0.70 rad of flexion to 0.06 rad over its first sixteen
samples, with the arm still parked at the bolt, which is what letting go looks
like in joint space.

That point is the one an intervention has to hand back to. When the operator
takes the arm at the bolt and threads the nut by hand, the part of the cycle
that was going to do it is spent; what still has to happen is the release and
the retreat, and the cycle after that. So the runner rejoins the stream at the
release sample of the cycle it was interrupted in, and the next cycle follows
normally.

The flag therefore has to survive into the artifact that gets replayed. A
Forge source capture carries ``replay_phase`` and ``cycle`` per sample, but
:mod:`~inspire_franka_trajectory_replay.make_cycles` writes the coordinated NPZ
form, which is arm and hand columns and nothing else. So ``make_cycles``
resolves the phases to one sample index per cycle and writes them into
``metadata.json`` as ``cycle_index``, and this module is where both ends of
that agree on the shape.

Sample indices here are always indices into the artifact's own samples, and
times are seconds in its own 15 Hz clock. Turning one into a time on the
controller's prepared stream needs the preparation itself, because that is what
holds the lead-in and the final time scale -- :func:`prepared_time_for_sample`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

#: The ``replay_phase`` value that releases the object and retreats.
#:
#: Named for what the recording calls it rather than for what it does: the
#: Forge rollout drives this phase from a scripted waypoint list instead of
#: from the policy, and ``threading_release_motion: manual`` in the source
#: metadata is the same fact stated from the other side.
RELEASE_PHASE = "follow_waypoints"

#: Per-sample fields a Forge capture needs before release points can be found.
SOURCE_FIELDS = ("cycle", "replay_phase")


@dataclass(frozen=True)
class CycleRelease:
    """One cycle of an artifact, and the sample its release phase begins on."""

    cycle: int
    start_sample: int
    end_sample: int
    release_sample: int

    def contains(self, sample: int) -> bool:
        return self.start_sample <= int(sample) <= self.end_sample

    def as_metadata(self, rate_hz: float) -> dict:
        return {
            "cycle": int(self.cycle),
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "release_sample": int(self.release_sample),
            "release_time_s": float(self.release_sample) / float(rate_hz),
        }


def releases_from_fields(
    cycle_field: Sequence,
    phase_field: Sequence,
    release_phase: str = RELEASE_PHASE,
    offset: int = 0,
) -> List[CycleRelease]:
    """Resolve per-sample ``cycle``/``replay_phase`` fields to one point per cycle.

    ``offset`` shifts every index, so a caller that sliced the source before
    calling this can report indices in its own output's numbering.

    A cycle whose rows are not contiguous, or which never enters the release
    phase, is an error rather than a cycle with no release: both mean the file
    is not the shape the intervention flow assumes, and silently dropping the
    cycle would leave a rollout that cannot hand back in the middle of it.
    """
    cycles = np.asarray(cycle_field).ravel()
    phases = np.asarray(phase_field).ravel().astype(str)
    if len(cycles) != len(phases):
        raise ValueError(
            f"cycle has {len(cycles)} samples and replay_phase has {len(phases)}"
        )
    if len(cycles) == 0:
        raise ValueError("cannot find release points in an empty recording")

    releases: List[CycleRelease] = []
    for value in [int(v) for v in np.unique(cycles)]:
        rows = np.flatnonzero(cycles == value)
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise ValueError(f"cycle {value} does not occupy contiguous samples")
        in_release = np.flatnonzero(phases[rows] == release_phase)
        if not len(in_release):
            raise ValueError(
                f"cycle {value} never enters the {release_phase!r} phase; its phases "
                f"are {sorted(set(phases[rows].tolist()))}"
            )
        releases.append(
            CycleRelease(
                cycle=value,
                start_sample=int(rows[0]) + int(offset),
                end_sample=int(rows[-1]) + int(offset),
                release_sample=int(rows[in_release[0]]) + int(offset),
            )
        )
    return releases


class ReleaseIndex:
    """The release points of a replayable artifact, in its own sample numbering."""

    def __init__(
        self,
        releases: Sequence[CycleRelease],
        release_phase: str = RELEASE_PHASE,
        rate_hz: float = 15.0,
        source: Optional[Path] = None,
    ) -> None:
        self.releases = sorted(releases, key=lambda entry: entry.start_sample)
        self.release_phase = str(release_phase)
        self.rate_hz = float(rate_hz)
        self.source = source

    def __len__(self) -> int:
        return len(self.releases)

    def cycle_for_sample(self, sample: int) -> Optional[CycleRelease]:
        """The cycle a sample falls in, or ``None`` past the last one.

        Past the last cycle is the return-to-home ramp that ``make_cycles``
        appends, which belongs to no cycle and has nothing to release.
        """
        for entry in self.releases:
            if entry.contains(sample):
                return entry
        return None

    def next_release_at_or_after(self, sample: int) -> Optional[CycleRelease]:
        """The release the run will reach next, counting one it is already past.

        A pause *after* the current cycle's release sample has already let go,
        so rejoining there would open the hand a second time on nothing and
        repeat the retreat. The next cycle's release is the one to aim at.
        """
        for entry in self.releases:
            if entry.release_sample >= int(sample):
                return entry
        return None

    def shifted(self, by: int) -> "ReleaseIndex":
        """The same index renumbered for an artifact sliced at sample ``by``.

        Cycles wholly before the slice are dropped; a cycle the slice starts
        inside keeps its release sample and gets a clamped start.
        """
        kept = []
        for entry in self.releases:
            if entry.end_sample < int(by):
                continue
            kept.append(
                CycleRelease(
                    cycle=entry.cycle,
                    start_sample=max(0, entry.start_sample - int(by)),
                    end_sample=entry.end_sample - int(by),
                    release_sample=entry.release_sample - int(by),
                )
            )
        return ReleaseIndex(kept, self.release_phase, self.rate_hz, self.source)

    def describe(self) -> str:
        parts = [
            f"cycle {entry.cycle}: samples {entry.start_sample}-{entry.end_sample}, "
            f"release at {entry.release_sample} "
            f"({entry.release_sample / self.rate_hz:.2f} s)"
            for entry in self.releases
        ]
        return "\n".join(parts)

    def as_metadata(self) -> List[dict]:
        return [entry.as_metadata(self.rate_hz) for entry in self.releases]


def from_metadata(metadata: dict, source: Optional[Path] = None) -> Optional[ReleaseIndex]:
    """The release index an artifact's ``metadata.json`` carries, or ``None``.

    ``None`` means the artifact predates the flag or was not built from a
    capture that had phases, not that it has no release: callers that need one
    say so with the command that regenerates the artifact.
    """
    entries = metadata.get("cycle_index")
    if not entries:
        return None
    releases = []
    for entry in entries:
        try:
            releases.append(
                CycleRelease(
                    cycle=int(entry["cycle"]),
                    start_sample=int(entry["start_sample"]),
                    end_sample=int(entry["end_sample"]),
                    release_sample=int(entry["release_sample"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed cycle_index entry {entry!r}: {exc}") from exc
    rate = float(
        metadata.get("rate_hz") or metadata.get("recording_frequency_hz") or 15.0
    )
    return ReleaseIndex(
        releases, metadata.get("release_phase", RELEASE_PHASE), rate, source
    )


def load(source) -> Optional[ReleaseIndex]:
    """The release index of a trajectory directory or its ``replay_data.npz``."""
    path = Path(source)
    directory = path.parent if path.suffix == ".npz" else path
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return from_metadata(metadata, metadata_path)


# --- placing a sample on the controller's clock ------------------------------------------


def prepared_time_for_sample(prepared, trajectory, sample: int) -> float:
    """When the prepared stream reaches an artifact sample, in its own seconds.

    The same mapping :func:`replay._hand_stream` uses to put the hand on the
    arm's clock: preparation prepends a hold and a lead-in ramp, so the capture
    starts at ``capture_start_index``, and every source interval is stretched by
    the final time scale -- the one the FR3 limit check settled on, which is not
    necessarily the one that was asked for.
    """
    index = int(np.clip(sample, 0, len(trajectory.time) - 1))
    capture_start = float(prepared.t[prepared.params["capture_start_index"]])
    return capture_start + float(trajectory.time[index]) * float(
        prepared.params["time_scale"]
    )


#: Times within a nanosecond of a sample boundary count as being on it.
#:
#: Undoing ``capture_start + t * scale`` does not land exactly back on ``t`` in
#: floating point, and without this a time built from sample n inverts to n - 1.
#: A nanosecond is seven orders of magnitude below the controller's own
#: millisecond period, so it cannot move a genuinely mid-interval time.
BOUNDARY_TOLERANCE_S = 1e-9


def sample_for_prepared_time(prepared, trajectory, seconds: float) -> int:
    """Which artifact sample the prepared stream is at, inverting the mapping.

    The sample at or before the given time. Clamped at both ends: the hold and
    lead-in before the capture report sample 0, and the lead-out and final hold
    report the last sample.
    """
    capture_start = float(prepared.t[prepared.params["capture_start_index"]])
    scale = float(prepared.params["time_scale"])
    source_seconds = (float(seconds) - capture_start) / scale
    index = int(
        np.searchsorted(
            trajectory.time, source_seconds + BOUNDARY_TOLERANCE_S, side="right"
        )
    ) - 1
    return int(np.clip(index, 0, len(trajectory.time) - 1))
