"""Hand-guided interventions inside a running replay: safe DAgger on the real rig.

The loop this implements, one cycle of it:

1. ``replay_trajectory --intervene`` rolls the trajectory out as usual.
2. **Enter** steps in. The trajectory clock ramps to zero, and once the
   controller reports it stopped, the replay controller is deactivated and
   ``gravity_compensation_example_controller`` takes the arm's effort
   interfaces, so the arm floats and is moved by hand. Be holding the arm
   before pressing it.
3. The operator does the work: tune the grip with the hand preset and jog keys
   against the object actually in front of them, then guide the arm by hand.
   **Enter** again records a pose -- the arm's and the hand's measured joints
   at that instant, written to ``events.jsonl`` as a ``pose_capture`` event in
   the shape ``extract_waypoints`` already reads.
4. **g** ends the supervision. The replay controller is reactivated -- it holds
   wherever the arm now is, by construction, because it initialises its
   reference from the measured joints -- and the run rejoins the trajectory at
   the current cycle's **release point** (see :mod:`.release_phase`), where the
   scripted phase opens the hand and retreats. The next cycle follows from
   there, and the operator can step in again.

**Nothing the operator did by hand is re-executed.** The arm goes from where it
was left to the release point and carries on; the recorded poses are a record,
not a motion to replay. That is deliberate for this task: after a nut has been
threaded by hand, driving the arm back through the turn with the nut already on
the bolt is not a correction, it is a collision. To replay a session's poses
as a motion of their own -- to check they are reachable, or to build training
material from them -- run ``extract_waypoints`` on the session afterwards,
which produces an ordinary artifact that goes through the ordinary runner.

What makes this safe rather than merely automatic is that nothing here moves
the arm while a person is touching it. During an intervention this module owns
no arm publisher and holds no arm command interface; the only thing it
publishes is the hand command, and the only thing it writes is the event log.
The one motion the handback makes happens after the arm is stiff again, through
the same guarded ``goto`` the ordinary runner uses, with the distance printed
per joint and a prompt in front of it.

The gravity-compensation controller is the same one
``inspire_franka_bringup ... gravity_compensation:=true`` uses for a
``capture_demo`` session, which is the hand-guiding mode already validated on
this rig. It commands zero torque; the arm is held up by libfranka's own
gravity compensation underneath, not by anything in this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import select
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from franka_trajectory_replay import limits

from inspire_hand_driver import kinematics as kin

from .capture import EventLog, HandController, format_capture
from .hand_presets import PresetTable
from .release_phase import CycleRelease
from .trajectory import ARM_JOINTS
from .waypoints import Waypoint, _from_snapshot, _snapshot_columns

#: Franka's own zero-torque controller. Declared in the replay controllers yaml
#: so the controller manager can load it without a second launch.
GRAVITY_CONTROLLER = "gravity_compensation_example_controller"

#: Both newline spellings: ``tty.setcbreak`` leaves ICRNL alone, so Enter
#: arrives as ``\\n`` on a normal terminal, but not on every one.
KEY_CAPTURE = ("\r", "\n")
KEY_RELEASE = "g"
KEY_ABORT = "q"
KEY_HELP = "?"

#: Keys this loop answers itself. A hand preset claiming one is a load error in
#: ``hand_presets.yaml`` (``reserved_keys``), so the two cannot silently clash.
RESERVED_KEYS = (KEY_RELEASE, KEY_ABORT, KEY_HELP) + KEY_CAPTURE

#: How long to let the controller manager settle after a switch before reading
#: joint states as "where the arm is". The switch is atomic, but the status and
#: joint-state streams are 50 Hz and 1 kHz publishers either side of it.
SETTLE_SECONDS = 0.3


@dataclass
class Intervention:
    """One hand-guided correction, from the pause that opened it to the handback."""

    index: int
    #: Where the rollout was when it was paused, in the artifact's own numbering.
    paused_sample: int
    paused_seconds: float
    #: The cycle the pause fell in, or ``None`` past the last one.
    cycle: Optional[CycleRelease]
    #: The release point the run will rejoin at, or ``None`` if none is left.
    rejoin: Optional[CycleRelease]
    waypoints: List[Waypoint] = field(default_factory=list)
    released: bool = False
    aborted: bool = False

    @property
    def rejoin_sample(self) -> Optional[int]:
        return None if self.rejoin is None else self.rejoin.release_sample


def pose_problems(arm: Sequence[float], velocity_margin: float = 1.0) -> List[str]:
    """Why the arm could not be *held* at this pose, if it could not.

    Checked when a waypoint is captured rather than when the correction is
    replayed, and that timing is the whole point: at capture the operator still
    has hold of a floating arm and can move the joint out: at handback they have
    let go, and a refusal there leaves a run that cannot continue.

    Two position-dependent reasons, neither of which slowing down can fix:

    * The joint is outside its own position limit.
    * The joint is hard enough against a limit that the FR3's velocity envelope
      at that angle has closed onto zero. The envelope follows the braking
      distance, so there the arm may hold the pose but may not move: it can
      neither be driven to the waypoint nor leave it, at any speed.

    The second is a narrow band - a couple of millirad of joint angle - so a
    hand-guided pose has to be pressed right up against the stop to fail it.
    """
    arm = np.asarray(arm, dtype=float)
    upper = velocity_margin * limits.upper_velocity_limits(arm)
    lower = velocity_margin * limits.lower_velocity_limits(arm)
    problems = []
    for index, joint in enumerate(ARM_JOINTS):
        if arm[index] > limits.POSITION_UPPER[index] or arm[index] < limits.POSITION_LOWER[index]:
            problems.append(
                f"{joint} is at {arm[index]:+.4f} rad, outside its position limit "
                f"[{limits.POSITION_LOWER[index]:+.4f}, {limits.POSITION_UPPER[index]:+.4f}]"
            )
        elif upper[index] <= 0.0 or lower[index] >= 0.0:
            nearest, bound = (
                (limits.POSITION_UPPER[index], "upper")
                if abs(arm[index] - limits.POSITION_UPPER[index])
                < abs(arm[index] - limits.POSITION_LOWER[index])
                else (limits.POSITION_LOWER[index], "lower")
            )
            problems.append(
                f"{joint} is at {arm[index]:+.4f} rad, "
                f"{abs(nearest - arm[index]):.4f} rad from its {bound} limit "
                f"{nearest:+.4f} rad and inside the braking zone, where the FR3 "
                f"velocity envelope is [{lower[index]:.3f}, {upper[index]:.3f}] rad/s "
                "and so has closed onto nought: the arm can hold this pose but cannot "
                "move here at any speed, so it can neither reach it nor leave it"
            )
    return problems


def gap_report(current: Sequence[float], target: Sequence[float], label: str) -> str:
    """Per-joint distance between two arm poses, worst joint first."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    order = np.argsort(-np.abs(target - current))
    parts = [
        f"{ARM_JOINTS[j]} {current[j]:+.6f} -> {target[j]:+.6f} "
        f"({abs(target[j] - current[j]):.3f} rad)"
        for j in order
        if abs(target[j] - current[j]) > 1e-4
    ]
    worst = float(np.max(np.abs(target - current)))
    if not parts:
        return f"{label}: already there (largest joint gap {worst:.2e} rad)"
    return f"{label}: {worst:.3f} rad at worst; " + "; ".join(parts)


def key_banner(presets: PresetTable) -> str:
    """The key map for an open intervention."""
    names = [preset.name for preset in presets]
    if presets.jog is not None:
        names += [control.label for control in presets.jog.controls]
    width = max([len(name) for name in names] + [8])

    lines = ["", "Intervention keys", "-----------------"]
    for preset in presets:
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
    lines += [
        "",
        f"  Enter  {'record':<{width}}  record this pose",
        f"  {KEY_RELEASE}  {'finished':<{width}}  end the supervision: the arm stiffens, then "
        f"goes to the release point and the trajectory carries on",
        f"  {KEY_ABORT}  {'abort':<{width}}  end the run here; the arm stiffens and holds",
        f"  {KEY_HELP}  {'help':<{width}}  reprint this map",
        "",
        "The arm is floating. Nothing in this mode commands it -- move it by hand.",
    ]
    return "\n".join(lines)


class InterventionSession:
    """The session directory, the event log, and the interventions in one rollout.

    One of these per ``replay_trajectory --intervene`` run. It owns the
    ``events.jsonl`` that ``extract_waypoints`` will read afterwards, so a
    session is a capture session as far as the rest of the toolchain is
    concerned -- with the rollout's own pauses and handbacks stamped into it
    alongside the operator's marks.
    """

    def __init__(
        self,
        directory: Path,
        node,
        capture_node,
        presets: PresetTable,
        log=print,
        velocity_margin: float = 1.0,
    ) -> None:
        #: The same ``prepare.velocity_margin`` the FR3 limit check will apply to
        #: the correction, so a pose accepted here is one the handback can hold.
        self.velocity_margin = float(velocity_margin)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.node = node
        self.capture_node = capture_node
        self.presets = presets
        self.log = log
        self.events = EventLog(self.directory / "events.jsonl")
        # The same hand controller a capture session uses, writing the same
        # hand_command events into the same log, so a posture found during an
        # intervention is recorded ground truth exactly as one found during a
        # standalone capture is.
        self.hand = HandController(capture_node, presets, self.events)
        self.interventions: List[Intervention] = []
        self._captures = 0

    def measured_pose(self):
        """Where the arm and hand are now: the correction replay's starting point.

        The arm comes from the replay client's own joint-state subscription and
        the hand from the capture node's, which is the same pair a waypoint is
        built from. The hand is ``None`` if its joint states have not arrived,
        and the caller then has to decide what to start the fingers from.
        """
        arm = np.asarray(self.node.current_joint_positions(), dtype=float)
        snapshot = self.capture_node.snapshot()
        hand = _snapshot_columns(snapshot.get("hand_joint_states"), kin.DRIVEN_JOINTS)
        return arm, hand

    # --- the event log ------------------------------------------------------------------

    def write_event(self, name: str, **fields) -> dict:
        return self.events.write(name, self.capture_node.clock_ns(), **fields)

    def rollout_event(self, name: str, **fields) -> dict:
        return self.write_event(name, **fields)

    # --- one intervention ---------------------------------------------------------------

    def run(
        self,
        paused_sample: int,
        paused_seconds: float,
        cycle: Optional[CycleRelease],
        rejoin: Optional[CycleRelease],
        read_key,
    ) -> Intervention:
        """Float the arm, take the operator's marks, and hand it back.

        Returns when the operator releases or aborts. The arm is stiff and
        holding in both cases: this never returns with the arm floating.
        """
        record = Intervention(
            index=len(self.interventions) + 1,
            paused_sample=int(paused_sample),
            paused_seconds=float(paused_seconds),
            cycle=cycle,
            rejoin=rejoin,
        )
        self.interventions.append(record)

        self.log("")
        self.log(f"--- intervention {record.index} ---")
        self.log(
            f"paused at sample {record.paused_sample} "
            f"({record.paused_seconds:.2f} s of the prepared stream)"
            + (f", inside cycle {cycle.cycle}" if cycle is not None else ", past the last cycle")
        )
        if rejoin is not None:
            self.log(
                f"on release the run will rejoin at cycle {rejoin.cycle}'s release "
                f"point, sample {rejoin.release_sample}"
            )
        else:
            self.log(
                "no release point is left in this trajectory; on release the run will "
                "continue from where it paused"
            )
        self.write_event(
            "intervention_open",
            index=record.index,
            paused_sample=record.paused_sample,
            paused_seconds=record.paused_seconds,
            cycle=None if cycle is None else cycle.cycle,
            rejoin_cycle=None if rejoin is None else rejoin.cycle,
            rejoin_sample=record.rejoin_sample,
        )

        self._float_arm()
        try:
            self._keyboard_loop(record, read_key)
        finally:
            # Whatever happened in there -- an abort, an exception, a keyboard
            # read that failed -- the arm does not stay floating.
            self._stiffen_arm()
        return record

    def _float_arm(self) -> None:
        self.log("")
        self.log("Handing the arm to gravity compensation. Do not let go of it before")
        self.log("the FLOATING line appears, and support it when it does.")
        stopped = self.node.ensure_active(self.log, controller=GRAVITY_CONTROLLER)
        time.sleep(SETTLE_SECONDS)
        self.write_event(
            "arm_floating", controller=GRAVITY_CONTROLLER, deactivated=list(stopped)
        )
        self.log("")
        self.log(
            "FLOATING: the arm commands zero torque and is held up by the robot's own "
            "gravity compensation. Move it by hand."
        )

    def _stiffen_arm(self) -> None:
        self.log("")
        self.log("Taking the arm back under the replay controller; let go of it now.")
        self.node.ensure_active(self.log)
        time.sleep(SETTLE_SECONDS)
        # The replay controller initialises its reference from the measured
        # joints on its first update, so it holds the pose the arm was left in
        # and there is no step to ramp out. Reported rather than assumed.
        arm, hand = self.measured_pose()
        self.write_event(
            "arm_stiff",
            controller=self.node.controller,
            measured=dict(zip(ARM_JOINTS, [float(v) for v in arm])),
            measured_hand=None if hand is None else dict(
                zip(kin.DRIVEN_JOINTS, [float(v) for v in hand])
            ),
        )
        self.log("HOLDING: the arm is under impedance control again at the pose it was left in.")

    def _keyboard_loop(self, record: Intervention, read_key) -> None:
        self.log(key_banner(self.presets))
        while True:
            key = read_key(0.2)
            if key is None:
                continue
            if key in KEY_CAPTURE:
                self._capture(record)
            elif key == KEY_RELEASE:
                # Legitimate with nothing recorded: the operator may have only
                # repositioned the workpiece by hand. Nothing is replayed from
                # these poses, so an empty intervention still hands back
                # correctly -- it just leaves no record behind, which is worth
                # saying out loud in a session whose purpose is the record.
                if not record.waypoints:
                    self.log(
                        "no poses were recorded, so this intervention leaves nothing in "
                        "the session; handing back anyway"
                    )
                record.released = True
                self.write_event(
                    "intervention_release",
                    index=record.index,
                    waypoints=len(record.waypoints),
                    rejoin_sample=record.rejoin_sample,
                )
                self.log(f"released after {len(record.waypoints)} waypoints")
                return
            elif key.lower() == KEY_ABORT:
                record.aborted = True
                self.write_event(
                    "intervention_abort", index=record.index, waypoints=len(record.waypoints)
                )
                self.log("aborting the run; the arm will stiffen and hold where it is")
                return
            elif key == KEY_HELP:
                self.log(key_banner(self.presets))
            else:
                self._hand_key(key)

    def _hand_key(self, key: str) -> None:
        preset = self.presets.by_key(key)
        if preset is not None:
            self.log(self.hand.apply_preset(preset))
            return
        if self.presets.jog is not None:
            resolved = self.presets.jog.control_for(key)
            if resolved is not None:
                control, delta = resolved
                self.log(self.hand.jog(control, delta))
                return
        self.log(f"{key!r} does nothing here; {KEY_HELP} reprints the keys")

    def _capture(self, record: Intervention) -> None:
        """Record where the arm and hand are, as a ``pose_capture`` event.

        The same event name and the same snapshot blocks ``capture_demo``
        writes, so ``extract_waypoints`` reads this session with no idea that
        it came out of a rollout rather than a standalone capture.
        """
        snapshot = self.capture_node.snapshot()
        self._captures += 1
        index = self._captures
        event = self.write_event(
            "pose_capture",
            index=index,
            intervention=record.index,
            **snapshot,
        )
        waypoint = _waypoint_from_event(event, index)
        if waypoint is None:
            self._captures -= 1
            self.log(
                "capture refused: the arm or hand joint states have not arrived yet, so "
                "there is nothing to record. Check the joint state topics and try again."
            )
            return
        problems = pose_problems(waypoint.arm, self.velocity_margin)
        if problems:
            # The event stays in the log -- it is a true record of a key that was
            # pressed -- but the waypoint does not join the correction, because
            # the correction would then be refused at the handback, with the
            # operator no longer holding the arm.
            self.write_event(
                "pose_capture_refused", index=index, intervention=record.index,
                problems=problems,
            )
            self.log("")
            self.log(f"waypoint {index} NOT recorded: the arm cannot hold this pose.")
            for problem in problems:
                self.log(f"  {problem}")
            self.log("Move that joint away from its limit and press Enter again.")
            return
        record.waypoints.append(waypoint)
        self.log("")
        self.log(format_capture(index, snapshot))
        self.log(f"  waypoint {len(record.waypoints)} of intervention {record.index}")

    # --- session bookkeeping ------------------------------------------------------------

    def write_manifest(self, extra: Optional[Dict[str, object]] = None) -> Path:
        path = self.directory / "manifest.json"
        document = {
            "tool": "replay_trajectory --intervene",
            "written_at": datetime.now(timezone.utc).isoformat(),
            "events": str(self.events.path.name),
            "event_counts": dict(self.events.counts),
            "release_key": KEY_RELEASE,
            "gravity_controller": GRAVITY_CONTROLLER,
            "presets": [preset.as_event() for preset in self.presets],
            "interventions": [
                {
                    "index": item.index,
                    "paused_sample": item.paused_sample,
                    "paused_seconds": item.paused_seconds,
                    "cycle": None if item.cycle is None else item.cycle.cycle,
                    "rejoin_cycle": None if item.rejoin is None else item.rejoin.cycle,
                    "rejoin_sample": item.rejoin_sample,
                    "waypoints": len(item.waypoints),
                    "released": item.released,
                    "aborted": item.aborted,
                }
                for item in self.interventions
            ],
            **(extra or {}),
        }
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def close(self) -> None:
        self.events.close()


def _waypoint_from_event(event: dict, index: int) -> Optional[Waypoint]:
    """A waypoint out of a ``pose_capture`` event, or ``None`` if it is incomplete.

    Deliberately built from the written event rather than from the snapshot it
    came out of, through the same ``_from_snapshot`` that ``extract_waypoints``
    will use on this file afterwards. What the handback replays now and what
    the artifact says later are then the same numbers by construction.
    """
    return _from_snapshot({**event, "index": index})


def read_key_from_stdin(timeout: float = 0.2) -> Optional[str]:
    """One key from a terminal already in cbreak mode, or ``None`` on timeout."""
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if readable else None


def session_directory(root, note: Optional[str] = None) -> Path:
    """``logs/dagger/<UTC stamp>[_<note>]``, in the shape capture sessions use."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "" if not note else "_" + "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in note.strip().replace(" ", "-")
    )
    return Path(root).expanduser() / f"{stamp}{slug}"
