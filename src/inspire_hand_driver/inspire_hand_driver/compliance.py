"""A synthesized fingertip spring for a hand that only takes position commands.

The FR3's gravity-compensation controller is *passive*: it commands zero torque
and the arm's own backdrivable joints do the rest. Nothing here can work that
way. The RH56 takes a position setpoint and nothing else, so "push the
fingertip and the finger gives" has to be manufactured -- measure the fingertip
force, and retreat the setpoint towards open in proportion to it. That is an
admittance law, and it is one-directional by construction: ``FORCE_ACT`` reads
compression of the fingertip pad, so pushing *into* the pad is the only input
it has. A finger yields open and never closes on its own.

The law, per DOF, in register counts (0 closed .. 1000 open):

    target_yield = clamp((force - zero - deadband) * counts_per_gram, 0, max_yield)

where ``zero`` is that channel's reading with nothing touching it, captured by
:meth:`FingerSpring.tare` -- not simply nought, because the sensor's zero moves
after a heavy push; see "The zero moves" below.

The applied yield slews towards that target -- at ``yield_rate`` while it grows,
at ``return_rate`` while it shrinks. :mod:`~inspire_hand_driver.driver_node`
adds it to whatever angle was last commanded, so the commanded pose stays the
spring's rest position: release the fingertip and the finger closes back onto
the object at ``return_rate``.

Why the slew limits are not optional
------------------------------------
The hand's own actuator lag is about 0.17 s and the driver polls at 50 Hz, so
this loop is two orders of magnitude slower than the arm's. A stiff gain on top
of that much lag gives a finger that buzzes against your hand instead of
yielding to it, and the rate limits are what keep the loop's own dynamics
slower than the plant's. Raise ``counts_per_gram`` until it feels responsive
and stop well before it feels alive.

``return_rate = 0`` turns the spring into a clutch: a finger that has yielded
stays where it was pushed to rather than closing back, until something commands
it again. Releasing compliance still ramps the yield out -- at ``yield_rate``,
so a finger never snaps shut on the hand that was holding it.

Two interactions to know before tuning
--------------------------------------
* ``FORCE_SET`` (the driver's ``startup_force``, 500 g by default) does **not**
  cap what ``FORCE_ACT`` reports. It was written here that it did, on the
  reasoning that the firmware stops a DOF when the threshold is reached; that
  reasoning does not survive contact with the hand. The threshold governs the
  *closing motion* -- a finger driving shut gives up at it -- and says nothing
  about what the sensor reports when an external load is put on a finger that
  is already holding station. Measured on this rig with ``FORCE_SET`` at 500 g,
  a firm push on a held finger reads 1400-1900 g. So the whole range is
  available to the law, and a gentle pinch preset does not starve it.
* The sensor is at the fingertip *pad*. Push a finger anywhere else -- the
  middle phalanx, the side of the tip -- and this reads nothing at all. That
  contact ends up at the stall guard instead, which is the right place for it.
* **The zero moves.** A fingertip that has been pushed hard does not return to
  the zero it had before. Measured on this rig: the index pad rested at -11 g,
  was pushed to 2511 g, and then sat at +219 g -- fully open, motor off,
  touching nothing -- and stayed there, with no decay over a minute, while its
  four neighbours held -6 to -11 g. Against a fixed deadband that is a standing
  183 g of phantom push, so the finger holds a permanent partial yield and
  never comes home. :meth:`FingerSpring.tare` is the answer: force is measured
  against a zero captured with nothing touching the hand, the driver takes one
  when compliant mode is engaged, and ``~/tare_force`` takes another on demand.
* The reading is **signed**, and an unloaded fingertip sits below its own zero.
  :func:`inspire_hand_driver.protocol.to_signed16` is what makes that true at
  the register boundary; before it existed a resting -90 g arrived here as
  65446 g, which is above every threshold in this file and would have driven
  each compliant finger straight to ``max_yield`` the moment the mode was
  switched on.

The gains below are starting points chosen to be too soft rather than too
stiff. The resting force numbers are measured; the gains are not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

#: Grams of fingertip force ignored before the finger gives at all. Covers the
#: sensor's noise floor and the preload of whatever the hand is already
#: holding, so that a grip does not slowly open itself. Measured on this rig:
#: the six fingertips read -2, -12, -26, -11, +1 and -90 g with nothing
#: touching them, so anything above zero clears the resting offset and 80 g
#: leaves room for a grip's own preload.
DEFAULT_DEADBAND = 80.0
#: Register counts of opening per gram above the deadband. 0.6 puts a firm
#: 400 g push at 192 counts, 19 % of travel, and reaches
#: :data:`DEFAULT_MAX_YIELD` at 500 g above the deadband -- so a hard shove
#: saturates rather than opening further, which is the trade this gain makes:
#: responsive to a light bump, capped against a heavy one.
DEFAULT_COUNTS_PER_GRAM = 0.6
#: The most a finger will ever give, in counts. Bounds the damage a stuck-high
#: force reading can do: the finger opens this far and no further.
DEFAULT_MAX_YIELD = 300.0
#: Counts per second, yielding. 400 = 40 % of travel per second.
DEFAULT_YIELD_RATE = 400.0
#: Counts per second, closing back once the push stops. Deliberately slower
#: than yielding: giving way fast is helpful, re-gripping fast is not.
DEFAULT_RETURN_RATE = 200.0


@dataclass(frozen=True)
class SpringGains:
    """One tuning of the spring, shared by all compliant DOF."""

    deadband: float = DEFAULT_DEADBAND
    counts_per_gram: float = DEFAULT_COUNTS_PER_GRAM
    max_yield: float = DEFAULT_MAX_YIELD
    yield_rate: float = DEFAULT_YIELD_RATE
    return_rate: float = DEFAULT_RETURN_RATE

    def __post_init__(self) -> None:
        negative = [
            name
            for name in ("deadband", "counts_per_gram", "max_yield", "yield_rate", "return_rate")
            if float(getattr(self, name)) < 0.0
        ]
        if negative:
            raise ValueError(f"compliance gains must not be negative: {', '.join(negative)}")
        # A zero rise rate cannot be slewed towards anything, so the finger
        # would never give however hard it is pushed. Silently doing nothing is
        # the worst answer to "why is compliance not working".
        if float(self.yield_rate) <= 0.0:
            raise ValueError("yield_rate must be positive, or no finger can ever give")

    def target_yield(self, force: float) -> float:
        """Counts of opening this force calls for, before any rate limiting."""
        over = float(force) - self.deadband
        if over <= 0.0:
            return 0.0
        return min(self.max_yield, over * self.counts_per_gram)


class FingerSpring:
    """Per-DOF opening offsets that track measured fingertip force.

    Stateful across cycles because the rate limits are: what the finger does
    this tick depends on where the last one left it.
    """

    def __init__(
        self, gains: Optional[SpringGains] = None, channels: Optional[Sequence[bool]] = None
    ) -> None:
        self._gains = gains if gains is not None else SpringGains()
        #: Which DOF participate. Thumb rotation carries no fingertip pad, so
        #: the driver leaves it out by default -- see its ``compliance_channels``.
        self._channels = [True] * 6 if channels is None else [bool(c) for c in channels]
        if len(self._channels) != 6:
            raise ValueError(f"expected 6 channel flags, got {len(self._channels)}")
        self._yield = [0.0] * 6
        #: Per DOF, the reading that means "nothing is touching this finger".
        #: Not nought: the sensor's zero shifts after a heavy push and stays
        #: shifted. See :meth:`tare`.
        self._zero = [0.0] * 6
        self._engaged = False

    # -- configuration -----------------------------------------------------
    @property
    def gains(self) -> SpringGains:
        return self._gains

    def retune(self, gains: SpringGains) -> None:
        """Swap the gains mid-flight, for tuning against a real fingertip.

        The yields already applied are kept: re-deriving them from the new
        gains would step every compliant finger at once, which is exactly the
        motion a person with a hand on the fingertips should not be given.
        """
        self._gains = gains

    @property
    def channels(self) -> List[bool]:
        return list(self._channels)

    def set_channels(self, channels: Sequence[bool]) -> None:
        if len(channels) != 6:
            raise ValueError(f"expected 6 channel flags, got {len(channels)}")
        self._channels = [bool(c) for c in channels]

    # -- state -------------------------------------------------------------
    @property
    def engaged(self) -> bool:
        """Whether force is currently being turned into yield."""
        return self._engaged

    @property
    def active(self) -> bool:
        """Whether the spring still has work to do this cycle.

        True while engaged, and still true after release until the yield has
        ramped back out -- the driver keeps polling force and rewriting targets
        for exactly this long.
        """
        return self._engaged or any(self._yield)

    @property
    def zeros(self) -> List[float]:
        """The per-DOF reading currently being treated as no contact."""
        return list(self._zero)

    def tare(self, forces: Sequence[float]) -> List[float]:
        """Take the current readings as "nothing is touching me", and return them.

        Must be called with the hand actually untouched, which is the one thing
        this cannot check. Everything downstream measures against it, so a tare
        taken while someone is holding a fingertip makes that push the new
        nothing, and the finger will then give only to a *harder* push than the
        one already being applied.
        """
        forces = list(forces[:6])
        self._zero = [float(f) for f in forces] + [0.0] * (6 - len(forces))
        return self.zeros

    @property
    def counts(self) -> List[int]:
        """The applied yield per DOF, in register counts towards open."""
        return [int(round(y)) for y in self._yield]

    def engage(self) -> None:
        self._engaged = True

    def release(self) -> None:
        """Stop yielding to force, and ramp out whatever yield is applied."""
        self._engaged = False

    def reset(self) -> None:
        """Drop the yield without ramping it out, staying in whatever mode.

        For when the applied yield has stopped meaning anything -- the hand
        rebooted, or went silent long enough to be declared lost -- and
        carrying it onto the next write would displace targets by an offset
        derived from a force reading against a pose the hand no longer holds.
        Engagement is deliberately untouched: the operator asked for compliant
        mode and is probably still standing there with a hand on the
        fingertips, so the mode is re-derived from the next force reading
        rather than silently abandoned.
        """
        self._yield = [0.0] * 6

    # -- the law -----------------------------------------------------------
    def update(self, forces: Sequence[float], dt: float) -> List[int]:
        """Advance every DOF's yield by one cycle and return the new counts."""
        dt = max(0.0, float(dt))
        for index in range(6):
            target = 0.0
            if self._engaged and self._channels[index] and index < len(forces):
                target = self._gains.target_yield(forces[index] - self._zero[index])
            self._yield[index] = self._slew(self._yield[index], target, dt)
        return self.counts

    def _slew(self, current: float, target: float, dt: float) -> float:
        if target > current:
            return min(target, current + self._gains.yield_rate * dt)
        if target == current:
            return current
        if self._engaged and self._gains.return_rate <= 0.0:
            # A clutch: what the push opened stays open until commanded.
            return current
        # Releasing always has a rate even when engaged holding does not, so
        # that leaving compliant mode cannot leave a finger stuck open.
        rate = self._gains.return_rate if self._gains.return_rate > 0.0 else self._gains.yield_rate
        return max(target, current - rate * dt)


def channel_mask(indices: Sequence[int]) -> List[bool]:
    """Turn a list of DOF indices into the six flags :class:`FingerSpring` takes."""
    mask = [False] * 6
    for index in indices:
        mask[int(index)] = True
    return mask


def describe(gains: SpringGains) -> str:
    """One line of gains, for the startup log and for tuning readback."""
    return (
        f"deadband={gains.deadband:g}g gain={gains.counts_per_gram:g}counts/g "
        f"max_yield={gains.max_yield:g} rates={gains.yield_rate:g}/"
        f"{gains.return_rate:g} counts/s"
    )


__all__ = [
    "FingerSpring",
    "SpringGains",
    "channel_mask",
    "describe",
]
