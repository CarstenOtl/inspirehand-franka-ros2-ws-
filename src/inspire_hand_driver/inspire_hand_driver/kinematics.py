"""Mapping between the hand's six register channels and the URDF's twelve joints.

The RH56 has **six actuators but twelve revolute joints**. Each finger's
``*_intermediate`` joint is driven off its ``*_proximal`` joint by a four-bar
linkage inside the finger, and the thumb carries two such followers. The
firmware exposes only the six driven DOF; the followers are mechanical.

The URDF this repo vendors (:mod:`inspire_hand_description`) does record the
coupling, as ``<mimic>`` tags. But a ``<mimic>`` is a hint to whoever reads the
description, not a mechanism: nothing propagates it at runtime unless something
is written to do so. So the same numbers are restated here, and every consumer
reads them from this table:

* the driver, to publish a complete ``JointState`` that ``robot_state_publisher``
  can turn into a full TF tree (without the followers, the fingertips have no
  frames at all);
* ``inspire_franka_sim/scripts/make_hand_mjcf.py``, to emit the equivalent
  MuJoCo ``<equality>`` constraints -- MuJoCo's URDF importer drops ``<mimic>``
  silently.

``inspire_hand_description/test/test_mimic_matches_driver.py`` fails if this
table and the URDF ever disagree, so the duplication cannot rot.

Units, and the two conventions that meet here
---------------------------------------------
The hand speaks **open ratios**: ``1.0`` fully open, ``0.0`` fully closed,
mapping onto the ``ANGLE`` registers (0 closed .. 1000 open). The URDF speaks
**radians**, and every driven joint has its *lower* limit at the open pose and
its *upper* limit at the closed pose. So::

    angle_rad = lower + (1 - open_ratio) * (upper - lower)

Where the coupling ratios come from
-----------------------------------
Straight out of the vendored URDF's ``<mimic>`` tags -- they are upstream's
numbers, not a re-derivation. Note that they are *not* what you get by assuming
a follower reaches its limit exactly when its driver does: that assumption gives
1.09214 for the fingers, where upstream says 1.06399. At the closed pose the
follower therefore stops at 1.519 rad, about 2.4 deg short of its own 1.56 rad
limit. The limits are simply a little loose; the multiplier is what the linkage
does.

This is a *linear* approximation of a four-bar, so it is good enough for TF,
RViz and collision checking, but it is not a substitute for a calibrated linkage
model if you need fingertip accuracy better than a few millimetres mid-travel.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence, Tuple

from .protocol import CHANNEL_IDS, DOF_ORDER


class Coupling(NamedTuple):
    """A follower joint: ``angle = multiplier * driver_angle + offset`` (radians).

    ``lower``/``upper`` are the follower's own URDF limits, and the result is
    clamped to them. That clamp is not cosmetic: upstream rounds the thumb's
    multipliers to 1.334 and 0.667 (rather than 4/3 and 2/3), which overshoots
    ``thumb_intermediate_joint`` and ``thumb_distal_joint`` by 4e-4 rad at the
    closed pose. Unclamped, TF would publish a joint fractionally outside the
    limit the description declares, while MuJoCo would clamp it -- so the two
    would disagree about where the thumb is.
    """

    joint: str
    multiplier: float
    offset: float
    lower: float
    upper: float


class Dof(NamedTuple):
    """One of the hand's six actuated degrees of freedom."""

    channel: str
    """Register channel id, "1".."6"."""

    name: str
    """Human name, matching :data:`inspire_hand_driver.protocol.DOF_ORDER`."""

    joint: str
    """The URDF joint this DOF drives."""

    lower: float
    """Joint angle at open_ratio 1.0 (fully open), radians."""

    upper: float
    """Joint angle at open_ratio 0.0 (fully closed), radians."""

    couplings: Tuple[Coupling, ...]
    """Passive joints that follow this one."""


def _finger(channel: str, name: str, prefix: str) -> Dof:
    """One of the four fingers: proximal driven, intermediate following."""
    return Dof(
        channel=channel,
        name=name,
        joint=f"{prefix}_proximal_joint",
        lower=0.0,
        upper=1.47,
        couplings=(
            Coupling(f"{prefix}_intermediate_joint", 1.06399, -0.04545, -0.04545, 1.56),
        ),
    )


#: The six DOF, in register order. Index i of any 6-register block is DOF[i].
DOFS: Tuple[Dof, ...] = (
    _finger("1", "pinky", "pinky"),
    _finger("2", "ring", "ring"),
    _finger("3", "middle", "middle"),
    _finger("4", "index", "index"),
    Dof(
        channel="5",
        name="thumb_bend",
        joint="thumb_proximal_pitch_joint",
        lower=0.0,
        upper=0.6,
        couplings=(
            Coupling("thumb_intermediate_joint", 1.334, 0.0, 0.0, 0.8),
            Coupling("thumb_distal_joint", 0.667, 0.0, 0.0, 0.4),
        ),
    ),
    Dof(
        channel="6",
        name="thumb_rotation",
        # Adduction/abduction of the thumb across the palm. It has no follower:
        # the yaw axis drives a single rigid body.
        joint="thumb_proximal_yaw_joint",
        lower=0.0,
        upper=1.308,
        couplings=(),
    ),
)

assert tuple(d.channel for d in DOFS) == CHANNEL_IDS, "DOFS must follow the register order"
assert tuple(d.name for d in DOFS) == DOF_ORDER, "DOFS must follow the register order"

#: The six joints the hand actually drives, in register order.
DRIVEN_JOINTS: Tuple[str, ...] = tuple(d.joint for d in DOFS)

#: The six joints that follow mechanically, in the order they are published.
PASSIVE_JOINTS: Tuple[str, ...] = tuple(
    c.joint for d in DOFS for c in d.couplings
)

#: Every revolute joint in the hand, driven first. This is the order the driver
#: publishes joint states in.
ALL_JOINTS: Tuple[str, ...] = DRIVEN_JOINTS + PASSIVE_JOINTS

_BY_CHANNEL: Dict[str, int] = {d.channel: i for i, d in enumerate(DOFS)}
_BY_JOINT: Dict[str, int] = {d.joint: i for i, d in enumerate(DOFS)}


def dof_index(name: str) -> int:
    """Resolve a channel id or a driven joint name to its index in :data:`DOFS`.

    Raises :class:`KeyError` for anything else -- including the *passive* joint
    names, which are deliberately not addressable: commanding a follower
    independently of its driver is not something the hardware can do.
    """
    key = str(name)
    if key in _BY_CHANNEL:
        return _BY_CHANNEL[key]
    return _BY_JOINT[key]


def open_ratio_to_rad(index: int, open_ratio: float) -> float:
    """Convert one DOF's open ratio to its driven joint angle, in radians."""
    dof = DOFS[index]
    clamped = max(0.0, min(1.0, float(open_ratio)))
    return dof.lower + (1.0 - clamped) * (dof.upper - dof.lower)


def rad_to_open_ratio(index: int, radians: float) -> float:
    """Convert one DOF's driven joint angle to an open ratio."""
    dof = DOFS[index]
    span = dof.upper - dof.lower
    if span == 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (float(radians) - dof.lower) / span))


def follower_angle(coupling: Coupling, driver_radians: float) -> float:
    """Apply one coupling, clamped to the follower's own limits."""
    value = coupling.multiplier * float(driver_radians) + coupling.offset
    return max(coupling.lower, min(coupling.upper, value))


def joint_positions(open_ratios: Sequence[float]) -> List[float]:
    """Expand six open ratios into twelve joint angles, ordered as :data:`ALL_JOINTS`.

    The driven joints come first, then every follower with its coupling applied.
    """
    if len(open_ratios) != len(DOFS):
        raise ValueError(f"expected {len(DOFS)} open ratios, got {len(open_ratios)}")
    driven = [open_ratio_to_rad(i, r) for i, r in enumerate(open_ratios)]
    passive = [
        follower_angle(c, driven[i])
        for i, dof in enumerate(DOFS)
        for c in dof.couplings
    ]
    return driven + passive
