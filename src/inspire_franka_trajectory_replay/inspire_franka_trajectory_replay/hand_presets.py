"""Named Inspire RH56 postures, loaded from ``config/hand_presets.yaml``.

The capture tool binds one keyboard key to one preset. Everything that decides
what a key does -- which joints move, to what open ratio, at what speed and
force -- lives in the YAML; this module only loads it and refuses anything the
hand could not actually be commanded to do.

The validation is deliberately strict, because a preset is the *ground truth*
hand action written into a demonstration: a typo that silently addressed the
wrong DOF would not show up as an error anywhere downstream, it would show up
as a training set whose action channel does not describe its own state channel.

Every joint name and every limit is resolved through
:mod:`inspire_hand_driver.kinematics` rather than restated here, so this file
cannot drift away from the driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from inspire_hand_driver import kinematics as kin


#: Register units for ``speed`` and ``force``; the driver clamps to this range.
REGISTER_MIN = 0
REGISTER_MAX = 1000

#: The commanding unit's range. Outside it the driver rejects the whole message.
RATIO_CLOSED = 0.0
RATIO_OPEN = 1.0

SUPPORTED_SCHEMA = 2


@dataclass(frozen=True)
class JogControl:
    """One DOF the operator can nudge from the keyboard, and its two keys.

    ``close_key`` lowers the open ratio (more flexion) and ``open_key`` raises
    it. Which way round that feels is a matter of taste, so both are data.
    """

    joint: str
    label: str
    close_key: str
    open_key: str
    #: Index into :data:`inspire_hand_driver.kinematics.DOFS`, resolved by name.
    dof: int

    def keys(self) -> Tuple[str, str]:
        return (self.close_key, self.open_key)


@dataclass(frozen=True)
class JogSettings:
    """Incremental hand control, as configured."""

    step: float
    speed: int
    force: int
    controls: Tuple[JogControl, ...]

    def control_for(self, key: str):
        """The control and the signed step that ``key`` asks for, or ``None``."""
        for control in self.controls:
            if key == control.close_key:
                return control, -self.step
            if key == control.open_key:
                return control, +self.step
        return None

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(key for control in self.controls for key in control.keys())


@dataclass(frozen=True)
class HandPreset:
    """One named posture, ready to be published to ``/inspire_hand/command``."""

    name: str
    key: str
    description: str
    speed: int
    force: int
    #: Open ratio per driven joint, in :data:`inspire_hand_driver.kinematics.DRIVEN_JOINTS` order.
    open_ratio: Tuple[float, ...]

    @property
    def joint_names(self) -> Tuple[str, ...]:
        return kin.DRIVEN_JOINTS

    def as_event(self) -> Dict[str, object]:
        """The JSON form written into a session's event log."""
        return {
            "preset": self.name,
            "key": self.key,
            "speed": self.speed,
            "force": self.force,
            "open_ratio": dict(zip(self.joint_names, self.open_ratio)),
        }


@dataclass(frozen=True)
class PresetTable:
    """Every preset in one file, indexed by key and by name."""

    presets: Tuple[HandPreset, ...]
    reserved_keys: Tuple[str, ...]
    source: Path
    #: DOF pinned for the whole session: index into ``DOFS`` -> open ratio.
    #: Neither a preset nor a jog key may move one.
    fixed: Dict[int, float] = field(default_factory=dict)
    jog: Optional[JogSettings] = None

    def fixed_names(self) -> Dict[str, float]:
        return {kin.DOFS[index].joint: value for index, value in self.fixed.items()}

    def apply_fixed(self, open_ratio: Sequence[float]) -> Tuple[float, ...]:
        """Overwrite every pinned DOF, whatever the caller believed it was.

        Called on every command the capture tool sends, so that a pinned joint
        stays pinned even if some future path forgets about it. Presets are
        validated against the same table at load time, so this is belt and
        braces rather than the only enforcement.
        """
        values = list(float(value) for value in open_ratio)
        for index, value in self.fixed.items():
            values[index] = float(value)
        return tuple(values)

    def by_key(self, key: str) -> Optional[HandPreset]:
        for preset in self.presets:
            if preset.key == key:
                return preset
        return None

    def by_name(self, name: str) -> HandPreset:
        for preset in self.presets:
            if preset.name == name:
                return preset
        raise KeyError(f"no preset named {name!r} in {self.source}")

    def __iter__(self):
        return iter(self.presets)

    def __len__(self) -> int:
        return len(self.presets)


def default_path() -> Path:
    """The packaged preset file."""
    from ament_index_python.packages import get_package_share_directory

    return Path(
        get_package_share_directory("inspire_franka_trajectory_replay")
    ) / "config" / "hand_presets.yaml"


def _register(value, label: str, where: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {label} must be a whole number, got {value!r}") from None
    if number != float(value):
        raise ValueError(f"{where}: {label} must be a whole number, got {value!r}")
    if not REGISTER_MIN <= number <= REGISTER_MAX:
        raise ValueError(
            f"{where}: {label} {number} is outside the hand's register range "
            f"[{REGISTER_MIN}, {REGISTER_MAX}]"
        )
    return number


def _open_ratios(block, where: str) -> Tuple[float, ...]:
    if not isinstance(block, Mapping) or not block:
        raise ValueError(f"{where}: open_ratio must be a non-empty mapping of joint name to ratio")

    named = {str(name) for name in block}
    driven = set(kin.DRIVEN_JOINTS)

    passive = sorted(named & set(kin.PASSIVE_JOINTS))
    if passive:
        # The driver rejects these outright. Catching it here means the operator
        # finds out when the file is loaded rather than when a key is pressed
        # mid-demonstration.
        raise ValueError(
            f"{where}: {passive} follow mechanically and cannot be commanded; "
            "name only the six driven joints"
        )
    unknown = sorted(named - driven)
    if unknown:
        raise ValueError(
            f"{where}: unknown hand joints {unknown}; "
            f"expected {list(kin.DRIVEN_JOINTS)}"
        )
    missing = [name for name in kin.DRIVEN_JOINTS if name not in named]
    if missing:
        # A partial command is legal on the wire -- unnamed DOF hold - but it
        # makes the recorded action ambiguous: the pose that results depends on
        # whatever ran before it. A preset states all six.
        raise ValueError(f"{where}: open_ratio is missing {missing}; a preset states all six DOF")

    ratios: List[float] = []
    for name in kin.DRIVEN_JOINTS:
        try:
            ratio = float(block[name])
        except (TypeError, ValueError):
            raise ValueError(
                f"{where}: {name} open ratio must be a number, got {block[name]!r}"
            ) from None
        if not RATIO_CLOSED <= ratio <= RATIO_OPEN:
            raise ValueError(
                f"{where}: {name} open ratio {ratio:g} is outside "
                f"[{RATIO_CLOSED:g}, {RATIO_OPEN:g}]"
            )
        ratios.append(ratio)
    return tuple(ratios)


def _fixed(block, where: str) -> Dict[int, float]:
    """Parse ``fixed_open_ratio`` into DOF index -> open ratio."""
    if block is None:
        return {}
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: fixed_open_ratio must be a mapping of joint name to ratio")
    fixed: Dict[int, float] = {}
    for name, value in block.items():
        name = str(name)
        if name in kin.PASSIVE_JOINTS:
            raise ValueError(
                f"{where}: fixed_open_ratio names {name!r}, which follows mechanically"
            )
        if name not in kin.DRIVEN_JOINTS:
            raise ValueError(
                f"{where}: fixed_open_ratio names unknown joint {name!r}; "
                f"expected one of {list(kin.DRIVEN_JOINTS)}"
            )
        ratio = float(value)
        if not RATIO_CLOSED <= ratio <= RATIO_OPEN:
            raise ValueError(
                f"{where}: fixed_open_ratio {name} = {ratio:g} is outside "
                f"[{RATIO_CLOSED:g}, {RATIO_OPEN:g}]"
            )
        fixed[kin.dof_index(name)] = ratio
    if len(fixed) == len(kin.DOFS):
        raise ValueError(f"{where}: fixed_open_ratio pins every DOF, leaving nothing to command")
    return fixed


def _jog(block, fixed: Dict[int, float], reserved: Sequence[str], where: str):
    """Parse the ``jog`` block, or return ``None`` if there is none."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: jog must be a mapping")

    step = float(block.get("step", 0.0))
    if not 0.0 < step <= 1.0:
        raise ValueError(f"{where}: jog step {step:g} must be within (0, 1]")

    entries = block.get("controls")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ValueError(f"{where}: jog.controls must be a non-empty list")

    controls = []
    for index, entry in enumerate(entries):
        label = f"{where}: jog.controls[{index}]"
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label} must be a mapping")
        joint = str(entry.get("joint", ""))
        if joint in kin.PASSIVE_JOINTS:
            raise ValueError(f"{label}: {joint!r} follows mechanically and cannot be commanded")
        if joint not in kin.DRIVEN_JOINTS:
            raise ValueError(
                f"{label}: unknown joint {joint!r}; expected one of {list(kin.DRIVEN_JOINTS)}"
            )
        dof = kin.dof_index(joint)
        if dof in fixed:
            # Otherwise a key would appear to work and silently do nothing,
            # because apply_fixed puts the pinned value back afterwards.
            raise ValueError(
                f"{label}: {joint!r} is pinned by fixed_open_ratio and cannot be jogged"
            )
        keys = (str(entry.get("close_key", "")), str(entry.get("open_key", "")))
        for key in keys:
            if len(key) != 1:
                raise ValueError(f"{label}: each key must be exactly one character, got {key!r}")
            if key in reserved:
                raise ValueError(f"{label}: key {key!r} is reserved by the capture tool")
        if keys[0] == keys[1]:
            raise ValueError(f"{label}: close_key and open_key are both {keys[0]!r}")
        controls.append(
            JogControl(
                joint=joint,
                label=" ".join(str(entry.get("label", joint)).split()),
                close_key=keys[0],
                open_key=keys[1],
                dof=dof,
            )
        )
    return JogSettings(
        step=step,
        speed=_register(block.get("speed", 0), "jog speed", where),
        force=_register(block.get("force", 0), "jog force", where),
        controls=tuple(controls),
    )


def _preset(entry, index: int, reserved: Sequence[str], fixed: Dict[int, float],
            where: str) -> HandPreset:
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where}: presets[{index}] must be a mapping")
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError(f"{where}: presets[{index}] has no name")
    label = f"{where}: preset {name!r}"

    key = str(entry.get("key", ""))
    if len(key) != 1:
        raise ValueError(f"{label}: key must be exactly one character, got {key!r}")
    if key in reserved:
        raise ValueError(
            f"{label}: key {key!r} is reserved by the capture tool ({list(reserved)})"
        )

    open_ratio = _open_ratios(entry.get("open_ratio"), label)
    for dof, pinned in fixed.items():
        if open_ratio[dof] != pinned:
            # Refused rather than silently overwritten: a preset that disagrees
            # with the pinned value was written under a different idea of what
            # the hand does, and the rest of it is suspect too.
            raise ValueError(
                f"{label}: {kin.DOFS[dof].joint} is pinned to {pinned:g} by "
                f"fixed_open_ratio, but this preset asks for {open_ratio[dof]:g}"
            )

    return HandPreset(
        name=name,
        key=key,
        description=" ".join(str(entry.get("description", "")).split()),
        speed=_register(entry.get("speed"), "speed", label),
        force=_register(entry.get("force"), "force", label),
        open_ratio=open_ratio,
    )


def load_presets(path=None) -> PresetTable:
    """Load and validate a preset file. ``path=None`` means the packaged one."""
    source = Path(path) if path is not None else default_path()
    where = str(source)
    document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        raise ValueError(f"{where}: the preset file must be a mapping")

    schema = document.get("schema_version")
    if schema != SUPPORTED_SCHEMA:
        raise ValueError(
            f"{where}: schema_version {schema!r} is not supported (expected {SUPPORTED_SCHEMA})"
        )

    reserved = tuple(str(key) for key in document.get("reserved_keys", ()))
    fixed = _fixed(document.get("fixed_open_ratio"), where)
    jog = _jog(document.get("jog"), fixed, reserved, where)

    entries = document.get("presets")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
        raise ValueError(f"{where}: presets must be a non-empty list")

    presets = tuple(
        _preset(entry, index, reserved, fixed, where) for index, entry in enumerate(entries)
    )

    for attribute in ("name", "key"):
        seen: Dict[str, str] = {}
        for preset in presets:
            value = getattr(preset, attribute)
            if value in seen:
                raise ValueError(
                    f"{where}: {attribute} {value!r} is used by both "
                    f"{seen[value]!r} and {preset.name!r}"
                )
            seen[value] = preset.name

    if jog is not None:
        # Presets and jog controls share one keyboard, and a collision between
        # them is the kind of thing that is only noticed when the wrong thing
        # moves during a demonstration.
        preset_keys = {preset.key: preset.name for preset in presets}
        for control in jog.controls:
            for key in control.keys():
                if key in preset_keys:
                    raise ValueError(
                        f"{where}: jog key {key!r} for {control.joint} is already "
                        f"the preset {preset_keys[key]!r}"
                    )
        jog_keys = list(jog.keys)
        duplicate = next((k for k in jog_keys if jog_keys.count(k) > 1), None)
        if duplicate is not None:
            raise ValueError(f"{where}: jog key {duplicate!r} is bound twice")

    # An `open` preset is what the session is required to start and end in, so
    # its absence is a load error rather than a surprise at shutdown.
    names = {preset.name for preset in presets}
    if "open" not in names:
        raise ValueError(f"{where}: a preset named 'open' is required; found {sorted(names)}")

    return PresetTable(
        presets=presets,
        reserved_keys=reserved,
        source=source,
        fixed=fixed,
        jog=jog,
    )


def open_ratio_to_radians(open_ratio: Sequence[float]) -> List[float]:
    """Six commanded open ratios as the six driven joint angles, in radians.

    This is the *commanded* pose in the units the replay artifacts use. It is
    not what the hand did: the driver's thumb-abduction overlay sits between the
    two, and contact stops a finger short of its target. Measured angles come
    from ``/inspire_hand/joint_states``.
    """
    if len(open_ratio) != len(kin.DOFS):
        raise ValueError(f"expected {len(kin.DOFS)} open ratios, got {len(open_ratio)}")
    return [kin.open_ratio_to_rad(index, ratio) for index, ratio in enumerate(open_ratio)]
