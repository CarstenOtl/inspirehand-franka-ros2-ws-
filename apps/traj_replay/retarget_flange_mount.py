#!/usr/bin/env python3
"""Retarget a legacy coordinated FR3 replay after changing hand flange clocking.

This compatibility tool is for ``traj_1``-convention source material only. The
current physical and unified-simulation mount is clocked another +90 degrees
around link8 Z relative to that legacy Forge replay model. The recorded world
pose of the hand is preserved by adding the same angle to fr3_joint7.

``traj_2`` and all later captures are developed with the hardware joint
orientation already correct. Do not pass them through this tool; replay their
joint-7 values unchanged.

This tool writes a new artifact and a matching homing YAML.  The source files
are never modified and an existing output directory is never overwritten.
Generated legacy artifacts are candidates, not automatically hardware
baselines. The validated legacy baseline is
demo_trajs/threading_cycle1_flange180; validate any derived selection with
--dry-run and on hardware before promoting it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml


ARM_KEYS = ("joint_pos_arm", "arm", "q_arm", "q")
JOINT = "fr3_joint7"
JOINT7_LIMITS_RAD = (-3.0508, 3.0508)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(path: Path) -> tuple[Path, dict]:
    source = path.expanduser().resolve()
    metadata_path = source / "metadata.json" if source.is_dir() else source.parent / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    if source.is_dir():
        source = source / metadata.get("data_file", "replay_data.npz")
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise ValueError(f"trajectory is not an NPZ file: {source}")
    return source, metadata


def _retarget_home(path: Path, offset_rad: float, source_ref: str) -> dict:
    home = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = [str(name) for name in home.get("joint_names", ())]
    positions = [float(value) for value in home.get("positions", ())]
    if len(names) != len(positions):
        raise ValueError("homing YAML joint_names and positions have different lengths")
    if names.count(JOINT) != 1:
        raise ValueError(f"homing YAML must contain {JOINT} exactly once")
    index = names.index(JOINT)
    positions[index] += offset_rad
    if not JOINT7_LIMITS_RAD[0] <= positions[index] <= JOINT7_LIMITS_RAD[1]:
        raise ValueError(
            f"retargeted home {JOINT}={positions[index]:.6f} rad leaves its "
            f"FR3 position range {JOINT7_LIMITS_RAD}"
        )
    home["name"] = f"{home.get('name', 'home')}_flange180"
    home["description"] = (
        f"{home.get('description', '').rstrip()} Retargeted for the 180-degree "
        "link8-Z hand mount."
    ).strip()
    home["positions"] = positions
    home["retargeting"] = {
        "joint": JOINT,
        "offset_rad": offset_rad,
        "offset_deg": float(np.rad2deg(offset_rad)),
        "reason": "preserve the recorded hand world pose after the +90-degree flange clocking change",
        "source_home": source_ref,
    }
    return home


def retarget(source_arg: Path, home_path: Path, output: Path, offset_rad: float) -> Path:
    source, source_metadata = _resolve_source(source_arg)
    home_path = home_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists; refusing to overwrite it: {output}")
    if not np.isfinite(offset_rad):
        raise ValueError("joint-7 offset must be finite")

    with np.load(source, allow_pickle=False) as data:
        payload = {key: np.array(data[key], copy=True) for key in data.files}
    arm_key = next((key for key in ARM_KEYS if key in payload), None)
    if arm_key is None:
        raise ValueError(f"coordinated NPZ needs one of these arm fields: {ARM_KEYS}")
    arm = payload[arm_key]
    if arm.ndim != 2 or arm.shape[1] != 7 or not np.issubdtype(arm.dtype, np.floating):
        raise ValueError(f"{arm_key} must be a floating-point array with shape (N, 7), got {arm.shape}")
    names = (
        [str(name) for name in payload["arm_joint_names"]]
        if "arm_joint_names" in payload
        else [f"fr3_joint{index}" for index in range(1, 8)]
    )
    if names.count(JOINT) != 1:
        raise ValueError(f"arm_joint_names must contain {JOINT} exactly once")
    joint_index = names.index(JOINT)
    arm[:, joint_index] += offset_rad
    if not np.all(np.isfinite(arm)):
        raise ValueError("retargeted arm trajectory contains non-finite values")
    q7_min = float(arm[:, joint_index].min())
    q7_max = float(arm[:, joint_index].max())
    if q7_min < JOINT7_LIMITS_RAD[0] or q7_max > JOINT7_LIMITS_RAD[1]:
        raise ValueError(
            f"retargeted {JOINT} range [{q7_min:.6f}, {q7_max:.6f}] leaves its "
            f"FR3 position range {JOINT7_LIMITS_RAD}"
        )
    payload[arm_key] = arm
    source_ref = os.path.relpath(source, output)
    home_ref = os.path.relpath(home_path, output)
    home = _retarget_home(home_path, offset_rad, home_ref)
    metadata = dict(source_metadata)
    metadata.update(
        {
            "schema_version": max(1, int(source_metadata.get("schema_version", 1))),
            "data_file": "replay_data.npz",
            "generated_by": "apps/traj_replay/retarget_flange_mount.py",
            "description": (
                f"{source_metadata.get('description', '').rstrip()} Retargeted for the "
                "180-degree link8-Z hand mount."
            ).strip(),
            "home": "homing.yaml",
            "retargeting": {
                "joint": JOINT,
                "offset_rad": offset_rad,
                "offset_deg": float(np.rad2deg(offset_rad)),
                "target_mount": "180-degree rotation around link8 Z",
                "purpose": "preserve the recorded hand world pose",
                "source_data": source_ref,
                "source_home": home_ref,
                "source_data_sha256": _sha256(source),
                "joint7_range_rad": [q7_min, q7_max],
                "fr3_joint7_limits_rad": list(JOINT7_LIMITS_RAD),
            },
        }
    )

    output.mkdir(parents=True)
    np.savez(output / "replay_data.npz", **payload)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (output / "homing.yaml").write_text(
        yaml.safe_dump(home, sort_keys=False), encoding="utf-8"
    )
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="coordinated NPZ or its trajectory directory")
    parser.add_argument("--home", required=True, help="matching source homing YAML")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument(
        "--joint7-offset-deg",
        type=float,
        required=True,
        help="explicit legacy fr3_joint7 compensation in degrees (+90 for traj_1 sources)",
    )
    args = parser.parse_args(argv)
    try:
        output = retarget(
            Path(args.trajectory),
            Path(args.home),
            Path(args.output),
            float(np.deg2rad(args.joint7_offset_deg)),
        )
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {output / 'replay_data.npz'}")
    print(f"matching home: {output / 'homing.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
