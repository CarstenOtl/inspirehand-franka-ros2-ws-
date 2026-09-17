#!/usr/bin/env python3
"""Read the D415's on-chip self-calibration Health-Check without changing it.

Runs librealsense's on-chip calibration only for the health number it reports,
then puts the original calibration table back into the camera's working copy.
The camera's flash is never written. The camera must not be streaming
elsewhere: stop the realsense2_camera node first.

Health-Check (absolute value), from librealsense's rs_device.hpp:
    [0, 0.25)     good
    [0.25, 0.75)  can be improved
    [0.75, ...)   requires calibration
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pyrealsense2 as rs


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEEDS = {"very-fast": 0, "fast": 1, "medium": 2, "slow": 3}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serial", default=None, help="camera serial (default: the only one)")
    parser.add_argument(
        "--emitter",
        choices=("on", "off"),
        default="off",
        help="projector state; Intel recommends off on a textured scene for the D415",
    )
    parser.add_argument("--speed", choices=tuple(SPEEDS), default="medium")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "camera_health",
        help="where the pre-run calibration table is saved",
    )
    return parser


def _classify(health: float) -> str:
    value = abs(health)
    if value < 0.25:
        return "good"
    if value < 0.75:
        return "can be improved"
    return "requires calibration"


def _find_device(context: rs.context, serial: str | None) -> rs.device:
    devices = list(context.query_devices())
    if serial is not None:
        devices = [d for d in devices if d.get_info(rs.camera_info.serial_number) == serial]
    if len(devices) != 1:
        raise SystemExit(f"expected exactly one RealSense device, found {len(devices)}")
    return devices[0]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = rs.context()
    device = _find_device(context, args.serial)
    serial = device.get_info(rs.camera_info.serial_number)
    print(
        f"{device.get_info(rs.camera_info.name)} serial {serial} "
        f"firmware {device.get_info(rs.camera_info.firmware_version)}"
    )

    calibrated = device.as_auto_calibrated_device()
    original_table = calibrated.get_calibration_table()
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.backup_dir / f"d415_{serial}_calibration_table_{stamp}.json"
    backup.write_text(json.dumps({"serial": serial, "table": list(original_table)}))
    print(f"saved current calibration table to {backup}")

    config = rs.config()
    config.enable_device(serial)
    # The on-chip calibration runs on this special depth profile.
    config.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)
    pipeline = rs.pipeline(context)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)
    # Let auto-exposure settle before measuring.
    for _ in range(60):
        pipeline.wait_for_frames()

    request = json.dumps(
        {
            "calib type": 0,
            "speed": SPEEDS[args.speed],
            "scan parameter": 0,
            "adjust both sides": 0,
            "white wall mode": 0,
            "host assistance": 0,
        }
    )
    results: list[float] = []
    try:
        for attempt in range(1, args.repeats + 1):
            started = time.monotonic()
            try:
                _new_table, health = calibrated.run_on_chip_calibration(request, 30000)
            except RuntimeError as error:
                print(f"run {attempt}: failed: {error}")
                continue
            elapsed = time.monotonic() - started
            print(
                f"run {attempt}: health {health[0]:+.3f} ({_classify(health[0])}), "
                f"raw {tuple(round(h, 4) for h in health)}, {elapsed:.1f} s"
            )
            results.append(health[0])
    finally:
        # The new table is discarded: restore the original to the working copy
        # so the camera keeps its flashed calibration. Nothing is written to flash.
        calibrated.set_calibration_table(original_table)
        pipeline.stop()

    if calibrated.get_calibration_table() != original_table:
        print("WARNING: calibration table differs from the backup after restore")
        return 2
    if not results:
        print("no Health-Check result (scene may lack texture; try --emitter on)")
        return 1
    worst = max(results, key=abs)
    print(
        f"emitter {args.emitter}, speed {args.speed}: "
        f"worst |health| {abs(worst):.3f} -> {_classify(worst)}; flash untouched"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
