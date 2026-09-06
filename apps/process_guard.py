#!/usr/bin/env python3
"""Run a command and tear down its process group if this guard's parent dies."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Optional, Sequence


_PR_SET_PDEATHSIG = 1
_requested_signal: Optional[int] = None


def _request_shutdown(signum: int, _frame: object) -> None:
    global _requested_signal
    _requested_signal = signum


def _arm_parent_death_signal() -> bool:
    """Ask Linux to send SIGTERM if the process which started us disappears."""
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        print(f"process_guard: prctl failed: {os.strerror(error)}", file=sys.stderr)
        return False
    # Close the race where the parent exits just before prctl takes effect.
    return os.getppid() == parent


def _process_group_is_running(process_group: int) -> bool:
    proc = Path("/proc")
    if proc.is_dir():
        for stat_path in proc.glob("[0-9]*/stat"):
            try:
                fields = stat_path.read_text().rsplit(")", 1)[1].split()
                state, group = fields[0], int(fields[2])
            except (IndexError, OSError, ValueError):
                continue
            if group == process_group and state != "Z":
                return True
        return False

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    process_group = process.pid
    if _process_group_is_running(process_group):
        try:
            # Let ros2 launch deliver one orderly SIGINT to each child. Sending
            # SIGINT to the whole group here would make Python nodes receive it
            # once from us and again from launch during their cleanup.
            os.kill(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 8.0
        while (
            _process_group_is_running(process_group)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)

    if _process_group_is_running(process_group):
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 3.0
        while (
            _process_group_is_running(process_group)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)

    if _process_group_is_running(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def main(command: Sequence[str]) -> int:
    if not command:
        print("usage: process_guard.py COMMAND [ARG ...]", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)
    if not _arm_parent_death_signal():
        return 1

    process = subprocess.Popen(command, start_new_session=True)
    returncode = 1
    try:
        while _requested_signal is None:
            try:
                returncode = process.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                continue
        if _requested_signal is not None:
            returncode = 128 + _requested_signal
    finally:
        # Also clean descendants if the command's group leader exits first.
        _stop_process_group(process)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
