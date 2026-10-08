"""Plot a ``--log`` CSV from the velocity check, properly.

The terminal trace is bounded by what characters can say: one column is one
sample and one row is a fortieth of the scale. This reads the CSV that the
same tool writes and renders what that resolution hides -- the quantisation
staircase in the ANGLE readings, the filter's lag against an unfiltered
reference, and the residual that is left once the lag is taken out.

    ros2 run inspire_hand_driver inspire_hand_velocity_check --sweep \\
        --channel 4 --log ~/vel.csv
    ros2 run inspire_hand_driver inspire_hand_velocity_plot ~/vel.csv

Deliberately not a live plot: it reads a file, so nothing about it needs a
running hand, a ROS graph or a display, and the figures it prints are computed
by the same functions that print them on the terminal. ``rqt_plot`` and
PlotJuggler are the live-plot answers, and neither is in this image.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, NamedTuple, Sequence, Tuple

from .velocity_check import (
    LOG_HEADER,
    best_lag_samples,
    interval_stats,
    rms,
    shifted_rms,
)


class Series(NamedTuple):
    """Everything one CSV phase recorded for one DOF."""

    phase: str
    channel: str
    joint: str
    stamp: List[float]
    count: List[int]
    rad: List[float]
    velocity: List[float]
    reference: List[float]

    @property
    def key(self) -> str:
        return f"{self.phase} / channel {self.channel}"

    @property
    def seconds(self) -> List[float]:
        """Stamps rebased to zero, because an epoch on an axis says nothing."""
        start = self.stamp[0] if self.stamp else 0.0
        return [t - start for t in self.stamp]


def read_log(path: str) -> List[Series]:
    """Load a ``--log`` CSV, one :class:`Series` per phase and channel.

    Grouped rather than returned as rows: every figure below is per-leg, and a
    standstill phase averaged together with a moving one is a number that
    describes neither.
    """
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    expected = LOG_HEADER.split(",")
    if lines[0].split(",") != expected:
        raise ValueError(
            f"{path} does not look like a velocity log; its first line should be\n"
            f"  {LOG_HEADER}"
        )
    grouped: Dict[Tuple[str, str], Series] = {}
    for number, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        if len(fields) != len(expected):
            raise ValueError(f"{path}:{number} has {len(fields)} fields, not {len(expected)}")
        phase, stamp, channel, joint, count, rad, velocity, reference = fields
        series = grouped.setdefault(
            (phase, channel),
            Series(phase, channel, joint, [], [], [], [], []),
        )
        try:
            series.stamp.append(float(stamp))
            series.count.append(int(count))
            series.rad.append(float(rad))
            series.velocity.append(float(velocity))
            series.reference.append(float(reference))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    return list(grouped.values())


def describe(series: Series) -> List[str]:
    """The figures worth printing beside the plot, same ones the sweep reports."""
    mean_dt, jitter, _, max_dt = interval_stats(series.stamp)
    lag = best_lag_samples(series.reference, series.velocity)
    lines = [
        f"{series.key} ({series.joint}), {len(series.stamp)} samples over "
        f"{series.seconds[-1] if series.seconds else 0.0:.2f} s",
        f"  interval   {mean_dt * 1e3:.1f} ms +-{jitter * 1e3:.1f} "
        f"(max {max_dt * 1e3:.1f})",
        f"  counts     {min(series.count)}..{max(series.count)}, "
        f"{len(set(series.count))} distinct",
        f"  peak rate  {max(series.velocity, key=abs, default=0.0):+.3f} reported, "
        f"{max(series.reference, key=abs, default=0.0):+.3f} from the positions",
    ]
    residual = [v - r for v, r in zip(series.velocity, series.reference)]
    if lag is None:
        lines.append(
            f"  residual   {rms(residual):.4f} rad/s; no lag measurable, which is "
            f"what a trace clipped flat looks like"
        )
    else:
        lines.append(
            f"  lag        {lag * mean_dt * 1e3:.1f} ms behind the positions"
        )
        lines.append(
            f"  residual   {rms(residual):.4f} rad/s raw, "
            f"{shifted_rms(series.reference, series.velocity, lag):.4f} with the "
            f"lag taken out"
        )
    return lines


def plot(series_list: Sequence[Series], out: str, title: str) -> str:
    """Render one column of panels per phase, and return the path written."""
    import matplotlib

    # Chosen before pyplot is imported: this runs in a container with no
    # display, and the default interactive backend would fail on import.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = len(series_list)
    figure, axes = plt.subplots(
        3,
        columns,
        figsize=(max(7.0, 6.0 * columns), 10.0),
        squeeze=False,
        sharex="col",
    )
    for column, series in enumerate(series_list):
        seconds = series.seconds
        mean_dt, _, _, _ = interval_stats(series.stamp)
        lag = best_lag_samples(series.reference, series.velocity)

        rate = axes[0][column]
        rate.plot(seconds, series.reference, linewidth=2.4, alpha=0.45,
                  label="centred difference of the positions")
        rate.plot(seconds, series.velocity, linewidth=1.1,
                  label="reported JointState.velocity")
        rate.set_title(series.key, fontsize=10)
        rate.set_ylabel("rad/s")
        rate.legend(fontsize=7, loc="best")
        rate.grid(alpha=0.25)

        # The staircase. Drawn as steps rather than a line because that is
        # what the register does: it holds a count until it flips to the next.
        counts = axes[1][column]
        counts.step(seconds, series.count, where="post", linewidth=1.0)
        counts.set_ylabel("ANGLE count")
        counts.grid(alpha=0.25)
        span = max(series.count) - min(series.count)
        counts.set_title(
            f"{span} counts of travel"
            + (
                f", {span / max(1e-9, seconds[-1]):.0f} counts/s"
                if seconds and seconds[-1] > 0
                else ""
            ),
            fontsize=9,
        )

        residual = axes[2][column]
        raw = [v - r for v, r in zip(series.velocity, series.reference)]
        residual.plot(seconds, raw, linewidth=0.9, label="reported - positions")
        if lag is not None:
            # The same residual with the lag slid out, which is the one that
            # reflects how noisy the reading is rather than how late.
            aligned = []
            for index in range(len(series.reference)):
                position = index + lag
                low = int(math.floor(position))
                if low < 0 or low + 1 >= len(series.velocity):
                    aligned.append(float("nan"))
                    continue
                fraction = position - low
                value = series.velocity[low] + fraction * (
                    series.velocity[low + 1] - series.velocity[low]
                )
                aligned.append(value - series.reference[index])
            residual.plot(
                seconds, aligned, linewidth=0.9,
                label=f"with {lag * mean_dt * 1e3:.0f} ms of lag taken out",
            )
        residual.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
        residual.set_ylabel("rad/s")
        residual.set_xlabel("seconds")
        residual.legend(fontsize=7, loc="best")
        residual.grid(alpha=0.25)

    figure.suptitle(title, fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out, dpi=130)
    plt.close(figure)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", help="CSV written by inspire_hand_velocity_check --log.")
    parser.add_argument(
        "--out", default=None,
        help="Where to write the PNG. Defaults to the CSV's path with .png.",
    )
    parser.add_argument(
        "--phase", action="append", default=None,
        help="Only plot these phases (standstill, closing, opening, live). "
             "Repeatable.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        series_list = read_log(args.log)
    except (OSError, ValueError) as exc:
        print(f"{exc}")
        return 2
    if args.phase:
        wanted = set(args.phase)
        series_list = [s for s in series_list if s.phase in wanted]
    if not series_list:
        print("nothing to plot")
        return 2

    for series in series_list:
        for line in describe(series):
            print(line)
        print("")

    out = args.out or os.path.splitext(args.log)[0] + ".png"
    try:
        plot(series_list, out, os.path.basename(args.log))
    except ImportError:
        print(
            "matplotlib is not installed, so only the figures above are available.\n"
            "  apt-get install -y python3-matplotlib"
        )
        return 2
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
