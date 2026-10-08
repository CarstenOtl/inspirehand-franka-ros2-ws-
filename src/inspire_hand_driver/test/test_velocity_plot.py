"""Reading a velocity log back, and the figures drawn from it.

The plot exists because the terminal's resolution is the ceiling on reading a
derived rate by eye. These tests cover the half that can be wrong silently:
parsing the log, grouping it so a standstill phase is never averaged in with a
moving one, and the figures printed beside the picture.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from inspire_hand_driver.velocity_check import LOG_HEADER, log_rows, write_log
from inspire_hand_driver.velocity_plot import Series, describe, plot, read_log

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_velocity_check import DT, INDEX, recording  # noqa: E402


@pytest.fixture
def log(tmp_path):
    """A log with a standstill phase and two legs, as a sweep writes one."""
    path = str(tmp_path / "vel.csv")
    still = recording(peak=0.0, moving=2, still=30)
    closing = recording(peak=1.8, moving=40, still=5)
    write_log(path, log_rows("standstill", still, [INDEX]), header=True)
    write_log(path, log_rows("closing", closing, [INDEX]), header=False)
    return path


def test_a_log_round_trips_through_the_reader(log):
    series = read_log(log)
    assert {s.phase for s in series} == {"standstill", "closing"}
    closing = next(s for s in series if s.phase == "closing")
    assert closing.joint.endswith("index_proximal_joint")
    assert len(closing.stamp) == len(closing.velocity) == len(closing.count)


def test_phases_are_kept_apart_rather_than_averaged_together(log):
    """A standstill averaged in with a moving leg describes neither.

    Same channel in both phases, so grouping by channel alone would merge
    them and hand every figure below a series that is half still.
    """
    series = read_log(log)
    still = next(s for s in series if s.phase == "standstill")
    closing = next(s for s in series if s.phase == "closing")
    assert still.channel == closing.channel
    assert set(still.velocity) == {0.0}, "nothing moving leaked into the standstill"
    assert max(abs(v) for v in closing.velocity) > 1.0
    # And neither series picked up the other's rows.
    assert len(still.stamp) + len(closing.stamp) == sum(
        1 for line in open(log) if line.strip()
    ) - 1


def test_each_channel_is_its_own_series(tmp_path):
    path = str(tmp_path / "two.csv")
    samples = recording(peak=1.0, moving=10, still=2)
    write_log(path, log_rows("live", samples, [0, INDEX]), header=True)
    series = read_log(path)
    assert len(series) == 2
    assert {s.channel for s in series} == {"1", "4"}


def test_the_seconds_axis_is_rebased_because_an_epoch_says_nothing(log):
    closing = next(s for s in read_log(log) if s.phase == "closing")
    assert closing.seconds[0] == 0.0
    assert closing.seconds[-1] == pytest.approx(
        closing.stamp[-1] - closing.stamp[0]
    )


def test_an_empty_file_is_refused_with_its_name(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_log(str(path))


def test_something_that_is_not_a_velocity_log_says_so(tmp_path):
    path = tmp_path / "other.csv"
    path.write_text("time,value\n1,2\n")
    with pytest.raises(ValueError, match="does not look like a velocity log"):
        read_log(str(path))


def test_a_truncated_row_names_its_line_number(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text(LOG_HEADER + "\nlive,1.0,4\n")
    with pytest.raises(ValueError, match=":2"):
        read_log(str(path))


def test_an_unparseable_number_names_its_line_number(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        LOG_HEADER + "\nlive,1.0,4,index_proximal_joint,x,0.0,0.0,0.0\n"
    )
    with pytest.raises(ValueError, match=":2"):
        read_log(str(path))


# -- the figures -------------------------------------------------------------
def test_the_figures_name_the_lag_and_both_residuals(log):
    closing = next(s for s in read_log(log) if s.phase == "closing")
    printed = "\n".join(describe(closing))
    assert "lag" in printed
    assert "with the lag taken out" in printed
    assert "counts" in printed
    assert "peak rate" in printed


def test_a_still_phase_reports_no_measurable_lag_rather_than_inventing_one(log):
    """There is no motion to align, and saying so beats reporting a number."""
    still = next(s for s in read_log(log) if s.phase == "standstill")
    printed = "\n".join(describe(still))
    assert "no lag measurable" in printed


def test_the_figures_survive_a_single_sample():
    one = Series("live", "4", "index_proximal_joint", [1.0], [500], [0.7], [0.0], [0.0])
    printed = "\n".join(describe(one))
    assert "1 samples" in printed


# -- the picture -------------------------------------------------------------
def test_a_png_is_written_for_every_phase(tmp_path, log):
    pytest.importorskip("matplotlib")
    out = str(tmp_path / "vel.png")
    assert plot(read_log(log), out, "test") == out
    assert Path(out).stat().st_size > 5000, "a real figure, not an empty canvas"


def test_plotting_needs_no_display(tmp_path, log, monkeypatch):
    """It runs in a container with no X, so the backend cannot be interactive."""
    pytest.importorskip("matplotlib")
    monkeypatch.delenv("DISPLAY", raising=False)
    out = str(tmp_path / "headless.png")
    plot(read_log(log), out, "headless")
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


def test_one_phase_alone_still_plots(tmp_path, log):
    pytest.importorskip("matplotlib")
    one = [s for s in read_log(log) if s.phase == "closing"]
    out = str(tmp_path / "one.png")
    plot(one, out, "one")
    assert Path(out).exists()
