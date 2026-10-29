"""The torque-candidate comparison reads what the hardware runner records."""

import json

import numpy as np
import pytest

from utils import compare_torque_candidates as ct


def _run(tmp_path, steps=6):
    data_dir = tmp_path / "hardware-x" / "data"
    data_dir.mkdir(parents=True)
    ramp = np.linspace(0.0, 1.0, steps)[:, None] * np.array([87.0, 0, 0, 0, 12.0, 0, 0])
    np.savez(
        data_dir / ct.DATA_FILENAME,
        process_phase=np.array(["policy"] * 3 + ["return_to_reset"] * (steps - 3)),
        arm_torque_external_nm=ramp,
        arm_torque_commanded_nm=-ramp,
    )
    (data_dir / ct.METADATA_FILENAME).write_text(json.dumps({"arm_joint_torque_source": "external"}))
    return data_dir.parent


def test_candidates_are_normalized_by_the_fr3_limits_and_split_by_phase(tmp_path):
    candidates, phases, metadata = ct.load_run(_run(tmp_path))
    assert set(candidates) == {"external", "commanded"}
    assert metadata["arm_joint_torque_source"] == "external"
    assert candidates["external"][-1].tolist() == pytest.approx([1, 0, 0, 0, 1, 0, 0])
    rows = ct.phase_rows(candidates["external"], phases)
    assert [(phase, count) for phase, count, _, _ in rows] == [
        ("all", 6), ("policy", 3), ("return_to_reset", 3)
    ]


def test_opposite_candidates_correlate_at_minus_one_and_flat_joints_are_nan(tmp_path):
    candidates, _, _ = ct.load_run(_run(tmp_path))
    value = ct.correlations(candidates)[("external", "commanded")]
    assert value[0] == pytest.approx(-1.0) and value[4] == pytest.approx(-1.0)
    assert np.isnan(value[1])


def test_the_script_runs_end_to_end(tmp_path, capsys):
    assert ct.main([str(_run(tmp_path))]) == 0
    assert "external <- policy" in capsys.readouterr().out
