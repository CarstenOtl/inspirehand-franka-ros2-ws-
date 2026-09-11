import json

import numpy as np
import pytest

from utils.data_collection import RolloutDataCollector, load_rollout
from utils.evaluation import evaluate_rollout


def _recording(path, *, position_offset=0.0, task_signals=False, record_rgbd=False):
    collector = RolloutDataCollector(
        path,
        metadata={"checkpoint": "/tmp/student.pt", "checkpoint_sha256": "abc123"},
        record_rgbd=record_rgbd,
    )
    for index in range(3):
        position = np.arange(10, dtype=np.float32) + position_offset + index * 0.1
        velocity = np.full(10, index * 0.2, dtype=np.float32)
        action = np.linspace(-0.5, 0.5, 9, dtype=np.float32) + index * 0.01
        signals = None
        if task_signals:
            signals = {
                "pickup_success": index == 2,
                "threading_entered": index >= 1,
                "completed_cycles": 6 if index == 2 else 0,
                "threading_turn_progress_rad": index * 0.25,
                "terminated": False,
                "truncated": False,
                "watchdog_stop": False,
            }
        camera = {
            "rgb": np.zeros((3, 2, 4), dtype=np.float32),
            "depth": np.full((1, 2, 4), 0.42, dtype=np.float32),
            "valid_mask": np.ones((1, 2, 4), dtype=bool),
        }
        collector.record(
            sample_time_s=10.0 + index * 0.1,
            joint_position=position,
            joint_velocity=velocity,
            proprio=np.concatenate((position, velocity, np.zeros(9))),
            policy_action=action,
            filtered_native_action=action * 0.5,
            clipped_elements=index == 2,
            trajectory_progress=index / 2,
            process_phase="policy" if index < 2 else "return_to_reset",
            task_signals=signals,
            **camera,
        )
    return collector.close()


def test_collector_writes_safe_aligned_npz_and_metadata(tmp_path):
    artifact = _recording(tmp_path / "run", record_rgbd=True)
    loaded = load_rollout(artifact.run_dir)
    assert artifact.sample_count == 3
    assert loaded.arrays["sample_time_s"].tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert loaded.arrays["joint_position"].shape == (3, 10)
    assert loaded.arrays["proprio"].shape == (3, 29)
    assert loaded.arrays["policy_action"].shape == (3, 9)
    assert loaded.arrays["head_rgb"].shape == (3, 3, 2, 4)
    assert loaded.arrays["head_depth"].shape == (3, 1, 2, 4)
    assert loaded.arrays["depth_valid"].dtype == np.bool_
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["sample_count"] == 3
    assert metadata["checkpoint_sha256"] == "abc123"


def test_collector_rejects_non_monotonic_time_and_unknown_task_signal(tmp_path):
    collector = RolloutDataCollector(tmp_path / "run")
    values = {
        "joint_position": np.zeros(10),
        "joint_velocity": np.zeros(10),
        "proprio": np.zeros(29),
        "policy_action": np.zeros(9),
        "filtered_native_action": np.zeros(9),
        "clipped_elements": 0,
    }
    collector.record(sample_time_s=1.0, **values)
    with pytest.raises(ValueError, match="increase strictly"):
        collector.record(sample_time_s=1.0, **values)
    with pytest.raises(ValueError, match="unsupported task signals"):
        collector.record(
            sample_time_s=2.0,
            task_signals={"geometric_success_proxy": True},
            **values,
        )


def test_evaluation_reports_task_outcome_and_reference_error(tmp_path):
    reference = _recording(tmp_path / "reference", task_signals=True)
    candidate = _recording(
        tmp_path / "candidate", position_offset=0.1, task_signals=True
    )
    report_path, report = evaluate_rollout(
        candidate.run_dir,
        reference=reference.run_dir,
    )
    assert report_path.is_file()
    assert report["metrics"]["task_outcome"]["status"] == "passed"
    comparison = report["reference_comparison"]
    assert comparison["compared_samples"] == 3
    assert comparison["channels"]["joint_position"]["overall"][
        "rmse"
    ] == pytest.approx(0.1)
    assert (report_path.parent / "reference_error_trace.npz").is_file()


def test_evaluation_does_not_infer_success_without_task_signals(tmp_path):
    artifact = _recording(tmp_path / "run")
    _, report = evaluate_rollout(artifact.run_dir)
    outcome = report["metrics"]["task_outcome"]
    assert outcome["status"] == "not_evaluable"
    assert "task adapter did not supply" in outcome["reason"]


def test_plotting_creates_standard_headless_artifacts(tmp_path):
    pytest.importorskip("matplotlib")
    from utils.plotting import plot_rollout

    artifact = _recording(tmp_path / "run")
    paths = plot_rollout(artifact.run_dir)
    assert {path.name for path in paths} == {
        "actions.png",
        "diagnostics.png",
        "joint_state.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
