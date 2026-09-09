"""Validate the checked-in task homing poses against the local robot model."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml")


APP_DIR = Path(__file__).resolve().parents[1]
HOMING_DIR = APP_DIR / "demo_trajs" / "homing"
ROBOT_XML = APP_DIR.parents[1] / "assets" / "fr3_inspirehand" / "fr3_inspirehand.xml"

ARM_JOINTS = tuple(f"fr3_joint{index}" for index in range(1, 8))
HAND_JOINTS = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)
CONTROLLED_JOINTS = ARM_JOINTS + HAND_JOINTS

EXPECTED_ARM_POSITIONS = {
    "pickup.yaml": (-0.392613, 0.004288, -0.072713, -1.811251,
                    0.592754, 2.280553, -2.620279),
    "threading.yaml": (-0.348916, -0.033274, -0.026434, -1.799153,
                       1.368818, 2.028677, -1.846102),
}
EXPECTED_HAND_POSITIONS = (1.0999, 1.0999, 1.0999, 0.44, 0.2, 1.14)


def test_threading_hardware_baseline_encodes_the_validated_mount_retarget():
    baseline_dir = APP_DIR / "demo_trajs" / "threading_cycle1_flange180"
    with (baseline_dir / "homing.yaml").open(encoding="utf-8") as stream:
        baseline_home = yaml.safe_load(stream)
    with (HOMING_DIR / "threading.yaml").open(encoding="utf-8") as stream:
        legacy_home = yaml.safe_load(stream)
    metadata = json.loads((baseline_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(baseline_dir / "replay_data.npz", allow_pickle=False) as data:
        arm = np.asarray(data["joint_pos_arm"])
        names = [str(name) for name in data["arm_joint_names"]]

    assert metadata["hardware_replay_status"] == "baseline"
    assert metadata["cycles"] == [1]
    assert names == list(ARM_JOINTS)
    assert baseline_home["positions"][:6] == pytest.approx(legacy_home["positions"][:6])
    assert baseline_home["positions"][6] == pytest.approx(
        legacy_home["positions"][6] + math.pi / 2
    )
    assert arm[0] == pytest.approx(baseline_home["positions"][:7], abs=1e-7)
    assert arm[-1] == pytest.approx(baseline_home["positions"][:7], abs=1e-12)


@pytest.mark.parametrize(
    "relative_metadata",
    (
        "traj_1/metadata.json",
        "threading_5x/metadata.json",
    ),
)
def test_other_threading_arm_artifacts_are_not_the_hardware_baseline(relative_metadata):
    path = APP_DIR / "demo_trajs" / relative_metadata
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["hardware_replay_status"].startswith("outdated")
    assert metadata["hardware_baseline"] == "../threading_cycle1_flange180"


def test_five_cycle_current_mount_artifact_is_a_validated_extended_run():
    path = APP_DIR / "demo_trajs" / "threading_5x_flange180" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["hardware_replay_status"] == "validated_extended_run"
    assert metadata["hardware_baseline"] == "../threading_cycle1_flange180"
    assert metadata["hardware_validation"]["time_scale"] == 5.0
    assert metadata["hardware_validation"]["interactive_pause"] is True


@pytest.mark.parametrize("filename", EXPECTED_ARM_POSITIONS)
def test_homing_pose_matches_franka_chi_and_local_joint_contract(filename):
    with (HOMING_DIR / filename).open(encoding="utf-8") as stream:
        pose = yaml.safe_load(stream)

    assert pose["schema_version"] == 1
    assert pose["units"] == "radians"
    assert tuple(pose["joint_names"]) == CONTROLLED_JOINTS
    assert len(pose["positions"]) == len(CONTROLLED_JOINTS)
    assert pose["positions"][:7] == pytest.approx(EXPECTED_ARM_POSITIONS[filename])
    assert pose["positions"][7:] == pytest.approx(EXPECTED_HAND_POSITIONS)


def test_homing_joint_names_exist_in_the_local_mujoco_model():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML))
    model_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    assert set(CONTROLLED_JOINTS) <= model_joints

    for pose_path in HOMING_DIR.glob("*.yaml"):
        with pose_path.open(encoding="utf-8") as stream:
            pose = yaml.safe_load(stream)
        for name, position in zip(pose["joint_names"], pose["positions"]):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low, high = model.jnt_range[joint_id]
            assert low <= position <= high, (
                f"{pose_path.name}: {name}={position} is outside [{low}, {high}]"
            )
