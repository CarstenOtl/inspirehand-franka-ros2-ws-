"""The preset file is the hand's action vocabulary, so it is checked like one."""

import textwrap

import pytest
import yaml

from inspire_franka_trajectory_replay import capture
from inspire_franka_trajectory_replay.hand_presets import (
    load_presets,
    open_ratio_to_radians,
)
from inspire_hand_driver import kinematics as kin


def _document(**overrides):
    open_ratio = {name: 1.0 for name in kin.DRIVEN_JOINTS}
    open_ratio["thumb_proximal_yaw_joint"] = 0.0
    document = {
        "schema_version": 2,
        "reserved_keys": ["w", "s", "c", "q", "?"],
        "fixed_open_ratio": {"thumb_proximal_yaw_joint": 0.0},
        "jog": {
            "step": 0.05,
            "speed": 250,
            "force": 150,
            "controls": [
                {
                    "joint": "thumb_proximal_pitch_joint",
                    "label": "thumb MCP flexion",
                    "close_key": "-",
                    "open_key": "=",
                },
                {
                    "joint": "index_proximal_joint",
                    "label": "index MCP flexion",
                    "close_key": "[",
                    "open_key": "]",
                },
            ],
        },
        "presets": [
            {
                "name": "open",
                "key": "o",
                "description": "everything open",
                "speed": 500,
                "force": 200,
                "open_ratio": open_ratio,
            }
        ],
    }
    document.update(overrides)
    return document


def _write(tmp_path, document):
    path = tmp_path / "presets.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# -- the packaged file -------------------------------------------------------


def test_packaged_presets_load():
    presets = load_presets()
    assert {preset.name for preset in presets} >= {"open", "pinch"}


def test_packaged_presets_name_exactly_the_driver_s_driven_joints():
    """The test the workspace cares about: this file and the driver cannot drift.

    Every preset addresses all six driven DOF and nothing else. A follower here
    would be rejected by the driver at the worst possible moment - mid-session,
    with an operator's hands on the arm.
    """
    for preset in load_presets():
        assert preset.joint_names == kin.DRIVEN_JOINTS
        assert len(preset.open_ratio) == len(kin.DOFS)


def test_packaged_preset_and_jog_keys_do_not_collide_with_anything():
    presets = load_presets()
    preset_keys = {preset.key for preset in presets}
    jog_keys = set(presets.jog.keys)
    reserved = set(capture.RESERVED_KEYS)

    assert not (preset_keys & reserved)
    assert not (jog_keys & reserved)
    assert not (preset_keys & jog_keys)
    assert len(jog_keys) == 2 * len(presets.jog.controls)


def test_packaged_reserved_key_list_matches_the_capture_tool():
    assert set(load_presets().reserved_keys) == set(capture.RESERVED_KEYS)


def test_packaged_preset_ratios_stay_inside_every_driven_joint_s_travel():
    for preset in load_presets():
        for index, radians in enumerate(open_ratio_to_radians(preset.open_ratio)):
            dof = kin.DOFS[index]
            assert dof.lower - 1e-12 <= radians <= dof.upper + 1e-12


# -- validation --------------------------------------------------------------


def test_a_follower_joint_is_rejected_by_name(tmp_path):
    document = _document()
    document["presets"][0]["open_ratio"]["index_intermediate_joint"] = 0.5
    with pytest.raises(ValueError, match="follow mechanically"):
        load_presets(_write(tmp_path, document))


def test_an_unknown_joint_is_rejected(tmp_path):
    document = _document()
    document["presets"][0]["open_ratio"]["ring_finger_joint"] = 0.5
    with pytest.raises(ValueError, match="unknown hand joints"):
        load_presets(_write(tmp_path, document))


def test_a_partial_posture_is_rejected(tmp_path):
    document = _document()
    document["presets"][0]["open_ratio"].pop("thumb_proximal_yaw_joint")
    with pytest.raises(ValueError, match="missing"):
        load_presets(_write(tmp_path, document))


@pytest.mark.parametrize("ratio", [-0.01, 1.01, 2.0])
def test_an_out_of_range_open_ratio_is_rejected(tmp_path, ratio):
    document = _document()
    document["presets"][0]["open_ratio"]["index_proximal_joint"] = ratio
    with pytest.raises(ValueError, match="outside"):
        load_presets(_write(tmp_path, document))


@pytest.mark.parametrize("field", ["speed", "force"])
@pytest.mark.parametrize("value", [-1, 1001, 2.5])
def test_speed_and_force_stay_in_register_units(tmp_path, field, value):
    document = _document()
    document["presets"][0][field] = value
    with pytest.raises(ValueError):
        load_presets(_write(tmp_path, document))


def test_a_preset_may_not_claim_a_reserved_key(tmp_path):
    document = _document()
    document["presets"][0]["key"] = "q"
    with pytest.raises(ValueError, match="reserved"):
        load_presets(_write(tmp_path, document))


def test_a_multi_character_key_is_rejected(tmp_path):
    document = _document()
    document["presets"][0]["key"] = "op"
    with pytest.raises(ValueError, match="one character"):
        load_presets(_write(tmp_path, document))


def test_duplicate_keys_are_rejected(tmp_path):
    document = _document()
    second = dict(document["presets"][0], name="also_open")
    document["presets"].append(second)
    with pytest.raises(ValueError, match="key 'o' is used by both"):
        load_presets(_write(tmp_path, document))


def test_duplicate_names_are_rejected(tmp_path):
    document = _document()
    second = dict(document["presets"][0], key="x")
    document["presets"].append(second)
    with pytest.raises(ValueError, match="name 'open' is used by both"):
        load_presets(_write(tmp_path, document))


def test_a_file_without_an_open_preset_is_rejected(tmp_path):
    document = _document()
    document["presets"][0]["name"] = "pinch"
    with pytest.raises(ValueError, match="'open' is required"):
        load_presets(_write(tmp_path, document))


def test_an_unsupported_schema_version_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        load_presets(_write(tmp_path, _document(schema_version=99)))


def test_an_empty_preset_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty list"):
        load_presets(_write(tmp_path, _document(presets=[])))


# -- units -------------------------------------------------------------------


def test_open_ratio_one_is_the_open_pose_and_zero_the_closed_one():
    wide = open_ratio_to_radians([1.0] * 6)
    shut = open_ratio_to_radians([0.0] * 6)
    assert wide == [dof.lower for dof in kin.DOFS]
    assert shut == [dof.upper for dof in kin.DOFS]


def test_open_ratio_to_radians_matches_the_driver_s_own_conversion():
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2]
    assert open_ratio_to_radians(ratios) == [
        kin.open_ratio_to_rad(index, ratio) for index, ratio in enumerate(ratios)
    ]


def test_open_ratio_to_radians_rejects_the_wrong_number_of_dof():
    with pytest.raises(ValueError, match="expected 6"):
        open_ratio_to_radians([1.0, 1.0])


def test_a_preset_event_records_every_joint_by_name(tmp_path):
    preset = load_presets(_write(tmp_path, _document())).by_name("open")
    event = preset.as_event()
    assert set(event["open_ratio"]) == set(kin.DRIVEN_JOINTS)
    assert event["preset"] == "open" and event["key"] == "o"


def test_a_malformed_file_is_a_load_error_not_a_crash(tmp_path):
    path = tmp_path / "presets.yaml"
    path.write_text(textwrap.dedent("- just\n- a\n- list\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_presets(path)


# -- the pinned thumb --------------------------------------------------------


def test_the_packaged_file_pins_thumb_abduction_to_the_bottom_of_its_range():
    presets = load_presets()
    assert presets.fixed_names() == {"thumb_proximal_yaw_joint": 0.0}
    for preset in presets:
        assert preset.open_ratio[kin.dof_index("thumb_proximal_yaw_joint")] == 0.0


def test_a_preset_that_disagrees_with_a_pinned_dof_is_refused(tmp_path):
    document = _document()
    document["presets"][0]["open_ratio"]["thumb_proximal_yaw_joint"] = 0.5
    with pytest.raises(ValueError, match="pinned to 0"):
        load_presets(_write(tmp_path, document))


def test_apply_fixed_overwrites_a_pinned_dof_whatever_the_caller_believed():
    presets = load_presets()
    index = kin.dof_index("thumb_proximal_yaw_joint")

    applied = presets.apply_fixed([0.9] * 6)

    assert applied[index] == 0.0
    assert all(value == 0.9 for position, value in enumerate(applied) if position != index)


def test_a_pinned_joint_may_not_also_be_jogged(tmp_path):
    document = _document()
    document["jog"]["controls"][0]["joint"] = "thumb_proximal_yaw_joint"
    with pytest.raises(ValueError, match="pinned .* cannot be jogged"):
        load_presets(_write(tmp_path, document))


def test_pinning_every_dof_is_refused(tmp_path):
    document = _document(fixed_open_ratio={name: 0.0 for name in kin.DRIVEN_JOINTS})
    with pytest.raises(ValueError, match="nothing to command"):
        load_presets(_write(tmp_path, document))


def test_a_follower_cannot_be_pinned(tmp_path):
    document = _document(fixed_open_ratio={"index_intermediate_joint": 0.0})
    with pytest.raises(ValueError, match="follows mechanically"):
        load_presets(_write(tmp_path, document))


# -- jog ---------------------------------------------------------------------


def test_jog_keys_resolve_to_a_signed_step_in_open_ratio():
    jog = load_presets().jog

    close, step_close = jog.control_for("-")
    open_, step_open = jog.control_for("=")

    assert close.joint == "thumb_proximal_pitch_joint"
    assert open_.joint == "thumb_proximal_pitch_joint"
    assert step_close == -jog.step
    assert step_open == +jog.step


def test_the_bracket_keys_drive_the_index_mcp():
    jog = load_presets().jog
    assert jog.control_for("[")[0].joint == "index_proximal_joint"
    assert jog.control_for("]")[0].joint == "index_proximal_joint"
    assert jog.control_for("[")[1] < 0 < jog.control_for("]")[1]


def test_an_unbound_key_is_not_a_jog():
    assert load_presets().jog.control_for("z") is None


@pytest.mark.parametrize("step", [0.0, -0.1, 1.5])
def test_an_out_of_range_jog_step_is_refused(tmp_path, step):
    document = _document()
    document["jog"]["step"] = step
    with pytest.raises(ValueError, match="jog step"):
        load_presets(_write(tmp_path, document))


def test_a_jog_key_that_is_already_a_preset_is_refused(tmp_path):
    document = _document()
    document["jog"]["controls"][0]["close_key"] = "o"
    with pytest.raises(ValueError, match="already the preset"):
        load_presets(_write(tmp_path, document))


def test_a_jog_key_that_is_reserved_is_refused(tmp_path):
    document = _document()
    document["jog"]["controls"][0]["close_key"] = "q"
    with pytest.raises(ValueError, match="reserved"):
        load_presets(_write(tmp_path, document))


def test_one_control_may_not_bind_the_same_key_both_ways(tmp_path):
    document = _document()
    document["jog"]["controls"][0]["open_key"] = "-"
    with pytest.raises(ValueError, match="both"):
        load_presets(_write(tmp_path, document))


def test_two_controls_may_not_share_a_key(tmp_path):
    document = _document()
    document["jog"]["controls"][1]["close_key"] = "-"
    with pytest.raises(ValueError, match="bound twice"):
        load_presets(_write(tmp_path, document))


def test_a_jog_control_naming_a_follower_is_refused(tmp_path):
    document = _document()
    document["jog"]["controls"][0]["joint"] = "index_intermediate_joint"
    with pytest.raises(ValueError, match="follows mechanically"):
        load_presets(_write(tmp_path, document))
