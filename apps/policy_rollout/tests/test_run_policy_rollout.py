from types import SimpleNamespace

import pytest

import run_policy_rollout as cli


def test_hardware_defaults_resolve_the_flow_steps_from_the_device():
    args = cli._hardware_parser().parse_args([])
    assert args.rate == 15.0
    assert args.device == "auto"
    assert args.integration_steps is None

    cpu = cli._hardware_parser().parse_args(["--device", "cpu"])
    assert str(cli._resolve_execution(cpu)) == "cpu"
    assert cpu.device == "cpu"
    assert cpu.integration_steps == 2

    from policy_rollout.flow_policy import default_integration_steps

    assert default_integration_steps("cuda") == 16
    assert default_integration_steps("cuda:0") == 16
    assert default_integration_steps("cpu") == 2


def test_explicit_flow_steps_survive_device_resolution():
    args = cli._hardware_parser().parse_args(
        ["--device", "cpu", "--integration-steps", "16"]
    )
    cli._resolve_execution(args)
    assert args.integration_steps == 16


def test_ros_sim_backend_shares_the_hardware_options():
    hardware = cli._hardware_parser().parse_args([])
    sim = cli._hardware_parser(sim=True).parse_args([])
    assert not hardware.sim and sim.sim
    assert {k: v for k, v in vars(sim).items() if k != "sim"} == {
        k: v for k, v in vars(hardware).items() if k != "sim"
    }
    # --home follows the task profile unless given.
    assert hardware.home is None and hardware.task_profile == "auto"
    assert str(cli.POLICY_HOME_CONFIGS["mimic_m24"]).endswith(
        "apps/policy_rollout/config/policy_home.yaml"
    )
    assert all(path.is_file() for path in cli.POLICY_HOME_CONFIGS.values())


def test_backend_limits_follow_isaac_in_ros_sim_only():
    for sim, steps, tolerance in ((True, 4500, 10.0), (False, 1500, 5.0)):
        args = cli._hardware_parser(sim=sim).parse_args([])
        cli.resolve_backend_defaults(args, sim)
        assert args.max_steps == steps
        assert args.return_tolerance_mm == tolerance and args.return_tolerance_deg == tolerance
    args = cli._hardware_parser(sim=True).parse_args(["--max-steps", "7", "--return-tolerance-mm", "12"])
    cli.resolve_backend_defaults(args, True)
    assert args.max_steps == 7 and args.return_tolerance_mm == 12.0 and args.return_tolerance_deg == 10.0


def test_hand_start_follows_the_students_reset_in_ros_sim():
    from policy_rollout import forge_osc as fo

    assert cli.resolve_hand_start("auto", fo.MIMIC_WINDBACK45_TILT30_PROFILE, sim=True) == "grasp"
    assert cli.resolve_hand_start("auto", fo.MIMIC_M24_PROFILE, sim=True) == "release"
    assert cli.resolve_hand_start("auto", fo.MIMIC_WINDBACK45_TILT30_PROFILE, sim=False) == "release"
    assert cli.resolve_hand_start("release", fo.MIMIC_WINDBACK45_TILT30_PROFILE, sim=True) == "release"


def test_every_task_profile_homes_to_its_own_reset():
    import yaml

    from policy_rollout import forge_osc as fo

    assert set(cli.POLICY_HOME_CONFIGS) == set(fo.TASK_PROFILES)
    for name, profile in fo.TASK_PROFILES.items():
        home = yaml.safe_load(cli.POLICY_HOME_CONFIGS[name].read_text())
        assert home["positions"][:7] == pytest.approx(profile.arm_reset_joints, abs=1e-6), name
        hand = dict(zip(home["joint_names"], home["positions"]))
        assert hand["thumb_proximal_yaw_joint"] == pytest.approx(profile.entry_pinch[0], abs=1e-6), name
        assert hand["thumb_proximal_pitch_joint"] == pytest.approx(profile.entry_pinch[1], abs=1e-6), name
        assert hand["index_proximal_joint"] == pytest.approx(profile.entry_pinch[2], abs=1e-6), name


def test_thumb_overlay_follows_the_backend_unless_overridden():
    """ros-sim's mock driver runs without the overlay; hardware's with it."""

    assert cli.resolve_thumb_overlay("auto", sim=False) is True
    assert cli.resolve_thumb_overlay("auto", sim=True) is False
    assert cli.resolve_thumb_overlay("on", sim=True) is True
    assert cli.resolve_thumb_overlay("off", sim=False) is False
    args = cli._hardware_parser(sim=True).parse_args([])
    assert args.thumb_overlay == "auto"
    assert args.hand_start == "auto"
    assert args.max_home_grasp_error == 0.002
    assert args.min_fingertip_separation == 0.030
    assert args.max_proxy_step_deg == 20.0
    assert args.expected_load_mass_kg is None


def test_hardware_parser_enables_matplotlib_viewer():
    args = cli._hardware_parser().parse_args(
        [
            "--viewer",
            "--viewer-depth-max",
            "1.25",
            "--viewer-hz",
            "6",
            "--integration-steps",
            "8",
        ]
    )
    assert args.viewer
    assert args.viewer_depth_max == 1.25
    assert args.viewer_hz == 6.0
    assert args.integration_steps == 8


def test_viewer_uses_policy_profile_topics(monkeypatch):
    calls = []

    class Process:
        pid = 123
        returncode = None

        @staticmethod
        def poll():
            return None

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    profile = SimpleNamespace(
        color_topic="/camera/camera/color/image_raw",
        depth_topic="/camera/camera/aligned_depth_to_color/image_raw",
    )

    process = cli._start_camera_viewer(profile, 1.25, 6.0)

    assert process.pid == 123
    command, options = calls[0]
    assert command[1].endswith("apps/camera_calibration/tests/test_camera.py")
    assert command[2:] == [
        "--no-launch",
        "--color-topic",
        profile.color_topic,
        "--depth-topic",
        profile.depth_topic,
        "--depth-max",
        "1.25",
        "--viewer-hz",
        "6.0",
    ]
    assert options == {"start_new_session": True}


def test_policy_warm_up_restores_seeded_runner_state():
    class Runner:
        config = SimpleNamespace(
            proprio_dim=29,
            cyclic_process_phase_conditioning=True,
            cyclic_process_phase_features=("policy", "return"),
            trajectory_progress_conditioning=True,
        )

        def __init__(self):
            self.seeds = []
            self.calls = []

        def reset(self, *, seed):
            self.seeds.append(seed)

        def step(self, **values):
            self.calls.append(values)

    runner = Runner()
    elapsed = cli._warm_up_runner(
        runner, SimpleNamespace(policy_shape=(180, 320)), seed=7
    )
    assert elapsed >= 0.0
    assert runner.seeds == [7, 7]
    assert runner.calls[0]["head_rgb"].shape == (3, 180, 320)
    assert runner.calls[0]["cyclic_process_phase"].tolist() == [1.0, 0.0]
