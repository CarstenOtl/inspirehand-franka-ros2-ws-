from types import SimpleNamespace

import run_policy_rollout as cli


def test_hardware_defaults_fit_the_cpu_policy_rate():
    args = cli._hardware_parser().parse_args([])
    assert args.rate == 15.0
    assert args.integration_steps == 2


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
