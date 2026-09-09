import importlib.util
from pathlib import Path
import signal


SCRIPT = Path(__file__).resolve().parents[1] / "calibrate.py"
SPEC = importlib.util.spec_from_file_location("camera_calibration_cli", SCRIPT)
calibrate_cli = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibrate_cli)


def parse(*arguments):
    return calibrate_cli._parser().parse_args(arguments)


def test_manual_mode_uses_one_shot_triggered_capture():
    command = calibrate_cli._launch_command(parse("--manual", "--no-camera"))
    assert "capture_mode:=triggered" in command
    assert "start_camera:=false" in command
    assert "world_frame:=fr3_link0" in command


def test_automatic_mode_remains_automatic():
    command = calibrate_cli._launch_command(parse("--auto"))
    assert "capture_mode:=auto" in command


def test_persistent_client_targets_private_capture_service():
    assert calibrate_cli.CAPTURE_SERVICE_NAME == "/camera_calibration/capture"


def test_manual_capture_reuses_one_client(monkeypatch):
    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, requested_signal):
            assert requested_signal == signal.SIGINT
            self.returncode = 0

        def wait(self, timeout):
            assert timeout == 8.0
            return self.returncode

    class FakeClient:
        def __init__(self):
            self.ready_checks = 0
            self.capture_requests = 0
            self.closed = False

        def wait_until_ready(self, process):
            self.ready_checks += 1
            return process.poll() is None

        def request_capture(self):
            self.capture_requests += 1
            return True, "armed"

        def close(self):
            self.closed = True

    process = FakeProcess()
    client = FakeClient()
    replies = iter(["", "", "q"])
    popen_arguments = {}

    def fake_popen(command, **kwargs):
        popen_arguments["command"] = command
        popen_arguments.update(kwargs)
        return process

    monkeypatch.setattr(calibrate_cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    assert calibrate_cli._run_enter_capture(
        ["ros2", "launch"], client_factory=lambda: client
    ) == 0
    assert client.ready_checks == 1
    assert client.capture_requests == 2
    assert client.closed
    assert popen_arguments["stdin"] is calibrate_cli.subprocess.DEVNULL
    assert popen_arguments["start_new_session"] is True


def test_capture_client_error_does_not_stop_launch_before_next_prompt(monkeypatch):
    class FakeProcess:
        returncode = None

        def poll(self):
            return self.returncode

        def send_signal(self, requested_signal):
            assert requested_signal == signal.SIGINT
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    class FailingClient:
        def wait_until_ready(self, process):
            return True

        def request_capture(self):
            raise RuntimeError("executor failure")

        def close(self):
            pass

    process = FakeProcess()
    replies = iter(["", "q"])
    monkeypatch.setattr(
        calibrate_cli.subprocess, "Popen", lambda command, **kwargs: process
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(replies))

    assert calibrate_cli._run_enter_capture(
        ["ros2", "launch"], client_factory=FailingClient
    ) == 0
