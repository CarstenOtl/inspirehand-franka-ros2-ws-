"""ROS-independent regression tests for manual capture control.

The complete calibration node depends on ROS message packages that are only
available in the development container.  These tests execute the two relevant
methods directly from their AST so the capture-state behavior is also covered
by the repository's ordinary host-side pytest run.
"""

import ast
from pathlib import Path
from types import SimpleNamespace


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "camera_calibration"
    / "camera_calibration"
    / "calibration_node.py"
)


def _class_definition() -> ast.ClassDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CameraCalibrationNode"
    )


def _method(name: str):
    definition = next(
        node
        for node in _class_definition().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    # Compile the method as a standalone function.  Removing annotations keeps
    # this test independent of sensor_msgs on non-ROS hosts.
    definition.decorator_list = []
    definition.returns = None
    for argument in (
        definition.args.posonlyargs
        + definition.args.args
        + definition.args.kwonlyargs
    ):
        argument.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[name]


def test_unarmed_triggered_mode_skips_image_processing():
    notices = []
    node = SimpleNamespace(
        _capture_mode="triggered",
        _capture_requested=False,
        _camera_matrix=None,
        _notice=lambda *arguments, **keywords: notices.append((arguments, keywords)),
    )

    _method("_image_callback")(node, object())

    assert notices == []


def test_armed_triggered_mode_resumes_image_processing():
    notices = []
    node = SimpleNamespace(
        _capture_mode="triggered",
        _capture_requested=True,
        _camera_matrix=None,
        _notice=lambda *arguments, **keywords: notices.append((arguments, keywords)),
    )

    _method("_image_callback")(node, object())

    assert notices[0][0][0] == "intrinsics"


def test_repeated_capture_request_does_not_queue_another_sample():
    callback = _method("_capture_callback")
    node = SimpleNamespace(
        _calibration_complete=False,
        _capture_mode="triggered",
        _capture_requested=False,
    )

    first = callback(node, object(), SimpleNamespace())
    second = callback(node, object(), SimpleNamespace())

    assert first.success is True
    assert node._capture_requested is True
    assert second.success is False
    assert "already armed" in second.message


def test_capture_service_has_a_dedicated_callback_group():
    constructor = next(
        node
        for node in _class_definition().body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    capture_services = []
    for node in ast.walk(constructor):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "create_service" or len(node.args) < 2:
            continue
        if isinstance(node.args[1], ast.Constant) and node.args[1].value == "~/capture":
            capture_services.append(node)

    assert len(capture_services) == 1
    callback_group = next(
        (
            keyword.value
            for keyword in capture_services[0].keywords
            if keyword.arg == "callback_group"
        ),
        None,
    )
    assert isinstance(callback_group, ast.Attribute)
    assert callback_group.attr == "_capture_service_group"

    group_initializers = [
        node.value
        for node in constructor.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "_capture_service_group"
            for target in node.targets
        )
    ]
    assert len(group_initializers) == 1
    assert isinstance(group_initializers[0], ast.Call)
    assert isinstance(group_initializers[0].func, ast.Name)
    assert group_initializers[0].func.id == "MutuallyExclusiveCallbackGroup"
