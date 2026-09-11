"""Strict adapter to the workspace-vendored Flow Matching package."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


def require_vendored_flow_matching() -> Path:
    """Import and verify ``third_party/flow_matching``, never a site package."""

    workspace_root = Path(__file__).resolve().parents[3]
    package_root = workspace_root / "third_party" / "flow_matching"
    expected = package_root / "flow_matching"
    if not (expected / "__init__.py").is_file():
        raise ModuleNotFoundError(
            f"vendored Flow Matching package is missing: {package_root}"
        )
    package_root_string = str(package_root)
    if package_root_string not in sys.path:
        sys.path.insert(0, package_root_string)
    import flow_matching

    module_path = Path(flow_matching.__file__).resolve()
    if not module_path.is_relative_to(expected.resolve()):
        raise ImportError(
            f"loaded non-vendored flow_matching from {module_path}; expected {expected}"
        )
    return expected.resolve()


def sample_local_ode(
    model,
    x_init: torch.Tensor,
    condition: torch.Tensor,
    *,
    steps: int,
) -> torch.Tensor:
    """Use ForgeUltra's vendored ODE solver with its fixed Euler adapter."""

    if steps < 1:
        raise ValueError("integration steps must be positive")
    require_vendored_flow_matching()
    try:
        from flow_matching.solver import ODESolver
        from flow_matching.utils import ModelWrapper
    except ModuleNotFoundError as exc:
        if exc.name == "torchdiffeq":
            raise ModuleNotFoundError(
                "vendored flow_matching requires torchdiffeq in the active environment"
            ) from exc
        raise

    class CachedConditionVelocity(ModelWrapper):
        def forward(self, value, time, **extras):
            if time.ndim == 0:
                time = time.expand(value.shape[0])
            return self.model(
                value,
                time.to(value),
                condition=extras["condition"],
            )

    solver = ODESolver(CachedConditionVelocity(model))
    return solver.sample(
        x_init=x_init,
        step_size=1.0 / steps,
        method="euler",
        time_grid=x_init.new_tensor([0.0, 1.0]),
        condition=condition,
    )


__all__ = ["require_vendored_flow_matching", "sample_local_ode"]
