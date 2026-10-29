"""Portable ForgeUltra student-policy inference for the FR3 workspace.

Submodules are intentionally imported lazily so camera-only preflight remains
available in an operator shell that does not yet have PyTorch loaded.
"""

from pathlib import Path

# The deployed student, which every backend and diagnostic tool runs unless
# given --checkpoint. One definition, so they cannot drift apart again.
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "checkpoints/m24-dp3-nominal-da-force-nohv-s0-20261029-113811-885629"
    / "nn/checkpoint.pt"
)

__all__: list[str] = ["DEFAULT_CHECKPOINT"]
