"""ONNX export for the feed-forward / single-pass models (BLUEPRINT section 13).

ONNX Runtime is the standard INT8 baseline path. The System 1 student (one forward
pass) exports cleanly; the recursive TRM can also be exported at a fixed depth
(the recursion loop unrolls into the graph). ``onnx``/``onnxruntime`` are optional
dependencies, so this module degrades gracefully when they are absent.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def onnx_available() -> bool:
    """True if ``onnx`` and ``onnxruntime`` are importable."""
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def export_to_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    opset: int = 17,
) -> Path:
    """Export ``model`` to ONNX with a single example input.

    Raises ``RuntimeError`` if ONNX is unavailable, with guidance to install it.
    """
    if not onnx_available():
        raise RuntimeError(
            "ONNX export requires `onnx` and `onnxruntime` "
            "(`pip install onnx onnxruntime`); they are optional and not installed."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        (example_input,),
        str(path),
        input_names=input_names or ["input"],
        output_names=output_names or ["logits"],
        opset_version=opset,
        dynamic_axes={"input": {0: "batch"}},
    )
    return path
