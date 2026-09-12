"""Experimental regional Inductor control; never an exact-runtime replacement.

Launch with TORCHINDUCTOR_FREEZING=1 before importing torch to use the frozen
max-autotune control. Only the recurrent cycle/head is compiled; the original
checker and early exit remain outside the graph. Compilation errors propagate.
"""
from __future__ import annotations

import copy
from typing import Any
import torch
from torch import nn
from spectra.fp_runtime import _validate, _environment
from deploy.m10_native import load_extension as load_checker

MODES = ("eager", "default", "max-autotune")


def compiler_counters() -> dict[str, dict[str, int]]:
    """Process-wide diagnostic snapshot, not per-request latency instrumentation."""
    from torch._dynamo.utils import counters
    return {str(k): {str(a): int(b) for a, b in v.items()} for k, v in counters.items()}


class _Cycle(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, emb: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        y, z = self.model.recursive_cycle(emb, y, z)
        return y, z, self.model.out_head(y).contiguous()


class CompilerFPSudoku:
    """Owning B=1 experimental eager/compiled control, within the FP graph scope.

    No exact numerical guarantee. A separate fresh checker and zero state are
    created per solve. The source is neither retained nor mutated. Compilation
    is lazy; call trace through all steps before warm measurements. Global
    dispatch/compiler configuration must be stable while this runtime is used.
    """
    @torch.inference_mode()
    def __init__(self, model: nn.Module, *, mode: str = "default"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        _validate(model)
        if mode != "eager":
            import torch._dynamo.config as dynamo_config
            if dynamo_config.suppress_errors:
                raise ValueError("compiler error suppression is not allowed")
        if mode == "max-autotune":
            import torch._inductor.config as inductor_config
            if not inductor_config.freezing:
                raise RuntimeError("launch with TORCHINDUCTOR_FREEZING=1 before importing torch; "
                                   "otherwise the frozen CPU comparator is not established")
        self._mode = mode
        self._model = copy.deepcopy(model)
        self._position = self._model.encode_positions(torch.zeros((1, 16), dtype=torch.int64), 4, 4).clone()
        self._module = _Cycle(self._model).eval()
        self._trained_steps = int(model.N_sup)
        self._block_count = len(model.blocks)
        self._checker = load_checker().SudokuProblem
        self._options: dict[str, Any] = {}
        if mode == "eager":
            self._cycle = self._module
        else:
            self._options = dict(torch._inductor.list_mode_options(mode))
            self._options.update({"freezing": mode == "max-autotune", "compile_threads": 1})
            # CUDA graphs are irrelevant to this CPU-only experiment.
            self._options["triton.cudagraphs"] = False
            self._cycle = torch.compile(self._module, backend="inductor", fullgraph=True,
                                        dynamic=False, options=self._options)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def trained_steps(self) -> int:
        return self._trained_steps

    @property
    def block_count(self) -> int:
        return self._block_count

    def _start(self, x: torch.Tensor, max_steps: int):
        _environment()
        if type(max_steps) is not int or not 1 <= max_steps <= self.trained_steps:
            raise ValueError("max_steps must be inside the trained supervision budget")
        if (type(x) is not torch.Tensor or x.device.type != "cpu" or x.dtype != torch.int64
                or x.layout != torch.strided or tuple(x.shape) != (1, 16)):
            raise ValueError("input must be a CPU int64 [1,16] tensor")
        x = x.contiguous()
        checker = self._checker(x, 2)
        try:
            emb = self._model.token_embed(x) + self._position
        except IndexError as exc:
            raise ValueError("input symbols must be in [0,4]") from exc
        return checker, emb, torch.zeros_like(emb), torch.zeros_like(emb)

    @torch.inference_mode()
    def solve(self, x: torch.Tensor, max_steps: int = 4):
        checker, emb, y, z = self._start(x, max_steps)
        for step in range(1, max_steps + 1):
            y, z, logits = self._cycle(emb, y, z)
            answer, valid = checker.decode(logits)
            if valid:
                break
        return answer, {"executed_steps": step, "block_applications": step * 2 * self.block_count,
                        "semantic_checks": step, "checker_constructions": 1,
                        "final_semantic": bool(valid), "target_used": False,
                        "checker": "native_exact_sudoku_v1",
                        "stop_reason": "semantic_valid" if valid else "budget_exhausted"}

    @torch.inference_mode()
    def trace(self, x: torch.Tensor, max_steps: int = 4):
        """Full-budget states for auditing; never substituted for a timed solve."""
        _, emb, y, z = self._start(x, max_steps)
        records = []
        for _ in range(max_steps):
            y, z, logits = self._cycle(emb, y, z)
            records.append((y.clone(), z.clone(), logits.clone()))
        return emb.clone(), records

    def identity(self) -> dict[str, Any]:
        return {"runtime": "experimental_compiler_fp_sudoku_v1", "mode": self.mode,
                "backend": "eager" if self.mode == "eager" else "inductor",
                "fullgraph": self.mode != "eager", "dynamic": False,
                "options": dict(self._options), "torch": str(torch.__version__),
                "trained_steps": self.trained_steps, "block_count": self.block_count,
                "precision": "FP32", "exact_claim": False,
                "region": "one_recurrent_cycle_plus_output_head",
                "checker_compiled": False, "fallback_on_error": False}
