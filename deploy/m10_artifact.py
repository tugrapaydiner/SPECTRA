"""Versioned Milestone-10 CPU deployment artifact for one trained W1.58A8 TRM.

The artifact contains only inference-required state: packed hard-ternary weights
plus scales/biases and the complete FP32 remainder of the supported graph. Master
ternary training weights are deliberately not required by the deployment loader.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from deploy.pack_ternary import pack_ternary_rows, unpack_ternary_rows
from model.bitlinear import FakeBitLinear
from model.fake_quant import FakeActQuant
from model.operators import SelfAttention, SwapBlock
from model.trm import TRM

M10_FORMAT = "spectra.cpu_recursive"
M10_VERSION = 1
M10_BACKEND_CONTRACT = "packed_ternary_fp32_linear_v1"


class DeploymentArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedCPUArtifact:
    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def architecture(self) -> dict[str, Any]:
        return dict(self.payload["architecture"])

    @property
    def task(self) -> dict[str, Any]:
        return dict(self.payload["task"])

    @property
    def precision(self) -> dict[str, Any]:
        return dict(self.payload["precision"])


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    c = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(c.dtype).encode("ascii"))
    h.update(np.asarray(c.shape, dtype=np.int64).tobytes())
    h.update(c.numpy().tobytes())
    return h.hexdigest()


def module_tensor_state_sha256(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        nb = name.encode("utf-8")
        h.update(len(nb).to_bytes(8, "little")); h.update(nb)
        tb = tensor_sha256(tensor).encode("ascii")
        h.update(tb)
    return h.hexdigest()


def _require_cpu_fp32_model(model: TRM) -> None:
    for name, p in model.named_parameters():
        if p.device.type != "cpu":
            raise DeploymentArtifactError(f"parameter {name} is not on CPU")
        if p.dtype is not torch.float32:
            raise DeploymentArtifactError(f"parameter {name} must be FP32, got {p.dtype}")


def _validate_supported_graph(model: nn.Module, *, height: int, width: int, box: int) -> TRM:
    if type(model) is not TRM:
        raise DeploymentArtifactError("M10 export supports exactly model.trm.TRM")
    if not model.ternary:
        raise DeploymentArtifactError("M10 artifact requires ternary=true")
    if not model.act8 or type(model.act_quant) is not FakeActQuant:
        raise DeploymentArtifactError("M10 fixed configuration requires FakeActQuant A8")
    if model.act_quant.bits != 8 or model.act_quant.qmin != -128 or model.act_quant.qmax != 127:
        raise DeploymentArtifactError("M10 supports exactly signed 8-bit recurrent fake quantization")
    if len(model.blocks) != 1:
        raise DeploymentArtifactError("M10 fixed artifact supports exactly one SwapBlock")
    if height <= 0 or width <= 0 or height * width != int(model.seq_len):
        raise DeploymentArtifactError("task geometry must exactly match model seq_len")
    if max(height, width) > int(model.max_grid_size):
        raise DeploymentArtifactError("task geometry exceeds model max_grid_size")
    if box <= 0 or height != width or box * box != width:
        raise DeploymentArtifactError("M10 Sudoku geometry requires square box*box grid")

    block = model.blocks[0]
    if type(block) is not SwapBlock or not bool(block.ternary_attn):
        raise DeploymentArtifactError("M10 requires ternary SwapBlock/SelfAttention")
    if type(block.attn) is not SelfAttention:
        raise DeploymentArtifactError("dense nn.MultiheadAttention is unsupported by M10 artifact")
    if type(block.norm1) is not nn.RMSNorm or type(block.norm2) is not nn.RMSNorm:
        raise DeploymentArtifactError("M10 supports exactly nn.RMSNorm block norms")
    if len(block.ff) != 3 or type(block.ff[0]) is not FakeBitLinear or type(block.ff[2]) is not FakeBitLinear:
        raise DeploymentArtifactError("M10 FFN must be FakeBitLinear -> GELU -> FakeBitLinear")
    if type(block.ff[1]) is not nn.GELU or getattr(block.ff[1], "approximate", "none") != "none":
        raise DeploymentArtifactError("M10 preserves GELU(approximate='none'); substitutions are rejected")
    for name in ("q", "k", "v", "proj"):
        if type(getattr(block.attn, name)) is not FakeBitLinear:
            raise DeploymentArtifactError(f"attention projection {name} must be FakeBitLinear")
    if type(model.out_head) is not FakeBitLinear:
        raise DeploymentArtifactError("M10 output head must be FakeBitLinear")
    if type(model.halt_head) is not nn.Linear:
        raise DeploymentArtifactError("M10 halt head must be nn.Linear")
    if type(model.norm_y) is not nn.RMSNorm or type(model.norm_z) is not nn.RMSNorm:
        raise DeploymentArtifactError("M10 recurrent output norms must be nn.RMSNorm")

    for name, module in model.named_modules():
        if isinstance(module, FakeBitLinear):
            rho = float(module.quant_strength.detach().cpu())
            if rho != 1.0:
                raise DeploymentArtifactError(
                    f"soft-ternary export rejected: {name}.quant_strength={rho}, expected exactly 1.0"
                )
    _require_cpu_fp32_model(model)
    return model


def _packed_entry(module: FakeBitLinear) -> dict[str, Any]:
    with torch.no_grad():
        w_q, scale = module._ternarize_hard(module.weight)
    codes = w_q.detach().cpu().to(torch.int8).numpy()
    packed_np, shape = pack_ternary_rows(codes)
    packed = torch.from_numpy(packed_np.copy()).to(torch.uint8).contiguous()
    row_scale = scale.detach().cpu().reshape(-1).to(torch.float32).contiguous()
    bias = (
        module.bias.detach().cpu().to(torch.float32).contiguous()
        if module.bias is not None
        else torch.empty(0, dtype=torch.float32)
    )
    reconstructed = torch.from_numpy(unpack_ternary_rows(packed_np, *shape)).to(torch.float32)
    reconstructed = reconstructed * row_scale[:, None]
    hard = (w_q.detach().cpu().to(torch.float32) * scale.detach().cpu().to(torch.float32))
    if not torch.equal(reconstructed, hard):
        raise DeploymentArtifactError("packed hard-ternary reconstruction is not exact")
    return {
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "packed": packed,
        "scale": row_scale,
        "bias": bias,
        "packing": "row_padded_lsb_first_2bit_00zero_01pos_10neg_11reserved",
        "reconstruction_exact": True,
    }


def _fp_tensor_inventory(model: TRM) -> dict[str, torch.Tensor]:
    block = model.blocks[0]
    return {
        "token_embed.weight": model.token_embed.weight.detach().cpu().to(torch.float32).contiguous(),
        "pos_encoder.row_embed.weight": model.pos_encoder.row_embed.weight.detach().cpu().to(torch.float32).contiguous(),
        "pos_encoder.col_embed.weight": model.pos_encoder.col_embed.weight.detach().cpu().to(torch.float32).contiguous(),
        "blocks.0.norm1.weight": block.norm1.weight.detach().cpu().to(torch.float32).contiguous(),
        "blocks.0.norm2.weight": block.norm2.weight.detach().cpu().to(torch.float32).contiguous(),
        "norm_y.weight": model.norm_y.weight.detach().cpu().to(torch.float32).contiguous(),
        "norm_z.weight": model.norm_z.weight.detach().cpu().to(torch.float32).contiguous(),
        "halt_head.weight": model.halt_head.weight.detach().cpu().to(torch.float32).contiguous(),
        "halt_head.bias": model.halt_head.bias.detach().cpu().to(torch.float32).contiguous(),
        "alpha_y": model.alpha_y.detach().cpu().reshape(()).to(torch.float32).contiguous(),
        "alpha_z": model.alpha_z.detach().cpu().reshape(()).to(torch.float32).contiguous(),
    }


def _assert_parameter_inventory_complete(
    model: TRM,
    packed_linears: Mapping[str, Mapping[str, Any]],
    fp_tensors: Mapping[str, torch.Tensor],
) -> None:
    handled = set(fp_tensors)
    for name, module in model.named_modules():
        if isinstance(module, FakeBitLinear):
            if name not in packed_linears:
                raise DeploymentArtifactError(f"missing packed inference layer {name}")
            handled.add(f"{name}.weight")
            if module.bias is not None:
                handled.add(f"{name}.bias")
    actual = {name for name, _ in model.named_parameters()}
    missing = sorted(actual - handled)
    extra = sorted(handled - actual)
    if missing or extra:
        raise DeploymentArtifactError(
            f"inference parameter inventory mismatch: missing={missing}, extra={extra}"
        )


def export_cpu_artifact(
    model: nn.Module,
    path: str | Path,
    *,
    height: int,
    width: int,
    box: int,
    source_checkpoint_sha256: str,
    source_checkpoint_tensor_sha256: str,
    training_seed: int,
    training_step: int,
    data_provenance: Mapping[str, Any],
    export_git_sha: str,
) -> dict[str, Any]:
    model = _validate_supported_graph(model, height=height, width=width, box=box)
    block = model.blocks[0]
    linears = {
        "blocks.0.attn.q": _packed_entry(block.attn.q),
        "blocks.0.attn.k": _packed_entry(block.attn.k),
        "blocks.0.attn.v": _packed_entry(block.attn.v),
        "blocks.0.attn.proj": _packed_entry(block.attn.proj),
        "blocks.0.ff.0": _packed_entry(block.ff[0]),
        "blocks.0.ff.2": _packed_entry(block.ff[2]),
        "out_head": _packed_entry(model.out_head),
    }
    fp = _fp_tensor_inventory(model)
    _assert_parameter_inventory_complete(model, linears, fp)

    heads = int(block.attn.heads)
    mlp_ratio = int(block.ff[0].out_features // model.dim)
    quant_strength = {
        name: float(module.quant_strength.detach().cpu())
        for name, module in model.named_modules()
        if isinstance(module, FakeBitLinear)
    }
    payload: dict[str, Any] = {
        "format": M10_FORMAT,
        "version": M10_VERSION,
        "backend_contract": M10_BACKEND_CONTRACT,
        "architecture": {
            "class": "TRM",
            "dim": int(model.dim),
            "num_tokens": int(model.num_tokens),
            "seq_len": int(model.seq_len),
            "n_layers": 1,
            "heads": heads,
            "mlp_ratio": mlp_ratio,
            "n": int(model.n),
            "T": int(model.T),
            "N_sup": int(model.N_sup),
            "max_grid_size": int(model.max_grid_size),
            "ternary": True,
            "act8": True,
            "gelu_approximate": "none",
            "rmsnorm_eps": None,
            "act_quant_bits": int(model.act_quant.bits),
            "act_quant_qmin": int(model.act_quant.qmin),
            "act_quant_qmax": int(model.act_quant.qmax),
            "act_quant_eps": float(model.act_quant.eps),
        },
        "task": {
            "task": "sudoku",
            "height": int(height),
            "width": int(width),
            "box": int(box),
        },
        "precision": {
            "packed_weights": "ternary_2bit_row_padded",
            "weight_scale": "float32_per_output_channel",
            "native_linear_input": "float32",
            "native_linear_accumulator": "float32_scalar_ordered",
            "native_linear_output": "float32",
            "attention": "pytorch_cpu_float32",
            "gelu": "pytorch_cpu_float32_approximate_none",
            "rmsnorm": "pytorch_cpu_float32_eps_none",
            "residual_arithmetic": "float32",
            "recurrent_a8": "dynamic_per_token_int8_round_dequant_float32_scale",
            "halt_head": "pytorch_cpu_float32",
            "final_logits": "float32",
        },
        "source": {
            "checkpoint_sha256": str(source_checkpoint_sha256),
            "checkpoint_tensor_sha256": str(source_checkpoint_tensor_sha256),
            "model_tensor_state_sha256": module_tensor_state_sha256(model),
            "training_seed": int(training_seed),
            "training_step": int(training_step),
            "quant_strength": quant_strength,
            "data_provenance": dict(data_provenance),
            "export_git_sha": str(export_git_sha),
        },
        "packed_linears": linears,
        "fp_tensors": fp,
    }
    payload["inventory"] = {
        "packed_linears": {
            name: {
                "in_features": int(entry["in_features"]),
                "out_features": int(entry["out_features"]),
                "packed_sha256": tensor_sha256(entry["packed"]),
                "scale_sha256": tensor_sha256(entry["scale"]),
                "bias_sha256": tensor_sha256(entry["bias"]),
            }
            for name, entry in linears.items()
        },
        "fp_tensors": {
            name: {
                "shape": list(t.shape),
                "dtype": str(t.dtype).replace("torch.", ""),
                "sha256": tensor_sha256(t),
            }
            for name, t in fp.items()
        },
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "format": M10_FORMAT,
        "version": M10_VERSION,
        "backend_contract": M10_BACKEND_CONTRACT,
        "packed_linear_count": len(linears),
        "fp_tensor_count": len(fp),
        "source_model_tensor_sha256": payload["source"]["model_tensor_state_sha256"],
    }


def _validate_packed_entry(name: str, entry: Mapping[str, Any]) -> None:
    required = {"in_features", "out_features", "packed", "scale", "bias", "packing", "reconstruction_exact"}
    if set(entry) != required:
        raise DeploymentArtifactError(f"packed layer {name} keys mismatch")
    i, o = int(entry["in_features"]), int(entry["out_features"])
    if i <= 0 or o <= 0:
        raise DeploymentArtifactError(f"packed layer {name} dimensions must be positive")
    packed, scale, bias = entry["packed"], entry["scale"], entry["bias"]
    for tensor, dtype, label in (
        (packed, torch.uint8, "packed"), (scale, torch.float32, "scale"), (bias, torch.float32, "bias")
    ):
        if not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype is not dtype or not tensor.is_contiguous():
            raise DeploymentArtifactError(f"packed layer {name} {label} tensor contract invalid")
    if packed.ndim != 1 or packed.numel() != o * ((i + 3) // 4):
        raise DeploymentArtifactError(f"packed layer {name} byte geometry invalid")
    if scale.ndim != 1 or scale.numel() != o or bias.ndim != 1 or bias.numel() not in {0, o}:
        raise DeploymentArtifactError(f"packed layer {name} scale/bias geometry invalid")
    if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
        raise DeploymentArtifactError(f"packed layer {name} scale invalid")
    if bias.numel() and not bool(torch.isfinite(bias).all()):
        raise DeploymentArtifactError(f"packed layer {name} bias invalid")
    raw = packed.numpy()
    rb = (i + 3) // 4
    tail = i % 4
    for row in range(o):
        r = raw[row * rb : (row + 1) * rb]
        for b, byte in enumerate(r.tolist()):
            valid = tail if b == rb - 1 and tail else 4
            for q in range(valid):
                if ((byte >> (2 * q)) & 3) == 3:
                    raise DeploymentArtifactError(f"packed layer {name} contains reserved code")
            for q in range(valid, 4):
                if ((byte >> (2 * q)) & 3) != 0:
                    raise DeploymentArtifactError(f"packed layer {name} nonzero row padding")


def load_cpu_artifact(path: str | Path) -> LoadedCPUArtifact:
    p = Path(path)
    if not p.is_file():
        raise DeploymentArtifactError(f"artifact not found: {p}")
    try:
        payload = torch.load(p, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(p, map_location="cpu")
    except (OSError, RuntimeError) as exc:
        raise DeploymentArtifactError(f"cannot load artifact: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DeploymentArtifactError("artifact root must be a mapping")
    payload = dict(payload)
    if payload.get("format") != M10_FORMAT or payload.get("version") != M10_VERSION:
        raise DeploymentArtifactError("unsupported M10 artifact format/version")
    if payload.get("backend_contract") != M10_BACKEND_CONTRACT:
        raise DeploymentArtifactError("unsupported M10 backend contract")
    arch = payload.get("architecture")
    task = payload.get("task")
    precision = payload.get("precision")
    if not isinstance(arch, Mapping) or not isinstance(task, Mapping) or not isinstance(precision, Mapping):
        raise DeploymentArtifactError("artifact metadata mappings are missing")
    expected_arch = {
        "class", "dim", "num_tokens", "seq_len", "n_layers", "heads", "mlp_ratio",
        "n", "T", "N_sup", "max_grid_size", "ternary", "act8", "gelu_approximate",
        "rmsnorm_eps", "act_quant_bits", "act_quant_qmin", "act_quant_qmax", "act_quant_eps",
    }
    if set(arch) != expected_arch:
        raise DeploymentArtifactError("artifact architecture metadata is incomplete or has unknown fields")
    if arch["class"] != "TRM" or arch["n_layers"] != 1 or arch["ternary"] is not True or arch["act8"] is not True:
        raise DeploymentArtifactError("artifact graph is outside the fixed M10 supported configuration")
    if arch["gelu_approximate"] != "none" or arch["rmsnorm_eps"] is not None:
        raise DeploymentArtifactError("artifact activation/normalization semantics are unsupported")
    if int(arch["act_quant_bits"]) != 8 or int(arch["act_quant_qmin"]) != -128 or int(arch["act_quant_qmax"]) != 127:
        raise DeploymentArtifactError("artifact recurrent quantization semantics are unsupported")
    if int(task.get("height", 0)) * int(task.get("width", 0)) != int(arch["seq_len"]):
        raise DeploymentArtifactError("artifact task geometry/seq_len mismatch")
    if task.get("task") != "sudoku":
        raise DeploymentArtifactError("M10 artifact supports Sudoku only")

    linears = payload.get("packed_linears")
    fp = payload.get("fp_tensors")
    if not isinstance(linears, Mapping) or not isinstance(fp, Mapping):
        raise DeploymentArtifactError("artifact inference tensors are missing")
    expected_linears = {
        "blocks.0.attn.q", "blocks.0.attn.k", "blocks.0.attn.v", "blocks.0.attn.proj",
        "blocks.0.ff.0", "blocks.0.ff.2", "out_head",
    }
    if set(linears) != expected_linears:
        raise DeploymentArtifactError("artifact packed-linear inventory mismatch")
    for name, entry in linears.items():
        if not isinstance(entry, Mapping):
            raise DeploymentArtifactError(f"packed layer {name} is not a mapping")
        _validate_packed_entry(str(name), entry)
    expected_fp = {
        "token_embed.weight", "pos_encoder.row_embed.weight", "pos_encoder.col_embed.weight",
        "blocks.0.norm1.weight", "blocks.0.norm2.weight", "norm_y.weight", "norm_z.weight",
        "halt_head.weight", "halt_head.bias", "alpha_y", "alpha_z",
    }
    if set(fp) != expected_fp:
        raise DeploymentArtifactError("artifact FP tensor inventory mismatch")
    for name, tensor in fp.items():
        if not torch.is_tensor(tensor) or tensor.device.type != "cpu" or tensor.dtype is not torch.float32 or not tensor.is_contiguous():
            raise DeploymentArtifactError(f"FP tensor {name} contract invalid")
        if not bool(torch.isfinite(tensor).all()):
            raise DeploymentArtifactError(f"FP tensor {name} contains non-finite values")

    inventory = payload.get("inventory")
    if not isinstance(inventory, Mapping):
        raise DeploymentArtifactError("artifact tensor hash inventory missing")
    for name, meta in inventory.get("fp_tensors", {}).items():
        if name not in fp or meta.get("sha256") != tensor_sha256(fp[name]):
            raise DeploymentArtifactError(f"artifact FP tensor hash mismatch: {name}")
    for name, meta in inventory.get("packed_linears", {}).items():
        if name not in linears:
            raise DeploymentArtifactError(f"artifact packed hash inventory unknown layer: {name}")
        entry = linears[name]
        if meta.get("packed_sha256") != tensor_sha256(entry["packed"]):
            raise DeploymentArtifactError(f"artifact packed-weight hash mismatch: {name}")
        if meta.get("scale_sha256") != tensor_sha256(entry["scale"]):
            raise DeploymentArtifactError(f"artifact scale hash mismatch: {name}")
        if meta.get("bias_sha256") != tensor_sha256(entry["bias"]):
            raise DeploymentArtifactError(f"artifact bias hash mismatch: {name}")
    return LoadedCPUArtifact(p, sha256_file(p), payload)
