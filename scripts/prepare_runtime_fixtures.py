"""Freeze untrained and newly trained CPU execution fixtures, not a capability study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.datasets import build_sudoku_arrays
from deploy.m10_artifact import export_cpu_artifact, module_tensor_state_sha256
from model.trm import TRM
from train.losses import deep_supervision_loss

CONFIG = {
    "seed": 916027, "untrained_dims": [16, 64], "untrained_depths": [1, 4, 16],
    "trained_dims": [32, 64], "trained_depth": 4, "training_steps": 128,
    "training_examples": 256, "training_batch": 16, "learning_rate": 0.001,
    "input_batches": [1, 4], "inputs_per_batch": 2,
    "scope": "runtime cost/equivalence fixtures; no learned generalization claim",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    write(out / "config.json", CONFIG)
    write(out / "preparation_source.json", {str(p.relative_to(ROOT)): sha(p) for directory in
          ["scripts", "model", "data", "deploy", "train", "common"]
          for p in sorted((ROOT / directory).rglob("*.py"))})
    data_rng = np.random.default_rng(CONFIG["seed"])
    tx, ty, _, _ = build_sudoku_arrays(2, n=CONFIG["training_examples"], num_clues=8,
                                     rng=data_rng, require_unique=True, augment=True)
    np.savez_compressed(out / "training_arrays.npz", inputs=tx, targets=ty)
    # Exclude exact training puzzles. Symmetry-orbit generalization is NOT claimed.
    forbidden = {x.tobytes() for x in tx}
    inputs = []
    while len(inputs) < 10:
        xx, _, _, _ = build_sudoku_arrays(2, n=16, num_clues=8, rng=data_rng,
                                        require_unique=True, augment=True)
        for x in xx:
            if x.tobytes() not in forbidden:
                inputs.append(x.copy()); forbidden.add(x.tobytes())
                if len(inputs) == 10: break
    input_files = []
    offset = 0
    for batch in CONFIG["input_batches"]:
        for index in range(CONFIG["inputs_per_batch"]):
            path = out / f"input-b{batch}-i{index}.npy"
            np.save(path, np.stack(inputs[offset:offset + batch])); offset += batch
            input_files.append(dict(path=path.name, batch=batch, index=index, sha256=sha(path)))
    specs = [(dim, depth, False) for dim in CONFIG["untrained_dims"] for depth in CONFIG["untrained_depths"]]
    specs += [(dim, CONFIG["trained_depth"], True) for dim in CONFIG["trained_dims"]]
    fixtures = []
    for dim, depth, trained in specs:
        seed = CONFIG["seed"] + dim + depth + (10000 if trained else 0)
        torch.manual_seed(seed)
        name = f"{'trained' if trained else 'untrained'}-d{dim}-s{depth}"
        model = TRM(dim=dim, num_tokens=5, seq_len=16, n_layers=1, n=1, T=1,
                    N_sup=depth, heads=4, max_grid_size=8, ternary=True, act8=True).cpu()
        before = module_tensor_state_sha256(model)
        curve = []
        start = time.perf_counter_ns()
        if trained:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
            xtrain, ytrain = torch.from_numpy(tx).long(), torch.from_numpy(ty).long()
            generator = torch.Generator().manual_seed(seed + 1)
            for step in range(CONFIG["training_steps"]):
                ids = torch.randint(len(tx), (CONFIG["training_batch"],), generator=generator)
                outputs = model(xtrain[ids], height=4, width=4)[1]
                loss = deep_supervision_loss(outputs, ytrain[ids])
                if not bool(torch.isfinite(loss)): raise ValueError("nonfinite fixture training loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(norm)): raise ValueError("nonfinite fixture gradient")
                optimizer.step()
                curve.append(dict(step=step + 1, loss=float(loss.detach()), grad_norm=float(norm)))
            write(out / f"{name}-training.json", curve)
        training_ns = time.perf_counter_ns() - start
        after = module_tensor_state_sha256(model)
        if trained and before == after: raise AssertionError("training did not change the checkpoint")
        checkpoint = out / f"{name}-source.pt"
        torch.save({"model_state": model.state_dict(), "seed": seed, "training_steps": len(curve),
                    "config": CONFIG, "dim": dim, "depth": depth}, checkpoint)
        model.eval()
        artifact = out / f"{name}.pt"
        export_cpu_artifact(model, artifact, height=4, width=4, box=2,
            source_checkpoint_sha256=sha(checkpoint), source_checkpoint_tensor_sha256=after,
            training_seed=seed, training_step=len(curve),
            data_provenance={"fixture_only": True, "training_arrays_sha256": sha(out / "training_arrays.npz"),
                             "exact_training_puzzles_excluded_from_inputs": True,
                             "symmetry_disjointness_or_quality_improvement_claimed": False},
            export_git_sha="content hashes in preparation_source.json")
        fixtures.append(dict(name=name, artifact=artifact.name, sha256=sha(artifact),
                             trained=trained, dim=dim, depth=depth, seed=seed,
                             training_steps=len(curve), training_ns=training_ns,
                             before_tensor_sha256=before, after_tensor_sha256=after))
        print(name, "training_steps", len(curve), flush=True)
    write(out / "fixtures.json", dict(schema="spectra.runtime_fixtures.v1", config=CONFIG,
                                      fixtures=fixtures, inputs=input_files))
    write(out / "SHA256.json", {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
