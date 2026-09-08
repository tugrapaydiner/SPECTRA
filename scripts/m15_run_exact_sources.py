#!/usr/bin/env python3
"""Run M15 using byte-identical accepted M14 source checkpoints.

This wrapper implements docs/M15_SOURCE_ARTIFACT_AMENDMENT.md. It changes only
source checkpoint materialization; all experiment logic remains in
scripts.m15_mechanism_ablations.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import torch

import scripts.m15_mechanism_ablations as m15
import scripts.m14_primary_experiment as m14
from eval.grounded_targets import tensor_state_sha256
from model.trm import TRM

M14_ARTIFACT_ID = 10077794916
M14_RUN_ID = 34280670349
M14_ARTIFACT_ZIP_SHA256 = "5729932600743b2cef19d9e0b9baf112ec75b627552af4d30a089266994a2f37"


def _find_one(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_file()]
    if len(matches) != 1:
        raise m15.M15Stop(f"expected exactly one {name!r} in accepted M14 artifact; found {len(matches)}")
    return matches[0]


def exact_reconstruct_sources(out: Path):
    archive_env = os.environ.get("M15_M14_ARTIFACT_ZIP")
    if not archive_env:
        raise m15.M15Stop("M15_M14_ARTIFACT_ZIP is required by the source-artifact amendment")
    archive = Path(archive_env)
    if not archive.is_file():
        raise m15.M15Stop(f"accepted M14 artifact archive is missing: {archive}")
    archive_sha = m15.sha256_file(archive)
    if archive_sha != M14_ARTIFACT_ZIP_SHA256:
        raise m15.M15Stop(
            f"accepted M14 artifact ZIP hash mismatch: observed={archive_sha} expected={M14_ARTIFACT_ZIP_SHA256}"
        )

    extracted = out / "accepted_m14_artifact"
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extracted)

    primary: dict[int, TRM] = {}
    records: list[dict[str, object]] = []
    for seed in m15.CORE_SEEDS:
        src = _find_one(extracted, f"fp_recursive_dim64_seed{seed}.pt")
        file_sha = m15.sha256_file(src)
        expected_file = m15.M14_FILE_SHA[seed]
        if file_sha != expected_file:
            raise m15.M15Stop(
                f"accepted M14 checkpoint file hash mismatch seed={seed}: observed={file_sha} expected={expected_file}"
            )
        model, kind, payload = m14.load_checkpoint(src)
        if kind != "fp_recursive_dim64" or not isinstance(model, TRM):
            raise m15.M15Stop(f"accepted M14 checkpoint restored wrong family for seed={seed}: {kind}")
        state_sha = tensor_state_sha256(model)
        expected_state = m15.EXPECTED_FP64_TENSOR_SHA[seed]
        if state_sha != expected_state:
            raise m15.M15Stop(
                f"accepted M14 tensor-state hash mismatch seed={seed}: observed={state_sha} expected={expected_state}"
            )
        primary[seed] = m15.freeze_model(model)
        records.append({
            "kind": kind,
            "seed": seed,
            "source": "exact_accepted_m14_actions_artifact",
            "artifact_id": M14_ARTIFACT_ID,
            "run_id": M14_RUN_ID,
            "artifact_zip_sha256": archive_sha,
            "checkpoint_path_in_extracted_artifact": str(src),
            "checkpoint_file_sha256": file_sha,
            "expected_checkpoint_file_sha256": expected_file,
            "checkpoint_file_identity_exact": True,
            "tensor_state_sha256": state_sha,
            "expected_tensor_state_sha256": expected_state,
            "tensor_identity_exact": True,
            "source_checkpoint_git_sha": payload.get("git_sha"),
        })

    report = {
        "materialization": "byte_identical_accepted_m14_actions_artifact",
        "source_amendment": "docs/M15_SOURCE_ARTIFACT_AMENDMENT.md",
        "failed_retrain_attempt_run": 34286178758,
        "failed_retrain_attempt_consumed_m15_scientific_result": False,
        "artifact_id": M14_ARTIFACT_ID,
        "artifact_run_id": M14_RUN_ID,
        "artifact_zip_sha256": archive_sha,
        "primary": records,
        "optional_precision_context": [
            {
                "status": "omitted_exact_context_state_not_supplied_in_primary_accepted_artifact",
                "mandatory_primary_precision_ablation": "fp32_vs_int8_search_state_storage_on_exact_fp64_cores_remains_enabled",
            }
        ],
    }
    m15.write_json(out / "source_checkpoint_reconstruction.json", report)
    return primary, {}, report


def main() -> int:
    m15.reconstruct_sources = exact_reconstruct_sources
    return m15.main()


if __name__ == "__main__":
    raise SystemExit(main())
