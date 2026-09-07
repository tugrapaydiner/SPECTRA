#!/usr/bin/env python3
"""Regenerate the M14 result table and Pareto plot directly from raw JSONL rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]
    for cid in sorted({str(r["config_id"]) for r in rows}):
        rr=[r for r in rows if str(r["config_id"])==cid]
        lats=[float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
        out.append({
            "config_id":cid,"family":rr[0]["family"],
            "n_rows":len(rr),"n_seeds":len({str(r["seed"]) for r in rr}),
            "semantic_validity":float(np.mean([float(r["semantic_success"]) for r in rr])),
            "exact_reference_match":float(np.mean([float(r["exact_reference_match"]) for r in rr])),
            "blank_cell_accuracy":float(np.mean([float(r["blank_cell_accuracy"]) for r in rr])),
            "latency_n":len(lats),"latency_median_ms":float(np.median(lats)) if lats else None,
            "latency_p95_ms":float(np.quantile(lats,.95)) if lats else None,
        })
    return out


def choose_final_rows(root: Path, summary: dict[str, Any]) -> tuple[str, Path, list[dict[str, Any]]]:
    if summary.get("confirmation_opened") and (root/"confirmation_rows.jsonl").exists():
        p=root/"confirmation_rows.jsonl"; return "confirmation",p,read_jsonl(p)
    if summary.get("reserve_intervention_used") and (root/"development_rows_reserve_dim64.jsonl").exists():
        p=root/"development_rows_reserve_dim64.jsonl"; return "development_reserve_dim64",p,read_jsonl(p)
    p=root/"development_rows_main.jsonl"; return "development_main",p,read_jsonl(p)


def pareto_front(rows: list[dict[str, Any]]) -> set[str]:
    pts=[r for r in rows if r.get("latency_median_ms") is not None]
    front=set()
    for a in pts:
        dominated=False
        for b in pts:
            if a is b: continue
            if float(b["latency_median_ms"]) <= float(a["latency_median_ms"]) and float(b["semantic_validity"]) >= float(a["semantic_validity"]):
                if float(b["latency_median_ms"]) < float(a["latency_median_ms"]) or float(b["semantic_validity"]) > float(a["semantic_validity"]):
                    dominated=True; break
        if not dominated: front.add(str(a["config_id"]))
    return front


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--experiment",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    root=Path(args.experiment); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    summary=read_json(root/"summary.json")
    surface,raw_path,raw=choose_final_rows(root,summary)
    agg=aggregate(raw)
    by={r["config_id"]:r for r in agg}; base=by.get("single_pass_fp")
    base_lat=None if base is None else base.get("latency_median_ms")
    front=pareto_front(agg)
    for r in agg:
        lat=r.get("latency_median_ms")
        ratio=None if lat is None or base_lat is None else float(lat)/float(base_lat)
        r["latency_ratio_vs_primary_baseline"]=ratio
        r["latency_match_status"]="matched" if ratio is not None and .85 <= ratio <= 1.15 else "unmatched"
        r["pareto_nondominated_on_plotted_surface"]=str(r["config_id"]) in front
        r["evidence_surface"]=surface
        r["primary_candidate"]=str(r["config_id"])==str(summary["selected_candidate"])
        r["primary_baseline"]=str(r["config_id"])=="single_pass_fp"
    table=out/"result_table.csv"
    if agg:
        keys=list(agg[0].keys())
        with table.open("w",newline="",encoding="utf-8") as fh:
            w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(agg)
    else: table.write_text("",encoding="utf-8")
    md=[f"# M14 result table — {surface}","",f"Claim status: **{summary['status']}**", "", "| config | semantic validity | median ms | match | Pareto |", "|---|---:|---:|---|---|"]
    for r in sorted(agg,key=lambda x:(-(x["semantic_validity"]),x.get("latency_median_ms") or 1e30)):
        lat="n/a" if r["latency_median_ms"] is None else f"{r['latency_median_ms']:.4f}"
        md.append(f"| {r['config_id']} | {r['semantic_validity']:.4f} | {lat} | {r['latency_match_status']} | {r['pareto_nondominated_on_plotted_surface']} |")
    (out/"result_table.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts=[r for r in agg if r.get("latency_median_ms") is not None]
    fig,ax=plt.subplots(figsize=(9.5,6.2))
    for r in pts:
        marker="*" if r["primary_candidate"] else ("s" if r["primary_baseline"] else "o")
        size=150 if marker=="*" else 70
        ax.scatter(float(r["latency_median_ms"]),float(r["semantic_validity"]),marker=marker,s=size)
        ax.annotate(str(r["config_id"]),(float(r["latency_median_ms"]),float(r["semantic_validity"])),xytext=(5,4),textcoords="offset points",fontsize=8)
    front_pts=sorted([r for r in pts if r["pareto_nondominated_on_plotted_surface"]],key=lambda x:float(x["latency_median_ms"]))
    if len(front_pts)>=2:
        ax.plot([float(r["latency_median_ms"]) for r in front_pts],[float(r["semantic_validity"]) for r in front_pts],linestyle="--",linewidth=1.2)
    ax.set_xlabel("Measured complete-solve latency (ms, median)")
    ax.set_ylabel("Strict Sudoku semantic validity")
    ax.set_title(f"M14 {surface}: learned-model quality / latency operating points")
    ax.grid(True,alpha=.25)
    fig.tight_layout(); plot=out/"pareto.png"; fig.savefig(plot,dpi=180); plt.close(fig)

    provenance={
        "renderer":"scripts/render_m14_results.py","evidence_surface":surface,
        "raw_rows":str(raw_path),"raw_rows_sha256":sha256(raw_path),
        "summary":str(root/"summary.json"),"summary_sha256":sha256(root/"summary.json"),
        "result_table":str(table),"result_table_sha256":sha256(table),
        "pareto_plot":str(plot),"pareto_plot_sha256":sha256(plot),
        "primary_candidate":summary["selected_candidate"],"primary_baseline":"single_pass_fp",
        "latency_matching_rule":"matched iff validation/final median ratio is within 0.85..1.15; unmatched points are not called iso-budget",
        "plot_source_of_truth":"raw JSONL rows; pixels are presentation only",
        "claim_confirmed":bool(summary.get("claim_confirmed")),
    }
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(provenance,indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
