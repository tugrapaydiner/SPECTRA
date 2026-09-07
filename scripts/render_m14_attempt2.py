#!/usr/bin/env python3
"""Render M14 Attempt 2 table/Pareto plot directly from raw paired rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def read_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def read_jsonl(path: Path): return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def aggregate(rows: list[dict[str, Any]]):
    out=[]
    for cid in sorted({str(r["config_id"]) for r in rows}):
        rr=[r for r in rows if str(r["config_id"])==cid]
        lats=[float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
        out.append({
            "config_id":cid,"family":rr[0]["family"],"n_rows":len(rr),
            "n_seeds":len({str(r["seed"]) for r in rr}),
            "semantic_validity":float(np.mean([float(r["semantic_success"]) for r in rr])),
            "exact_reference_match":float(np.mean([float(r["exact_reference_match"]) for r in rr])),
            "blank_cell_accuracy":float(np.mean([float(r["blank_cell_accuracy"]) for r in rr])),
            "latency_n":len(lats),"latency_median_ms":float(np.median(lats)) if lats else None,
            "latency_p95_ms":float(np.quantile(lats,.95)) if lats else None,
        })
    return out


def front_ids(rows):
    pts=[r for r in rows if r.get("latency_median_ms") is not None]; out=set()
    for a in pts:
        dominated=False
        for b in pts:
            if a is b: continue
            if float(b["latency_median_ms"]) <= float(a["latency_median_ms"]) and float(b["semantic_validity"]) >= float(a["semantic_validity"]):
                if float(b["latency_median_ms"]) < float(a["latency_median_ms"]) or float(b["semantic_validity"]) > float(a["semantic_validity"]):
                    dominated=True; break
        if not dominated: out.add(str(a["config_id"]))
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--experiment",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    root=Path(args.experiment); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    summary=read_json(root/"summary.json")
    raw_path=root/("confirmation_rows.jsonl" if summary.get("confirmation_opened") else "development_rows.jsonl")
    surface="confirmation" if summary.get("confirmation_opened") else "development"
    rows=read_jsonl(raw_path); agg=aggregate(rows); by={r["config_id"]:r for r in agg}; base=by["single_pass_fp"]
    base_lat=float(base["latency_median_ms"]); front=front_ids(agg)
    for r in agg:
        lat=r.get("latency_median_ms"); ratio=None if lat is None else float(lat)/base_lat
        r["latency_ratio_vs_primary_baseline"]=ratio
        r["latency_match_status"]="matched" if ratio is not None and .85 <= ratio <= 1.15 else "unmatched"
        r["pareto_nondominated_on_surface"]=str(r["config_id"]) in front
        r["primary_candidate"]=str(r["config_id"])==str(summary["selected_candidate"])
        r["primary_baseline"]=str(r["config_id"])=="single_pass_fp"
        r["evidence_surface"]=surface
    table=out/"result_table.csv"
    keys=list(agg[0].keys())
    with table.open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows(agg)
    md=[f"# M14 Attempt 2 — {surface}","",f"Status: **{summary['status']}**","","| config | validity | median ms | matched | Pareto |","|---|---:|---:|---|---|"]
    for r in sorted(agg,key=lambda x:(-(x["semantic_validity"]),x.get("latency_median_ms") or 1e30)):
        md.append(f"| {r['config_id']} | {r['semantic_validity']:.4f} | {r['latency_median_ms']:.4f} | {r['latency_match_status']} | {r['pareto_nondominated_on_surface']} |")
    (out/"result_table.md").write_text("\n".join(md)+"\n",encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,6))
    for r in agg:
        if r.get("latency_median_ms") is None: continue
        marker="*" if r["primary_candidate"] else ("s" if r["primary_baseline"] else "o")
        ax.scatter(float(r["latency_median_ms"]),float(r["semantic_validity"]),marker=marker,s=150 if marker=="*" else 65)
        ax.annotate(str(r["config_id"]),(float(r["latency_median_ms"]),float(r["semantic_validity"])),xytext=(4,4),textcoords="offset points",fontsize=8)
    fp=sorted([r for r in agg if r["pareto_nondominated_on_surface"] and r.get("latency_median_ms") is not None],key=lambda x:float(x["latency_median_ms"]))
    if len(fp)>1: ax.plot([r["latency_median_ms"] for r in fp],[r["semantic_validity"] for r in fp],linestyle="--",linewidth=1)
    ax.set_xlabel("Measured complete-solve latency (ms, median)"); ax.set_ylabel("Strict Sudoku semantic validity")
    ax.set_title(f"M14 Attempt 2 {surface}: quality / complete-solve latency")
    ax.grid(True,alpha=.25); fig.tight_layout(); plot=out/"pareto.png"; fig.savefig(plot,dpi=180); plt.close(fig)
    prov={
        "attempt":2,"surface":surface,"raw_rows":str(raw_path),"raw_rows_sha256":sha(raw_path),
        "summary":str(root/"summary.json"),"summary_sha256":sha(root/"summary.json"),
        "result_table":str(table),"result_table_sha256":sha(table),"pareto_plot":str(plot),"pareto_plot_sha256":sha(plot),
        "primary_candidate":summary["selected_candidate"],"primary_baseline":"single_pass_fp",
        "latency_matching_rule":"matched iff ratio within 0.85..1.15; unmatched points are not iso-budget",
        "source_of_truth":"raw JSONL rows",
    }
    (out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(prov,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
