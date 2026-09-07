#!/usr/bin/env python3
"""Render Amendment-A result table and Pareto plot from raw M14 JSONL artifacts."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRIMARY = "single_pass_fp"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = sorted({str(r["config_id"]) for r in rows})
    out=[]
    for cid in ids:
        rr=[r for r in rows if str(r["config_id"])==cid]
        lats=[float(r["latency_ms"]) for r in rr if r.get("latency_ms") is not None]
        seeds=sorted({str(r["seed"]) for r in rr})
        out.append({
            "config_id":cid,
            "family":rr[0].get("family"),
            "n_rows":len(rr),
            "n_seeds":len([s for s in seeds if s!="symbolic"]),
            "semantic_validity":float(np.mean([float(r["semantic_success"]) for r in rr])),
            "exact_reference_match":float(np.mean([float(r["exact_reference_match"]) for r in rr])),
            "blank_cell_accuracy":float(np.mean([float(r["blank_cell_accuracy"]) for r in rr])),
            "latency_n":len(lats),
            "latency_median_ms":float(np.median(lats)) if lats else None,
            "latency_p95_ms":float(np.quantile(lats,.95)) if lats else None,
        })
    by={r["config_id"]:r for r in out}; base=by.get(PRIMARY)
    base_lat=None if base is None else base.get("latency_median_ms")
    for r in out:
        lat=r.get("latency_median_ms")
        ratio=None if base_lat is None or lat is None else float(lat)/float(base_lat)
        r["latency_ratio_vs_primary_baseline"]=ratio
        r["latency_match_status"]=("matched" if ratio is not None and 0.85 <= ratio <= 1.15 else "unmatched")
        r["is_primary_baseline"]=r["config_id"]==PRIMARY
    for r in out:
        q=float(r["semantic_validity"]); l=r.get("latency_median_ms")
        dominated=False
        if l is not None:
            for s in out:
                sl=s.get("latency_median_ms")
                if sl is None or s is r: continue
                if float(s["semantic_validity"]) >= q and float(sl) <= float(l) and (
                    float(s["semantic_validity"]) > q or float(sl) < float(l)
                ):
                    dominated=True; break
        r["pareto_nondominated"]=not dominated
    return out

def write_table(rows:list[dict[str,Any]], csv_path:Path, md_path:Path)->None:
    keys=["config_id","family","n_rows","n_seeds","semantic_validity","exact_reference_match","blank_cell_accuracy","latency_n","latency_median_ms","latency_p95_ms","latency_ratio_vs_primary_baseline","latency_match_status","pareto_nondominated","is_primary_baseline"]
    with csv_path.open("w",newline="",encoding="utf-8") as fh:
        w=csv.DictWriter(fh,fieldnames=keys); w.writeheader(); w.writerows([{k:r.get(k) for k in keys} for r in rows])
    lines=["| config | semantic valid | median ms | p95 ms | ratio vs FP baseline | budget label | Pareto |","|---|---:|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['config_id']} | {r['semantic_validity']:.4f} | {r['latency_median_ms']:.4f} | {r['latency_p95_ms']:.4f} | {r['latency_ratio_vs_primary_baseline']:.3f} | {r['latency_match_status']} | {r['pareto_nondominated']} |")
    md_path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def render_plot(rows:list[dict[str,Any]], out:Path, candidate:str)->None:
    fig,ax=plt.subplots(figsize=(8.2,5.6))
    for r in rows:
        x=r.get("latency_median_ms")
        if x is None: continue
        y=float(r["semantic_validity"])
        marker="*" if r["config_id"]==candidate else ("s" if r["config_id"]==PRIMARY else "o")
        ax.scatter([x],[y],marker=marker,s=95 if marker=="*" else 55)
        ax.annotate(r["config_id"],(x,y),xytext=(5,4),textcoords="offset points",fontsize=8)
    ax.set_xlabel("Complete-solve median latency (ms)")
    ax.set_ylabel("Strict semantic validity")
    ax.set_title("M14 Amendment A — raw-artifact Pareto surface")
    ax.grid(True,alpha=.25)
    fig.tight_layout(); fig.savefig(out,dpi=180); plt.close(fig)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--experiment",required=True); ap.add_argument("--out",required=True); args=ap.parse_args()
    root=Path(args.experiment); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/"summary.json").read_text())
    if summary.get("confirmation_opened") and (root/"confirmation_rows.jsonl").is_file():
        raw=root/"confirmation_rows.jsonl"; surface="confirmation"
    else:
        raw=root/"development_rows_amendment_a.jsonl"; surface="development_amendment_a"
    rows=aggregate(load_jsonl(raw)); candidate=str(summary["selected_candidate"])
    csv_path=out/"result_table.csv"; md_path=out/"result_table.md"; plot=out/"pareto.png"
    write_table(rows,csv_path,md_path); render_plot(rows,plot,candidate)
    prov={
        "milestone":14,"development_amendment":"A","evidence_surface":surface,
        "raw_rows":str(raw),"raw_rows_sha256":sha256(raw),
        "summary":str(root/"summary.json"),"summary_sha256":sha256(root/"summary.json"),
        "result_table":str(csv_path),"result_table_sha256":sha256(csv_path),
        "pareto_plot":str(plot),"pareto_plot_sha256":sha256(plot),
        "primary_candidate":candidate,"primary_baseline":PRIMARY,
        "plot_source_of_truth":"raw JSONL rows; pixels are presentation only",
        "latency_matching_rule":"matched iff median ratio is within 0.85..1.15; unmatched points are not called iso-budget",
        "primary_gate_unchanged_from_m14":True,
    }
    (out/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n")
    print(json.dumps(prov,indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
