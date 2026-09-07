#!/usr/bin/env python3
"""Render M14 Amendment-B table/Pareto plot directly from raw paired rows."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
PRIMARY='single_pass_fp'
def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p:Path):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def aggregate(rr:list[dict[str,Any]]):
    out=[]
    for cid in sorted({str(r['config_id']) for r in rr}):
        x=[r for r in rr if str(r['config_id'])==cid]; l=[float(r['latency_ms']) for r in x if r.get('latency_ms') is not None]
        steps=[int(r['work_realized_steps']) for r in x if r.get('work_realized_steps') is not None]
        out.append({'config_id':cid,'family':x[0].get('family'),'n_rows':len(x),'semantic_validity':float(np.mean([r['semantic_success'] for r in x])),
                    'exact_reference_match':float(np.mean([r['exact_reference_match'] for r in x])),'blank_cell_accuracy':float(np.mean([r['blank_cell_accuracy'] for r in x])),
                    'latency_n':len(l),'latency_median_ms':float(np.median(l)) if l else None,'latency_p95_ms':float(np.quantile(l,.95)) if l else None,
                    'mean_realized_steps':float(np.mean(steps)) if steps else None,'fraction_step1_stop':float(np.mean(np.asarray(steps)==1)) if steps else None})
    b=next(r for r in out if r['config_id']==PRIMARY); bl=b['latency_median_ms']
    for r in out:
        ratio=float(r['latency_median_ms'])/float(bl); r['latency_ratio_vs_primary_baseline']=ratio; r['latency_match_status']='matched' if .85<=ratio<=1.15 else 'unmatched'
    for r in out:
        r['pareto_nondominated']=not any(s is not r and s['semantic_validity']>=r['semantic_validity'] and s['latency_median_ms']<=r['latency_median_ms'] and (s['semantic_validity']>r['semantic_validity'] or s['latency_median_ms']<r['latency_median_ms']) for s in out)
    return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();root=Path(a.experiment);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    s=json.loads((root/'summary.json').read_text()); raw=root/('confirmation_rows.jsonl' if s.get('confirmation_opened') and (root/'confirmation_rows.jsonl').exists() else 'development_rows_amendment_b.jsonl'); surface='confirmation' if raw.name.startswith('confirmation') else 'development_amendment_b'
    agg=aggregate(rows(raw)); keys=list(agg[0].keys());
    with (out/'result_table.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(agg)
    md=['| config | semantic | median ms | ratio | mean steps | stop@1 | budget | Pareto |','|---|---:|---:|---:|---:|---:|---|---|']
    for r in agg:md.append(f"| {r['config_id']} | {r['semantic_validity']:.4f} | {r['latency_median_ms']:.4f} | {r['latency_ratio_vs_primary_baseline']:.3f} | {r['mean_realized_steps'] if r['mean_realized_steps'] is not None else 'n/a'} | {r['fraction_step1_stop'] if r['fraction_step1_stop'] is not None else 'n/a'} | {r['latency_match_status']} | {r['pareto_nondominated']} |")
    (out/'result_table.md').write_text('\n'.join(md)+'\n')
    fig,ax=plt.subplots(figsize=(8.2,5.6)); cand=s['selected_candidate']
    for r in agg:
        marker='*' if r['config_id']==cand else ('s' if r['config_id']==PRIMARY else 'o'); ax.scatter([r['latency_median_ms']],[r['semantic_validity']],marker=marker,s=95 if marker=='*' else 55);ax.annotate(r['config_id'],(r['latency_median_ms'],r['semantic_validity']),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Complete-solve median latency (ms)');ax.set_ylabel('Strict semantic validity');ax.set_title('M14 Amendment B — validator-gated adaptive depth');ax.grid(True,alpha=.25);fig.tight_layout();fig.savefig(out/'pareto.png',dpi=180);plt.close(fig)
    p={'milestone':14,'development_amendment':'B','evidence_surface':surface,'raw_rows':str(raw),'raw_rows_sha256':sha(raw),'summary':str(root/'summary.json'),'summary_sha256':sha(root/'summary.json'),'result_table':str(out/'result_table.csv'),'result_table_sha256':sha(out/'result_table.csv'),'pareto_plot':str(out/'pareto.png'),'pareto_plot_sha256':sha(out/'pareto.png'),'primary_candidate':cand,'primary_baseline':PRIMARY,'plot_source_of_truth':'raw JSONL rows; pixels are presentation only','latency_matching_rule':'matched iff median ratio is within 0.85..1.15; unmatched points are not called iso-budget','primary_gate_unchanged_from_m14':True}
    (out/'provenance.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');print(json.dumps(p,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())