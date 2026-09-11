"""Paired puzzle-level summaries; repeated model seeds are not new puzzles."""
from __future__ import annotations
import math
from typing import Any
import numpy as np


def finite_array(values,*,ndim=None):
    out=np.asarray(values,dtype=np.float64)
    if (ndim is not None and out.ndim!=ndim) or not np.isfinite(out).all():
        raise ValueError('invalid/non-finite numeric array')
    return out


def probability_metrics(labels,probs) -> dict[str,Any]:
    y=finite_array(labels,ndim=1);p=finite_array(probs,ndim=1)
    if not len(y) or y.shape!=p.shape or (y<0).any() or (y>1).any() or (p<0).any() or (p>1+1e-6).any():
        raise ValueError('probability/label domain mismatch')
    p=np.clip(p,0,1);binary=bool(np.isin(y,[0,1]).all());bins=[];ece=0.
    bucket=np.minimum((p*10).astype(np.int64),9)
    for i in range(10):
        low=i/10
        selected=bucket==i
        if selected.any():
            frequency=float(y[selected].mean());confidence=float(p[selected].mean())
            ece+=float(selected.mean())*abs(frequency-confidence)
            bins.append({'low':float(low),'n':int(selected.sum()),'mean_label':frequency,'mean_prediction':confidence})
    auc=None
    if binary and y.min()!=y.max():
        # Mann-Whitney rank AUC with average ranks for ties; no quadratic pair matrix.
        order=np.argsort(p,kind='stable');rank=np.empty(len(p));i=0
        while i<len(order):
            j=i+1
            while j<len(order) and p[order[j]]==p[order[i]]:j+=1
            rank[order[i:j]]=(i+1+j)/2;i=j
        positives=int(y.sum());negatives=len(y)-positives
        auc=float((rank[y==1].sum()-positives*(positives+1)/2)/(positives*negatives))
    return {'n':len(y),'mean_label':float(y.mean()),'mean_prediction':float(p.mean()),
            'brier_or_soft_target_mse':float(np.mean((p-y)**2)),'ece10':ece,'binary_labels':binary,
            'roc_auc':auc,'bins':bins,'calibration_guarantee':False}


def paired_bootstrap(a,b,*,replicates=2000,seed=160990,statistic='difference') -> dict[str,Any]:
    """Arrays [fixed cores, common puzzles]. Resample the SAME puzzle indices.

    The interval is conditional on these fitted cores. It does not estimate a
    population-level training-seed distribution, and is not simultaneous across
    a family of exploratory comparisons.
    """
    aa=finite_array(a,ndim=2);bb=finite_array(b,ndim=2)
    if aa.shape!=bb.shape or min(aa.shape)<1 or type(replicates)is not int or replicates<100:
        raise ValueError('paired bootstrap shape/count mismatch')
    if statistic not in ('difference','ratio'):raise ValueError('unsupported statistic')
    if statistic=='ratio' and (bb<=0).any():raise ValueError('ratio denominator must be positive')
    x=aa.mean(axis=0);y=bb.mean(axis=0);rng=np.random.default_rng(seed);samples=[]
    for start in range(0,replicates,128):
        idx=rng.integers(0,len(x),size=(min(128,replicates-start),len(x)))
        mx=x[idx].mean(axis=1);my=y[idx].mean(axis=1)
        samples.extend((mx-my if statistic=='difference' else mx/my).tolist())
    point=float(x.mean()-y.mean()) if statistic=='difference' else float(x.mean()/y.mean())
    return {'statistic':statistic,'point':point,'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'replicates':replicates,'bootstrap_seed':seed,'unique_puzzles':aa.shape[1],'fixed_cores':aa.shape[0],
            'unit':'common puzzle resampled jointly across fixed cores',
            'training_seed_population_inference':False,'simultaneous_multiple_comparison_interval':False}


def latency_summary(values) -> dict[str,Any]:
    a=finite_array(values,ndim=1)
    if not len(a) or (a<0).any():raise ValueError('latencies must be nonnegative and nonempty')
    return {'n':len(a),'mean_ms':float(a.mean()),'median_ms':float(np.median(a)),
            'p95_ms':float(np.quantile(a,.95)),'min_ms':float(a.min()),'max_ms':float(a.max())}


def pool_selection(validity, scores) -> dict[str,Any]:
    valid=np.asarray(validity)
    if valid.ndim!=2 or valid.dtype!=np.bool_ or min(valid.shape)<1:raise ValueError('validity requires bool[N,P]')
    score=finite_array(scores,ndim=2)
    if score.shape!=valid.shape:raise ValueError('pool score geometry mismatch')
    selected=score.argmax(axis=1);coverage=valid.any(axis=1);returned=valid[np.arange(len(valid)),selected]
    failures=coverage&~returned
    return {'selected_indices':selected.tolist(),'coverage':coverage.tolist(),'returned':returned.tolist(),
            'covered_puzzles':int(coverage.sum()),'returned_valid':int(returned.sum()),
            'selection_failures_given_coverage':int(failures.sum()),
            'conditional_selection_success':float(returned.sum()/coverage.sum()) if coverage.any() else None,
            'tie_rule':'first maximum in fixed pool order'}
