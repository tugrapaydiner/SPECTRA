"""Risk-control calibration, NOT a per-input or distribution-shift guarantee."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta
THRESHOLDS=(0.,.25,.5,1.,2.,3.,4.,6.,8.,12.,16.,24.,32.)

def upper_bound(k:int,n:int,delta:float)->float:
    if type(k) is not int or type(n) is not int or not 0<=k<=n or n<1:
        raise ValueError('integer binomial count required')
    if type(delta) not in (float,int) or not 0<delta<1:raise ValueError('invalid confidence level')
    return 1. if k==n else float(beta.ppf(1-delta,k+1,n-k))

def calibrate(gap,fast,slow,truth,*,delta=.05,budgets=(.005,.01,.02)):
    gap=np.asarray(gap,dtype=np.float64);fast=np.asarray(fast);slow=np.asarray(slow);truth=np.asarray(truth)
    if gap.ndim!=1 or not len(gap) or any(x.shape!=gap.shape for x in (fast,slow,truth)):
        raise ValueError('aligned nonempty calibration arrays required')
    if not np.isfinite(gap).all() or np.any(gap<0):raise ValueError('invalid gap')
    if any(type(x) not in (float,int) or not 0<x<1 for x in budgets):raise ValueError('invalid risk budget')
    harmful=(fast!=truth)&(slow==truth);rows=[]
    for threshold in THRESHOLDS:
        accept=gap>=threshold;k=int(np.sum(accept&harmful))
        rows.append({'threshold':threshold,'accepted':int(accept.sum()),'n':len(gap),'harmful':k,
                     'upper_added_harm':upper_bound(k,len(gap),delta/len(THRESHOLDS))})
    policies={}
    for budget in budgets:
        candidates=[r for r in rows if r['upper_added_harm']<=budget]
        chosen=candidates[0] if candidates else {'threshold':None,'accepted':0,'n':len(gap),'harmful':0,'upper_added_harm':0.}
        policies[str(budget)]={**chosen,'risk_budget':budget,'accept_fraction':chosen['accepted']/len(gap),
                              'always_strong':chosen['threshold'] is None}
    return {'rows':rows,'policies':policies,'delta':delta,'finite_threshold_count':len(THRESHOLDS),
            'scope':'relative added harm under iid independent calibration; no absolute error or shift guarantee'}
