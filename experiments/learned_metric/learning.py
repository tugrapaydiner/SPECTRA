"""Training-only diagonal metric learning and deterministic integer compilation.

This is a bounded supervised metric-learning experiment, not a novel general
metric-learning algorithm. Inference consumes only the exported integer weights.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import math
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit, logsumexp
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def checked_codes(values, maximum: int) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 2 or not 1 <= x.shape[1] <= 4096 or x.shape[0] < 1:
        raise ValueError('expected nonempty two-dimensional codes')
    if type(maximum) is not int or not 1 <= maximum <= 255:
        raise ValueError('maximum must be an integer in1..255')
    if x.dtype.kind not in 'iuf' or not np.isfinite(x).all() or (x < 0).any() or (x > maximum).any() or not np.equal(x, np.rint(x)).all():
        raise ValueError('only exact finite integer grid members are accepted')
    return np.ascontiguousarray(x, dtype=np.float64)


def quantize_weights(values, *, units: int = 2) -> np.ndarray:
    """Largest remainder quantization, preserving exactly units*dimensions mass.

    All weights nonnegative; a deterministic lower-index tie rule. This is a
    learned metric definition, not an approximation to be substituted at inference.
    """
    w = np.asarray(values, dtype=np.float64)
    if w.ndim != 1 or not 1 <= len(w) <= 4096 or not np.isfinite(w).all() or (w < 0).any() or not w.sum() > 0:
        raise ValueError('nonnegative finite nonzero weight vector required')
    if type(units) is not int or not 1 <= units <= 8:
        raise ValueError('units must be in1..8')
    target = units * len(w)
    scaled = w * (target / w.sum())
    integer = np.floor(scaled).astype(np.int64)
    remainder = target - int(integer.sum())
    order = np.lexsort((np.arange(len(w)), -(scaled - integer)))
    integer[order[:remainder]] += 1
    if integer.max() > 255 or integer.sum() != target:
        raise ValueError('quantized geometry exceeds supported bounds')
    return integer.astype(np.uint16)


def subset(y, cap, seed):
    indices = np.arange(len(y))
    if len(y) <= cap:
        return indices
    chosen, _ = train_test_split(indices, train_size=cap, stratify=y, random_state=seed)
    return np.sort(chosen)


def split_development(q, y, seed):
    """Keep identical feature vectors together before applying common size caps."""
    _, groups = np.unique(q, axis=0, return_inverse=True)
    train, validation = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(q, y, groups))
    train = train[subset(y[train], 6000, seed + 1)]
    validation = validation[subset(y[validation], 2000, seed + 2)]
    if set(groups[train]) & set(groups[validation]):
        raise AssertionError('duplicate feature group crosses validation boundary')
    return train, validation


def _triplets(x, y, weights, seed):
    gallery = subset(y, 4096, seed)
    anchors = subset(y, 2048, seed + 1)
    target = x[gallery]
    ys = y[gallery]
    pull, push = [], []
    for start in range(0, len(anchors), 128):
        ix = anchors[start:start+128]
        distances = cdist(x[ix], target, 'sqeuclidean', w=weights)
        distances[ix[:, None] == gallery[None, :]] = np.inf
        for r, i in enumerate(ix):
            same = np.flatnonzero(ys == y[i]); other = np.flatnonzero(ys != y[i])
            if len(same) < 4 or len(other) < 3:
                raise ValueError('insufficient training neighbors')
            pos = same[np.argsort(distances[r, same], kind='stable')[:3]]
            neg = other[np.argsort(distances[r, other], kind='stable')[:3]]
            pd = (x[i] - target[pos]) ** 2
            nd = (x[i] - target[neg]) ** 2
            pull.append(np.repeat(pd, 3, axis=0))
            push.append(np.tile(nd, (3, 1)))
    return np.concatenate(pull), np.concatenate(push)


def fit_weights(q, y, maximum, *, method='margin', seed=20260928):
    x = checked_codes(q, maximum) / maximum
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != len(x):
        raise ValueError('training labels do not match rows')
    d = x.shape[1]
    begin = time.process_time()
    if method == 'uniform':
        continuous = np.ones(d)
        history = []
    elif method == 'variance':
        variance = np.var(x, axis=0)
        continuous = 1 / np.maximum(variance, .01 * float(np.mean(variance)) + 1e-12)
        continuous = np.minimum(continuous / np.mean(continuous), 8.)
        history = []
    elif method == 'nca':
        continuous, history = fit_neighborhood(x, labels, seed)
    elif method == 'margin':
        continuous = np.ones(d)
        history = []
        for iteration in range(3):
            positive, negative = _triplets(x, labels, continuous, seed + iteration * 97)
            scale = max(float(np.median(np.sum(positive, axis=1))), 1e-4)
            delta = (positive - negative) / scale
            pull = positive.mean(axis=0) / scale
            def objective(w):
                margin = 1. + delta @ w
                # Smooth logistic large-margin loss plus pull and identity prior.
                loss = np.logaddexp(0., margin).mean() + .05 * pull @ w + .01 * np.mean((w - 1.)**2)
                gradient = delta.T @ expit(margin) / len(delta) + .05 * pull + .02 * (w - 1.) / d
                return float(loss), gradient
            result = minimize(objective, continuous, jac=True, method='SLSQP',
                bounds=[(0., 8.)]*d,
                constraints={'type':'eq','fun':lambda w: w.sum()-d,'jac':lambda w: np.ones(d)},
                options={'maxiter':120,'ftol':1e-8})
            if not np.isfinite(result.x).all() or (result.x < -1e-6).any():
                raise RuntimeError('invalid metric optimizer result')
            continuous = np.maximum(result.x, 0.)
            continuous *= d / continuous.sum()
            history.append({'round':iteration,'triplets':len(delta),'scale':scale,
                'objective':float(result.fun),'iterations':int(result.nit),
                'converged':bool(result.success),'message':str(result.message),
                'weights':continuous.tolist()})
    else:
        raise ValueError('unknown metric method')
    integer = quantize_weights(continuous)
    return integer, {'method':method,'seed':seed,'training_rows':len(x),'features':d,
        'continuous_weights':continuous.tolist(),'integer_weights':integer.tolist(),
        'history':history,'cpu_seconds':time.process_time()-begin}


def kernel_table(weights, maximum, gamma):
    w = np.asarray(weights)
    if w.ndim != 1 or w.dtype.kind not in 'iu' or (w < 0).any() or w.max() > 255 or not w.sum()>0:
        raise ValueError('invalid integer weights')
    length = int(w.sum()) * maximum * maximum + 1
    if length > 4_194_304 or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('kernel table exceeds budget or invalid gamma')
    coefficient = float(gamma)/(maximum*maximum*float(w.mean()))
    table = np.fromiter((math.exp(-coefficient * s) for s in range(length)), dtype=np.float64, count=length)
    return table, coefficient


def kernel_matrix(a, b, weights, table, *, path=None):
    """Bounded-block exact integer signatures, optionally backed by a disk file."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or len(w)!=a.shape[1]:
        raise ValueError('kernel geometry mismatch')
    result = np.empty((len(a),len(b)),dtype=np.float64) if path is None else np.memmap(path, mode='w+',dtype=np.float64,shape=(len(a),len(b)))
    for start in range(0,len(a),128):
        distances = cdist(a[start:start+128],b,'sqeuclidean',w=w)
        # All factors are bounded integers; no rounding or input quantization.
        codes=distances.astype(np.int64)
        if not np.equal(codes,distances).all() or (codes<0).any() or (codes>=len(table)).any():
            raise ValueError('invalid integer signature')
        result[start:start+128]=table[codes]
    if path is not None: result.flush()
    return result


def fit_neighborhood(x, y, seed):
    """NCA-style supervised probability objective, restricted to positive diagonal.

    One fixed gallery and anchor sample. Self matches excluded. The optimizer
    sees training labels only. A .5 floor keeps every original feature alive.
    """
    d=x.shape[1]
    gallery=subset(y,2048,seed)
    anchors=subset(y,512,seed+1)
    costs=np.stack([(x[anchors,j,None]-x[gallery,j][None,:])**2 for j in range(d)])
    same=y[anchors,None]==y[gallery][None,:]
    self_match=anchors[:,None]==gallery[None,:]
    same[self_match]=False
    distance=costs.sum(axis=0)
    distance[~same]=np.inf
    nearest=np.partition(distance,2,axis=1)[:,2]
    beta=1./max(float(np.median(nearest)),1e-4)
    history=[]
    def objective(w):
        logits=-beta*np.tensordot(w,costs,axes=1)
        logits[self_match]=-np.inf
        z=logsumexp(logits,axis=1,keepdims=True)
        positive=np.where(same,logits,-np.inf)
        pz=logsumexp(positive,axis=1,keepdims=True)
        loss=float(np.mean(z-pz)+.02*np.mean((w-1.)**2))
        residual=np.exp(positive-pz)-np.exp(logits-z)
        grad=beta*np.tensordot(costs,residual,axes=((1,2),(0,1)))/len(anchors)+.04*(w-1.)/d
        return loss,grad
    result=minimize(objective,np.ones(d),jac=True,method='SLSQP',bounds=[(.5,4.)]*d,
        constraints={'type':'eq','fun':lambda w:w.sum()-d,'jac':lambda w:np.ones(d)},
        options={'maxiter':100,'ftol':1e-8})
    w=np.asarray(result.x);w=np.maximum(w,.5);w*=d/w.sum()
    history.append({'objective':float(result.fun),'iterations':int(result.nit),'converged':bool(result.success),
                    'message':str(result.message),'gallery':len(gallery),'anchors':len(anchors),'beta':beta,'weights':w.tolist()})
    return w,history
