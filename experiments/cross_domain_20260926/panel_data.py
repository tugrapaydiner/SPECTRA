"""Deterministic parsing/partitioning for the frozen UCI executor panel.

No estimator is imported here. Original row IDs and provided test boundaries are
retained. Sampling caps operate only on fit/validation portions.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

SEED = 20260926
TASKS = ('wine', 'vehicle', 'satellite', 'har', 'sensorless')
DIMS = dict(wine=13, vehicle=18, satellite=36, har=561, sensorless=48)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(folder: Path, task: str):
    if task not in TASKS:
        raise ValueError('unknown panel task')
    archive = zipfile.ZipFile(folder / f'{task}.zip')
    if archive.testzip() is not None:
        raise ValueError('archive CRC failure')
    boundary = None
    groups = None
    if task == 'wine':
        a = np.loadtxt(io.BytesIO(archive.read('wine.data')), delimiter=',')
        x, y = a[:, 1:], a[:, 0].astype(np.int64)
    elif task == 'vehicle':
        names = sorted(n for n in archive.namelist() if n.endswith('.dat'))
        rows = [line.split() for n in names for line in archive.read(n).decode().splitlines() if line.strip()]
        x = np.asarray([r[:-1] for r in rows], dtype=np.float64)
        y = np.asarray([r[-1] for r in rows])
    elif task == 'satellite':
        a, b = [np.loadtxt(io.BytesIO(archive.read(n))) for n in ('sat.trn', 'sat.tst')]
        boundary = len(a)
        a = np.concatenate([a, b])
        x, y = a[:, :-1], a[:, -1].astype(np.int64)
    elif task == 'har':
        inner = zipfile.ZipFile(io.BytesIO(archive.read('UCI HAR Dataset.zip')))
        arrays, targets, subjects = [], [], []
        for part in ('train', 'test'):
            prefix = f'UCI HAR Dataset/{part}/'
            arrays.append(np.loadtxt(io.BytesIO(inner.read(prefix+f'X_{part}.txt'))))
            targets.append(np.loadtxt(io.BytesIO(inner.read(prefix+f'y_{part}.txt')), dtype=np.int64))
            subjects.append(np.loadtxt(io.BytesIO(inner.read(prefix+f'subject_{part}.txt')), dtype=np.int64))
        boundary = len(arrays[0])
        x, y, groups = np.concatenate(arrays), np.concatenate(targets), np.concatenate(subjects)
    else:
        a = np.loadtxt(io.BytesIO(archive.read('Sensorless_drive_diagnosis.txt')))
        x, y = a[:, :-1], a[:, -1].astype(np.int64)
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != DIMS[task] or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError('invalid data dimensions or values')
    return x, y, groups, boundary


def cap_indices(ids, y, cap):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) <= cap:
        return np.sort(ids)
    kept, _ = train_test_split(ids, train_size=cap, stratify=y[ids], random_state=SEED)
    return np.sort(kept)


def split_indices(x, y, task, groups=None, boundary=None):
    ids = np.arange(len(y))
    if task in ('wine', 'vehicle'):
        dev, test = train_test_split(ids, test_size=.25, stratify=y, random_state=SEED)
        fit, val = train_test_split(dev, test_size=.20, stratify=y[dev], random_state=SEED)
    elif task in ('satellite', 'har'):
        if not boundary or not 0 < boundary < len(y):
            raise ValueError('missing official boundary')
        dev, test = ids[:boundary], ids[boundary:]
        if task == 'har':
            if groups is None or set(groups[dev]) & set(groups[test]):
                raise ValueError('official subject overlap')
            a, b = next(GroupShuffleSplit(n_splits=1, test_size=.20, random_state=SEED).split(dev, y[dev], groups[dev]))
            fit, val = dev[a], dev[b]
        else:
            fit, val = train_test_split(dev, test_size=.20, stratify=y[dev], random_state=SEED)
    elif task == 'sensorless':
        fit, val, test = [], [], []
        for label in np.unique(y):
            cls = ids[y == label]; a, b = int(.6*len(cls)), int(.8*len(cls))
            fit.extend(cls[:a]); val.extend(cls[a:b]); test.extend(cls[b:])
        dev = np.asarray(fit + val)
    else:
        raise ValueError('unknown task')
    fit, val = cap_indices(fit, y, 4096), cap_indices(val, y, 2048)
    test = np.sort(np.asarray(test, dtype=np.int64))
    if (set(fit) & set(val)) or (set(fit) & set(test)) or (set(val) & set(test)):
        raise ValueError('partition overlap')
    labels = set(np.unique(y))
    if any(set(np.unique(y[s])) != labels for s in (fit, val, test)):
        raise ValueError('a partition is missing a class')
    timing = np.sort(np.random.default_rng(SEED).choice(len(test), min(256, len(test)), replace=False))
    # Duplicates are disclosed, never silently removed.
    dev_bytes = set(row.tobytes() for row in x[np.asarray(dev)])
    overlap = [int(i) for i in test if x[i].tobytes() in dev_bytes]
    result = dict(fit=fit.tolist(), validation=val.tolist(), test=test.tolist(),
                  timing_test_positions=timing.tolist(), duplicate_feature_test_ids=overlap)
    if groups is not None:
        result['subjects'] = {name: sorted(map(int, np.unique(groups[s]))) for name, s in [('fit', fit), ('validation', val), ('test', test)]}
        if set(result['subjects']['fit']) & set(result['subjects']['validation']):
            raise ValueError('training subject overlap')
    return result


def prepare(data: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    for task in TASKS:
        x, y, groups, boundary = read_rows(data, task)
        partition = split_indices(x, y, task, groups, boundary)
        folder = output/task; folder.mkdir()
        np.savez_compressed(folder/'original.npz', x=x, y=y)
        report = dict(task=task, dimensions=x.shape[1], rows=len(x), labels=np.unique(y).tolist(),
                      source_sha256=sha(data/f'{task}.zip'), seed=SEED, **partition)
        (folder/'split.json').write_text(json.dumps(report, indent=2)+'\n')
        print(task, 'shape', x.shape, 'split', {n:len(partition[n]) for n in ('fit','validation','test')}, flush=True)
