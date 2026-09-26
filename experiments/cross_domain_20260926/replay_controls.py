"""Retain full predictions of frozen task controls; no training or reselection.

Pickles below are our own sealed fitting artifacts, not untrusted public uploads.
"""
from pathlib import Path
import argparse,json,pickle
import numpy as np
from threadpoolctl import threadpool_limits
from evaluate_panel import check_freeze
from panel_data import TASKS

def replay(root):
    check_freeze(root/'panel')
    for task in TASKS:
        folder=root/'formal'/task;models=root/'panel'/task/'models'
        data=np.load(folder/'test_inputs.npz',allow_pickle=False);pred={}
        original=json.loads((folder/'quality_fidelity.json').read_text())['quality']
        for name in ('logistic','mlp'):
            model=pickle.loads((models/(name+'.pkl')).read_bytes())
            pred[name]=model.predict(data['x'])
            if int(np.sum(pred[name]==data['y']))!=original[name]['correct']:
                raise AssertionError('task-control replay changed')
        destination=folder/'control_predictions.npz'
        if destination.exists():raise FileExistsError(destination)
        np.savez_compressed(destination,**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):replay(a.evidence)
