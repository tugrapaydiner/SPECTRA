"""Adversarial tests for the independent parser, certificate and timing auditor."""
import copy,itertools,io,random,struct
import numpy as np
import pytest
from audit_panel import read_npy,certificate,check_rows,SEED

@pytest.mark.parametrize('dtype',['<f8','<i8','<i4','<u4','<U7'])
def test_npy_reader(dtype):
    a=np.asarray([[1,2],[3,4]],dtype=dtype);stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
    shape,values=read_npy(stream.getvalue());assert shape==a.shape and list(values)==a.ravel().tolist()
    with pytest.raises(ValueError):read_npy(stream.getvalue()[:-1])

def test_certificate_completion_oracle():
    pairs=[(a,b) for a in range(4) for b in range(a+1,4)]
    for edges in itertools.product(*[(-1,a,b) for a,b in pairs]):
        winners=set()
        unknown=[i for i,v in enumerate(edges) if v==-1]
        for fill in itertools.product(*[pairs[i] for i in unknown]):
            full=list(edges)
            for i,v in zip(unknown,fill):full[i]=v
            wins=[full.count(c) for c in range(4)];winners.add(max(range(4),key=lambda c:wins[c]))
        for w in range(4):assert certificate(4,w,edges)==(winners=={w})
    assert not certificate(3,0,[True,0,1])


def rows_fixture():
    positions=[0,2,5];names=['one','two'];rng=random.Random(SEED);rows=[]
    for repeat in range(11):
        order=list(positions);rng.shuffle(order)
        for case in order:
            modes=list(names);rng.shuffle(modes)
            for arm in modes:rows.append(dict(repeat=repeat,case=case,arm=arm,ns=100+case,class_index=1))
    return positions,names,rows

@pytest.mark.parametrize('corruption',['none','missing','extra','order','ns_bool','ns_negative','case','arm','output','field'])
def test_timing_inventory_rejects_corruption(corruption):
    positions,names,rows=rows_fixture()
    if corruption=='missing':rows.pop()
    elif corruption=='extra':rows.append(rows[-1].copy())
    elif corruption=='order':rows[0],rows[1]=rows[1],rows[0]
    elif corruption=='ns_bool':rows[0]['ns']=True
    elif corruption=='ns_negative':rows[0]['ns']=-1
    elif corruption=='case':rows[0]['case']=999
    elif corruption=='arm':rows[0]['arm']='unknown'
    elif corruption=='output':rows[0]['class_index']=0
    elif corruption=='field':rows[0]['extra']='discarded'
    if corruption=='none':assert len(check_rows(rows,positions,names,[1]*6))==6
    else:
        with pytest.raises(ValueError):check_rows(rows,positions,names,[1]*6)
