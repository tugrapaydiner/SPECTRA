"""Frozen compact comparison inventory, semantic and archive regression checks."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('compact_bench',Path(__file__).resolve().parents[1]/'scripts/bench_compact_cnf.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)

@pytest.fixture
def evidence(tmp_path,monkeypatch):
    for k,v in dict(sizes=[4096],families=['uniform'],per_cell=1,search_seeds=[17001],rounds=1,moves=0,bootstrap_repeats=8).items():
        monkeypatch.setitem(bench.CONFIG,k,v)
    # Use a small generator but retain a large-cell label for the analysis gate.
    def generate():
        p,_=bench.random_3sat(4,16,91226000)
        return [dict(id='4096:uniform:0',nvars=4096,family='uniform',seed=91226000,sha256=p.sha256(),formula=p.record())]
    monkeypatch.setattr(bench,'generate',generate)
    out=tmp_path/'evidence';bench.run(out)
    cases=json.loads((out/'cases.json').read_text())
    with gzip.open(out/'timings.jsonl.gz','rt') as f:rows=[json.loads(x) for x in f]
    memory=[json.loads(x) for x in (out/'memory.jsonl').read_text().splitlines()]
    return out,cases,rows,memory


def test_replay_roundtrip(evidence):
    result=bench.verify(evidence[0],True)
    assert result['integrity']=='PASS' and result['answers']==result['path_replays']==2

@pytest.mark.parametrize('case',['missing','order','negative_time','bool_time','wrong_status','witness','path','memory_missing','memory_negative','peak','extra'])
def test_semantic_tamper(evidence,case):
    _,cases,rows,memory=copy.deepcopy(evidence)
    if case=='missing':rows.pop()
    elif case=='order':rows.reverse()
    elif case=='negative_time':rows[0]['result']['elapsed_ns']=-1
    elif case=='bool_time':rows[0]['result']['elapsed_ns']=True
    elif case=='wrong_status':rows[0]['result']['status']='UNSAT'
    elif case=='witness':rows[0]['result']['witness'][0]=1
    elif case=='path':rows[0]['result']['path_sha256']='0'*64
    elif case=='memory_missing':memory.pop()
    elif case=='memory_negative':memory[0]['current_python_bytes']=-1
    elif case=='peak':memory[0]['current_python_bytes']=memory[0]['peak_python_bytes']+1
    else:rows[0]['extra']=1
    with pytest.raises((ValueError,TypeError)):bench.analyze(cases,rows,memory)

@pytest.mark.parametrize('name',['summary.json','config.json','source_sha256.json','cases.json'])
def test_forgery_with_recomputed_checksum(evidence,name):
    out,*_=evidence;p=out/name;value=json.loads(p.read_text())
    if name=='cases.json':value[0]['seed']+=1
    elif name=='summary.json':value['gate']='FORGED'
    elif name=='config.json':value['moves']+=1
    else:value['data/cnf.py']='0'*64
    bench.write(p,value)
    inventory=json.loads((out/'SHA256.json').read_text());inventory[name]=hashlib.sha256(p.read_bytes()).hexdigest();bench.write(out/'SHA256.json',inventory)
    with pytest.raises(ValueError):bench.verify(out)
