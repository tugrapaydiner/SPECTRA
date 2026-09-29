"""Run one raw integer feature row; default is the first supplied example."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from experiments.selective_refinement.deployment import load_deployment
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',default='letter',choices=['letter','pendigits','satellite','optdigits']);p.add_argument('--row',help='JSON integer array');p.add_argument('--target',choices=['portable','avx2'],default='portable');p.add_argument('--policy',choices=['primary','strict','loose'],default='primary');a=p.parse_args()
    m=json.loads((ROOT/'MANIFEST.json').read_text());entry=m['tasks'][a.task]
    if a.row is None:
        with (ROOT/'inputs'/f'{a.task}.u8').open('rb') as f:q=bytearray(f.read(entry['features']))
    else:
        raw=json.loads(a.row)
        if type(raw) is not list or len(raw)!=entry['features'] or any(type(v) is not int or not 0<=v<=entry['maximum'] for v in raw):raise ValueError('expected bounded integer features in the model schema')
        q=bytearray(raw)
    with load_deployment(ROOT/'models'/a.task,ROOT/'native'/a.target/'refinement.so',policy=a.policy,expected_policy_sha256=entry['policy_sha256']) as model:
        pred,work=model.inspect_buffer(q)
    print(json.dumps({'label':pred[0],'work':work,'experimental':True,'note':'calibration does not guarantee accuracy or reliability under distribution shift'}))
