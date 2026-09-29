"""Fresh-process prepared footprint; filesystem cache may be warm, no ML imports."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.certified_trees.session import TreeSession
from experiments.certified_trees.controls import CatBoostSession
from experiments.certified_trees.pipeline import RefinementSession

def run(root,task,kind):
    root=Path(root);m=json.loads((root/'resource-inputs.json').read_text())[task];q=bytearray(m['query']);start=time.perf_counter_ns()
    compact=root/'compiled'/task/'q16.sct';original=root/'models'/task/'source.cbm'
    if kind=='compact':w=TreeSession(compact,root/'tiled_bench/trees.so',expected_sha256=m['compact_sha256']);expected=m['compact_expected']
    elif kind=='official':w=CatBoostSession(original,root/'catboost_tiled_bench/control.so',m['maximum'],expected_sha256=m['cbm_sha256']);expected=m['source_expected']
    elif kind=='full_coverage':w=RefinementSession(compact,original,root/'catboost_owned_fixed/control.so',compact_sha256=m['compact_sha256'],cbm_sha256=m['cbm_sha256'],source_json_sha256=m['source_sha256']);expected=m['source_expected']
    else:raise ValueError('unknown deployment')
    with w:
        setup=time.perf_counter_ns()-start;pred=w.predict_buffer(q)
        if pred!=[expected]:raise ValueError('resource probe output mismatch')
        status=Path('/proc/self/status').read_text().splitlines()
        memory={k:int(next(l.split()[1] for l in status if l.startswith(k+':'))) for k in ('VmHWM','VmRSS')}
        if {'numpy','scipy','sklearn','catboost','torch'}&sys.modules.keys():raise ValueError('ML Python framework imported')
        return {'task':task,'kind':kind,'setup_ns':setup,'memory_kib':memory,'matched':True,'first_input_only':True,
                'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_record':m,
                'scope':'own Linux process high-water after full preparation and one fixed query; not maximum-batch RSS or cold-filesystem evidence'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',required=True);p.add_argument('--kind',required=True);a=p.parse_args();print(json.dumps(run(a.root,a.task,a.kind)))
