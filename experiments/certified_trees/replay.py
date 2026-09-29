"""All-row native acceptance against two official libraries and original C++.

Slow oracle covers fixed stratified-in-index samples, not just certified rows.
Every native retained decision is checked against its own frozen source model.
"""
from pathlib import Path
from contextlib import ExitStack
import argparse,hashlib,json,time
import numpy as np
from .session import TreeSession,VerifiedCompact
from .reference import certificate_oracle as co
from .export_control import build,ExportSession
TASKS=('letter','pendigits','satellite','optdigits')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(a):
    a.out.mkdir(parents=True,exist_ok=False);records=[]
    for task in TASKS:
        fit=json.loads((a.models/task/'FIT.json').read_text());d=fit['features'];classes=len(fit['classes']);q=np.fromfile(a.evaluation/task/'input.u8',dtype=np.uint8).reshape(-1,d)
        expected=np.fromfile(a.evaluation/task/'indices.i32',dtype='<i4').tolist();folder=a.out/task;folder.mkdir()
        proofs={b:VerifiedCompact.from_files(a.models/task/'model.json',a.compiled/f'{task}-{b}.sct') for b in (8,16)}
        build(a.models/task,folder/'cpp');export=ExportSession(folder/'cpp')
        assert export.predict_buffer(q)==expected
        source_scores=np.fromfile(a.evaluation/task/'source_scores.f64',dtype='<f8').reshape(len(q),classes)
        export_scores=np.frombuffer(export.scores(q),dtype='<f8').reshape(len(q),classes)
        detail={'task':task,'rows':len(q),'source_cpp_indices_match':True,'cpp_vs_python_max_score_error':float(np.max(np.abs(source_scores-export_scores))),'runs':[]}
        with ExitStack() as stack:
            for v in ('1.2.8','1.2.10'):
                lib=a.upstream/f'libcatboostmodel-linux-x86_64-{v}.so'
                official=stack.enter_context(TreeSession(a.library,official_model=a.models/task/'model.cbm',official_library=lib,features=d,maximum=fit['maximum'],classes=classes))
                assert official.predict_buffer(q)==expected,(task,v,'native CatBoost')
                detail['official_'+v]={'matched':len(q),'library_sha256':sha(lib)}
            sessions={b:stack.enter_context(TreeSession(a.library,first=proofs[b])) for b in (8,16)}
            hybrid=stack.enter_context(TreeSession(a.library,first=proofs[8],second=proofs[16],official_model=a.models/task/'model.cbm',official_library=a.upstream/'libcatboostmodel-linux-x86_64-1.2.10.so'))
            sixteen=stack.enter_context(TreeSession(a.library,first=proofs[16],official_model=a.models/task/'model.cbm',official_library=a.upstream/'libcatboostmodel-linux-x86_64-1.2.10.so'))
            oracles={b:co.CertifiedOracle((a.models/task/'model.json').read_bytes(),co.loads((a.compiled/f'{task}-{b}.oracle.json').read_bytes())) for b in (8,16)}
            samples=np.linspace(0,len(q)-1,33,dtype=int)
            for b in (8,16):
                for mode in ('scalar','tiled'):
                    for checkpoint in (0,16):
                        got=sessions[b].inspect_buffer(q,mode=mode,checkpoint=checkpoint);indices=got['indices']
                        assert all(p==-1 or p==e for p,e in zip(indices,expected)),(task,b,mode,checkpoint,'false certificate')
                        for i in samples:
                            r=oracles[b].predict(q[i].tolist(),checkpoint=checkpoint)
                            assert indices[i]==(r['class_index'] if r['class_index'] is not None else -1)
                            assert got['trees_evaluated'][i]==r['trees_evaluated']
                        tag=f'{b}-{mode}-{checkpoint}';np.asarray(indices,dtype='<i4').tofile(folder/(tag+'.indices'))
                        np.asarray(got['trees_evaluated'],dtype='<u4').tofile(folder/(tag+'.work'))
                        detail['runs'].append({'tag':tag,'work':got['work'],'false_certificates':0,'oracle_samples':len(samples),'indices_sha256':sha(folder/(tag+'.indices'))})
            for engine,name,refine,fallback in ((hybrid,'8to16',True,False),(hybrid,'8to16toofficial',True,True),(sixteen,'16toofficial',False,True)):
                for checkpoint in (0,16):
                    got=engine.inspect_buffer(q,checkpoint=checkpoint,refine=refine,fallback=fallback)
                    assert all(p==e or (not fallback and p==-1) for p,e in zip(got['indices'],expected))
                    tag=name+'-'+str(checkpoint);np.asarray(got['indices'],dtype='<i4').tofile(folder/(tag+'.indices'))
                    detail['runs'].append({'tag':tag,'work':got['work'],'matched_or_explicitly_unresolved':True,'indices_sha256':sha(folder/(tag+'.indices'))})
        (folder/'replay.json').write_text(json.dumps(detail,indent=2));records.append(detail)
        print(task,{x['tag']:x['work'] for x in detail['runs'] if x['tag'] in ('8-tiled-0','16-tiled-0','8to16toofficial-0')},flush=True)
    (a.out/'REPLAY.json').write_text(json.dumps({'status':'PASS','models':records,'source':{p.name:sha(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cpp')},'scope':'same-source numerical execution, exposed compatibility data; not new classifier intelligence'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('models','evaluation','compiled','library','upstream','out'):p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
