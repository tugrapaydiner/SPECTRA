"""All-retained-row same-certificate check after model-only native layout change."""
from pathlib import Path
from array import array
from contextlib import ExitStack
import argparse,hashlib,json,struct
from .session import TreeSession,VerifiedCompact

def digest(x,fmt):return hashlib.sha256(struct.pack('<'+str(len(x))+fmt,*x)).hexdigest()
def run(a):
    a.out.mkdir(parents=True,exist_ok=False);records=[]
    for task in ('letter','pendigits','satellite','optdigits'):
        q=array('B',(a.evaluation/task/'input.u8').read_bytes())
        p={b:VerifiedCompact.from_files(a.models/task/'model.json',a.compiled/f'{task}-{b}.sct') for b in (8,16)}
        with ExitStack() as stack:
            sessions={}
            for name,library in (('original',a.original),('vector',a.vector),('register',a.register)):
                sessions[name]={b:stack.enter_context(TreeSession(library,first=p[b])) for b in (8,16)}
                sessions[name]['mixed']=stack.enter_context(TreeSession(library,first=p[8],second=p[16],official_model=a.models/task/'model.cbm',official_library=a.upstream/'libcatboostmodel-linux-x86_64-1.2.10.so'))
            for b in (8,16,'mixed'):
                for checkpoint in (0,16):
                    kwargs={'checkpoint':checkpoint,'refine':b=='mixed','fallback':b=='mixed'}
                    baseline=sessions['original'][b].inspect_buffer(q,**kwargs)
                    for name in ('vector','register'):
                        actual=sessions[name][b].inspect_buffer(q,**kwargs)
                        assert actual['indices']==baseline['indices']
                        assert actual['trees_evaluated']==baseline['trees_evaluated']
                        assert {k:v for k,v in actual['work'].items() if k!='compact_prepared_bytes'}=={k:v for k,v in baseline['work'].items() if k!='compact_prepared_bytes'}
                        rec={'task':task,'layout':name,'precision':str(b),'checkpoint':checkpoint,'rows':len(actual['indices']),
                             'indices_sha256':digest(actual['indices'],'i'),'work_sha256':digest(actual['trees_evaluated'],'I'),'work':actual['work'],'matches_original':True}
                        records.append(rec)
        print(task,'all layout states/work identical',flush=True)
    result={'status':'PASS','comparisons':records,'source':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cpp')},
            'libraries':{name:hashlib.sha256(getattr(a,name).read_bytes()).hexdigest() for name in ('original','vector','register')}}
    (a.out/'LAYOUT_REPLAY.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('models','compiled','evaluation','original','vector','register','upstream','out'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
