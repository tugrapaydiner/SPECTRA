"""Deliberately mutate copied evidence. Originals are never modified in place."""
import argparse, json, os, shutil, tempfile
from pathlib import Path
from audit import audit, read, sha

def replace(path,raw):
    temporary=path.with_name(path.name+'.replacement')
    temporary.write_bytes(raw);os.replace(temporary,path)
def json_replace(path,doc):replace(path,(json.dumps(doc)+'\n').encode())

def main(a):
    cases=('missing-time','reordered-time','boolean-time','wrong-label-digest','false-accuracy','false-correct-count',
           'false-gate','altered-model','altered-data','wrong-selection','changed-binary','changed-weight')
    results=[]
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='admission-negative-') as temp:
            temp=Path(temp);root=temp/'study';builds=temp/'builds';run=temp/'timing'
            for old,new in ((a.root,root),(a.builds,builds),(a.run,run)):
                shutil.copytree(old,new,copy_function=os.link)
            if case.endswith('time') or case=='wrong-label-digest':
                p=run/'timings.jsonl';rows=[json.loads(s) for s in p.read_text().splitlines()]
                if case=='missing-time':rows.pop()
                elif case=='reordered-time':rows[0],rows[1]=rows[1],rows[0]
                elif case=='boolean-time':rows[0]['ns']=True
                else:rows[0]['predictions_sha256']='0'*64
                replace(p,(''.join(json.dumps(r)+'\n' for r in rows)).encode())
            elif case in ('false-accuracy','false-correct-count','false-gate'):
                p=root/'evaluation/QUALITY.json';doc=read(p)
                if case=='false-gate':doc['tasks']['isolet']['svm_admitted_test']=not doc['tasks']['isolet']['svm_admitted_test']
                elif case=='false-correct-count':doc['tasks']['isolet']['models']['svm']['correct']+=1
                else:doc['tasks']['isolet']['models']['svm']['accuracy']=.123
                json_replace(p,doc)
            elif case=='altered-model':replace(root/'final/isolet/svm/model.pkl',b'changed')
            elif case=='altered-data':replace(root/'data/isolet/test.npz',b'changed')
            elif case=='wrong-selection':
                p=root/'SELECTED.json';doc=read(p);doc['selected']['isolet']['configs']['svm']['C']=999.;json_replace(p,doc)
                f=root/'MODEL_FREEZE.json';doc=read(f);doc['selected_sha256']=sha(p);json_replace(f,doc)
                q=root/'evaluation/QUALITY.json';doc=read(q);doc['freeze_sha256']=sha(f);json_replace(q,doc)
            elif case=='changed-binary':replace(builds/'libsvm.so',b'changed')
            else:replace(builds/'isolet/linear.weights',b'changed')
            try:audit(root,builds,run,a.source)
            except (ValueError,KeyError) as error:results.append({'case':case,'rejected':True,'reason':str(error)})
            else:raise AssertionError('auditor accepted '+case)
    with a.out.open('x') as f:json.dump({'status':'PASS','cases':results},f,indent=2)
    print(json.dumps(results,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','builds','run','source','out'):p.add_argument('--'+n,type=Path,required=True)
    main(p.parse_args())
