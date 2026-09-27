"""Secondary same-preprocessor raw Chess input; controls share input work."""
from pathlib import Path
from array import array
import argparse,hashlib,json,os,random,statistics,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_pipeline import Preprocessor
from spectra.svm_preprocess_native import NativePreprocessor
from spectra.svm_shared import PreparedModel
from experiments.native_baselines.runtime import NativeSession
p=argparse.ArgumentParser()
for n in ('models','controls','library','preprocessor','out'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
folder=a.models/'chess-101';rows=[json.loads(x) for x in (folder/'original.jsonl').read_text().splitlines()];expected=json.loads((folder/'expected.json').read_text())
pre=NativePreprocessor(Preprocessor.load(folder/'preprocessing.json'),a.preprocessor)
assert pre.transform(rows).tobytes()==(folder/'X.f64').read_bytes()
resources=[];funcs={}
for label,b,lib in [('parent','off',a.controls/'parent.so'),('off','off',a.library),('packed','packed',a.library),('lookup','lookup',a.library)]:
 owner=PreparedModel(folder/'model.srt',lib,boolean=b,input_dtype='float64');w=owner.session();resources.extend([owner,w]);funcs[label]=w.predict_buffer
native=NativeSession(a.controls/'chess-101.so',73,['nowin','won']);resources.append(native);funcs['generated_c']=native.predict_buffer
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol={'schema':'spectra.boolean_kernel.raw.v1','seed':20260929,'repeats':11,'chunks':[1,32,256],'arms':list(funcs),'rows':799,
 'input_sha256':{n:sha(folder/n) for n in ('original.jsonl','model.srt','preprocessing.json','X.f64','expected.json')},
 'sources':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('RAW_PROTOCOL.md'),ROOT/'spectra/svm_shared.py',ROOT/'spectra/svm_preprocess_native.py',ROOT/'spectra/_native/ovo/shared.hpp']},
 'libraries':{str(x):sha(x) for x in [a.library,a.preprocessor,a.controls/'parent.so',a.controls/'chess-101.so']},
 'affinity':sorted(os.sched_getaffinity(0)), 'scope':'complete raw rows through identical compiled preprocessing, checked buffer native calls and fresh labels; no file parse, setup or compile'}
(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
batches={c:[rows[s:s+c] for s in range(0,799,c)] for c in (1,32,256)}
def run(arm,chunk):
 out=[]
 for x in batches[chunk]:out.extend(funcs[arm](pre.transform(x)))
 return out
for arm in funcs:assert run(arm,256)==expected
records=[]
try:
 for repeat in range(11):
  order=[(c,k) for c in (1,32,256) for k in funcs];random.Random(20260929+repeat).shuffle(order)
  for c,k in order:
   t=time.perf_counter_ns();out=run(k,c);ns=time.perf_counter_ns()-t;assert out==expected
   records.append(dict(repeat=repeat,chunk=c,arm=k,rows=799,ns=ns,output_sha256=hashlib.sha256(json.dumps(out,separators=(',',':')).encode()).hexdigest()))
finally:
 for x in reversed(resources):x.close()
(a.out/'rows.json').write_text(json.dumps(records,indent=2))
result={str(c):{k:statistics.median(x['ns'] for x in records if x['chunk']==c and x['arm']==k)/799/1000 for k in funcs} for c in (1,32,256)}
(a.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
