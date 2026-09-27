"""Independent secondary raw-input schedule/hash/arithmetic reconstruction."""
import argparse,hashlib,json,math,random,statistics
from pathlib import Path
from audit import ANCHOR,load,sha,label_hash,member
p=argparse.ArgumentParser()
for n in ('run','models','source','library','preprocessor','controls','out'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();manifest=a.models.parent/'SHA256.json'
assert sha(manifest)==ANCHOR
anchor=load(manifest);protocol=load(a.run/'protocol.json');rows=load(a.run/'rows.json');summary=load(a.run/'summary.json')
arms=['parent','off','packed','lookup','generated_c'];chunks=[1,32,256]
assert protocol['arms']==arms and protocol['chunks']==chunks and protocol['repeats']==11 and protocol['seed']==20260929 and protocol['rows']==799
files={'original.jsonl','model.srt','preprocessing.json','X.f64','expected.json'}
assert set(protocol['input_sha256'])==files
for f in files:
 path=member(a.models,'chess-101/'+f)
 assert sha(path)==protocol['input_sha256'][f]==anchor['models/chess-101/'+f]['sha256']
expected=load(a.models/'chess-101/expected.json');digest=label_hash(expected)
sourcefiles={'experiments/boolean_kernel/bench_raw.py','experiments/boolean_kernel/RAW_PROTOCOL.md','spectra/svm_shared.py','spectra/svm_preprocess_native.py','spectra/_native/ovo/shared.hpp'}
assert set(protocol['sources'])==sourcefiles
for f,v in protocol['sources'].items():assert sha(member(a.source,f))==v
provided={p.name:sha(p) for p in (a.library,a.preprocessor,a.controls/'parent.so',a.controls/'chess-101.so')}
assert {Path(k).name:v for k,v in protocol['libraries'].items()}==provided
assert len(rows)==11*3*5
index=0
for repeat in range(11):
 order=[(c,arm) for c in chunks for arm in arms];random.Random(20260929+repeat).shuffle(order)
 for c,arm in order:
  r=rows[index];index+=1
  e=dict(repeat=repeat,chunk=c,arm=arm,rows=799,output_sha256=digest)
  assert set(r)==set(e)|{'ns'}
  assert all(type(r[k]) is type(v) and r[k]==v for k,v in e.items())
  assert type(r['ns']) is int and r['ns']>0
result={str(c):{k:statistics.median(x['ns'] for x in rows if x['chunk']==c and x['arm']==k)/799/1000 for k in arms} for c in chunks}
assert result==summary
record={'status':'PASS','records':len(rows),'repeated_predictions':len(rows)*799,'medians_us_per_row':result,
        'ratios_lookup_over':{str(c):{k:result[str(c)]['lookup']/result[str(c)][k] for k in arms if k!='lookup'} for c in chunks},
        'scope':'fixed raw-row same-preprocessor pipeline, not external reproduction or input-file I/O'}
with a.out.open('x') as f:json.dump(record,f,indent=2)
print('Raw audit PASS',len(rows))
