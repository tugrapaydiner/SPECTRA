"""Literal-source replay and fixed source-domain stress; no training.

The reference directly decodes JSON and sums original leaves, without consulting
compact coefficients, dictionary IDs or certificate bounds.
"""
from __future__ import annotations
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import time
from .compiler import compile_bytes, VerifiedTotal
from .session import TotalSession
from ..certified_trees.export_control import ExportSession
TASKS=('letter','pendigits','satellite','optdigits')
DOMAINS={'letter':(16,15),'pendigits':(16,100),'satellite':(36,255),'optdigits':(64,16)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

class ScalarSource:
    def __init__(self,raw,maximum):
        doc=json.loads(raw);fs=doc['features_info']['float_features']
        self.width=max(f['flat_feature_index'] for f in fs)+1
        mapping={f['feature_index']:f['flat_feature_index'] for f in fs}
        self.scale=float(doc['scale_and_bias'][0]);self.bias=list(map(float,doc['scale_and_bias'][1]))
        self.binary=len(self.bias)==1
        if self.binary:self.bias=[0.,self.bias[0]]
        self.classes=len(self.bias);self.trees=[];self.maximum=maximum
        for t in doc['oblivious_trees']:
            splits=[(mapping[s['float_feature_index']],float(s['border'])) for s in (t.get('splits') or [])]
            values=list(map(float,t['leaf_values']))
            if self.binary:values=[v for x in values for v in (0.,x)]
            self.trees.append((splits,values))
    def scores(self,rows):
        out=array('d');c=self.classes
        for q in rows:
            if len(q)!=self.width or any(not 0<=v<=self.maximum for v in q):raise ValueError('reference input domain')
            scores=[0.]*c
            for splits,values in self.trees:
                leaf=0
                for bit,(f,border) in enumerate(splits):leaf|=int(q[f]>border)<<bit
                offset=leaf*c
                for j in range(c):scores[j]+=values[offset+j]
            out.extend(self.scale*v+b for v,b in zip(scores,self.bias))
        return out.tobytes()
    def indices(self,scores):
        a=array('d');a.frombytes(scores)
        return [max(range(self.classes),key=lambda j:a[start+j]) for start in range(0,len(a),self.classes)]

def stress(source,d,D,n=2048):
    rng=random.Random(2026092961+d+D)
    uniform=[bytes(rng.randrange(D+1) for _ in range(d)) for _ in range(n)]
    boundary=[{0,D} for _ in range(d)]
    for splits,_ in source.trees:
        for f,border in splits:
            cut=math.floor(border)
            boundary[f].update(x for x in (cut-1,cut,cut+1,cut+2) if 0<=x<=D)
    choices=[sorted(x) for x in boundary]
    near=[bytes(rng.choice(choices[f]) for f in range(d)) for _ in range(n)]
    return uniform,near

def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'LOCK.json',{'source_files':{p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()},
      'models':{t:{'source':sha(a.sdk/'models'/t/'model.json'),'inputs':sha(a.sdk/'models'/t/'input.u8'),'expected':sha(a.sdk/'models'/t/'indices.i32')} for t in TASKS},
      'library_sha256':sha(a.library),'stress_rows_per_kind':2048,'seed_rule':'2026092961+d+D',
      'scope':'frozen sources plus fixed synthetic source-domain inputs, not accuracy selection'})
    results={}
    for task in TASKS:
        d,D=DOMAINS[task];dest=a.out/task;dest.mkdir();folder=a.sdk/'models'/task
        raw=(folder/'model.json').read_bytes();x=(folder/'input.u8').read_bytes();n=len(x)//d
        rows=[x[i*d:(i+1)*d] for i in range(n)];expected=list(struct.unpack('<'+'i'*n,(folder/'indices.i32').read_bytes()))
        source=ScalarSource(raw,D);start=time.perf_counter();scores=source.scores(rows);elapsed=time.perf_counter()-start
        (dest/'reference_scores.f64').write_bytes(scores);assert source.indices(scores)==expected
        exported=ExportSession(a.exports/task/'cpp');cpp=exported.scores(bytearray(x))
        old=array('d');old.frombytes(cpp);new=array('d');new.frombytes(scores)
        differences=sum(cpp[i:i+8]!=scores[i:i+8] for i in range(0,len(scores),8));models={}
        for layout in ('flat','interned'):
            start=time.perf_counter();binary=compile_bytes(raw,D,layout=layout);compile_time=time.perf_counter()-start
            (dest/(layout+'.sctt')).write_bytes(binary);proof=VerifiedTotal(raw,binary)
            with TotalSession(proof,a.library) as engine:
                assert engine.scores(bytearray(x))==scores,(task,layout,'source scores')
                work={}
                for policy in ('total','exact','audit','certificate_only'):
                    got=engine.inspect_buffer(bytearray(x),policy=policy)
                    assert all(i==j or (policy=='certificate_only' and i==-1) for i,j in zip(got['indices'],expected))
                    work[policy]=got['work']
                assert engine.scores(bytearray(x),traversal='scalar')==scores
                models[layout]={'bytes':len(binary),'sha256':sha(dest/(layout+'.sctt')),'compile_seconds':compile_time,'info':engine.info,'work':work}
        st={};proof=VerifiedTotal(raw,(dest/'interned.sctt').read_bytes())
        with TotalSession(proof,a.library) as engine:
            for kind,queries in zip(('uniform','boundary'),stress(source,d,D)):
                buf=bytearray(b''.join(queries));wanted=source.scores(queries)
                assert engine.scores(buf)==wanted,(task,kind,'source scores')
                got=engine.inspect_buffer(buf,policy='audit');indices=source.indices(wanted);assert got['indices']==indices
                cpp_indices=exported.predict_buffer(buf)
                packed=struct.pack('<'+'i'*len(indices),*indices)
                st[kind]={'rows':len(queries),'work':got['work'],'source_scores_sha256':hashlib.sha256(wanted).hexdigest(),
                          'indices_sha256':hashlib.sha256(packed).hexdigest(),'exported_cpp_label_disagreements':sum(i!=j for i,j in zip(indices,cpp_indices))}
                (dest/(kind+'.u8')).write_bytes(buf);(dest/(kind+'-scores.f64')).write_bytes(wanted);(dest/(kind+'-indices.i32')).write_bytes(packed)
        results[task]={'rows':n,'classes':source.classes,'score_values':len(scores)//8,'scalar_reference_seconds':elapsed,
          'cpp_score_bit_differences':differences,'cpp_max_score_absolute_difference':max(abs(a-b) for a,b in zip(old,new)),
          'cpp_labels_match':exported.predict_buffer(bytearray(x))==expected,'models':models,'stress':st}
        write(dest/'result.json',results[task]);print(task,'PASS; CPP different scores',differences,'source rows',n,flush=True)
    write(a.out/'report.json',{'status':'PASS','results':results,'source_score_values':sum(r['score_values'] for r in results.values()),
                              'retained_rows':sum(r['rows'] for r in results.values()),'synthetic_rows':len(TASKS)*4096})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sdk','exports','library','out'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
