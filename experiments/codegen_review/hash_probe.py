"""Count real AST hash calls; deliberately no timings under instrumentation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
from m2cgen import ast, interpreters
sys.setrecursionlimit(12000)

def chain(n, reused=False):
    value=ast.FeatureRef(0)
    for i in range(n):value=ast.BinNumExpr(value,ast.NumVal(i+1),ast.BinNumOpType.ADD)
    value.to_reuse=reused
    return value

original={cls:cls.__hash__ for cls in vars(ast).values() if isinstance(cls,type) and cls.__module__==ast.__name__ and '__hash__' in cls.__dict__ and cls.__hash__ is not None}
counts={}
for cls,func in original.items():
    def counted(self,_f=func,_name=cls.__name__):
        counts[_name]=counts.get(_name,0)+1;return _f(self)
    cls.__hash__=counted
records=[]
try:
 for n in (16,32,64,128,256):
  for kind in ('empty_chain','same_type_pollution','reused_hit'):
    if kind=='empty_chain':root=chain(n)
    elif kind=='same_type_pollution':root=ast.VectorVal([chain(1,True),chain(n)])
    else:
        e=ast.ExpExpr(chain(n),to_reuse=True);root=ast.VectorVal([e]*8)
    counts.clear();text=interpreters.CInterpreter().interpret(root).encode()
    records.append({'kind':kind,'n':n,'hash_calls':dict(counts),'total':sum(counts.values()),'output_sha256':hashlib.sha256(text).hexdigest()})
finally:
 for cls,func in original.items():cls.__hash__=func
a.out.write_text(json.dumps({'rows':records},indent=2)+'\n')
