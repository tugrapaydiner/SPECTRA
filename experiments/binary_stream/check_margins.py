"""Compare separately compiled old-source and new-source actual binary margins."""
import argparse
import ctypes as C
import hashlib,json,math,random,struct,sys,zlib,time
from array import array
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
for name in ('inputs','base_library','candidate_library','out'):
 parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
args=parser.parse_args();ROOT=args.inputs
if args.out.exists():raise FileExistsError(args.out)
libs=[]
for name in [args.base_library,args.candidate_library]:
 lib=C.CDLL(str(name));lib.probe_margins.argtypes=[C.c_void_p,C.c_size_t,C.c_int,C.POINTER(C.c_double),C.c_int,C.POINTER(C.c_double),C.c_int];libs.append(lib)
def call(lib,raw,x,d,tables,path):
 blob=C.create_string_buffer(raw,len(raw));values=(C.c_double*len(x)).from_buffer(x);out=(C.c_double*(len(x)//d))()
 status=lib.probe_margins(blob,len(raw),tables,values,len(out),out,path)
 assert status==0,status
 return bytes(out)
def decode(raw):
 _,c,n,d,nl,ps,crc=struct.unpack('<8sIIIIII',raw[:32]);off=32+nl
 gamma=struct.unpack_from('<d',raw,off)[0];off+=8+4*c
 sv=struct.unpack_from('<'+'d'*(n*d),raw,off);off+=8*n*d
 coef=struct.unpack_from('<'+'d'*n,raw,off);off+=8*n
 bias=struct.unpack_from('<d',raw,off)[0]
 return d,gamma,sv,coef,bias
def independent(raw,x):
 d,gamma,sv,coef,bias=decode(raw);out=array('d')
 for start in range(0,len(x),d):
  total=0.
  for t,a in enumerate(coef):
   if a==0:continue
   distance=0.
   for f in range(d):
    v=x[start+f]-sv[t*d+f];distance+=v*v
   total+=a*math.exp(-gamma*distance)
  out.append(total+bias)
 return out.tobytes()
start=time.time();rows=variants=synthetic=python_rows=0;receipts=[]
for folder in sorted(ROOT.iterdir()):
 raw=(folder/'model.srt').read_bytes();_,c,n,d,*_=struct.unpack('<8sIIIIII',raw[:32])
 if c!=2:continue
 x=array('d');x.frombytes((folder/'transformed.f64').read_bytes());nrows=len(x)//d
 base=call(libs[0],raw,x,d,False,0)
 py=independent(raw,x);assert py==base,(folder,'ordered interpreter');python_rows+=nrows
 for tables in [False,True]:
  old=call(libs[0],raw,x,d,tables,0);assert old==base
  for path in [0,1,2]:
   candidate=call(libs[1],raw,x,d,tables,path);assert candidate==base,(folder,tables,path)
   variants+=nrows
 receipts.append({'model':folder.name,'rows':nrows,'margin_sha256':hashlib.sha256(base).hexdigest(),'model_sha256':hashlib.sha256(raw).hexdigest()});rows+=nrows
 print(folder.name,nrows,'bitwise PASS',flush=True)
rng=random.Random(93471)
for d in [1,3,16,17,73,257]:
 for n in [2,3,4,5,7,8,9,17,65]:
  for kind in ['random','cancel','subnormal']:
   coef=[rng.uniform(-3,3) for _ in range(n)] if kind=='random' else [(-1 if i%2 else 1)*(1e290 if kind=='cancel' else 2**-1074) for i in range(n)]
   sv=[rng.randint(-3,3)/8 for _ in range(n*d)] if kind=='random' else [0.]*(n*d)
   bias=-2**-1074 if kind!='random' else rng.uniform(-1,1)
   meta=b'{"labels":[-1,1]}';vals=sv+coef+[bias];payload=struct.pack('<dII',.25/d,n//2,n-n//2)+struct.pack('<'+'d'*len(vals),*vals);body=meta+payload
   raw=struct.pack('<8sIIIIII',b'SPCSVM02',2,n,d,len(meta),len(payload),zlib.crc32(body))+body
   x=array('d',(rng.uniform(-2,2) for _ in range(17*d)))
   for r,v in enumerate([0.,-0.,2**-1074,-2**-1074,sys.float_info.max,-sys.float_info.max]):
    for f in range(d):x[r*d+f]=v
   base=call(libs[0],raw,x,d,False,0);assert independent(raw,x)==base;python_rows+=17
   for tables in [False,True]:
    for path in [0,1,2]:
     assert call(libs[1],raw,x,d,tables,path)==base,(d,n,kind,tables,path);variants+=17
   synthetic+=17
report={'status':'PASS','natural_pairs':rows,'synthetic_pairs':synthetic,'repeated_native_margin_checks':variants,'independent_ordered_python_margins':python_rows,'synthetic_fixtures':162,'models':receipts,'elapsed_seconds':time.time()-start,'scope':'computed FP64 margins in this environment, not exact-real RBF arithmetic; repeated variants are not independent examples'}
args.out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='models'},indent=2))
