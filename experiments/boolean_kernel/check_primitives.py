"""Independent integer primitive/original-score tests, without fitted new models."""
import argparse,ctypes as C,hashlib,json,math,random,struct,zlib
from array import array
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('library','models','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();lib=C.CDLL(str(a.library.resolve()))
lib.bk_population.argtypes=[C.c_uint64];lib.bk_population.restype=C.c_uint
lib.bk_epoch_wrap.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_double)]
lib.bk_observe.argtypes=[C.c_void_p,C.c_size_t,C.c_int,C.c_int,C.POINTER(C.c_double),C.c_int,C.c_int,C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_size_t,C.POINTER(C.c_double),C.c_size_t]
rng=random.Random(414717)
values=list(range(65536))+[rng.getrandbits(64) for _ in range(10000)]+[0,2**64-1,*[1<<i for i in range(64)]]
for v in values: assert lib.bk_population(v)==v.bit_count()
d=6;sv=[float((i>>f)&1) for i in range(64) for f in range(d)]
coef=[(-1 if i%2 else 1)*(1+i%3)/8 for i in range(64)]
meta=b'{"labels":[0,1]}';numbers=sv+coef+[.125]
payload=struct.pack('<dII',.25,32,32)+struct.pack('<'+'d'*len(numbers),*numbers);body=meta+payload
raw=struct.pack('<8sIIIIII',b'SPCSVM02',2,64,d,len(meta),len(payload),zlib.crc32(body))+body
blob=C.create_string_buffer(raw);x=(C.c_double*len(sv))(*sv)
expected_dist=array('d');expected_kernel=array('d');scores=array('d')
for i in range(64):
 total=0.
 for j in range(64):
  dist=float((i^j).bit_count());expected_dist.append(dist)
  k=math.exp(-.25*dist);expected_kernel.append(k);total+=coef[j]*k
 scores.append(total+.125)
for mode in (0,1,2):
 for tables in (0,1):
  distance=(C.c_double*4096)();kernels=(C.c_double*4096)();score=(C.c_double*64)()
  assert lib.bk_observe(blob,len(raw),tables,mode,x,64,d,distance,kernels,4096,score,64)==0
  assert bytes(distance)==expected_dist.tobytes() and bytes(kernels)==expected_kernel.tobytes()
  assert bytes(score)==scores.tobytes()
path=a.models/'chess-101';raw=(path/'model.srt').read_bytes();blob=C.create_string_buffer(raw)
query=array('d');query.frombytes((path/'X.f64').read_bytes()[:73*8]);x=(C.c_double*73).from_buffer(query)
assert lib.bk_epoch_wrap(blob,len(raw),x)==0
result={'status':'PASS','population_checks':len(values),'exhaustive_six_bit_pairs':4096,
        'ordered_pair_scores':64,'repeated_mode_table_checks':6,'generation_wrap_checks':1,
        'observer_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
        'scope':'test-only parameter-free fixtures, no learned accuracy evidence'}
with a.out.open('x') as f:json.dump(result,f,indent=2)
print(result)
