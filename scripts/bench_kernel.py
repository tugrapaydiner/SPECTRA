#!/usr/bin/env python3
"""Microbenchmark the checked SPECTRA weight-stationary native kernel."""
from __future__ import annotations
import argparse,csv,ctypes,json,platform,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(ROOT))
from deploy.pack_ternary import pack_ternary_rows
from deploy.torch_kernel import cpu_supports_avx2
from eval.roofline import arithmetic_intensity
SRC=ROOT/'deploy'/'cpp_sparse_kernel'/'spectra_kernel.cpp'; SHIFT=15
I8=ctypes.POINTER(ctypes.c_int8);U8=ctypes.POINTER(ctypes.c_uint8);I32=ctypes.POINTER(ctypes.c_int32);SZ=ctypes.c_size_t

def compile_kernel(avx2,builddir):
    out=builddir/f"libspectra_{'avx2' if avx2 else 'scalar'}.so"; flags=['g++','-O3','-std=c++17','-shared','-fPIC']
    if avx2:flags.append('-mavx2')
    subprocess.run(flags+[str(SRC),'-o',str(out)],check=True);return out

def load(path):
    lib=ctypes.CDLL(str(path));lib.spectra_weight_stationary_gemv.restype=ctypes.c_int
    lib.spectra_weight_stationary_gemv.argtypes=[I8,SZ,U8,SZ,I32,SZ,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,I8,SZ];return lib

def make_inputs(out_dim,hidden,K,seed=0):
    rng=np.random.default_rng(seed);W=rng.integers(-1,2,size=(out_dim,hidden),dtype=np.int8);packed,_=pack_ternary_rows(W);X=rng.integers(-128,128,size=(K,hidden),dtype=np.int16).astype(np.int8);mult=np.full(out_dim,1000,dtype=np.int32);Y=np.zeros((K,out_dim),dtype=np.int8);return np.ascontiguousarray(packed),np.ascontiguousarray(X),mult,Y

def call_ws(lib,packed,X,mult,K,hidden,out_dim,Y):
    st=lib.spectra_weight_stationary_gemv(X.ctypes.data_as(I8),X.size,packed.ctypes.data_as(U8),packed.size,mult.ctypes.data_as(I32),mult.size,SHIFT,K,hidden,out_dim,Y.ctypes.data_as(I8),Y.size)
    if st!=0:raise RuntimeError(f'native weight-stationary status={st}')

def timeit(fn,target=.30):
    fn();reps=1
    while True:
        t0=time.perf_counter()
        for _ in range(reps):fn()
        dt=time.perf_counter()-t0
        if dt>=target:return dt/reps
        reps=max(reps*2,int(reps*target/max(dt,1e-9))+1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='assets/data');args=ap.parse_args();out=(ROOT/args.out) if not Path(args.out).is_absolute() else Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if not cpu_supports_avx2():raise SystemExit('AVX2 comparison requested but CPU does not advertise AVX2; use scalar backend instead')
    build=Path(tempfile.mkdtemp(prefix='spectra_kbuild_'));avx2=load(compile_kernel(True,build));scalar=load(compile_kernel(False,build))
    packed,X,mult,Yv=make_inputs(256,257,8,seed=1);_,_,_,Ys=make_inputs(256,257,8,seed=1);call_ws(avx2,packed,X,mult,8,257,256,Yv);call_ws(scalar,packed,X,mult,8,257,256,Ys);bit_exact=bool(np.array_equal(Yv,Ys));print(f'[correctness] AVX2 == scalar: {bit_exact}')
    if not bit_exact:raise SystemExit('AVX2 diverged from scalar')
    host={'cpu':platform.processor() or platform.machine(),'uname':platform.platform(),'bit_exact':bit_exact}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):host['cpu']=line.split(':',1)[1].strip();break
    except OSError:pass
    out_dim,hidden=512,512;rows_ws=[];peak=0.0
    for K in [1,2,4,8,16,32,64,128,256]:
        packed,X,mult,Y=make_inputs(out_dim,hidden,K,seed=2)
        for name,lib in (('avx2',avx2),('scalar',scalar)):
            t=timeit(lambda l=lib,p=packed,x=X,m=mult,k=K,Y=Y:call_ws(l,p,x,m,k,hidden,out_dim,Y));macs=K*out_dim*hidden;gops=2*macs/t/1e9;ai=arithmetic_intensity(out_dim,hidden,weight_bits=2,reuse=K);rows_ws.append({'K':K,'build':name,'hidden':hidden,'out_dim':out_dim,'latency_ms':t*1e3,'gops':gops,'arithmetic_intensity':ai})
            if name=='avx2':peak=max(peak,gops)
            print(f'[ws] K={K:<4} {name:<6} {gops:8.2f} GOP/s AI={ai:8.1f}')
    Kfix,out_dim=64,512;rows_simd=[]
    for hidden in [64,128,256,512,1024,2048]:
        packed,X,mult,Y=make_inputs(out_dim,hidden,Kfix,seed=3);g={}
        for name,lib in (('avx2',avx2),('scalar',scalar)):
            t=timeit(lambda l=lib,p=packed,x=X,m=mult,h=hidden,Y=Y:call_ws(l,p,x,m,Kfix,h,out_dim,Y));g[name]=2*Kfix*out_dim*hidden/t/1e9
        speed=g['avx2']/g['scalar'];rows_simd.append({'hidden':hidden,'gops_avx2':g['avx2'],'gops_scalar':g['scalar'],'speedup':speed});print(f'[simd] hidden={hidden:<5} avx2={g["avx2"]:7.2f} scalar={g["scalar"]:7.2f} x{speed:5.2f}')
    host['peak_avx2_gops']=peak;(out/'host.json').write_text(json.dumps(host,indent=2))
    with open(out/'bench_weight_stationary.csv','w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(rows_ws[0].keys()));w.writeheader();w.writerows(rows_ws)
    with open(out/'bench_simd_scaling.csv','w',newline='') as fh:w=csv.DictWriter(fh,fieldnames=list(rows_simd[0].keys()));w.writeheader();w.writerows(rows_simd)
if __name__=='__main__':main()
