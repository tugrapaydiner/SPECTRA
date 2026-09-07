#!/usr/bin/env python3
"""Empirical cache-residency sweep using the checked native ABI."""
from __future__ import annotations
import csv, ctypes, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT))
from deploy.pack_ternary import pack_ternary_rows, packed_size_bytes
from deploy.torch_kernel import cpu_supports_avx2
SRC=ROOT/"deploy"/"cpp_sparse_kernel"/"spectra_kernel.cpp"; DATA=ROOT/"assets"/"data"
I8=ctypes.POINTER(ctypes.c_int8); U8=ctypes.POINTER(ctypes.c_uint8); I32=ctypes.POINTER(ctypes.c_int32); SZ=ctypes.c_size_t
NA,HIDDEN,SHIFT=8,1024,15

def host_caches():
    base=Path('/sys/devices/system/cpu/cpu0/cache'); out={}
    try:
        for idx in base.glob('index*'):
            lvl=int((idx/'level').read_text()); typ=(idx/'type').read_text().strip(); s=(idx/'size').read_text().strip()
            kb=float(s.rstrip('K')) if s.endswith('K') else float(s.rstrip('M'))*1024
            if lvl in (2,3) and typ in ('Unified','Data'): out[lvl]=kb/1024.0
    except OSError: pass
    return out.get(2),out.get(3)

def timeit(fn,target=.15):
    fn(); reps=1
    while True:
        t0=time.perf_counter()
        for _ in range(reps): fn()
        dt=time.perf_counter()-t0
        if dt>=target:return dt/reps
        reps=max(reps*2,int(reps*target/max(dt,1e-9))+1)

def main():
    build=Path(tempfile.mkdtemp(prefix='spectra_cache_')); so=build/'lib.so'; flags=['g++','-O3','-std=c++17','-shared','-fPIC']
    backend='avx2' if cpu_supports_avx2() else 'scalar'
    if backend=='avx2':flags.append('-mavx2')
    subprocess.run(flags+[str(SRC),'-o',str(so)],check=True); lib=ctypes.CDLL(str(so))
    lib.spectra_sparse_ternary_gemv.restype=ctypes.c_int
    lib.spectra_sparse_ternary_gemv.argtypes=[I8,SZ,I32,SZ,U8,SZ,I32,SZ,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,I8,SZ]
    rng=np.random.default_rng(11); idx=np.ascontiguousarray(np.arange(NA,dtype=np.int32)); X=np.ascontiguousarray(rng.integers(-128,128,size=(NA,HIDDEN),dtype=np.int16).astype(np.int8)); rows=[]
    for od in [128,512,2048,4096,8192,16384,32768,65536,131072]:
        W=rng.integers(-1,2,size=(od,HIDDEN),dtype=np.int8); packed,_=pack_ternary_rows(W); mult=np.full(od,1000,dtype=np.int32); Y=np.zeros((NA,od),dtype=np.int8)
        def call():
            st=lib.spectra_sparse_ternary_gemv(X.ctypes.data_as(I8),X.size,idx.ctypes.data_as(I32),idx.size,packed.ctypes.data_as(U8),packed.size,mult.ctypes.data_as(I32),mult.size,SHIFT,NA,HIDDEN,od,Y.ctypes.data_as(I8),Y.size)
            if st!=0: raise RuntimeError(f'native status={st}')
        t=timeit(call); wmb=packed_size_bytes(od*HIDDEN)/(1024*1024); gops=2*NA*od*HIDDEN/t/1e9
        rows.append({'out_dim':od,'weight_mb':wmb,'gops':gops,'latency_ms':t*1e3,'backend':backend}); print(f'[cache] {backend} W={wmb:8.3f} MB {gops:7.2f} GOP/s')
        del W,packed,Y
    DATA.mkdir(parents=True,exist_ok=True)
    with open(DATA/'bench_cache_residency.csv','w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    l2,l3=host_caches(); hp=DATA/'host.json'; host=json.loads(hp.read_text()) if hp.exists() else {}; host.update({'l2_mb':l2,'l3_mb':l3,'packed_core_mb':packed_size_bytes(5_600_000)/(1024*1024)}); hp.write_text(json.dumps(host,indent=2))
if __name__=='__main__':main()
