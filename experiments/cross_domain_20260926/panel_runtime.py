"""Actual native comparators; no Python reference substituted for LIBSVM."""
from __future__ import annotations
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from spectra.svm_shared import PreparedModel, SHARED_SCHEDULES, _check
from spectra.svm import build_runtime
from panel_data import sha

ARMS={'exhaustive_direct':(False,'exhaustive'),'cert_direct':(False,'beretta_cert'),
      'exhaustive_tables':(True,'exhaustive'),'cert_tables':(True,'beretta_cert'),
      'knockout_direct':(False,'knockout_cert')}
DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int)


def build(out:Path, upstream:Path):
    out.mkdir(parents=True,exist_ok=False)
    library=build_runtime(out/'spectra',target='avx2')
    cmd=['g++','-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared',
         '-I'+str(upstream),str(Path(__file__).with_name('native_bench.cpp')),str(upstream/'svm.cpp'),'-o',str(out/'libpanel.so')]
    p=subprocess.run(cmd,capture_output=True,text=True)
    report=dict(command=cmd,status=p.returncode,stdout=p.stdout,stderr=p.stderr,
                source={str(p):sha(p) for p in (upstream/'svm.cpp',upstream/'svm.h',upstream/'COPYRIGHT',Path(__file__).with_name('native_bench.cpp'))})
    (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    p.check_returncode()
    return library


class Timer:
    def __init__(self,path):
        self.lib=C.CDLL(str(Path(path).resolve()))
        self.lib.panel_error.restype=C.c_char_p
        self.lib.panel_open.argtypes=[C.c_char_p,C.c_int];self.lib.panel_open.restype=C.c_void_p
        self.lib.panel_close.argtypes=[C.c_void_p];self.lib.panel_close.restype=None
        self.lib.panel_batch.argtypes=[C.c_void_p,DP,C.c_int,C.c_int,IP]
        self.lib.panel_time_libsvm.argtypes=[C.c_void_p,DP,C.c_int,IP];self.lib.panel_time_libsvm.restype=C.c_uint64
        self.lib.panel_time_shared.argtypes=[C.c_void_p,C.c_void_p,DP,C.c_int,C.c_int,IP];self.lib.panel_time_shared.restype=C.c_uint64
        self.lib.panel_time_noop.restype=C.c_uint64


class Libsvm:
    def __init__(self,timer,path,d,labels):
        self.timer,self.d,self.labels=timer,d,labels
        self.handle=timer.lib.panel_open(str(path).encode(),d)
        if not self.handle:raise ValueError(timer.lib.panel_error().decode())
    def batch(self,x):
        x=np.ascontiguousarray(x,dtype=np.float64)
        output=np.empty(len(x),dtype=np.int32)
        code=self.timer.lib.panel_batch(self.handle,x.ctypes.data_as(DP),len(x),self.d,output.ctypes.data_as(IP))
        if code:raise ValueError(self.timer.lib.panel_error().decode())
        return output
    def predict(self,x):return self.labels[int(self.batch(np.asarray(x).reshape(1,-1))[0])]
    def close(self):
        if self.handle:self.timer.lib.panel_close(self.handle);self.handle=None


class Arm:
    def __init__(self,path,library,name):
        tables,schedule=ARMS[name]
        self.model=PreparedModel(path,library,tables=tables,input_dtype='float64')
        self.session=self.model.session();self.schedule=schedule
        self.mode=SHARED_SCHEDULES[schedule]
        self.function=C.cast(self.session._lib.sp_worker_run,C.c_void_p)
        self.stats=(C.c_uint64*5)()
    def batch(self,x):
        x=np.ascontiguousarray(x,dtype=np.float64);output=np.empty(len(x),dtype=np.int32)
        status=self.session._lib.sp_worker_run(self.session._handle,x.ctypes.data_as(DP),len(x),x.shape[1],self.mode,-1,
                                             output.ctypes.data_as(IP),len(x),self.stats,5,0)
        _check(self.session._lib,status);return output
    def predict(self,row):return self.session.predict(row,schedule=self.schedule)
    def close(self):self.session.close();self.model.close()
