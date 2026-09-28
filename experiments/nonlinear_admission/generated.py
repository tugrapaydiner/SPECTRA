"""Bounded unmodified m2cgen export/compile. Missing results remain missing."""
import argparse,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.nonlinear_admission.native import execute,FLAGS
from experiments.nonlinear_admission.study import TASKS,sha,save,verify_freeze
from experiments.native_baselines.build import wrapper

def generate(a):
    sys.path.insert(0,str(a.package));import m2cgen
    with a.model.open('rb') as f:model=pickle.load(f)['model']
    sys.setrecursionlimit(12000);t=time.perf_counter();text=m2cgen.export_to_c(model);elapsed=time.perf_counter()-t
    a.output.write_text(text,encoding='utf-8')
    print(json.dumps({'generator':m2cgen.__version__,'generation_seconds':elapsed,'source_bytes':a.output.stat().st_size,'source_sha256':sha(a.output)}))

def run(a):
    verify_freeze(a.root);build=json.loads((a.builds/'BUILD.json').read_text())
    for task in TASKS:
        folder=a.builds/'generated'/task;folder.mkdir(parents=True,exist_ok=False)
        model=a.root/'final'/task/'svm/model.pkl';source=folder/'model.c'
        command=[sys.executable,str(Path(__file__).resolve()),'generate','--package',str(a.package),'--model',str(model),'--output',str(source)]
        result=execute(command,folder/'generation',120)
        status={'status':'GENERATION_FAILED','generate':result,'fit_pickle_sha256':sha(model)}
        if result['status']=='COMPLETE':
            info=build['models'][task]['svm'];(folder/'wrapper.cpp').write_text(wrapper(info['classes'],info['features']))
            compiled=execute(['g++',*FLAGS,str(folder/'wrapper.cpp'),'-o',str(folder/'model.so')],folder/'compile',180)
            status.update(compile=compiled,status='COMPILE_FAILED',source_sha256=sha(source),source_bytes=source.stat().st_size)
            if compiled['status']=='COMPLETE':status.update(status='COMPLETE',binary_sha256=sha(folder/'model.so'),binary_bytes=(folder/'model.so').stat().st_size)
        save(folder/'STATUS.json',status);print(task,status['status'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('operation',choices=['run','generate'])
    for n in ('root','builds','package','model','output'):p.add_argument('--'+n,type=Path)
    a=p.parse_args();globals()[a.operation](a)
