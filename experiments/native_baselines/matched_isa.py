"""Apply the prospective stronger-ISA amendment, preserving the original run.

Recompiles existing unmodified sources only. Model selection, generated-code
budget and golden test predictions remain fixed. Run bench.py separately.
"""
from __future__ import annotations
import argparse,json,pickle,shutil,sys,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.native_baselines.build import execute,sha
from experiments.native_baselines.runtime import NativeSession

def main(root):
    root=Path(root).resolve();out=root/'matched_isa';out.mkdir(exist_ok=False)
    (out/'models').symlink_to(root/'models',target_is_directory=True)
    build=out/'builds-final';shutil.copytree(root/'builds-final',build)
    valid=out/'validation';shutil.copytree(root/'validation',valid)
    report=json.loads((build/'builds.json').read_text())
    flags=['-std=c++17','-O3','-mavx2','-fno-fast-math','-ffp-contract=off','-fPIC','-shared']
    vendor=root/'external/libsvm';bridge=Path(__file__).with_name('libsvm_bridge.cpp')
    cmd=['g++',*flags,'-I'+str(vendor),str(vendor/'svm.cpp'),str(bridge),'-o',str(build/'libsvm/libnative_svm.so')]
    r=execute(cmd,build/'libsvm','matched_build',180)
    if r['returncode']!=0:raise RuntimeError('matched LIBSVM build failed')
    report['libsvm']['build']=r;report['libsvm']['sha256']=sha(build/'libsvm/libnative_svm.so')
    for name,entry in report['generated'].items():
        if entry['status']!='PASS':continue
        folder=build/'m2cgen'/name
        cmd=['g++',*flags,str(folder/'wrapper.cpp'),'-o',str(folder/'model.so')]
        r=execute(cmd,folder,'matched_compile',180)
        if r['returncode']!=0:raise RuntimeError('matched m2cgen build failed: '+name)
        entry['compile']=r;entry['binary_sha256']=sha(folder/'model.so');entry['binary_bytes']=(folder/'model.so').stat().st_size
    report['baseline_isa']='avx2';report['amendment_commit']='66d0e799f9206e09e4cce2e6d0b110156a51d546'
    (build/'builds.json').write_text(json.dumps(report,indent=2))
    frozen=json.loads((root/'models/MODEL_FREEZE.json').read_text());result=json.loads((valid/'VALIDATION.json').read_text())
    with threadpool_limits(1):
        for name,m in frozen['models'].items():
            folder=root/'models'/name
            with (folder/'model.pkl').open('rb') as f:model=pickle.load(f)
            x=np.fromfile(folder/'X.f64',dtype=np.float64).reshape(m['rows'],m['features'])
            golden=json.loads((valid/f'{name}-sklearn.json').read_text());margin=model._decision_function(x).reshape(len(x),-1)
            arms=[('libsvm',build/'libsvm/libnative_svm.so',folder/'libsvm.model')]
            if report['generated'][name]['status']=='PASS':arms += [('m2cgen',build/f'm2cgen/{name}/model.so',None)]
            for arm,library,text in arms:
                with NativeSession(library,m['features'],m['classes'],text) as w:
                    labels,raw=w.predict_buffer(x,return_scores=True)
                errors=sum(a!=b for a,b in zip(labels,golden))
                if errors:raise ValueError(f'new matched-ISA disagreement: {name}/{arm}/{errors}')
                p=result['models'][name]['backends'][arm];p['disagreements']=errors
                p['max_abs_margin_difference']=float(np.max(np.abs(np.frombuffer(raw,dtype=np.float64).reshape(margin.shape)-margin)))
                (valid/f'{name}-{arm}.json').write_text(json.dumps(labels,indent=2))
            (valid/f'{name}-checks.json').write_text(json.dumps(result['models'][name],indent=2))
    result['builds_sha256']=sha(build/'builds.json');result['amendment_commit']=report['amendment_commit']
    (valid/'VALIDATION.json').write_text(json.dumps(result,indent=2))
    print('all matched-AVX2 controls build and preserve frozen predictions')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
