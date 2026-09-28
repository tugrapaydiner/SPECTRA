"""Fixed feasible generated-code fallbacks; original failures are not replaced."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.nonlinear_admission.native import execute,FLAGS
from experiments.nonlinear_admission.study import sha,save,verify_freeze,TASKS
from experiments.native_baselines.build import wrapper

def compile_source(folder,info,level):
    (folder/'wrapper.cpp').write_text(wrapper(info['classes'],info['features']))
    name='model.so' if level=='-O3' else 'model-O1.so'
    rec=execute(['g++',*[level if f=='-O3' else f for f in FLAGS],str(folder/'wrapper.cpp'),'-o',str(folder/name)],folder/('compile-'+level[1:]),180)
    result={'compile':rec,'status':'COMPLETE' if rec['status']=='COMPLETE' else 'COMPILE_FAILED',
            'optimization':level,'source_sha256':sha(folder/'model.c')}
    if rec['status']=='COMPLETE':result.update(binary=name,binary_sha256=sha(folder/name),binary_bytes=(folder/name).stat().st_size)
    return result

def run(a):
    verify_freeze(a.root);build=json.loads((a.builds/'BUILD.json').read_text());results={}
    for task in TASKS:
        info=build['models'][task]['svm'];original=a.builds/'generated'/task
        previous=json.loads((original/'STATUS.json').read_text())
        if previous['status']=='COMPLETE':
            results[task]={'variant':'stock-O3','binary':str(original/'model.so'),'sha256':sha(original/'model.so')};continue
        folder=original;variant='stock'
        if previous['status']=='GENERATION_FAILED':
            variant='guarded';folder=a.builds/'generated_guarded'/task;folder.mkdir(parents=True,exist_ok=False)
            model=a.root/'final'/task/'svm/model.pkl'
            rec=execute([sys.executable,str(Path(__file__).with_name('generated.py')),'generate','--package',str(a.package),'--model',str(model),'--output',str(folder/'model.c')],folder/'generation',120)
            save(folder/'GENERATION.json',{'generate':rec,'variant':'existing-PR36-single-lookup','model_sha256':sha(model)})
            if rec['status']!='COMPLETE':results[task]={'status':'GENERATION_FAILED','variant':variant};continue
            result=compile_source(folder,info,'-O3');save(folder/'O3.json',result)
            if result['status']=='COMPLETE':
                results[task]={'variant':'guarded-O3','binary':str(folder/result['binary']),'sha256':result['binary_sha256']};continue
        result=compile_source(folder,info,'-O1');save(folder/'O1.json',result)
        if result['status']=='COMPLETE':results[task]={'variant':variant+'-O1','binary':str(folder/result['binary']),'sha256':result['binary_sha256']}
        else:results[task]={'status':'COMPILE_FAILED','variant':variant+'-O1'}
        print(task,results[task],flush=True)
    save(a.builds/'GENERATED_SELECTION.json',{'order':['stock-O3','stock-O1','guarded-O3','guarded-O1'],'models':results,
         'amendment_sha256':sha(Path(__file__).with_name('COMPARATOR_AMENDMENT.md')),'script_sha256':sha(__file__),
         'patched_package':{p.relative_to(a.package).as_posix():sha(p) for p in sorted((a.package/'m2cgen').rglob('*.py'))}})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('root','builds','package'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
