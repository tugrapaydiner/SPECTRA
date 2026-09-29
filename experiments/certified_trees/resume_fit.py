"""Resume only missing fixed fits after an execution timeout; preserve completed bytes."""
import argparse,json,time,sys
from pathlib import Path
from .reproduce import TASKS,PARAMS,sha,write,load

def main(a):
    import catboost
    lock=json.loads((a.out/'FIT_PROTOCOL.json').read_text())
    if lock['parameters']!=PARAMS or lock['script_sha256']!=sha(Path(__file__).with_name('reproduce.py')):raise ValueError('original fitting source changed')
    for name,h in lock['inputs'].items():
        if sha(a.data/name)!=h:raise ValueError('data changed')
    if catboost.__version__!='1.2.8':raise ValueError('version')
    records=[]
    for task in TASKS:
        dest=a.out/task
        if (dest/'FIT.json').is_file():
            rec=json.loads((dest/'FIT.json').read_text())
            for name,h in rec['files'].items():
                if sha(dest/name)!=h:raise ValueError('completed model changed')
            records.append(rec);print('retained',task,flush=True);continue
        if dest.exists() and list(dest.iterdir()):raise ValueError('unclassified partial fit; do not overwrite')
        dest.mkdir(exist_ok=True)
        q,y,D=load(a.data,task,'train');start=time.process_time();wall=time.perf_counter()
        model=catboost.CatBoostClassifier(**PARAMS).fit(q,y)
        seconds=time.process_time()-start;elapsed=time.perf_counter()-wall
        for fmt in ('cbm','json','cpp'):model.save_model(str(dest/('model.'+fmt)),format=fmt)
        rec={'task':task,'training_rows':len(q),'features':q.shape[1],'maximum':D,'classes':model.classes_.tolist(),
             'fit_cpu_seconds':seconds,'fit_wall_seconds':elapsed,'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}}
        write(dest/'FIT.json',rec);records.append(rec);print('completed',task,seconds,flush=True)
    write(a.out/'MODEL_LOCK.json',{'models':records,'files':{p.relative_to(a.out).as_posix():sha(p) for p in a.out.rglob('*') if p.is_file()},
          'script_sha256':sha(Path(__file__).with_name('reproduce.py')),'resume_script_sha256':sha(__file__),
          'test_predictions_made':False,'scope':'new fixed model set; first invocation timed out after two completed model exports; only missing fits resumed'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
