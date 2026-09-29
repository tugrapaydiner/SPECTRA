"""Fresh-process footprint of an unchanged trained MLP deployed as FP32."""
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.budgeted_prototypes.float_control import FloatSession

def run(a):
    meta=json.loads((a.models/'LOCK.json').read_text())['models'][a.task]
    path=a.models/(a.task+'.sfn')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('FP32 bytes changed')
    with a.input.open('rb') as f:raw=f.read(256)
    begin=time.perf_counter_ns()
    with FloatSession(path,a.library) as w:
        setup=time.perf_counter_ns()-begin
        expected=json.loads(a.expected.read_text())[0]
        if w.predict_buffer(bytearray(raw[:w.features]))!=[expected]:raise ValueError('FP32 memory probe output')
        text=Path('/proc/self/status').read_text().splitlines()
        memory={key:int(next(line.split()[1] for line in text if line.startswith(key+':'))) for key in ('VmHWM','VmRSS')}
        if {'numpy','scipy','torch','sklearn','pandas'}&sys.modules.keys():raise ValueError('numerical Python dependency imported')
        return {'task':a.task,'arm':'mlp_float32','setup_ns':setup,'memory_kib':memory,'runtime_info':w.info,
                'matched':True,'model_sha256':meta['sha256'],'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
                'scope':'one model/one prediction, fresh process, warm filesystem possible; ordinary FP32 cast, not retraining'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True)
    for name in ('models','library','input','expected'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(p.parse_args()),sort_keys=True))
