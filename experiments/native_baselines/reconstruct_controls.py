"""Produce the six native-baseline controls without private attachment access.

Wrap the already versioned byte-identical 18-model reconstruction. Capture its
trusted in-memory SVC before disposal, then retain the six predetermined seed101
models. Model/plan/input identities are checked against the pre-existing corpus.
No new parameter selection or model change is permitted.
"""
from __future__ import annotations
import argparse,hashlib,json,pickle,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.streaming import corpus
from spectra import svm_pipeline_export
TASKS=('wine','wdbc','chess','penguins','titanic','zoo')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    original=svm_pipeline_export.export_pipeline
    def capture(preprocessor,svc,folder):
        result=original(preprocessor,svc,folder)
        with (Path(folder)/'reference_svc.pkl').open('xb') as f:pickle.dump(svc,f,protocol=5)
        return result
    try:
        svm_pipeline_export.export_pipeline=capture
        result=corpus.reconstruct(destination/'reconstructed')
    finally:svm_pipeline_export.export_pipeline=original
    expected=json.loads((ROOT/'experiments/streaming/corpus_manifest.json').read_text())
    manifest={'format':'spectra.native_controls.v1','origin':'versioned original corpus reconstruction','models':{},'reconstruction':result}
    for task in TASKS:
        name=task+'-101';src=destination/'reconstructed'/name;out=destination/name;out.mkdir()
        for filename in ('model.srt','preprocessing.json','expected.json','reference_svc.pkl'):
            shutil.copyfile(src/filename,out/filename)
        shutil.copyfile(src/'input.jsonl',out/'original.jsonl')
        assert sha(out/'model.srt')==expected['models'][name]['files']['model.srt']['sha256']
        manifest['models'][name]={'original_model_sha256':sha(out/'model.srt'),
          'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir())}}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('six specified control models reconstructed exactly')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
