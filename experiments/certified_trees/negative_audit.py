"""Corrupt only a new disposable copy of empirical records. Never alter originals."""
from pathlib import Path
import argparse,json,shutil,zipfile
try:
    from .audit import audit
except ImportError:
    from audit import audit

def run(root,source,out):
    root,source,out=map(Path,(root,source,out));out.mkdir(parents=True,exist_ok=False)
    target=out/'copy';target.mkdir()
    for name in ('models','compiled','fidelity','source_evaluation','benchmark','benchmark_tiled','benchmark_final','sources_v1','sources_v2','sources_v3','native','catboost_native','catboost_refine','tiled_bench','catboost_tiled_bench','catboost_owned_fixed','exports','upstream'):
        shutil.copytree(root/name,target/name)
    audit(target,source)
    def change_json(path,fn):
        doc=json.loads(path.read_text());fn(doc);return json.dumps(doc).encode()
    def truncate_line(path):return b'\n'.join(path.read_bytes().splitlines()[:-1])+b'\n'
    cases=[
      ('compiled-byte','compiled/letter/q16.sct',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
      ('source-byte','models/letter/source.json',lambda p:p.read_bytes()+b' '),
      ('missing-source-inventory','models/FINAL_LOCK.json',lambda p:change_json(p,lambda x:x.update(files={}))),
      ('missing-compiled-inventory','compiled/COMPILED_LOCK.json',lambda p:change_json(p,lambda x:x.update(files={}))),
      ('false-certified-count','fidelity/report.json',lambda p:change_json(p,lambda x:x['tasks']['letter']['results']['q16_cp0'].update(certified=4000))),
      ('truncated-first-grid','benchmark/observations.jsonl',truncate_line),
      ('truncated-tiled-grid','benchmark_tiled/observations.jsonl',truncate_line),
      ('truncated-final-grid','benchmark_final/observations.jsonl',truncate_line),
      ('wrong-output-hash','benchmark_final/observations.jsonl',lambda p:p.read_bytes().replace(b'"output_sha256": "',b'"output_sha256": "f',1)),
      ('negative-time','benchmark_final/observations.jsonl',lambda p:p.read_bytes().replace(b'"ns": ',b'"ns": -',1)),
      ('wrong-final-ratio','benchmark_final/summary.json',lambda p:change_json(p,lambda x:x.update(equal_task_owned_refine_ratio=.1))),
      ('empty-library-inventory','benchmark_final/PROTOCOL.json',lambda p:change_json(p,lambda x:x.update(files={}))),
      ('empty-timed-source','benchmark_final/PROTOCOL.json',lambda p:change_json(p,lambda x:x.update(sources={}))),
      ('changed-timed-source','sources_v3/runtime.cpp',lambda p:p.read_bytes()+b'\n// altered\n'),
      ('changed-native-library','tiled_bench/trees.so',lambda p:p.read_bytes()[:-1]+bytes([p.read_bytes()[-1]^1])),
      ('duplicate-json-key','benchmark_final/PROTOCOL.json',lambda p:p.read_bytes().replace(b'"seed":',b'"seed":0,"seed":',1)),
    ]
    records=[]
    for name,file,change in cases:
        path=target/file;old=path.read_bytes()
        try:
            altered=change(path)
            if altered==old:raise AssertionError('mutation did not change file: '+name)
            path.write_bytes(altered)
            try:audit(target,source)
            except (ValueError,zipfile.BadZipFile) as err:records.append({'case':name,'rejected':True,'reason':str(err)})
            else:raise AssertionError('corruption was accepted: '+name)
        finally:path.write_bytes(old)
    final={'status':'PASS','rejected':len(records),'cases':records,'scope':'modified copies of recorded data, not new classifier examples'}
    (out/'report.json').write_text(json.dumps(final,indent=2));return final
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','source','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.root,a.source,a.out),indent=2))
