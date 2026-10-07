"""Replay one occurrence of every custom solver path; timings are not reproduced."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.cover_search import audit_corrected,study
from spectra.cnf.cover import build_cover_runtime


def replay(folder: Path, native_directory: Path):
    cases,rows,_=audit_corrected.validate(folder)
    library=build_cover_runtime(native_directory)
    lookup={c['id']:c for c in cases}
    fields=('reason','nodes','decisions','propagations','backtracks','trace_fingerprint',
            'index_payload_bytes','state_word_bytes_peak','max_nodes','max_state_bytes')
    completed=0
    for row in rows:
        if row['round']!=0 or row['arm'] not in ('cover','dense','cover_cnf'):continue
        study.require(row['status']!='TIMEOUT','timed-out path cannot be claimed exact replay')
        actual=study.attempt(lookup[row['case']],row['arm'],library,controls=(None,None))
        study.require(actual['status']==row['status'],'replayed status differs')
        study.require(actual['witness']==row['witness'],'replayed witness differs')
        for key in fields:
            study.require(actual['details'][key]==row['details'][key],'replayed '+key+' differs')
        completed+=1
    study.require(completed==len(cases)*3,'incomplete replay inventory')
    return {'unique_cases':len(cases),'paths_replayed':completed,'timings_replayed':False,
            'mismatches':0,'library':str(library)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('--native-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=replay(a.folder,a.native_dir)
    with a.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))
