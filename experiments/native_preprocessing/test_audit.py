"""Corrupt actual accepted receipts; requires explicit input paths, not CI data downloads."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import pytest

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('preprocessing_receipt_audit',HERE/'audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

@pytest.mark.parametrize('change',['missing','extra','reorder','bool_time','zero_time','matched','summary','duplicate_key'])
def test_corruption_rejected(tmp_path,change):
    run=Path(os.environ['SPECTRA_PREPROCESS_RUN'])
    for p in run.iterdir():
        if p.is_file():shutil.copy2(p,tmp_path/p.name)
    rows=(tmp_path/'rows.jsonl').read_text().splitlines()
    if change=='missing':rows.pop()
    elif change=='extra':rows.append(rows[0])
    elif change=='reorder':rows[0],rows[1]=rows[1],rows[0]
    elif change in ('bool_time','zero_time','matched'):
        r=json.loads(rows[0]);r['matched' if change=='matched' else 'ns']=False if change=='matched' else True if change=='bool_time' else 0
        rows[0]=json.dumps(r)
    elif change=='duplicate_key':rows[0]=rows[0][:-1]+',"ns":1}'
    else:
        p=tmp_path/'summary.json';r=json.loads(p.read_text());r['cells']+=1;p.write_text(json.dumps(r))
    (tmp_path/'rows.jsonl').write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError):audit.check(tmp_path,Path(os.environ['SPECTRA_PREPROCESS_INPUTS']),Path(os.environ['SPECTRA_PREPROCESS_SOURCE']))
