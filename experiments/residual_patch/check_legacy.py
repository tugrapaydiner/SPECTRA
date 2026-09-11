"""Require every base blob and mode unchanged; allow only isolated additions."""
from __future__ import annotations
import json
import subprocess

BASE='e73dc05379b38ab2689e0e3054f6eee74e0290c0'


def tree(ref):
    output=subprocess.check_output(['git','ls-tree','-rz',ref])
    result={}
    for entry in output.split(b'\0'):
        if not entry:continue
        header,path=entry.split(b'\t',1)
        mode,kind,sha=header.decode().split()
        result[path.decode()]=(mode,kind,sha)
    return result


def main():
    base=tree(BASE);current=tree('HEAD')
    for path,value in base.items():
        if current.get(path)!=value:raise ValueError('legacy file/mode changed: '+path)
    added=sorted(set(current)-set(base))
    for path in added:
        if not (path.startswith('experiments/residual_patch/') or path=='.github/workflows/residual-patch-overhaul.yml'):
            raise ValueError('unrelated addition: '+path)
    print(json.dumps({'legacy_integrity':'PASS','base':BASE,'unchanged_files':len(base),'isolated_additions':added},indent=2))

if __name__=='__main__':main()
