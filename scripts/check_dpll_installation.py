"""Check the actual dependency-free installed DPLL outside the source checkout."""
from __future__ import annotations
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def main():
    with tempfile.TemporaryDirectory(prefix='spectra-dpll-installed-') as temporary:
        os.chdir(temporary)
        import spectra.cnf.dpll
        from data.cnf import CNF
        from spectra.cnf.dpll import solve_dpll
        require(Path(spectra.cnf.dpll.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),
                'DPLL must come from installed wheel')
        for dependency in ('torch','numpy','pytest','pysat'):
            require(importlib.util.find_spec(dependency) is None,'unexpected dependency: '+dependency)
        clauses=[(),(1,),(-1,),(2,),(-2,),(1,2),(1,-2),(-1,2),(-1,-2)]
        for mask in range(512):
            p=CNF(2,tuple(c for i,c in enumerate(clauses) if mask & (1<<i)))
            valid=lambda w:all(any(w[abs(l)-1]==(l>0) for l in c) for c in p.clauses)
            expected=any(valid(w) for w in itertools.product((False,True),repeat=2))
            result=solve_dpll(p,max_decisions=32)
            require((result.status=='SAT_VERIFIED')==expected and valid(result.witness)==expected,
                    'truth-table or witness mismatch')
            require(result.decisions<=32,'decision budget exceeded')
            require(json.loads(json.dumps(result.record()))==result.record(),'record is not JSON-stable')
        print(json.dumps({'status':'PASS','exhaustive_formulas':512,'optional_dependencies':False,
                          'scope':'actual installed wheel; isolated Python; outside checkout'},indent=2))


if __name__=='__main__':
    main()
