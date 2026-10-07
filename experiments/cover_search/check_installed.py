"""Run with a clean installed environment's python -I; never prepend checkout paths."""
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import spectra
from data.cnf import CNF
from spectra.cnf.cover import ChoiceProblem,PreparedCover,build_cover_runtime,solve_cover


def require(ok,message):
    if not ok:raise AssertionError(message)


def main():
    package=Path(spectra.__file__).resolve()
    require('site-packages' in package.parts,'installed package not imported')
    require(not any(x in sys.modules for x in ('numpy','torch','scipy','sklearn','pysat')),'unexpected numerical dependency')
    with tempfile.TemporaryDirectory(prefix='spectra-cover-installed-') as temp:
        library=build_cover_runtime(Path(temp)/'native')
        bank=[(),(1,),(2,),(1,2),(-1,),(-2,),(-1,-2),(1,-1),(2,2)]
        for mask in range(512):
            problem=CNF(2,tuple(c for i,c in enumerate(bank) if mask>>i&1))
            answer=solve_cover(problem,library)
            possible=any(all(any(w[abs(v)-1] == (v>0) for v in c) for c in problem.clauses)
                         for w in product((False,True),repeat=2))
            require((answer.status=='SAT_VERIFIED')==possible,'truth-table discrepancy')
            if possible:require(not problem.violated(answer.witness),'invalid installed witness')
        p=ChoiceProblem(4,((1,2),(3,4)),((1,3),(2,4)))
        with PreparedCover(p,library) as index:
            require(index.solve().status=='SAT_VERIFIED','direct groups fail')
            require(index.solve(max_nodes=0).status=='UNKNOWN','zero budget ignored')
        try:index.solve()
        except RuntimeError:pass
        else:raise AssertionError('closed index used')
        print(json.dumps({'status':'PASS','exhaustive_formulas':512,'direct_lifecycle_checks':3,
                          'installed_package':str(package),'numerical_dependencies':False,
                          'native_source_sha256':json.loads((Path(temp)/'native/build.json').read_text())['source_sha256']}))

if __name__=='__main__':main()
