from io import StringIO
import json
import subprocess
import sys

import pytest
from spectra.cnf import CNF, read_dimacs, write_dimacs, solve
from spectra.cnf.search import _execute
from eval.cnf_cached import CachedCNFRepairState
from spectra.cli import main

@pytest.mark.parametrize("text", ["p cnf 0 0\n", "p cnf 0 1\n0\n",
    "c hi\np cnf 3 2\n1 -2\nc between\n3 0 0\n", "p cnf 2 2\n1 1 -1 0 -2 0"])
def test_roundtrip(text):
    p=read_dimacs(StringIO(text));out=StringIO();write_dimacs(p,out)
    assert read_dimacs(StringIO(out.getvalue())) == p

@pytest.mark.parametrize("text", ["", "1 0", "p cnf 1 1\n", "p cnf 1 0\n0",
    "p cnf 1 1\n2 0", "p cnf 1 1\n1", "p cnf -1 0", "p cnf 1 0\np cnf 1 0",
    "p cnf 1 1\n1.0 0", "p dnf 1 0", "p cnf 1 0 junk", "p cnf 1 1\n0 junk"])
def test_reject_bad_dimacs(text):
    with pytest.raises(ValueError):read_dimacs(StringIO(text))

def test_resource_limits():
    for kwargs in ({"max_variables":1}, {"max_clauses":0}, {"max_literals":0}):
        with pytest.raises(ValueError): read_dimacs(StringIO("p cnf 2 1\n1 0"), **kwargs)
    with pytest.raises(ValueError): read_dimacs(StringIO("p cnf 0 0"),max_variables=True)

@pytest.mark.parametrize("seed", range(12))
def test_identical_policy_and_verified_witness(seed):
    from data.cnf import random_3sat
    p,_=random_3sat(32,134,seed)
    a=solve(p,seed=seed,max_flips=128)
    b=_execute(p,seed=seed,max_flips=128,state_class=CachedCNFRepairState)
    ar,br=a.record(),b.record()
    ar.pop("elapsed_ns");br.pop("elapsed_ns")
    assert ar==br
    assert (a.status=="SAT_VERIFIED")==p.satisfied(a.witness)
    assert a.flips<=128

@pytest.mark.parametrize("kwargs", [{"seed":True},{"seed":-1},{"seed":2**64},
    {"max_flips":True},{"max_flips":-1},{"max_flips":1.0}])
def test_bad_search_settings(kwargs):
    with pytest.raises(ValueError):solve(CNF(0,()),**kwargs)

def test_empty_and_capped_unknown():
    assert solve(CNF(0,())).status=="SAT_VERIFIED"
    assert solve(CNF(0,((),))).status=="UNKNOWN"
    r=solve(CNF(1,((1,),(-1,))),max_flips=0)
    assert r.flips==0 and r.status=="UNKNOWN"

def test_cli_round_trip_and_no_overwrite(tmp_path,capsys):
    p=tmp_path/"problem.cnf";p.write_text("p cnf 1 1\n1 0")
    out=tmp_path/"answer.json"
    assert main(["cnf","solve",str(p),"--out",str(out)])==0
    before=out.read_bytes()
    assert main(["cnf","solve",str(p),"--out",str(out)])==2
    assert out.read_bytes()==before
    assert main(["cnf","check",str(p),str(out)])==0
    p.write_text("p cnf 1 1\n-1 0")
    assert main(["cnf","check",str(p),str(out)])==2

def test_cli_rejects_false_witness_and_missing_files(tmp_path):
    p=tmp_path/"problem.cnf";p.write_text("p cnf 1 1\n1 0")
    out=tmp_path/"answer.json";out.write_text('{"witness":[false]}')
    assert main(["cnf","check",str(p),str(out)])==1
    out.write_text('{"witness":[1]}')
    assert main(["cnf","check",str(p),str(out)])==2
    assert main(["cnf","solve",str(tmp_path/"missing")])==2

def test_import_is_dependency_light():
    script = "import spectra, spectra.cnf, spectra.evidence; import sys; assert not ({'torch','numpy'} & sys.modules.keys())"
    subprocess.run([sys.executable,"-S","-c",script],check=True)


@pytest.mark.parametrize("text", ["p cnf 1_0 0", "p cnf ١ 0", "p cnf +1 0", "p cnf 10 1\n1_0 0", "p cnf 1 1\n+1 0"])
def test_ascii_dimacs_lexical_contract(text):
    with pytest.raises(ValueError):
        read_dimacs(StringIO(text))
