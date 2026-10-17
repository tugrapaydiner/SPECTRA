"""Contracts for current strong-solver adapters and gauntlet reporting.

The native fixtures are intentionally tiny brute-force implementations of the
published C APIs.  They test our ownership, assumptions, model decoding, rebuild,
checking and lifecycle code without pretending to benchmark the real solvers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest

from .binary_list import BinaryListProblem
from .champion_competitors import CaDiCaL3Session, Kissat4RebuildSession
from .champion_gauntlet import gmean, _svg_bar


class _Checker:
    def __init__(self, problem):
        self.problem = problem
        self.payload_bytes = 0
        self.closed = False

    def check(self, labels, restrictions):
        if self.closed:
            raise RuntimeError('checker closed')
        return self.problem.check(labels, restrictions)

    def close(self):
        self.closed = True


class _Runtime:
    def checker(self, n, palette, edges, masks, _max_bytes):
        return _Checker(BinaryListProblem.create(n, palette, edges, masks))


_CADICAL = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct { int data[4096], used, assumptions[256], n_assumptions, max_var; uint64_t model; } CCaDiCaL;
const char *ccadical_signature(void){ return "fake-cadical-3"; }
CCaDiCaL *ccadical_init(void){ return (CCaDiCaL*) calloc(1,sizeof(CCaDiCaL)); }
void ccadical_release(CCaDiCaL *s){ free(s); }
void ccadical_set_option(CCaDiCaL *s,const char*n,int v){ (void)s;(void)n;(void)v; }
void ccadical_freeze(CCaDiCaL *s,int lit){ if(abs(lit)>s->max_var)s->max_var=abs(lit); }
void ccadical_add(CCaDiCaL *s,int lit){ s->data[s->used++]=lit; if(abs(lit)>s->max_var)s->max_var=abs(lit); }
void ccadical_assume(CCaDiCaL *s,int lit){ s->assumptions[s->n_assumptions++]=lit; if(abs(lit)>s->max_var)s->max_var=abs(lit); }
static int value(uint64_t m,int lit){ int bit=(m>>(abs(lit)-1))&1; return lit>0?bit:!bit; }
static int clauses(CCaDiCaL*s,uint64_t m){ int ok=0; for(int i=0;i<s->used;i++){int lit=s->data[i]; if(!lit){if(!ok)return 0;ok=0;} else if(value(m,lit))ok=1;} return ok==0; }
int ccadical_solve(CCaDiCaL *s){ uint64_t limit=1ull<<s->max_var; for(uint64_t m=0;m<limit;m++){ int ok=clauses(s,m); for(int i=0;ok&&i<s->n_assumptions;i++)ok=value(m,s->assumptions[i]); if(ok){s->model=m;s->n_assumptions=0;return 10;} } s->n_assumptions=0; return 20; }
int ccadical_val(CCaDiCaL *s,int lit){ int bit=(s->model>>(abs(lit)-1))&1; return bit?abs(lit):-abs(lit); }
'''

_KISSAT = r'''
#include <stdint.h>
#include <stdlib.h>
typedef struct { int data[4096], used, max_var; uint64_t model; } kissat;
const char *kissat_signature(void){ return "fake-kissat-4"; }
const char *kissat_version(void){ return "4.0.4-test"; }
kissat *kissat_init(void){ return (kissat*) calloc(1,sizeof(kissat)); }
void kissat_release(kissat *s){ free(s); }
void kissat_set_option(kissat*s,const char*n,int v){(void)s;(void)n;(void)v;}
void kissat_reserve(kissat*s,int n){if(n>s->max_var)s->max_var=n;}
void kissat_add(kissat*s,int lit){s->data[s->used++]=lit;if(abs(lit)>s->max_var)s->max_var=abs(lit);}
static int value(uint64_t m,int lit){int bit=(m>>(abs(lit)-1))&1;return lit>0?bit:!bit;}
static int clauses(kissat*s,uint64_t m){int ok=0;for(int i=0;i<s->used;i++){int lit=s->data[i];if(!lit){if(!ok)return 0;ok=0;}else if(value(m,lit))ok=1;}return ok==0;}
int kissat_solve(kissat*s){uint64_t limit=1ull<<s->max_var;for(uint64_t m=0;m<limit;m++)if(clauses(s,m)){s->model=m;return 10;}return 20;}
int kissat_value(kissat*s,int lit){int bit=(s->model>>(abs(lit)-1))&1;return bit?abs(lit):-abs(lit);}
'''


def _compile(tmp: Path, name: str, source: str) -> Path:
    cc = shutil.which('cc') or shutil.which('gcc')
    if not cc:
        pytest.skip('C compiler unavailable')
    c = tmp / f'{name}.c'; so = tmp / f'lib{name}.so'
    c.write_text(source)
    subprocess.run([cc, '-shared', '-fPIC', '-O2', str(c), '-o', str(so)], check=True)
    return so


@pytest.fixture
def problem():
    return BinaryListProblem.create(2, 2, ((0, 1),), (0b11, 0b11))


@pytest.mark.parametrize('kind', ('cadical', 'kissat'))
def test_native_competitor_adapters_return_complete_checked_models_and_exact_decisions(
        tmp_path, problem, kind):
    runtime = _Runtime()
    if kind == 'cadical':
        session = CaDiCaL3Session(problem, runtime, _compile(tmp_path, 'cadical', _CADICAL), 1 << 20)
        assert session.info['persistent_learned_state'] is True
    else:
        session = Kissat4RebuildSession(problem, runtime, _compile(tmp_path, 'kissat', _KISSAT), 1 << 20)
        assert session.info['rebuild_per_query'] is True
    try:
        sat = session.solve(())
        assert sat.status == 'SAT_VERIFIED'
        assert len(sat.labels) == 2 and problem.check(sat.labels, ())
        pinned = session.solve(((0, 0b01),))
        assert pinned.status == 'SAT_VERIFIED'
        assert pinned.labels == bytes((0, 1))
        unsat = session.solve(((0, 0b01), (1, 0b01)))
        assert unsat.status == 'UNSAT' and unsat.labels == b''
        empty = session.solve(((0, 0b01), (0, 0b10)))
        assert empty.status == 'UNSAT' and empty.labels == b''
        assert session.info['solves'] == 3  # empty intersection is decided before native search
    finally:
        session.close()
    with pytest.raises(RuntimeError):
        session.solve(())


def test_missing_external_library_fails_closed(problem, tmp_path):
    with pytest.raises(FileNotFoundError):
        CaDiCaL3Session(problem, _Runtime(), tmp_path/'missing.so', 1 << 20)


def test_chart_and_geometric_mean_are_deterministic(tmp_path):
    assert gmean([.5, 2.0]) == pytest.approx(1.0)
    target = tmp_path/'chart.svg'
    _svg_bar(target, 'A < B & honest', [('x', .8), ('y', 1.2)])
    text = target.read_text()
    assert 'A &lt; B &amp; honest' in text
    assert 'tie = 1.0' in text and '0.800' in text and '1.200' in text
    assert text == target.read_text()


def test_ortools_adapter_if_dependency_is_installed(problem):
    pytest.importorskip('ortools')
    from .champion_competitors import ORToolsCPSATSession
    session = ORToolsCPSATSession(problem, _Runtime(), 1 << 20)
    try:
        assert session.solve(((0, 1),)).labels == bytes((0, 1))
        assert session.solve(((0, 1), (1, 1))).status == 'UNSAT'
        assert session.info['workers'] == 1
    finally:
        session.close()


def test_report_preserves_fixed_and_oracle_labels(tmp_path, monkeypatch):
    import experiments.compiled_query.champion_gauntlet as module
    times = {
        'exact-raw': 100.0, 'exact-scc': 80.0, 'exact-route': 90.0,
        'cadical3': 160.0, 'minicard': 200.0, 'cadical195': 180.0,
        'cudd-reduced-bfs': 140.0, 'ortools-cpsat': 220.0,
        'kissat4-rebuild': 400.0,
    }
    rows = []
    for count in module.COUNTS:
        for arm in module.ARMS:
            rows.append(dict(case='case-a', cluster='case-a', count=count, arm=arm,
                             eligible=True, mean_complete_ns=times[arm]))
    audit = dict(all_scheduled_decisions_complete=True, scheduled_jobs=len(rows),
                 outcomes={'COMPLETE': len(rows)}, audited_decisions={'SAT_VERIFIED': 10},
                 session_summaries=rows)
    monkeypatch.setattr(module, 'audit_study', lambda _root: audit)
    output = tmp_path/'report'
    summary = module.report(tmp_path/'unused', output)
    assert summary['primary_mode'] == 'exact-scc'
    direct = summary['counts']['1024']['exact-raw']
    assert direct['fixed']['exact-scc']['geomean_ratio'] == pytest.approx(.8)
    assert direct['oracle_best']['selected_modes'] == {'case-a': 'exact-scc'}
    assert (output/'gauntlet-primary-q1024.svg').is_file()
    assert (output/'gauntlet-oracle-diagnostic-q1024.svg').is_file()
    assert 'Fixed SCC ratio' in (output/'REPORT.md').read_text()
    assert 'oracle-best-of-raw-scc-route' in (output/'matrix.csv').read_text()
