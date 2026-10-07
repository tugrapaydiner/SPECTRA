"""Direct problem adapters and reproducible synthetic development generators."""
from itertools import combinations
from data.cnf import CNF, SplitMix64
from spectra.cnf.cover import ChoiceProblem
from experiments.structured_search.task import validate_puzzle


def sudoku(puzzle: str) -> ChoiceProblem:
    validate_puzzle(puzzle)
    def v(r,c,d): return 81*r+9*c+d+1
    groups=[tuple(v(r,c,d) for d in range(9)) for r in range(9) for c in range(9)]
    groups += [tuple(v(r,c,d) for c in range(9)) for r in range(9) for d in range(9)]
    groups += [tuple(v(r,c,d) for r in range(9)) for c in range(9) for d in range(9)]
    groups += [tuple(v(r,c,d) for r in range(br,br+3) for c in range(bc,bc+3))
               for br in (0,3,6) for bc in (0,3,6) for d in range(9)]
    clues=tuple((9*i+int(c),) for i,c in enumerate(puzzle) if c in '123456789')
    return ChoiceProblem(729,tuple(groups)+clues,tuple(groups))


def cnf(problem: ChoiceProblem) -> CNF:
    return CNF(problem.nvars,problem.covers+tuple((-a,-b) for g in problem.exclusive for a,b in combinations(g,2)))


def colouring(n: int, colours: int, seed: int, *, planted: bool) -> dict:
    """Independent G(n,p) edges with expected degree about 4.5; no solver filtering.

    Planted graphs omit same-label edges (different, explicitly easier family).
    Full colour domains; this is a synthetic graph workload, not real scheduling.
    """
    rng=SplitMix64(seed)
    labels=[rng.below(colours) for _ in range(n)]
    edges=[]
    # Correct for removed same-label pairs only in the planted generator.
    denom=(n-1)*(colours-1 if planted else colours)
    for a in range(n):
        for b in range(a+1,n):
            if (not planted or labels[a]!=labels[b]) and rng.below(denom*2)<9*colours:
                edges.append((a,b))
    return {'kind':'colouring','n':n,'colours':colours,'seed':seed,'planted':planted,'edges':edges}


def build(case: dict) -> ChoiceProblem:
    if case['kind']=='sudoku': return sudoku(case['puzzle'])
    if case['kind']!='colouring': raise ValueError('unknown task')
    n,k=case['n'],case['colours']
    groups=tuple(tuple(i*k+d+1 for d in range(k)) for i in range(n))
    excludes=tuple((a*k+d+1,b*k+d+1) for a,b in case['edges'] for d in range(k))
    return ChoiceProblem(n*k,groups,groups+excludes)


def check(case: dict, witness: tuple[bool,...]) -> bool:
    """Original task checker, independent of the choice/CNF group encoder."""
    if type(witness) is not tuple or any(type(v) is not bool for v in witness): return False
    if case['kind']=='sudoku':
        from experiments.structured_search.task import decode,check_grid
        try: return check_grid(case['puzzle'],decode(witness))
        except ValueError: return False
    n,k=case['n'],case['colours']
    if len(witness)!=n*k: return False
    labels=[]
    for i in range(n):
        active=[d for d in range(k) if witness[i*k+d]]
        if len(active)!=1:return False
        labels.append(active[0])
    return all(labels[a]!=labels[b] for a,b in case['edges'])
