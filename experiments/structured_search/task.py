"""Sudoku task encoding and independent original-grid solution checking."""
from itertools import combinations
from data.cnf import CNF


def validate_puzzle(puzzle):
    if type(puzzle) is not str or len(puzzle) != 81 or any(c not in '.0123456789' for c in puzzle):
        raise ValueError('expected exactly 81 ASCII Sudoku cells')


def encode(puzzle):
    validate_puzzle(puzzle)
    def variable(r, c, d):
        return 81*r+9*c+d+1
    groups = [tuple(variable(r, c, d) for d in range(9)) for r in range(9) for c in range(9)]
    groups += [tuple(variable(r, c, d) for c in range(9)) for r in range(9) for d in range(9)]
    groups += [tuple(variable(r, c, d) for r in range(9)) for c in range(9) for d in range(9)]
    groups += [tuple(variable(r, c, d) for r in range(br, br+3) for c in range(bc, bc+3))
               for br in (0, 3, 6) for bc in (0, 3, 6) for d in range(9)]
    clauses = []
    for group in groups:
        clauses.append(group)
        clauses.extend((-a, -b) for a, b in combinations(group, 2))
    clauses.extend((9*i+int(value),) for i, value in enumerate(puzzle) if value in '123456789')
    return CNF(729, tuple(clauses))


def decode(witness):
    if len(witness) != 729 or any(type(v) is not bool for v in witness):
        raise ValueError('expected complete Boolean CNF witness')
    grid = []
    for i in range(81):
        active = [d+1 for d in range(9) if witness[9*i+d]]
        if len(active) != 1:
            raise ValueError('witness violates one-hot cell constraint')
        grid.append(str(active[0]))
    return ''.join(grid)


def check_grid(puzzle, grid):
    """Check clues, rows, columns and boxes without consulting the encoder."""
    validate_puzzle(puzzle)
    if type(grid) is not str or len(grid) != 81 or any(c not in '123456789' for c in grid):
        return False
    if any(p in '123456789' and p != value for p, value in zip(puzzle, grid)):
        return False
    expected = set('123456789')
    for i in range(9):
        if set(grid[9*i:9*i+9]) != expected or set(grid[i::9]) != expected:
            return False
    return all({grid[9*r+c] for r in range(br, br+3) for c in range(bc, bc+3)} == expected
               for br in (0, 3, 6) for bc in (0, 3, 6))


def check_cnf(problem, witness):
    return (len(witness) == problem.nvars and all(type(v) is bool for v in witness)
            and all(any(witness[abs(lit)-1] == (lit > 0) for lit in clause) for clause in problem.clauses))
