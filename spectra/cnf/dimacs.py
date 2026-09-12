"""Strict streaming DIMACS CNF interchange; original clause order is retained."""
from __future__ import annotations
from typing import TextIO
import re
from data.cnf import CNF


def read_dimacs(stream: TextIO, *, max_variables: int = 1_000_000,
                max_clauses: int = 10_000_000, max_literals: int = 30_000_000) -> CNF:
    """Read one header and zero-terminated clauses; c-prefixed lines are comments.

    Clauses can span lines or share a line. A zero with no pending literals is
    an empty clause, not end of file. Resource limits are checked before building
    state; they bound declarations/parsed literals, not arbitrary input byte size.
    """
    for limit in (max_variables, max_clauses, max_literals):
        if type(limit) is not int or limit < 0:
            raise ValueError("DIMACS limits must be nonnegative integers")
    header = None
    clauses: list[tuple[int, ...]] = []
    pending: list[int] = []
    total = 0
    for line_number, raw in enumerate(stream, 1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        tokens = line.split()
        if tokens[0] == "p":
            if header is not None or len(tokens) != 4 or tokens[1] != "cnf":
                raise ValueError(f"line {line_number}: expected one 'p cnf N M' header")
            try:
                if not all(re.fullmatch(r"[0-9]+", t) for t in tokens[2:]):
                    raise ValueError("nondecimal count")
                n, m = int(tokens[2]), int(tokens[3])
            except ValueError as error:
                raise ValueError(f"line {line_number}: invalid CNF counts") from error
            if not (0 <= n <= max_variables and 0 <= m <= max_clauses):
                raise ValueError("CNF counts exceed configured limits")
            header = (n, m)
            continue
        if header is None:
            raise ValueError(f"line {line_number}: clause precedes header")
        n, m = header
        for token in tokens:
            try:
                if not re.fullmatch(r"-?[0-9]+", token):
                    raise ValueError("nondecimal literal")
                value = int(token)
            except ValueError as error:
                raise ValueError(f"line {line_number}: invalid literal {token!r}") from error
            if value == 0:
                clauses.append(tuple(pending))
                pending.clear()
                if len(clauses) > m:
                    raise ValueError("more clauses than declared")
            else:
                total += 1
                if abs(value) > n or total > max_literals:
                    raise ValueError("literal outside declared range or resource limit")
                pending.append(value)
    if header is None:
        raise ValueError("missing CNF header")
    if pending:
        raise ValueError("unterminated final clause")
    if len(clauses) != header[1]:
        raise ValueError("clause count differs from header")
    return CNF(header[0], tuple(clauses))


def write_dimacs(problem: CNF, stream: TextIO) -> None:
    if not isinstance(problem, CNF):
        raise TypeError("CNF required")
    stream.write(f"p cnf {problem.nvars} {len(problem.clauses)}\n")
    for clause in problem.clauses:
        stream.write(" ".join(map(str, (*clause, 0))) + "\n")
