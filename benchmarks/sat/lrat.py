"""Write bounded ASCII LRAT certificates for original-CNF unit propagation."""
from __future__ import annotations

from collections import defaultdict
from io import BytesIO
from shutil import copyfileobj
from tempfile import SpooledTemporaryFile


class ProofLimit(Exception):
    """The next addition would exceed an explicit certificate bound."""


class LRATWriter:
    def __init__(self, stream, original_count, max_rows, max_literals, max_bytes):
        self.stream = stream
        self.next_id = original_count + 1
        self.remaining_rows = max_rows
        self.remaining_literals = max_literals
        self.remaining_bytes = max_bytes

    def add(self, clause, hints=()):
        if not self.remaining_rows or len(clause) > self.remaining_literals:
            raise ProofLimit
        line = (" ".join(map(str, (self.next_id, *clause, 0, *hints, 0))) + "\n").encode("ascii")
        if len(line) > self.remaining_bytes:
            raise ProofLimit
        self.stream.write(line)
        self.remaining_rows -= 1
        self.remaining_literals -= len(clause)
        self.remaining_bytes -= len(line)
        self.next_id += 1
        return self.next_id - 1


def write_input_empty_lrat(path, clauses, max_rows, max_literals, max_bytes):
    """Cite the original empty clause directly, preserving its DIMACS clause number."""
    buffer = BytesIO()
    writer = LRATWriter(buffer, len(clauses), max_rows, max_literals, max_bytes)
    try:
        writer.add((), (clauses.index(()) + 1,))
    except ProofLimit:
        return False
    with path.open("xb") as destination:
        destination.write(buffer.getvalue())
    return True


class _UnitReasons:
    def __init__(self):
        self.empty = None
        self.units = []
        self.occurrences = defaultdict(list)

    def add(self, number, clause):
        if not clause:
            self.empty = number
        elif len(clause) == 1:
            self.units.append((number, clause))
        else:
            reason = (number, clause)
            for literal in clause:
                self.occurrences[literal].append(reason)

    def explain(self, clause):
        if self.empty is not None:
            return [self.empty]
        values = {abs(v): -v for v in clause}
        queue = list(values.values())
        trail = []
        for reason in self.units:
            literal = reason[1][0]
            assigned = values.get(abs(literal))
            if assigned == -literal:
                return self._conflict(reason, trail)
            if assigned is None:
                values[abs(literal)] = literal
                queue.append(literal)
                trail.append((literal, reason))
        for assigned in queue:
            for reason in self.occurrences.get(-assigned, ()):
                unit = 0
                for literal in reason[1]:
                    value = values.get(abs(literal))
                    if value == literal:
                        break
                    if value is None:
                        if unit:
                            break
                        unit = literal
                else:
                    if not unit:
                        return self._conflict(reason, trail)
                    values[abs(unit)] = unit
                    queue.append(unit)
                    trail.append((unit, reason))
        raise ValueError("native proof addition failed original-CNF unit propagation")

    @staticmethod
    def _conflict(reason, trail):
        number, clause = reason
        hints = [number]
        needed = {abs(v) for v in clause}
        for literal, (number, antecedent) in reversed(trail):
            if abs(literal) in needed:
                needed.remove(abs(literal))
                needed.update(abs(v) for v in antecedent if v != literal)
                hints.append(number)
        hints.reverse()
        return hints


def write_rup_lrat(path, clauses, proof, max_rows, max_literals, max_bytes):
    """Return False on a size limit. Publish the file only after deriving the empty clause."""
    from spectra.sat.native_cdcl import DRUPProof, PROOF_SCHEMA

    if (not isinstance(proof, DRUPProof) or proof.schema != PROOF_SCHEMA
            or not proof.additions or proof.additions[-1] != ()):
        raise ValueError("expected a complete native DRUP proof")
    if len(proof.additions) > max_rows or max_bytes == 0:
        return False
    database = _UnitReasons()
    duplicates = {}
    with SpooledTemporaryFile(max_size=1 << 20, dir=path.parent) as stream:
        writer = LRATWriter(stream, len(clauses), max_rows, max_literals, max_bytes)
        try:
            for number, row in enumerate(clauses, 1):
                literals = dict.fromkeys(row)
                if any(-v in literals for v in literals):
                    continue
                if len(literals) != len(row):
                    row = tuple(literals)
                    duplicates[number] = row, None
                database.add(number, row)
            for row in proof.additions:
                hints = database.explain(row)
                # Only unit hints need normalization. Every literal in the final conflict is already false.
                for position, original in enumerate(hints[:-1]):
                    if original in duplicates:
                        normalized, number = duplicates[original]
                        if number is None:
                            number = writer.add(normalized, (original,))
                            duplicates[original] = normalized, number
                        hints[position] = number
                number = writer.add(row, hints)
                database.add(number, row)
        except ProofLimit:
            return False
        stream.seek(0)
        with path.open("xb") as destination:
            copyfileobj(stream, destination, 65536)
    return True

