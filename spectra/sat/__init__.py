"""General CNF solving with independently checked answers.

``native_cdcl`` is a deterministic CDCL solver (watched literals, first-UIP
learning, restarts) whose SAT models are checked against the original clauses
and whose UNSAT answers carry DRUP proofs replayed by a separate checker.
``binary_core`` is a proof-replayable unit/binary simplifier, ``admission``
strictly validates DIMACS inputs, and ``drat`` connects to an external DRAT-trim
checker. Native code is compiled only by an explicit build call.
"""
