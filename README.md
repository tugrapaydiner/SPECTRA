# SPECTRA

**CPU-efficient verified search, compact reasoning experiments, and replayable evidence.**

SPECTRA is a research workbench, not a frontier-model replacement. The usable
public interface is `spectra`: dependency-light CNF tools and artifact checking.
The historical neural experiments remain reproducible, with their failed gates
visible rather than promoted into the default system.

[Current results](docs/STATUS.md) · [Documentation](docs/README.md) ·
[Development](docs/DEVELOPMENT.md) · [Historical experiments](docs/history/README.md)

## Measured implementation result

The optional indexed backend preserves seeded search paths while removing
per-flip sorting. In a frozen local comparison, larger 4096-flip executions take
**4.21x less time**, including preparation. Small cases can be slower and cold
allocation increases. Large cases in that comparison return `UNKNOWN`: this is
cheaper identical bounded search, not SAT superiority or a new learned result.
The [efficiency guide](docs/EFFICIENCY_GUIDE.md) includes all cells, ablations,
uncertainty, memory scope and reproduction commands.

Output-only CPU inference additionally avoids retaining diagnostic trajectories
and preserves final numerical outputs; returned tensor storage is not peak RAM.

## Start here

Use Python 3.10 or newer. Install from this checkout; this command does not
claim that version 0.7.1 is published to a package index.

```bash
python -m pip install .
spectra doctor
spectra cnf solve examples/tiny.cnf --seed 7 --max-flips 1024 --out answer.json
spectra cnf check examples/tiny.cnf answer.json
# Opt in; the historical compact backend remains the default.
spectra cnf solve examples/tiny.cnf --backend indexed --seed 7 --max-flips 4096 --out indexed.json
```

The base install needs neither PyTorch nor a C++ compiler. Output is JSON and
existing output files are never silently replaced. `SAT_VERIFIED` means the
complete Boolean assignment passed the original formula; `UNKNOWN` means no
witness was found within the flip cap. This is not a complete SAT decision
procedure and never reports an unproved `UNSAT` result.

```python
from spectra.cnf import CNF, CompactCNFRepairState, solve

problem = CNF(2, ((1, 2), (-1, 2)))
result = solve(problem, seed=7, max_flips=128)
assert result.status in {"SAT_VERIFIED", "UNKNOWN"}

state = CompactCNFRepairState(CNF(2, ((1, 2),)), (True, True))
assert state.make_break(0) == (0, 0)
assert state.make_break_patch((0, 1)) == (0, 1)  # joint flips are not additive
```

## What is supported

| Area | Entry point | Boundary |
|---|---|---|
| Exact CNF state and capped search | `spectra.cnf`, `spectra cnf` | Classical tools, not a learned solver |
| Portable artifact integrity | `spectra.evidence`, `spectra evidence` | Hash/size integrity, not scientific validity |
| Neural training and historical replay | Existing `model/`, `train/`, `eval/`, `scripts/` | Optional research dependencies and original protocols |
| Native kernels | `deploy/` | Sources ship in wheels; compilation is explicit |

The compact count/XOR state avoids global-index bitsets in every clause. Its
bounded comparison retains identical search paths while reducing large-instance
Python allocation footprint. Small-instance memory can increase. See the
[protocol and measurements](docs/STATUS.md#compact-cnf-state) before citing speed
or memory numbers. Neither this optimization nor packaging cleanup establishes
a new learned capability.

## Research installation and checks

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -r requirements-cpu-research.txt
python -m pip install --no-deps -e .
python -m pytest -m "not slow"
python scripts/verify_retained_results.py --out outputs/retained-check.json
```

Alternatively, `pip install ".[research]"` installs unpinned research extras.
The pinned environment above is used for historical CPU evidence. Long retraining
tests are separate from the fast suite; excluded tests are never counted as passes.

## Repository map

```text
spectra/       public API and CLI
examples/      small runnable inputs
model/ train/  historical neural implementation
common/ data/ eval/ deploy/   compatibility and evidence-bound research modules
scripts/       experiments, replays, and bounded maintenance tools
config/        historical experiment configurations
results/       retained raw evidence and reports
tests/         regression suite; public/ is dependency-light
docs/          current guides and pinned protocols; history/ indexes retired work
maintenance/   exact branch tips and file-retention receipts
```

Existing module paths and scientific records are intentionally preserved. Older
launch notes and one-off workflows are indexed at immutable commits instead of
crowding the active interface. Branch retirement first tags every exact tip;
advanced branches are rejected rather than overwritten. The active neighborhood
repair branch remains experimental: its new neighborhood component is a protocol,
not an already demonstrated learned improvement.
