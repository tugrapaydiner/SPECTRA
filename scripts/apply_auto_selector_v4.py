from __future__ import annotations

from pathlib import Path
import re

CPP_FILES = (
    Path("spectra/_native/exact_binary.cpp"),
    Path("experiments/compiled_query/exact_native.cpp"),
)
PY_FILES = (
    Path("spectra/exact.py"),
    Path("experiments/compiled_query/exact.py"),
)

OLD_BLOCK = '''        if(mode==4) {
            // Structural auto policy v3. The optimized tile64 SCC engine wins
            // long low-pin banks even when contraction is modest. Strong SCC
            // contraction also pays before one complete 64-row tile. Selection
            // uses only immutable graph/request geometry, never answers or time.
            if(bad!=NIL) {
                condensed=false;routed=false;auto_reason="base_infeasible";
            } else if(W(bank_pins)>=8*W(bank_size)) {
                condensed=false;routed=false;auto_reason="high_pin_density";
            } else if(bank_size>=64) {
                condensed=true;routed=false;auto_reason="scc_tile64";
            } else if(bank_size>=8 && W(2)*W(nc)<=W(n)) {
                condensed=true;routed=false;auto_reason="scc_strong_compression";
            } else {
                condensed=false;routed=false;auto_reason="short_weak_compression";
            }
        }
'''

NEW_BLOCK = '''        if(mode==4) {
            // Structural auto policy v4. Absolute pin count is not enough: a
            // large graph can carry many pins while still remaining sparse.
            // Use raw only when both the absolute and graph-relative density
            // tests fire. Selection consults immutable geometry only.
            if(bad!=NIL) {
                condensed=false;routed=false;auto_reason="base_infeasible";
            } else if(W(bank_pins)>=8*W(bank_size)
                    && W(32)*W(bank_pins)>=W(n)*W(bank_size)) {
                condensed=false;routed=false;auto_reason="dense_pin_fraction";
            } else if(bank_size>=64) {
                condensed=true;routed=false;auto_reason="scc_tile64";
            } else if(bank_size>=8 && W(2)*W(nc)<=W(n)) {
                condensed=true;routed=false;auto_reason="scc_strong_compression";
            } else {
                condensed=false;routed=false;auto_reason="short_weak_compression";
            }
        }
'''

DOC_PATTERN = re.compile(
    r"The default ``auto`` mode uses structural policy v3\..*?"
    r"structural heuristic rather than a measured-timing oracle\.",
    re.DOTALL,
)
DOC_REPLACEMENT = '''The default ``auto`` mode uses structural policy v4. It keeps a
        base-infeasible graph on raw traversal. A request bank also stays on raw
        when it averages at least eight pins per request and those pins cover at
        least one original vertex in 32. Feasible banks of at least 64 requests
        use SCC; banks from eight requests use SCC earlier when SCCs compress the
        literal graph by at least fourfold. Selection uses immutable graph/request
        geometry only, never answers or measured timings. ``info['selected_mode']``
        and ``info['auto_policy_reason']`` expose the decision. Explicit modes keep
        their meaning; the automatic policy is a frozen structural dispatcher.'''

for path in CPP_FILES:
    text = path.read_text(encoding="utf-8")
    if text.count(OLD_BLOCK) != 1:
        raise RuntimeError(f"expected one v3 selector block in {path}")
    text = text.replace(OLD_BLOCK, NEW_BLOCK)
    if text.count('which==4?"structural-v3":"explicit"') != 1:
        raise RuntimeError(f"expected one v3 policy identity in {path}")
    text = text.replace(
        'which==4?"structural-v3":"explicit"',
        'which==4?"structural-v4":"explicit"',
    )
    path.write_text(text, encoding="utf-8")

for path in PY_FILES:
    text = path.read_text(encoding="utf-8")
    text, count = DOC_PATTERN.subn(DOC_REPLACEMENT, text, count=1)
    if count != 1:
        raise RuntimeError(f"expected one v3 public-policy paragraph in {path}")
    path.write_text(text, encoding="utf-8")

exact_test = Path("tests/public/test_exact_api.py")
text = exact_test.read_text(encoding="utf-8")
if "'structural-v3'" not in text:
    raise RuntimeError("exact API test does not contain the v3 identity")
exact_test.write_text(text.replace("'structural-v3'", "'structural-v4'"), encoding="utf-8")

readme = Path("experiments/compiled_query/README.md")
text = readme.read_text(encoding="utf-8")
text = text.replace("structural policy v3", "structural policy v4")
text = text.replace("high-pin", "dense-pin")
readme.write_text(text, encoding="utf-8")

if CPP_FILES[0].read_bytes() != CPP_FILES[1].read_bytes():
    raise RuntimeError("packaged and research exact sources differ")
