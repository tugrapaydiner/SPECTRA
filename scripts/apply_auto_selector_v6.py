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

OLD_CPP = '''        if(mode==4) {
            // Structural auto policy v5. Dense request banks stay raw, but the
            // density threshold is graph-relative. A separate warmup guard
            // handles edge-heavy graphs with no SCC contraction before the
            // quotient amortises. Selection consults immutable geometry only.
            if(bad!=NIL) {
                condensed=false;routed=false;auto_reason="base_infeasible";
            } else if(W(bank_pins)>=8*W(bank_size)
                    && W(24)*W(bank_pins)>=W(n)*W(bank_size)) {
                condensed=false;routed=false;auto_reason="dense_pin_fraction";
            } else if(bank_size<=128 && W(nc)==2*W(n)
                    && W(p.edges.size())>=4*W(n)) {
                condensed=false;routed=false;auto_reason="dense_uncompressed_warmup";
            } else if(bank_size>=64) {
                condensed=true;routed=false;auto_reason="scc_tile64";
            } else if(bank_size>=8 && W(2)*W(nc)<=W(n)) {
                condensed=true;routed=false;auto_reason="scc_strong_compression";
            } else {
                condensed=false;routed=false;auto_reason="short_weak_compression";
            }
        }
'''

NEW_CPP = '''        if(mode==4) {
            // Structural auto policy v6. Large tiny banks take the exact raw
            // path directly at the public boundary; keep the same rule here for
            // native callers. Remaining choices use immutable geometry only.
            if(bank_size<=8 && n>512) {
                condensed=false;routed=false;auto_reason="large_tiny_bank";
            } else if(bad!=NIL) {
                condensed=false;routed=false;auto_reason="base_infeasible";
            } else if(W(bank_pins)>=8*W(bank_size)
                    && W(24)*W(bank_pins)>=W(n)*W(bank_size)) {
                condensed=false;routed=false;auto_reason="dense_pin_fraction";
            } else if(bank_size<=128 && W(nc)==2*W(n)
                    && W(p.edges.size())>=4*W(n)) {
                condensed=false;routed=false;auto_reason="dense_uncompressed_warmup";
            } else if(bank_size>=64) {
                condensed=true;routed=false;auto_reason="scc_tile64";
            } else if(bank_size>=8 && W(2)*W(nc)<=W(n)) {
                condensed=true;routed=false;auto_reason="scc_strong_compression";
            } else {
                condensed=false;routed=false;auto_reason="short_weak_compression";
            }
        }
'''

for path in CPP_FILES:
    text = path.read_text(encoding="utf-8")
    if text.count(OLD_CPP) != 1:
        raise RuntimeError(f"expected one v5 policy block in {path}")
    text = text.replace(OLD_CPP, NEW_CPP)
    if text.count('which==4?"structural-v5":"explicit"') != 1:
        raise RuntimeError(f"expected one v5 identity in {path}")
    text = text.replace(
        'which==4?"structural-v5":"explicit"',
        'which==4?"structural-v6":"explicit"',
    )
    path.write_text(text, encoding="utf-8")

DOC_PATTERN = re.compile(
    r"The default ``auto`` mode uses structural policy v5\..*?"
    r"frozen structural dispatcher\.",
    re.DOTALL,
)
DOC_REPLACEMENT = '''The default ``auto`` mode uses structural policy v6. Banks of
        at most eight requests on graphs above 512 vertices take the exact raw
        path directly. Other base-infeasible graphs stay raw. A request bank also
        stays raw when it averages at least eight pins per request and those pins
        cover at least one original vertex in 24, or when an uncompressed graph
        with at least four original edges per vertex has at most 128 requests.
        Feasible banks of at least 64 requests otherwise use SCC; banks from eight
        requests use SCC earlier when SCCs compress the literal graph by at least
        fourfold. Selection uses immutable graph/request geometry only, never
        answers or measured timings. ``info['selected_mode']`` and
        ``info['auto_policy_reason']`` expose the decision. Explicit modes keep
        their meaning; the automatic policy is a frozen structural dispatcher.'''

OLD_CALL = '''        values = self._module.solve_session_packet(
            packet, {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3, 'auto': 4}[mode], max_bytes)
        output, count, sat, unsat, proof_bytes, visits, arcs, info = values
        info = {**info, 'backend': 'native-exact-binary', 'mode': mode,
'''
NEW_CALL = '''        native_mode = {'raw': 0, 'scc': 1, 'sparse': 2, 'route': 3, 'auto': 4}[mode]
        direct_reason = None
        if mode == 'auto' and len(packet) >= SESSION_PACKET_HEADER.size:
            magic, vertices, _palette, _edges, rows, _pins, _r0, _r1 = (
                SESSION_PACKET_HEADER.unpack_from(packet)
            )
            if magic == SESSION_PACKET_MAGIC and rows <= 8 and vertices > 512:
                native_mode = 0
                direct_reason = 'large_tiny_bank'
        values = self._module.solve_session_packet(packet, native_mode, max_bytes)
        output, count, sat, unsat, proof_bytes, visits, arcs, info = values
        if direct_reason is not None:
            info = {**info, 'auto_policy': 'structural-v6',
                    'selected_mode': 'raw',
                    'auto_policy_reason': direct_reason}
        info = {**info, 'backend': 'native-exact-binary', 'mode': mode,
'''

for path in PY_FILES:
    text = path.read_text(encoding="utf-8")
    text, count = DOC_PATTERN.subn(DOC_REPLACEMENT, text, count=1)
    if count != 1:
        raise RuntimeError(f"expected one v5 policy paragraph in {path}")
    if text.count(OLD_CALL) != 1:
        raise RuntimeError(f"expected one session-packet native call in {path}")
    text = text.replace(OLD_CALL, NEW_CALL)
    path.write_text(text, encoding="utf-8")

exact_test = Path("tests/public/test_exact_api.py")
text = exact_test.read_text(encoding="utf-8")
if text.count("'structural-v5'") != 1:
    raise RuntimeError("exact API policy identity differs")
exact_test.write_text(text.replace("'structural-v5'", "'structural-v6'"), encoding="utf-8")

readme = Path("experiments/compiled_query/README.md")
text = readme.read_text(encoding="utf-8")
text = text.replace("structural policy v5", "structural policy v6")
readme.write_text(text, encoding="utf-8")

benchmark = Path("experiments/compiled_query/auto_selector_benchmark.py")
text = benchmark.read_text(encoding="utf-8")
for old, new in (
    ('SCHEMA = "spectra.auto_selector_benchmark.v3"',
     'SCHEMA = "spectra.auto_selector_benchmark.v4"'),
    ('POLICY = "structural-v5"', 'POLICY = "structural-v6"'),
    ('SPECTRA-auto-selector-v5-confirmation',
     'SPECTRA-auto-selector-v6-confirmation'),
    ('SPECTRA-auto-selector-v5-schedule',
     'SPECTRA-auto-selector-v6-schedule'),
):
    if text.count(old) != 1:
        raise RuntimeError(f"benchmark identity differs: {old}")
    text = text.replace(old, new)
old_expected = '''    if not base:
        return "raw", "base_infeasible"
    if pins >= 8 * count and 24 * pins >= vertices * count:
'''
new_expected = '''    if count <= 8 and vertices > 512:
        return "raw", "large_tiny_bank"
    if not base:
        return "raw", "base_infeasible"
    if pins >= 8 * count and 24 * pins >= vertices * count:
'''
if text.count(old_expected) != 1:
    raise RuntimeError("benchmark v5 policy function differs")
benchmark.write_text(text.replace(old_expected, new_expected), encoding="utf-8")

wap = Path("experiments/compiled_query/auto_selector_wap_guard.py")
text = wap.read_text(encoding="utf-8")
for old, new in (
    ('SCHEMA = "spectra.auto_selector_wap_guard.v3"',
     'SCHEMA = "spectra.auto_selector_wap_guard.v4"'),
    ('POLICY = "structural-v5"', 'POLICY = "structural-v6"'),
    ('SPECTRA-auto-wap-v5/', 'SPECTRA-auto-wap-v6/'),
):
    if text.count(old) != 1:
        raise RuntimeError(f"WAP guard identity differs: {old}")
    text = text.replace(old, new)
wap.write_text(text, encoding="utf-8")

transfer = Path("experiments/compiled_query/itc_exam_transfer.py")
text = transfer.read_text(encoding="utf-8")
text = text.replace(
    '("explicit", "structural-v5")',
    '("explicit", "structural-v6")',
)
text = text.replace(
    'current["auto_policy"] != "structural-v5"',
    'current["auto_policy"] != "structural-v6"',
)
text = text.replace(
    '("explicit", "structural-v2", "structural-v3", "structural-v4", "structural-v5")',
    '("explicit", "structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6")',
)
text = text.replace(
    '("structural-v2", "structural-v3", "structural-v4", "structural-v5")',
    '("structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6")',
)
if "structural-v6" not in text:
    raise RuntimeError("ITC transfer policy did not advance")
transfer.write_text(text, encoding="utf-8")

suite = Path("experiments/compiled_query/itc_exam_suite.py")
text = suite.read_text(encoding="utf-8")
text = text.replace("structural-v5 selector", "structural-v6 selector")
text = text.replace('"selector_policy": "structural-v5"', '"selector_policy": "structural-v6"')
text = text.replace(
    '("structural-v2", "structural-v3", "structural-v4", "structural-v5")',
    '("structural-v2", "structural-v3", "structural-v4", "structural-v5", "structural-v6")',
)
if '"selector_policy": "structural-v6"' not in text:
    raise RuntimeError("ITC suite policy did not advance")
suite.write_text(text, encoding="utf-8")

transfer_test = Path("experiments/compiled_query/test_itc_exam_transfer.py")
text = transfer_test.read_text(encoding="utf-8")
if text.count('== "structural-v5"') != 1:
    raise RuntimeError("ITC transfer test policy identity differs")
transfer_test.write_text(text.replace('== "structural-v5"', '== "structural-v6"'), encoding="utf-8")

v5_test = Path("tests/public/test_auto_selector_v5.py")
if v5_test.exists():
    v5_test.unlink()

if CPP_FILES[0].read_bytes() != CPP_FILES[1].read_bytes():
    raise RuntimeError("packaged and research exact sources differ")
