from __future__ import annotations

from pathlib import Path


NATIVE_FILES = (
    Path("spectra/_native/exact_binary.cpp"),
    Path("experiments/compiled_query/exact_native.cpp"),
)
PYTHON_FILES = (
    Path("spectra/exact.py"),
    Path("experiments/compiled_query/exact.py"),
)

OLD_DENSE = '''            } else if(W(bank_pins)>=8*W(bank_size)
                    && W(24)*W(bank_pins)>=W(n)*W(bank_size)) {
                condensed=false;routed=false;auto_reason="dense_pin_fraction";
            } else if(bank_size<=128 && W(nc)==2*W(n)
'''
NEW_DENSE = '''            } else if(W(bank_pins)>=8*W(bank_size)
                    && W(24)*W(bank_pins)>=W(n)*W(bank_size)) {
                if(bank_size>=64 && n>=128) {
                    condensed=true;routed=false;auto_reason="dense_long_tile64";
                } else {
                    condensed=false;routed=false;auto_reason="dense_pin_fraction";
                }
            } else if(bank_size<=128 && W(nc)==2*W(n)
'''

for path in NATIVE_FILES:
    text = path.read_text(encoding="utf-8")
    text = text.replace("Structural auto policy v6", "Structural auto policy v7")
    text = text.replace('"structural-v6"', '"structural-v7"')
    if text.count(OLD_DENSE) != 1:
        raise RuntimeError(f"dense v6 block differs in {path}")
    text = text.replace(OLD_DENSE, NEW_DENSE)
    path.write_text(text, encoding="utf-8")

for path in PYTHON_FILES:
    text = path.read_text(encoding="utf-8")
    text = text.replace("structural policy v6", "structural policy v7")
    text = text.replace(
        "A request bank also\n        stays raw when it averages at least eight pins per request and those pins\n        cover at least one original vertex in 24, or when an uncompressed graph\n        with at least four original edges per vertex has at most 128 requests.\n        Feasible banks of at least 64 requests otherwise use SCC; banks from eight\n",
        "A dense request bank averages at least eight pins per request and covers\n        at least one original vertex in 24. Dense banks of at least 64 requests\n        use SCC when the graph has at least 128 vertices; smaller dense graphs\n        stay raw. An uncompressed graph with at least four original edges per\n        vertex and at most 128 requests also stays raw. Other feasible banks of\n        at least 64 requests use SCC; banks from eight\n",
    )
    text = text.replace("'auto_policy': 'structural-v6'", "'auto_policy': 'structural-v7'")
    path.write_text(text, encoding="utf-8")

research_test = Path("experiments/compiled_query/test_exact.py")
text = research_test.read_text(encoding="utf-8").replace("structural-v6", "structural-v7")
marker = "\n\ndef test_auto_does_not_hide_malformed_packets_or_resource_limits"
if marker not in text:
    raise RuntimeError("research selector test insertion point differs")
new_test = '''


def test_auto_structural_v7_dense_long_boundary(exact_runtime):
    def answer(n):
        problem = BinaryListProblem.create(n, 2, (), (3,) * n)
        rows = tuple(tuple((v, 3) for v in range(8)) for _ in range(64))
        return exact_runtime.solve_session_packet(
            pack_session_packet(problem, rows), mode='auto'
        )

    below = answer(127)
    boundary = answer(128)
    assert below.info['selected_mode'] == 'raw'
    assert below.info['auto_policy_reason'] == 'dense_pin_fraction'
    assert boundary.info['selected_mode'] == 'scc'
    assert boundary.info['auto_policy_reason'] == 'dense_long_tile64'
'''
if "test_auto_structural_v7_dense_long_boundary" not in text:
    text = text.replace(marker, new_test + marker)
research_test.write_text(text, encoding="utf-8")

public_exact = Path("tests/public/test_exact_api.py")
public_exact.write_text(
    public_exact.read_text(encoding="utf-8").replace("structural-v6", "structural-v7"),
    encoding="utf-8",
)

old_public = Path("tests/public/test_auto_selector_v6.py")
if old_public.exists():
    old_public.unlink()
