from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path, old, new, count=1):
    path = ROOT / path
    text = path.read_text()
    if text.count(old) != count:
        raise RuntimeError(f"{path}: expected {count} occurrences, found {text.count(old)}")
    path.write_text(text.replace(old, new))


old_cpp = '''        if(mode==4) {
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
new_cpp = '''        if(mode==4) {
            // Structural auto policy v7. Short, contradictory, and small
            // high-pin banks stay raw. Long feasible banks use the tile64 SCC
            // engine, except for the measured 64-request dense warmup boundary.
            if(bank_size<=8 && n>512) {
                condensed=false;routed=false;auto_reason="large_tiny_bank";
            } else if(bad!=NIL) {
                condensed=false;routed=false;auto_reason="base_infeasible";
            } else if(W(bank_pins)>=8*W(bank_size)
                    && W(24)*W(bank_pins)>=W(n)*W(bank_size)
                    && (bank_size<64 || n<256)) {
                condensed=false;routed=false;
                auto_reason=n<256?"dense_small_high_pin":"dense_pin_short";
            } else if(bank_size==64 && W(nc)==2*W(n)
                    && W(p.edges.size())>=4*W(n)) {
                condensed=false;routed=false;auto_reason="dense_uncompressed_tile64";
            } else if(bank_size>=64) {
                condensed=true;routed=false;auto_reason="scc_long_bank";
            } else if(bank_size>=8 && W(2)*W(nc)<=W(n)) {
                condensed=true;routed=false;auto_reason="scc_strong_compression";
            } else {
                condensed=false;routed=false;auto_reason="short_weak_compression";
            }
        }
'''
for path in ("spectra/_native/exact_binary.cpp", "experiments/compiled_query/exact_native.cpp"):
    replace(path, old_cpp, new_cpp)

old_doc = '''        The default ``auto`` mode uses structural policy v6. Banks of
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
        their meaning; the automatic policy is a frozen structural dispatcher.
'''
new_doc = '''        The default ``auto`` mode uses structural policy v7. Banks of
        at most eight requests on graphs above 512 vertices take the exact raw
        path directly, and fixed contradictions stay raw. Dense high-pin work
        stays raw before 64 requests and at every length below 256 vertices.
        At exactly 64 requests, dense uncompressed graphs also remain raw.
        Other feasible banks of at least 64 requests use SCC; banks from eight
        requests use SCC earlier when SCCs compress the literal graph by at least
        fourfold. Selection uses immutable graph/request geometry only, never
        answers or measured timings. ``info['selected_mode']`` and
        ``info['auto_policy_reason']`` expose the decision. Explicit modes keep
        their meaning; the automatic policy is a frozen structural dispatcher.
'''
for path in ("spectra/exact.py", "experiments/compiled_query/exact.py"):
    replace(path, old_doc, new_doc)
    replace(path, "'auto_policy': 'structural-v6'", "'auto_policy': 'structural-v7'")

bench = ROOT / "experiments/compiled_query/auto_selector_benchmark.py"
text = bench.read_text()
text = text.replace("structural auto policy v4", "structural auto policy v7")
text = text.replace('POLICY = "structural-v6"', 'POLICY = "structural-v7"')
text = text.replace('name = f"auto-v4-', 'name = f"auto-v7-')
text = text.replace('namespace = "SPECTRA-auto-selector-v6-confirmation"',
                    'namespace = "SPECTRA-auto-selector-v7-confirmation"')
text = text.replace('"SPECTRA-auto-selector-v6-schedule"',
                    '"SPECTRA-auto-selector-v7-schedule"')
old_regimes = '''    regimes = (
        ("strong-sparse", 320, 2, True, False, 0.0),
        ("below-density", 640, 25, False, False, 0.0),
        ("above-density", 448, 19, False, False, 0.0),
        ("edge-heavy-warmup", 768, 12, False, False, 5.0),
        ("low-density-large", 1280, 24, False, False, 0.0),
        ("base-infeasible", 512, 4, False, True, 0.0),
    )
'''
new_regimes = '''    regimes = (
        ("strong-sparse", 352, 2, True, False, 0.0),
        ("below-density", 704, 27, False, False, 0.0),
        ("above-density", 480, 21, False, False, 0.0),
        ("edge-heavy-warmup", 832, 13, False, False, 5.0),
        ("low-density-large", 1408, 26, False, False, 0.0),
        ("small-high-pin", 224, 12, False, False, 0.0),
        ("base-infeasible", 576, 4, False, True, 0.0),
    )
'''
if text.count(old_regimes) != 1:
    raise RuntimeError("confirmation regimes differ")
text = text.replace(old_regimes, new_regimes)
old_policy = '''def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    edges: int, components: int) -> tuple[str, str]:
    if count <= 8 and vertices > 512:
        return "raw", "large_tiny_bank"
    if not base:
        return "raw", "base_infeasible"
    if pins >= 8 * count and 24 * pins >= vertices * count:
        return "raw", "dense_pin_fraction"
    if count <= 128 and components == 2 * vertices and edges >= 4 * vertices:
        return "raw", "dense_uncompressed_warmup"
    if count >= 64:
        return "scc", "scc_tile64"
    if count >= 8 and 2 * components <= vertices:
        return "scc", "scc_strong_compression"
    return "raw", "short_weak_compression"
'''
new_policy = '''def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    edges: int, components: int) -> tuple[str, str]:
    if count <= 8 and vertices > 512:
        return "raw", "large_tiny_bank"
    if not base:
        return "raw", "base_infeasible"
    if (pins >= 8 * count and 24 * pins >= vertices * count
            and (count < 64 or vertices < 256)):
        return "raw", "dense_small_high_pin" if vertices < 256 else "dense_pin_short"
    if count == 64 and components == 2 * vertices and edges >= 4 * vertices:
        return "raw", "dense_uncompressed_tile64"
    if count >= 64:
        return "scc", "scc_long_bank"
    if count >= 8 and 2 * components <= vertices:
        return "scc", "scc_strong_compression"
    return "raw", "short_weak_compression"
'''
if text.count(old_policy) != 1:
    raise RuntimeError("expected policy block differs")
bench.write_text(text.replace(old_policy, new_policy))

# Update test and application policy identities.
for path in (
    "experiments/compiled_query/test_exact.py",
    "experiments/compiled_query/itc_exam_suite.py",
    "experiments/compiled_query/itc_exam_transfer.py",
    "experiments/compiled_query/test_itc_exam_transfer.py",
):
    p = ROOT / path
    data = p.read_text().replace("structural-v6", "structural-v7")
    p.write_text(data)

old_test = ROOT / "tests/public/test_auto_selector_v6.py"
new_test = ROOT / "tests/public/test_auto_selector_v7.py"
test = old_test.read_text().replace("selector-v6-runtime", "selector-v7-runtime")
test = test.replace('"structural-v6"', '"structural-v7"')
test = test.replace('"scc_tile64"', '"scc_long_bank"')
test = test.replace(
'''def test_pin_density_at_one_in_24_uses_raw(exact_runtime):
    problem = independent_problem(192)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_pin_fraction"
''',
'''def test_small_high_pin_bank_stays_raw_even_when_long(exact_runtime):
    problem = independent_problem(192)
    row = tuple((vertex, 3) for vertex in range(8))
    answer = solve(exact_runtime, problem, (row,) * 1024)
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_small_high_pin"


def test_large_high_pin_bank_uses_scc_at_tile64(exact_runtime):
    problem = independent_problem(448)
    row = tuple((vertex, 3) for vertex in range(19))
    answer = solve(exact_runtime, problem, (row,) * 64)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"
''')
test = test.replace(
'''def test_dense_uncompressed_graph_uses_raw_through_128_requests(exact_runtime):
    problem = dense_uncompressed_problem()
    for count in (64, 128):
        answer = solve(exact_runtime, problem, ((),) * count)
        assert answer.info["components"] == 2 * problem.n
        assert answer.info["selected_mode"] == "raw"
        assert answer.info["auto_policy_reason"] == "dense_uncompressed_warmup"


def test_dense_uncompressed_graph_uses_scc_after_warmup_boundary(exact_runtime):
    problem = dense_uncompressed_problem()
    answer = solve(exact_runtime, problem, ((),) * 129)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"
''',
'''def test_dense_uncompressed_graph_uses_raw_at_tile64(exact_runtime):
    problem = dense_uncompressed_problem()
    answer = solve(exact_runtime, problem, ((),) * 64)
    assert answer.info["components"] == 2 * problem.n
    assert answer.info["selected_mode"] == "raw"
    assert answer.info["auto_policy_reason"] == "dense_uncompressed_tile64"


def test_dense_uncompressed_graph_uses_scc_after_tile64(exact_runtime):
    problem = dense_uncompressed_problem()
    answer = solve(exact_runtime, problem, ((),) * 128)
    assert answer.info["selected_mode"] == "scc"
    assert answer.info["auto_policy_reason"] == "scc_long_bank"
''')
if "structural-v6" in test or "dense_pin_fraction" in test or "dense_uncompressed_warmup" in test:
    raise RuntimeError("stale v6 selector expectation remains in public test")
new_test.write_text(test)
old_test.unlink()

# The research mirror and packaged implementation must remain byte-identical.
if (ROOT / "spectra/_native/exact_binary.cpp").read_bytes() != (ROOT / "experiments/compiled_query/exact_native.cpp").read_bytes():
    raise RuntimeError("native exact source copies differ")
