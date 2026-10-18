from __future__ import annotations

from pathlib import Path

benchmark = Path("experiments/compiled_query/auto_selector_benchmark.py")
text = benchmark.read_text(encoding="utf-8")

replacements = {
    'SCHEMA = "spectra.auto_selector_benchmark.v2"':
        'SCHEMA = "spectra.auto_selector_benchmark.v3"',
    'POLICY = "structural-v4"': 'POLICY = "structural-v5"',
    'namespace: str,\n) -> tuple[BinaryListProblem, list[int], random.Random]:':
        'namespace: str,\n    minimum_edge_factor: float = 0.0,\n) -> tuple[BinaryListProblem, list[int], random.Random]:',
    '    palette = 8\n    if strong_contraction:':
        '''    if minimum_edge_factor:
        target = math.ceil(minimum_edge_factor * n)
        while len(edges) < target:
            a = left[rng.randrange(len(left))]
            b = right[rng.randrange(len(right))]
            edges.add((min(a, b), max(a, b)))

    palette = 8
    if strong_contraction:''',
    '    namespace: str,\n) -> Case:':
        '    namespace: str,\n    minimum_edge_factor: float = 0.0,\n) -> Case:',
    '        namespace=namespace,\n    )\n    rows = _queries(':
        '        namespace=namespace,\n        minimum_edge_factor=minimum_edge_factor,\n    )\n    rows = _queries(',
    '    namespace = "SPECTRA-auto-selector-v4-confirmation"':
        '    namespace = "SPECTRA-auto-selector-v5-confirmation"',
    '''    regimes = (
        ("strong-sparse", 320, 2, True, False),
        ("below-density", 512, 14, False, False),
        ("above-density", 384, 14, False, False),
        ("low-density-large", 1536, 28, False, False),
        ("base-infeasible", 512, 4, False, True),
    )
    for family in FAMILIES:
        for regime, n, width, strong, infeasible in regimes:
''':
        '''    regimes = (
        ("strong-sparse", 320, 2, True, False, 0.0),
        ("below-density", 640, 25, False, False, 0.0),
        ("above-density", 448, 19, False, False, 0.0),
        ("edge-heavy-warmup", 768, 12, False, False, 5.0),
        ("low-density-large", 1280, 24, False, False, 0.0),
        ("base-infeasible", 512, 4, False, True, 0.0),
    )
    for family in FAMILIES:
        for regime, n, width, strong, infeasible, edge_factor in regimes:
''',
    '                namespace=namespace,\n            )\n            result.append(PanelCase(case, family, regime, CONFIRMATION_COUNTS))':
        '                namespace=namespace,\n                minimum_edge_factor=edge_factor,\n            )\n            result.append(PanelCase(case, family, regime, CONFIRMATION_COUNTS))',
    '''def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    components: int) -> tuple[str, str]:
    if not base:
        return "raw", "base_infeasible"
    if pins >= 8 * count and 32 * pins >= vertices * count:
        return "raw", "dense_pin_fraction"
    if count >= 64:
        return "scc", "scc_tile64"
    if count >= 8 and 2 * components <= vertices:
        return "scc", "scc_strong_compression"
    return "raw", "short_weak_compression"
''':
        '''def expected_policy(*, base: bool, pins: int, count: int, vertices: int,
                    edges: int, components: int) -> tuple[str, str]:
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
''',
    '                    vertices=problem.n,\n                    components=int(info["components"]),':
        '                    vertices=problem.n,\n                    edges=len(problem.edges),\n                    components=int(info["components"]),',
    '"SPECTRA-auto-selector-v4-schedule",':
        '"SPECTRA-auto-selector-v5-schedule",',
}

for old, new in replacements.items():
    if text.count(old) != 1:
        raise RuntimeError(f"benchmark replacement differs: {old[:90]!r}")
    text = text.replace(old, new)
benchmark.write_text(text, encoding="utf-8")

wap = Path("experiments/compiled_query/auto_selector_wap_guard.py")
text = wap.read_text(encoding="utf-8")
for old, new in (
    ('SCHEMA = "spectra.auto_selector_wap_guard.v2"',
     'SCHEMA = "spectra.auto_selector_wap_guard.v3"'),
    ('POLICY = "structural-v4"', 'POLICY = "structural-v5"'),
    ('SPECTRA-auto-wap-v4/', 'SPECTRA-auto-wap-v5/'),
):
    if text.count(old) != 1:
        raise RuntimeError(f"WAP guard replacement differs: {old}")
    text = text.replace(old, new)
wap.write_text(text, encoding="utf-8")
