from __future__ import annotations

from pathlib import Path
import re


REPLACEMENT = '''        if(mode==4) {
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

SELECTOR = re.compile(
    r'        if\(mode==4\) \{\n.*?        \}\n'
    r'        if\(condensed && bad==NIL\) \{',
    re.DOTALL,
)


def patch_native(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    updated, count = SELECTOR.subn(
        REPLACEMENT + '        if(condensed && bad==NIL) {', text, count=1
    )
    if count != 1:
        raise RuntimeError(f'expected one selector block in {path}, got {count}')
    updated = updated.replace(
        'which==4?"structural-v2":"explicit"',
        'which==4?"structural-v3":"explicit"',
    )
    if updated.count('structural-v3') != 1:
        raise RuntimeError(f'policy identity differs in {path}')
    path.write_text(updated, encoding='utf-8')


def patch_python_doc(path: Path) -> None:
    old = '''The default ``auto`` mode uses structural policy v2. It selects raw traversal below
        64 requests, on a base-infeasible graph, or when the immutable bank carries
        at least eight pins per request on average. Otherwise it uses routed proofs
        when SCCs at least halve the literal graph, and plain SCC batching when they
        do not.'''
    new = '''The default ``auto`` mode uses structural policy v3. It keeps base-infeasible,
        high-pin, and short weakly compressed banks on raw traversal. It selects SCC
        for every feasible low-pin bank of at least 64 requests, and from eight
        requests when SCCs compress the literal graph by at least fourfold.'''
    text = path.read_text(encoding='utf-8')
    if old not in text:
        raise RuntimeError(f'auto-policy documentation missing from {path}')
    path.write_text(text.replace(old, new), encoding='utf-8')


def main() -> None:
    for name in (
        'spectra/_native/exact_binary.cpp',
        'experiments/compiled_query/exact_native.cpp',
    ):
        patch_native(Path(name))

    for name in ('spectra/exact.py', 'experiments/compiled_query/exact.py'):
        patch_python_doc(Path(name))

    test = Path('tests/public/test_exact_api.py')
    text = test.read_text(encoding='utf-8')
    text = text.replace("'structural-v2'", "'structural-v3'")
    text = text.replace("'bank_below_tile64'", "'short_weak_compression'")
    test.write_text(text, encoding='utf-8')

    readme = Path('experiments/compiled_query/README.md')
    text = readme.read_text(encoding='utf-8')
    old = '''It chooses raw for fewer than 64 requests or an
impossible fixed graph, then route or SCC according to graph compression.'''
    new = '''It keeps base-infeasible, high-pin, and short weakly compressed banks on raw;
it chooses SCC for feasible low-pin banks at 64 requests, or from eight requests
when literal contraction is at least fourfold.'''
    if old not in text:
        raise RuntimeError('README selector description differs')
    readme.write_text(text.replace(old, new), encoding='utf-8')


if __name__ == '__main__':
    main()
