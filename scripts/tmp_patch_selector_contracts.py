from __future__ import annotations

from pathlib import Path
import textwrap


path = Path("experiments/compiled_query/test_exact.py")
text = path.read_text(encoding="utf-8")

text = text.replace("structural-v2", "structural-v6")
text = text.replace("high_pin_density", "dense_pin_fraction")
text = text.replace(
    "test_auto_structural_v6_freezes_the_eight_pin_density_boundary",
    "test_auto_structural_v6_freezes_the_pin_density_boundary",
)

high_reason = "    assert high.info['auto_policy_reason'] == 'dense_pin_fraction'"
if high_reason in text and "    assert high.info['auto_policy'] == 'structural-v6'\n" + high_reason not in text:
    text = text.replace(
        high_reason,
        "    assert high.info['auto_policy'] == 'structural-v6'\n" + high_reason,
        1,
    )

start_marker = (
    "@pytest.mark.parametrize('count', (0, 1, 63, 64, 65, 129))\n"
    "@pytest.mark.parametrize('structure', ('empty', 'independent', 'chain', 'odd_cycle'))\n"
    "def test_auto_packet_exposes_and_matches_its_selected_engine"
)
start = text.index(start_marker)
end = text.index(
    "\n\ndef test_auto_does_not_hide_malformed_packets_or_resource_limits",
    start,
)
replacement = textwrap.dedent(
    '''
    @pytest.mark.parametrize('count', (0, 1, 63, 64, 65, 129))
    @pytest.mark.parametrize('structure', ('empty', 'independent', 'chain', 'odd_cycle'))
    def test_auto_packet_exposes_and_matches_its_selected_engine(exact_runtime, count, structure):
        n = 0 if structure == 'empty' else 9
        edges = () if structure in ('empty', 'independent') else tuple(
            (v, v + 1) for v in range(n - 1)
        )
        if structure == 'odd_cycle':
            edges += ((0, n - 1),)
        p = BinaryListProblem.create(n, 2, edges, (3,) * n)
        requests = ((),) if not n else (
            (), ((0, 1),), ((0, 1), (1, 1)), ((0, 0),)
        )
        rows = tuple(requests[i % len(requests)] for i in range(count))
        packet = pack_session_packet(p, rows)
        answer = exact_runtime.solve_session_packet(packet, mode='auto')
        pins = sum(map(len, rows))

        if structure == 'odd_cycle':
            selected, reason = 'raw', 'base_infeasible'
        elif pins >= 8 * count and 24 * pins >= n * count:
            selected, reason = 'raw', 'dense_pin_fraction'
        else:
            components = {'empty': 0, 'independent': 18, 'chain': 2}[structure]
            if count <= 128 and components == 2 * n and len(edges) >= 4 * n:
                selected, reason = 'raw', 'dense_uncompressed_warmup'
            elif count >= 64:
                selected, reason = 'scc', 'scc_tile64'
            elif count >= 8 and 2 * components <= n:
                selected, reason = 'scc', 'scc_strong_compression'
            else:
                selected, reason = 'raw', 'short_weak_compression'

        assert answer.info['mode'] == 'auto'
        assert answer.info['auto_policy'] == 'structural-v6'
        assert answer.info['auto_policy_reason'] == reason
        assert answer.info['bank_query_count'] == count
        assert answer.info['selected_mode'] == selected
        assert answer.output == exact_runtime.solve_session_packet(
            packet, mode=selected
        ).output
        for query, (decision, labels, certificate) in zip(
                rows, decode_certified_output(answer.output, n, count)):
            assert bool(decision) == bool(brute(p, query))
            assert verify_answer(
                p,
                query,
                'SAT_VERIFIED' if decision else 'UNSAT_VERIFIED',
                labels if decision else b'',
                certificate,
            )
    '''
).lstrip()
text = text[:start] + replacement.rstrip() + text[end:]

if "structural-v2" in text or "high_pin_density" in text:
    raise RuntimeError("stale selector contract remains")

path.write_text(text, encoding="utf-8")
