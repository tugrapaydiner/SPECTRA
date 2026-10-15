"""Build and exercise the proof-carrying exact API from an installed wheel.

Run with the installation venv's Python and ``-I`` so repository packages cannot
satisfy imports. Native compilation is explicit and uses the host compiler.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix='spectra-exact-installed-') as temporary:
        root = Path(temporary)
        os.chdir(root)
        import spectra.exact as exact

        module_path = Path(exact.__file__).resolve()
        source_path = exact.SOURCE.resolve()
        prefix = Path(sys.prefix).resolve()
        require(module_path.is_relative_to(prefix), 'exact API must come from installed wheel')
        require(source_path.is_relative_to(prefix) and source_path.is_file(),
                'installed wheel must include exact native source')
        for dependency in ('torch', 'numpy', 'pytest', 'pysat'):
            require(importlib.util.find_spec(dependency) is None,
                    'unexpected optional dependency: ' + dependency)
        require(importlib.util.find_spec('experiments') is None,
                'installed contract must not import repository experiments')

        library = exact.build_exact_runtime(root / 'native')
        runtime = exact.ExactRuntime(library)
        problem = exact.ExactProblem.create(2, 2, ((0, 1),), (3, 3))
        queries = ((), ((0, 1),), ((0, 1), (1, 1)))
        packet = exact.pack_session_packet(problem, queries)
        result = runtime.solve_session_packet(packet, mode='auto')
        require((result.query_count, result.sat_count, result.unsat_count) == (3, 2, 1),
                'unexpected installed exact decisions')
        require(exact.verify_session_output(packet, result.output),
                'independent installed replay rejected output')
        rows = exact.decode_certified_output(result.output, problem.n, len(queries))
        require(tuple(code for code, _, _ in rows) == (1, 1, 0),
                'unexpected installed SCQ2 rows')
        receipt = json.loads(library.with_name('build.json').read_text())
        require(receipt['exact_api_version'] == exact.EXACT_API_VERSION == 1,
                'exact API version differs')
        require(receipt['native_abi'] == exact.EXACT_NATIVE_ABI == 3,
                'native ABI differs')
        report = {
            'status': 'PASS',
            'scope': 'actual installed wheel; isolated Python; explicit native build',
            'module': str(module_path),
            'source': str(source_path),
            'exact_api_version': exact.EXACT_API_VERSION,
            'native_abi': exact.EXACT_NATIVE_ABI,
            'queries': result.query_count,
            'sat': result.sat_count,
            'unsat': result.unsat_count,
            'selected_mode': result.info['selected_mode'],
            'optional_dependencies': False,
            'receipt_source_sha256': receipt['source_sha256'],
            'receipt_library_sha256': receipt['library_sha256'],
        }
        text = json.dumps(report, indent=2, sort_keys=True) + '\n'
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding='utf-8')
        print(text, end='')


if __name__ == '__main__':
    main()
