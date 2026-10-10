"""Correct the inherited supervisor RSS floor; preserve original memory rows.

The original benchmark supervisor retained all timing rows, so fork/exec children
inherited a high-water floor. An intermediate fresh interpreter creates the actual
measurement worker from a small process. This reruns memory only on the consumed
evaluation cases; it is not another independent quality/timing evaluation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--dependencies', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    helper = 'import subprocess,sys; r=subprocess.run(sys.argv[1:],check=True,capture_output=True,text=True); print(r.stdout,end="")'
    corpus = [json.loads(line) for line in (a.evaluation/'cases.jsonl').read_text().splitlines()]
    runner = Path(__file__).with_name('bench.py').resolve()
    temporary = a.out.with_suffix('.input.json')
    with a.out.open('x') as output:
        try:
            for case in corpus:
                temporary.write_text(json.dumps(case))
                for arm in ('indexed', 'deductive', 'glucose4'):
                    worker = [sys.executable, '-I', '-S', str(runner), '--dependencies', str(a.dependencies.resolve()),
                              '--rss-case', str(temporary.resolve()), '--arm', arm]
                    result = subprocess.run([sys.executable, '-I', '-S', '-c', helper, *worker],
                                            check=True, capture_output=True, text=True, timeout=30)
                    record = {'id': case['id'], 'arm': arm, **json.loads(result.stdout)}
                    output.write(json.dumps(record)+'\n')
                    output.flush()
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
