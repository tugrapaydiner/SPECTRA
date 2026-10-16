"""Build current and pinned prior native controls with identical compiler flags."""
from pathlib import Path
import argparse
import subprocess

from . import exact
from .cases import digest
from .transfer_panel import PRIOR_SOURCE_SHA256


def build(root: Path):
    root.mkdir(parents=True, exist_ok=False)
    current = exact.build_exact_runtime(root/'current')
    prior_source = root/'prior_source.cpp'
    prior_source.write_bytes(exact.SOURCE.read_bytes())
    patch = Path(__file__).with_name('TRANSFER_PRIOR_NATIVE.patch')
    subprocess.run(['patch','--batch','--forward',str(prior_source),str(patch.resolve())],check=True)
    if digest(prior_source.read_bytes()) != PRIOR_SOURCE_SHA256:
        raise ValueError('restored historical source hash differs')
    original = exact.SOURCE
    try:
        exact.SOURCE = prior_source.resolve()
        prior = exact.build_exact_runtime(root/'prior')
    finally:
        exact.SOURCE = original
    print('current:',current)
    print('prior:',prior)
    return current,prior


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    build(parser.parse_args().directory)
