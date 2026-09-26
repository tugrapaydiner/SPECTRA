from pathlib import Path
import os
import pytest
from spectra.svm import build_runtime


@pytest.fixture(scope='session')
def library(tmp_path_factory):
    value = os.environ.get('SPECTRA_SVM_LIBRARY')
    if value:
        return Path(value).resolve(strict=True)
    return build_runtime(tmp_path_factory.mktemp('native') / 'build')
