"""Replay failures must remain failures under isolated, optimized Python."""
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from .test_total import library


@pytest.fixture
def replay_inputs(tmp_path, library):
    # Real tiny source models and the real total runtime; the exported-C++ control
    # is a Python stand-in in the subprocess, solely to test orchestration failures.
    sdk = tmp_path / 'sdk'
    for task, features in (('letter', 16), ('pendigits', 16), ('satellite', 36), ('optdigits', 64)):
        folder = sdk / 'models' / task
        folder.mkdir(parents=True)
        model = {'features_info': {'float_features': [
            {'feature_index': i, 'flat_feature_index': i, 'borders': []} for i in range(features)]},
            'scale_and_bias': [1., [0.]], 'oblivious_trees': [{
                'splits': [{'split_type': 'FloatFeature', 'float_feature_index': 0, 'border': .5}],
                'leaf_values': [-1., 1.]}]}
        (folder / 'model.json').write_text(json.dumps(model))
        (folder / 'input.u8').write_bytes(bytes(features) + bytes([1] * features))
        (folder / 'indices.i32').write_bytes(struct.pack('<ii', 0, 1))
    return sdk, Path(library).resolve()


CHILD = '''
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from experiments.tree_total import replay
a = SimpleNamespace(sdk=Path(sys.argv[2]), library=Path(sys.argv[3]),
                    exports=Path(sys.argv[4]), out=Path(sys.argv[5]))
fault = sys.argv[6]
class ExportControl:
    def __init__(self, folder):
        task = folder.parent.name
        self.source = replay.ScalarSource((a.sdk / 'models' / task / 'model.json').read_bytes(),
                                          replay.DOMAINS[task][1])
    def scores(self, data):
        d = self.source.width
        return self.source.scores([data[i:i+d] for i in range(0, len(data), d)])
    def predict_buffer(self, data):
        return self.source.indices(self.scores(data))
replay.ExportSession = ExportControl
if fault in ('scores', 'stress'):
    original = replay.TotalSession.scores
    def corrupted(self, data, **kwargs):
        raw = bytearray(original(self, data, **kwargs))
        if fault == 'scores' or len(data) > 2 * self.features:
            raw[0] ^= 1
        return bytes(raw)
    replay.TotalSession.scores = corrupted
if fault == 'short':
    original = replay.TotalSession.inspect_buffer
    def shortened(self, data, **kwargs):
        result = original(self, data, **kwargs)
        if len(data) == 2 * self.features:
            result['indices'] = result['indices'][:1]
        return result
    replay.TotalSession.inspect_buffer = shortened
replay.run(a)
'''


def invoke(tmp_path, inputs, optimized, fault):
    sdk, library = inputs
    source = Path(__file__).resolve().parents[2]
    destination = tmp_path / 'replay-output'
    if fault == 'expected':
        (sdk / 'models/letter/indices.i32').write_bytes(struct.pack('<ii', 1, 1))
    command = [sys.executable, '-I', '-S'] + (['-O'] if optimized else [])
    result = subprocess.run(command + ['-c', CHILD, str(source), str(sdk), str(library),
                                      str(tmp_path / 'exports'), str(destination), fault],
                            cwd=tmp_path, capture_output=True, text=True)
    return result, destination


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('fault', ['expected', 'scores', 'short', 'stress'])
def test_inconsistent_replay_cannot_publish_success(tmp_path, replay_inputs, optimized, fault):
    result, destination = invoke(tmp_path, replay_inputs, optimized, fault)
    assert result.returncode != 0, result.stdout
    assert not (destination / 'report.json').exists()


@pytest.mark.parametrize('optimized', [False, True])
def test_valid_synthetic_replay_remains_available(tmp_path, replay_inputs, optimized):
    result, destination = invoke(tmp_path, replay_inputs, optimized, 'none')
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / 'report.json').read_text())
    assert report['status'] == 'PASS' and report['retained_rows'] == 8
    assert report['synthetic_rows'] == 16384
