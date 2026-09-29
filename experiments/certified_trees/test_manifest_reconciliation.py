"""Bundle metadata, source pairing and output audit contracts; synthetic fixtures."""
import hashlib
import io
import json
from pathlib import Path
import struct

import pytest
from .manifest import verify_manifest, source_labels
from .packed import compile_bytes


def write_manifest(root, models):
    files = {p.relative_to(root).as_posix(): {'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in root.rglob('*') if p.is_file() and p.name != 'SDK_MANIFEST.json'}
    doc = {'format': 'spectra.tree.sdk.v2', 'files': files, 'models': models}
    (root / 'SDK_MANIFEST.json').write_text(json.dumps(doc))
    return doc


@pytest.fixture
def kit(tmp_path):
    root = tmp_path / 'kit'; folder = root / 'models/toy'; folder.mkdir(parents=True)
    doc = {'model_info': {'class_params': {'class_names': [10, 20], 'class_to_label': [0, 1]}},
           'features_info': {'float_features': [{'feature_index': 0, 'flat_feature_index': 0}]},
           'scale_and_bias': [1.0, [0.0, 0.0]], 'oblivious_trees': [{
               'splits': [{'split_type': 'FloatFeature', 'float_feature_index': 0, 'border': 0.5}],
               'leaf_values': [0.0, 1.0, 1.0, 0.0]}]}
    raw = json.dumps(doc).encode(); (folder / 'model.json').write_bytes(raw)
    (folder / 'model.cbm').write_bytes(b'synthetic manifest fixture; never executed')
    for bits in (8, 16):
        packed, _ = compile_bytes(raw, 1, features=1, bits=bits)
        (folder / f'model-{bits}.sct').write_bytes(packed)
    (folder / 'input.u8').write_bytes(bytes([0, 1]))
    (folder / 'indices.i32').write_bytes(struct.pack('<2i', 1, 0))
    binding = {'status': 'PASS', 'classes': [10, 20],
               'json_sha256': hashlib.sha256(raw).hexdigest(),
               'cbm_sha256': hashlib.sha256((folder / 'model.cbm').read_bytes()).hexdigest()}
    (folder / 'source_binding.json').write_text(json.dumps(binding))
    for path in ('native/portable/trees.so', 'native/avx2/trees.so',
                 'official/libcatboostmodel-linux-x86_64-1.2.10.so'):
        file = root / path; file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'inert synthetic manifest fixture, not native code')
    models = {'toy': {'rows': 2, 'features': 1, 'maximum': 1, 'classes': [10, 20],
                      'certified_8': 2, 'certified_16': 2, 'certified_refined': 2}}
    write_manifest(root, models)
    return root


def test_complete_metadata_fixture(kit):
    assert verify_manifest(kit)['models']['toy']['classes'] == [10, 20]


@pytest.mark.parametrize('change', ['reversed_labels', 'bool_count', 'bool_size', 'missing_asset',
                                  'binding', 'wrong_domain', 'wrong_features', 'extra_file',
                                  'unknown_index', 'bad_input', 'unsafe_model_name', 'digest',
                                  'duplicate_key', 'nonfinite', 'symlink', 'empty_models'])
def test_malformed_bundle_rejected(kit, change):
    path = kit / 'SDK_MANIFEST.json'; m = json.loads(path.read_text()); e = m['models']['toy']
    if change == 'reversed_labels': e['classes'] = [20, 10]
    elif change == 'bool_count': e['certified_8'] = True
    elif change == 'bool_size': m['files']['models/toy/input.u8']['bytes'] = True
    elif change == 'wrong_domain': e['maximum'] = 2
    elif change == 'wrong_features': e['features'] = 2
    elif change == 'unsafe_model_name': m['models']['../toy'] = m['models'].pop('toy')
    elif change == 'empty_models': m['models'] = {}
    elif change == 'digest': m['files']['models/toy/model.json']['sha256'] = 'z' * 64
    elif change == 'duplicate_key':
        path.write_text(path.read_text().replace('"models":', '"models":{},"models":', 1))
    elif change == 'nonfinite': path.write_text(path.read_text().replace('"rows": 2', '"rows": 1e999'))
    elif change == 'extra_file': (kit / 'unlisted').write_text('unexpected')
    elif change == 'symlink': (kit / 'extra-link').symlink_to('models/toy/model.json')
    else:
        if change == 'missing_asset': (kit / 'models/toy/model.cbm').unlink()
        elif change == 'binding':
            b = kit / 'models/toy/source_binding.json'; r = json.loads(b.read_text())
            r['cbm_sha256'] = '0' * 64; b.write_text(json.dumps(r))
        elif change == 'unknown_index': (kit / 'models/toy/indices.i32').write_bytes(struct.pack('<2i', 1, 2))
        elif change == 'bad_input': (kit / 'models/toy/input.u8').write_bytes(bytes([0, 2]))
        m = write_manifest(kit, m['models'])  # Rehash changes; semantic rejection must still work.
    if change not in ('duplicate_key', 'nonfinite'):
        path.write_text(json.dumps(m))
    with pytest.raises(ValueError): verify_manifest(kit)


@pytest.mark.parametrize('mapping', [[1, 0], [True, False], [0], None])
def test_unsupported_original_class_order(mapping):
    with pytest.raises(ValueError):
        source_labels({'model_info': {'class_params': {'class_names': [7, 99], 'class_to_label': mapping}}})




def test_certificate_acceptance_is_not_assumed_nested(tmp_path):
    from array import array
    from .test_native import source, verified
    from .session import TreeSession
    from .build import build
    import os
    library = Path(os.environ['TREE_LIBRARY']) if 'TREE_LIBRARY' in os.environ else build(tmp_path / 'native')
    raw = source([[.2519531, .2519532], [100., 100.], [-100., -100.]], bias=[-.251950])
    first, _ = verified(raw, 8, D=1)
    second, _ = verified(raw, 16, D=1)
    with TreeSession(library, first=first) as a, TreeSession(library, first=second) as b, TreeSession(library, first=first, second=second) as both:
        assert a.predict_buffer(array('B', [0, 1])) == [1, 1]
        assert b.predict_buffer(array('B', [0, 1])) == [-1, 1]
        result = both.inspect_buffer(array('B', [0, 1]), refine=True)
        assert result['indices'] == [1, 1]
        assert result['work']['unresolved_rows'] == 0
    # Source margins are positive in both cases. No incorrect certificate is
    # involved: independent outward interval enclosures need not be nested.
