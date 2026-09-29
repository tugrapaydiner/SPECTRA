"""Real source-pair checks and independent output corruption tests.

Toy fitted models isolate packaging correctness, not benchmark quality.
"""
import hashlib
import io
import json
from pathlib import Path
import pytest
from .bundle import verify_fallback_pair
from .deployment import stream_predict
from .verify_output import verify_output


@pytest.fixture(scope='module')
def model_pair(tmp_path_factory):
    import numpy as np
    from catboost import CatBoostClassifier
    folder = tmp_path_factory.mktemp('cbm-source-binding')
    x = np.array([[i % 8, (i*3) % 7] for i in range(96)], dtype=np.uint8)
    y = (x[:,0] % 3).astype(int)
    m = CatBoostClassifier(iterations=6, depth=3, loss_function='MultiClass',
                          thread_count=1, random_seed=37, bootstrap_type='No',
                          allow_writing_files=False, verbose=False).fit(x,y)
    m.save_model(str(folder/'model.cbm'))
    m.save_model(str(folder/'model.json'), format='json')
    leaves = m.get_leaf_values()
    leaves[1] += 1.0
    m.set_leaf_values(leaves)
    m.save_model(str(folder/'changed.cbm'))
    return folder


def test_cbm_source_all_tree_binding(model_pair):
    r = verify_fallback_pair(model_pair/'model.json', model_pair/'model.cbm',
                            features=2, maximum=7, labels=[0,1,2])
    assert r['status'] == 'PASS' and r['trees'] == 6 and r['leaf_scalars'] > 0


def test_same_shaped_wrong_fallback_rejected(model_pair):
    with pytest.raises(ValueError, match='numerical structure'):
        verify_fallback_pair(model_pair/'model.json', model_pair/'changed.cbm',
                             features=2, maximum=7, labels=[0,1,2])


@pytest.mark.parametrize('labels', [[2,1,0], [False,1,2], ['0','1','2'], [0,1]])
def test_wrong_original_label_mapping_rejected(model_pair, labels):
    with pytest.raises(ValueError, match='class mapping'):
        verify_fallback_pair(model_pair/'model.json', model_pair/'model.cbm',
                             features=2, maximum=7, labels=labels)


def test_metadata_reordering_is_not_a_changed_function(model_pair, tmp_path):
    p = tmp_path/'reformatted.json'
    doc = json.loads((model_pair/'model.json').read_text())
    doc['model_info']['irrelevant_metadata'] = 'formatting is not a numerical parameter'
    p.write_text(json.dumps(doc, sort_keys=True, indent=1))
    assert verify_fallback_pair(p, model_pair/'model.cbm', features=2, maximum=7,
                                labels=[0,1,2])['status'] == 'PASS'


class ToyEngine:
    features, maximum, classes = 2, 3, 2
    def inspect_buffer(self, data, *, refine, fallback):
        values = [int(data[i] > data[i+1]) for i in range(0,len(data),2)]
        return {'indices':values, 'work':{'certified_first':len(values),
                'certified_second':0, 'official_rows':0, 'unresolved_rows':0}}


def produce(tmp_path):
    raw = b'[0,1]\n[3,0]\n'
    path = tmp_path/'out.jsonl'
    stream_predict(ToyEngine(), [7,99], io.BytesIO(raw), path,
                   identity={'source_sha256':'a'*64})
    kwargs = {'expected_indices':[0,1], 'labels':[7,99],
              'input_sha256':hashlib.sha256(raw).hexdigest(),
              'identity':{'source_sha256':'a'*64}, 'features':2, 'maximum':3,
              'compact_only':False, 'refine':False}
    return path, kwargs


def test_independent_file_reconstruction(tmp_path):
    p, kwargs = produce(tmp_path)
    r = verify_output(p, **kwargs)
    assert r['rows'] == 2 and r['unresolved_rows'] == 0
    assert r['output_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['label','index','bool-index','bool-row','skip',
    'order','missing-footer','trailing','input-digest','identity','prefix','counter',
    'bool-counter','duplicate','newline','nonfinite'])
def test_independent_output_rejects_corruptions(tmp_path, change):
    p, kwargs = produce(tmp_path)
    lines = p.read_bytes().splitlines(keepends=True)
    docs = [json.loads(x) for x in lines]
    if change=='label': docs[1]['label'] = 99
    elif change=='index': docs[1]['class_index'] = 1
    elif change=='bool-index': docs[1]['class_index'] = False
    elif change=='bool-row': docs[1]['row'] = False
    elif change=='skip': docs.pop(1)
    elif change=='order': docs[1],docs[2] = docs[2],docs[1]
    elif change=='missing-footer': docs.pop()
    elif change=='input-digest': docs[-1]['input_sha256'] = 'b'*64
    elif change=='identity': docs[0]['identity']['source_sha256'] = 'b'*64
    elif change=='prefix': docs[-1]['prefix_sha256'] = '0'*64
    elif change=='counter': docs[-1]['certified_rows'] = 1
    elif change=='bool-counter': docs[-1]['official_rows'] = False
    if change in ('trailing','duplicate','newline','nonfinite'):
        raw = b''.join(lines)
        if change=='trailing': raw += b'{}\n'
        elif change=='duplicate': raw = raw.replace(b'"row":0',b'"row":8,"row":0',1)
        elif change=='newline': raw = raw[:-1]
        else: raw = raw.replace(b'"row":0',b'"row":NaN',1)
    else:
        # Refresh the hash to isolate structural/outcome validation from ordinary
        # byte corruption; a self-consistent forged prefix must still be refused.
        prefix = b''.join((json.dumps(x,separators=(',',':'))+'\n').encode() for x in docs[:-1])
        if docs and docs[-1].get('type')=='complete' and change!='prefix':
            docs[-1]['prefix_sha256'] = hashlib.sha256(prefix).hexdigest()
        raw = b''.join((json.dumps(x,separators=(',',':'))+'\n').encode() for x in docs)
    p.write_bytes(raw)
    with pytest.raises(ValueError): verify_output(p, **kwargs)


def test_explicit_unresolved_is_not_a_false_label(tmp_path):
    p, kwargs = produce(tmp_path)
    docs = [json.loads(x) for x in p.read_bytes().splitlines()]
    docs[0]['compact_only'] = True
    docs[1].update(class_index=None, label=None, status='UNRESOLVED')
    docs[-1].update(certified_rows=1, unresolved_rows=1)
    prefix = b''.join((json.dumps(x)+'\n').encode() for x in docs[:-1])
    docs[-1]['prefix_sha256'] = hashlib.sha256(prefix).hexdigest()
    p.write_bytes(prefix + (json.dumps(docs[-1])+'\n').encode())
    kwargs['compact_only'] = True
    assert verify_output(p, **kwargs)['unresolved_rows'] == 1
    kwargs['compact_only'] = False
    with pytest.raises(ValueError): verify_output(p, **kwargs)


@pytest.mark.parametrize('field', ['certified_first','certified_second','official_rows','unresolved_rows'])
def test_boolean_native_counter_is_rejected_before_publishing(tmp_path,field):
    class Broken(ToyEngine):
        def inspect_buffer(self, *a, **kw):
            r = super().inspect_buffer(*a, **kw)
            r['work'][field] = bool(r['work'][field])
            return r
    p = tmp_path/'out'
    with pytest.raises(ValueError, match='work counters'):
        stream_predict(Broken(),[7,99],io.BytesIO(b'[0,1]\n'),p,identity={})
    assert not p.exists()


def test_failed_temporary_cleanup_does_not_hide_published_success(tmp_path,monkeypatch):
    unlink = Path.unlink
    def fail_on_partial(self, *a, **kw):
        if self.suffix == '.partial': raise OSError('simulated cleanup failure')
        return unlink(self, *a, **kw)
    monkeypatch.setattr(Path, 'unlink', fail_on_partial)
    p, kwargs = produce(tmp_path)
    assert verify_output(p, **kwargs)['rows'] == 2
