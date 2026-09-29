"""Bind exported numeric source to its actual CBM; diagnostic only, no fitting.

Every leaf value and source split border is checked, plus every retained row's
leaf routing. Unguarded quantization is diagnostic, never a served classifier.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import catboost

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(a):
    records=[]
    for task in ('letter','pendigits','satellite','optdigits'):
        folder=a.models/task;fit=json.loads((folder/'FIT.json').read_text());m=catboost.CatBoostClassifier();m.load_model(str(folder/'model.cbm'))
        source=json.loads((folder/'model.json').read_text())
        leaf=np.asarray([x for tr in source['oblivious_trees'] for x in tr['leaf_values']],dtype=np.float64)
        assert leaf.tobytes()==m.get_leaf_values().tobytes()
        assert [2**len(tr.get('splits') or []) for tr in source['oblivious_trees']]==m.get_tree_leaf_counts().tolist()
        assert list(m.get_scale_and_bias())==source['scale_and_bias']
        borders={f['flat_feature_index']:f['borders'] for f in source['features_info']['float_features']}
        assert borders==m.get_borders()
        q=np.fromfile(a.evaluation/task/'input.u8',dtype=np.uint8).reshape(-1,fit['features'])
        expected=np.fromfile(a.evaluation/task/'indices.i32',dtype='<i4')
        native_leaves=m.calc_leaf_indexes(q,thread_count=1);wrong={}
        for bits in (8,16):
            doc=json.loads((a.compiled/f'{task}-{bits}.oracle.json').read_text());s=np.repeat(np.asarray(doc['bias'],dtype=np.int32)[None,:],len(q),axis=0)
            for k,tr in enumerate(doc['trees']):
                indices=np.zeros(len(q),dtype=np.int32)
                for b,(f,t) in enumerate(zip(tr['features'],tr['thresholds'])):indices|=(q[:,f].astype(np.int32)>t).astype(np.int32)<<b
                assert np.array_equal(indices,native_leaves[:,k])
                s+=np.asarray(tr['leaves'],dtype=np.int32)[indices]
            raw=s.argmax(1);cert=np.fromfile(a.replay/task/f'{bits}-tiled-0.indices',dtype='<i4')
            assert np.all((cert<0)|(cert==expected))
            wrong[str(bits)]={'unguarded_source_disagreements':int(np.count_nonzero(raw!=expected)),
                              'rejected_unguarded_disagreements':int(np.count_nonzero((raw!=expected)&(cert<0))),
                              'certified':int(np.count_nonzero(cert>=0))}
        records.append({'task':task,'json_sha256':sha(folder/'model.json'),'cbm_sha256':sha(folder/'model.cbm'),
          'leaf_values_bit_identical':int(leaf.size),'borders_equal':True,'scale_bias_equal':True,
          'routes_equal':int(native_leaves.size),'rows':len(q),'quantization':wrong})
    report={'status':'PASS','catboost':catboost.__version__,'source_sha256':sha(__file__),'models':records,
            'scope':'same frozen CBM/export and all retained routes; no new models or universal backend semantics'}
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('models','compiled','evaluation','replay','out'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
