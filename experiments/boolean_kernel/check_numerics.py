"""Exact-bit replay against separately compiled original source, plus Boolean oracle."""
from array import array
import argparse,ctypes as C,hashlib,itertools,json,random,struct,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from spectra.svm_shared import PreparedModel


def bind(path):
    lib=C.CDLL(str(Path(path).resolve()))
    lib.bk_observe.argtypes=[C.c_void_p,C.c_size_t,C.c_int,C.c_int,C.POINTER(C.c_double),C.c_int,C.c_int,
        C.POINTER(C.c_double),C.POINTER(C.c_double),C.c_size_t,C.POINTER(C.c_double),C.c_size_t]
    return lib


def observe(lib,raw,values,rows,d,nsv,classes,mode,tables):
    data=(C.c_double*len(values)).from_buffer(values);blob=C.create_string_buffer(raw)
    dist=(C.c_double*(rows*nsv))();kernel=(C.c_double*(rows*nsv))();margin=(C.c_double*(rows*classes*(classes-1)//2))()
    assert lib.bk_observe(blob,len(raw),tables,mode,data,rows,d,dist,kernel,len(kernel),margin,len(margin))==0
    return bytes(dist),bytes(kernel),bytes(margin)


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    dst=a.out/(a.model+'.json')
    if dst.exists():raise FileExistsError(dst)
    frozen=json.loads((a.models/'MODEL_FREEZE.json').read_text())['models'][a.model]
    folder=a.models/a.model
    inventory=json.loads((a.models.parent/'SHA256.json').read_text())
    for f in ('model.srt','X.f64','expected.json'):
        expected_hash=inventory[f'models/{a.model}/{f}']['sha256']
        assert hashlib.sha256((folder/f).read_bytes()).hexdigest()==expected_hash
        if f in frozen['files']:assert expected_hash==frozen['files'][f]
    raw=(folder/'model.srt').read_bytes();d=frozen['features'];n=frozen['rows'];nsv=frozen['supports'];c=len(frozen['classes'])
    values=array('d');values.frombytes((folder/'X.f64').read_bytes())
    old,new=bind(a.old),bind(a.new);hashes=[hashlib.sha256() for _ in range(3)]
    started=time.monotonic()
    for first in range(0,n,128):
        count=min(128,n-first);x=values[first*d:(first+count)*d]
        reference=observe(old,raw,x,count,d,nsv,c,0,False)
        for tables in (False,True):
            for mode in (0,1,2):
                result=observe(new,raw,x,count,d,nsv,c,mode,tables)
                if result!=reference:raise AssertionError((a.model,first,mode,tables))
        for h,b in zip(hashes,reference):h.update(b)
    expected=json.loads((folder/'expected.json').read_text())
    enabled=None;stats=[];metadata={}
    for mode in ('off','packed','lookup'):
        with PreparedModel(folder/'model.srt',a.runtime,boolean=mode,input_dtype='float64') as owner,owner.session() as w:
            metadata[mode]={'model':owner.info,'worker':w.info}
            for schedule in ('exhaustive','beretta_cert','binary_stream'):
                assert w.predict_buffer(values,schedule=schedule)==expected
            if mode!='off':enabled=owner.info['boolean_enabled']
            if mode=='lookup' and enabled:
                for row in range(n):
                    w.predict(values[row*d:(row+1)*d],schedule='exhaustive');stats.append(w.boolean_stats)
    record={'status':'PASS','model':a.model,'rows':n,'distance_values':n*nsv,'kernel_values':n*nsv,
        'margin_values':n*c*(c-1)//2,'native_variants':6,'boolean_eligible':enabled,
        'hashes':dict(zip(('distances','kernels','margins'),(h.hexdigest() for h in hashes))),
        'elapsed_seconds':time.monotonic()-started,'metadata':metadata,'per_input_work':stats,
        'scope':'actual bitwise equality on this host; not exact-real exponentials or independent researcher replication'}
    dst.write_text(json.dumps(record,indent=2));print(a.model,record['status'],round(record['elapsed_seconds'],2),enabled)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ('models','old','new','runtime','out'):p.add_argument('--'+field,type=Path,required=True)
    p.add_argument('--model',required=True);main(p.parse_args())
