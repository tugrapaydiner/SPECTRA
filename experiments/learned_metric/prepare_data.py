"""Extract the previously acquired official partitions without reselecting rows."""
import argparse,hashlib,json,zipfile,io
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--archives',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
records={}
for task,D in [('letter',15),('pendigits',100)]:
 src=a.archives/(task+'.zip')
 with zipfile.ZipFile(src) as z:
  if task=='letter':
   raw=z.read('letter-recognition.data');lines=raw.decode().splitlines()
   q=np.array([[int(v) for v in l.split(',')[1:]] for l in lines],dtype=np.uint8)
   y=np.array([ord(l[0])-ord('A') for l in lines],dtype=np.int64)
   partitions={'train':(q[:16000],y[:16000]),'test':(q[16000:],y[16000:])}
  else:
   partitions={}
   for name,f in [('train','pendigits.tra'),('test','pendigits.tes')]:
    m=np.loadtxt(io.BytesIO(z.read(f)),delimiter=',',dtype=np.int64)
    partitions[name]=(m[:,:-1].astype(np.uint8),m[:,-1])
 for part,(q,y) in partitions.items():
  if (q>D).any() or q.shape[1]!=16:raise ValueError('schema mismatch')
  out=a.out/(task+'-'+part+'.npz');np.savez_compressed(out,q=q,y=y,maximum=D)
  records[out.name]={'rows':len(q),'features':16,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 records[task+'-archive']={'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'bytes':src.stat().st_size}
(a.out/'manifest.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
