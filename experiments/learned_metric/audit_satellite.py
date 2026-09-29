"""Independent recorded-data/selection/outcome/timing audit for Satellite."""
from __future__ import annotations
import argparse,hashlib,json,math,random,statistics,struct,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.learned_metric.audit import arrays,rows,read,decode,sha,compare,outcome
ARMS=('uniform_compiled','variance_compiled','nca_compiled','nca_scalar_integer','nca_scalar_exp','nca_exhaustive','linear_native','mlp_native','qda_native','uniform_sklearn','nca_sklearn')

def audit(root,source):
 root=Path(root);models=root/'models';selection=root/'selection';evaluation=root/'evaluation';data=root/'data';source=Path(source)
 manifest=read(data/'manifest.json')
 if sha(root/'acquisition/satellite.zip')!=manifest['archive_sha256']:raise ValueError('source dataset changed')
 with zipfile.ZipFile(root/'acquisition/satellite.zip') as archive:
  for name,file in [('train','sat.trn'),('test','sat.tst')]:
   raw=archive.read(file);parsed=[[int(v) for v in line.split()] for line in raw.decode().splitlines()]
   if hashlib.sha256(raw).hexdigest()!=manifest['files'][name]['raw_sha256']:raise ValueError('original rows changed')
   if sha(data/(name+'.npz'))!=manifest['files'][name]['npz_sha256']:raise ValueError('parsed data changed')
   stored=arrays(data/(name+'.npz'))
   if rows(stored['q'])!=[tuple(r[:-1]) for r in parsed] or stored['y']['values']!=[r[-1] for r in parsed]:raise ValueError('parsed labels differ from official data')
 train=arrays(data/'train.npz');test=arrays(data/'test.npz');qtrain=rows(train['q']);ytrain=train['y']['values'];ytest=test['y']['values']
 records=[decode(s) for s in (selection/'selection.jsonl').read_text().splitlines()];seen=set()
 for seed in (611,977,1543):
  split=arrays(selection/f'split-{seed}.npz');f=split['fit']['values'];v=split['validation']['values']
  if set(f)&set(v) or {qtrain[i] for i in f}&{qtrain[i] for i in v}:raise ValueError('split leakage')
  for r in records:
   if r['seed']!=seed:continue
   key=(seed,r['arm'],r['C'],r['gamma'])
   if key in seen:raise ValueError('duplicate tuning cell')
   seen.add(key);p=arrays(selection/f"{seed}-{r['arm']}-C{r['C']:g}-g{r['gamma']:g}.npz")
   correct_y=[ytrain[i] for i in v]
   if p['expected']['values']!=correct_y:raise ValueError('validation labels changed')
   compare(r,{'correct':sum(a==b for a,b in zip(p['prediction']['values'],correct_y)),'validation_rows':len(v),'fit_rows':len(f)})
 expected={(s,a,c,g) for s in (611,977,1543) for a in ('uniform','variance','nca') for c in (1.,10.,100.) for g in (2.,8.,32.)}
 if seen!=expected:raise ValueError('incomplete model search')
 chosen=read(selection/'selected.json')
 for arm in ('uniform','variance','nca'):
  options=[]
  for i,(c,g) in enumerate(( (c,g) for c in (1.,10.,100.) for g in (2.,8.,32.))):
   rr=[r for r in records if r['arm']==arm and r['C']==c and r['gamma']==g]
   options.append((sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr),-sum(r['support_vectors'] for r in rr),-i,c,g))
  best=max(options);compare(chosen[arm],{'C':best[3],'gamma':best[4],'validation_accuracy':best[0]})
 lock=read(models/'FINAL_LOCK.json')
 for name,h in lock['files'].items():
  if sha(models/name)!=h:raise ValueError('final model changed')
 if sha(source/'experiments/learned_metric/satellite.py')!=lock['script_sha256'] or sha(source/'experiments/learned_metric/learning.py')!=lock['learning_sha256']:raise ValueError('training source changed')
 opening=read(evaluation/'TEST_OPENING.json')
 if opening['final_lock_sha256']!=sha(models/'FINAL_LOCK.json'):raise ValueError('wrong evaluation model lock')
 quality=read(evaluation/'quality.json');preds={};keys=set(qtrain);mask=[r not in keys for r in rows(test['q'])]
 for arm in ('uniform','variance','nca','linear','mlp','qda'):
  p=arrays(evaluation/(arm+'-predictions.npz'));pred=p['prediction']['values'];preds[arm]=pred
  if p['expected']['values']!=ytest or p['nonoverlap']['values']!=mask:raise ValueError('evaluation identity')
  compare(quality[arm],outcome(pred,ytest));compare(quality[arm]['nonoverlap'],outcome([x for x,m in zip(pred,mask) if m],[x for x,m in zip(ytest,mask) if m]))
 wins=sum(a==y and b!=y for a,b,y in zip(preds['nca'],preds['uniform'],ytest));loss=sum(a!=y and b==y for a,b,y in zip(preds['nca'],preds['uniform'],ytest));n=wins+loss;k=min(wins,loss)
 pval=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
 compare(quality['difference'],{'difference_points':100*(wins-loss)/len(ytest),'candidate_only_correct':wins,'baseline_only_correct':loss,'exact_discordance_pvalue':pval})
 protocol=read(evaluation/'timing_protocol.json')
 if protocol['arms']!=list(ARMS) or protocol['chunks']!=[1,32,256] or protocol['repeats']!=7 or protocol['seed']!=2026092811 or protocol['source_sha256']!=sha(source/'experiments/learned_metric/satellite_evaluate.py'):raise ValueError('timing protocol changed')
 timings=[decode(s) for s in (evaluation/'timings.jsonl').read_text().splitlines()];expected=[];rng=random.Random(2026092811)
 for rep in range(7):
  jobs=[(c,a) for c in (1,32,256) for a in ARMS];rng.shuffle(jobs)
  expected.extend((rep,c,a) for c,a in jobs)
 if len(timings)!=len(expected):raise ValueError('missing timed row')
 for r,key in zip(timings,expected):
  if (r['repeat'],r['chunk'],r['arm'])!=key or type(r['ns']) is not int or r['ns']<=0:raise ValueError('invalid timing row')
  pred=preds[r['arm'].split('_')[0]];digest=hashlib.sha256(struct.pack('<'+str(len(pred))+'q',*pred)).hexdigest()
  if r['prediction_sha256']!=digest or r['rows']!=len(ytest):raise ValueError('timed output changed')
 med={str(c):{a:statistics.median(r['ns'] for r in timings if r['chunk']==c and r['arm']==a)/len(ytest)/1000 for a in ARMS} for c in (1,32,256)}
 ratio=med['32']['nca_compiled']/med['32']['uniform_compiled'];gain=100*(wins-loss)/len(ytest);summary=read(evaluation/'timing_summary.json')
 compare(summary,{'cells':len(timings),'us_per_row':med,'nca_over_uniform':ratio,'accuracy_gain_points':gain,'joint_gate':gain>=.5 and ratio<=1.25})
 return {'status':'PASS','selection_cells':len(records),'timing_cells':len(timings),'underlying_test_rows':len(ytest),'accuracy_gain_points':gain,'joint_gate':summary['joint_gate'],'scope':'data/selection/outcome/recorded arithmetic; no spatial-independence or clock-authentication claim'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('root','source','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();result=audit(a.root,a.source)
 with a.out.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
