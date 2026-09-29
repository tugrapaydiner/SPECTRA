"""Separate standard-library audit: splits, selection, labels, timings and identity.

Reads only JSON/NPY data, never unpickles a model or imports the learner/runtime.
Hashes bind recorded bytes; they do not authenticate clocks or prove absence of
all human benchmark adaptation. The official evaluation partitions are reused.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,math,random,statistics,struct,zipfile
from pathlib import Path
TASKS=('letter','pendigits');SEEDS=(611,977,1543);LEARNERS=('uniform','variance','nca')
ARMS=('uniform_compiled','variance_compiled','nca_compiled','nca_scalar_integer','nca_exhaustive','nca_scalar_exp','linear_native','mlp_native','uniform_sklearn','nca_sklearn','historical_finite')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unique(pairs):
 r={}
 for k,v in pairs:
  if k in r:raise ValueError('duplicate JSON key')
  r[k]=v
 return r

def reject(x):raise ValueError('nonfinite JSON token')
def read(p):return json.loads(Path(p).read_text(),object_pairs_hook=unique,parse_constant=reject)
def decode(text):return json.loads(text,object_pairs_hook=unique,parse_constant=reject)

def arrays(path):
 result={}
 with zipfile.ZipFile(path) as z:
  if len(z.namelist())!=len(set(z.namelist())):raise ValueError('duplicate numpy member')
  for name in z.namelist():
   if not name.endswith('.npy') or '/' in name or z.getinfo(name).file_size>100_000_000:raise ValueError('invalid array member')
   data=z.read(name)
   if data[:6]!=b'\x93NUMPY':raise ValueError('numpy magic')
   major=data[6];size=2 if major==1 else 4;offset=8+size
   length=int.from_bytes(data[8:offset],'little');header=ast.literal_eval(data[offset:offset+length].decode('latin1'))
   if header['fortran_order']:raise ValueError('Fortran array not accepted')
   shape=header['shape'];count=math.prod(shape)
   formats={'<i8':'q','<u8':'Q','|u1':'B','|b1':'?','<u2':'H','<f8':'d','<i4':'i','<f4':'f'}
   if header['descr'] not in formats:raise ValueError('unsupported receipt array dtype')
   payload=data[offset+length:];fmt=formats[header['descr']]
   if len(payload)!=count*struct.calcsize('<'+fmt):raise ValueError('array inventory')
   result[name[:-4]]={'shape':shape,'values':list(struct.unpack('<'+str(count)+fmt,payload)),'bytes':payload}
 return result

def rows(a):
 n,d=a['shape'];return [tuple(a['values'][i*d:(i+1)*d]) for i in range(n)]
def outcome(pred,y):
 if len(pred)!=len(y):raise ValueError('prediction count')
 labels=sorted(set(y));index={v:i for i,v in enumerate(labels)};matrix=[[0]*len(labels) for _ in labels]
 for t,p in zip(y,pred):matrix[index[t]][index[p]]+=1
 correct=sum(a==b for a,b in zip(pred,y));f1=[]
 for i in range(len(labels)):
  denominator=sum(matrix[i])+sum(r[i] for r in matrix)
  f1.append(2*matrix[i][i]/denominator if denominator else 0.)
 return {'rows':len(y),'correct':correct,'accuracy':correct/len(y),'macro_f1':statistics.mean(f1),'confusion':matrix}

def compare(actual,expected):
 if isinstance(expected,dict):
  for k,v in expected.items():compare(actual[k],v)
 elif isinstance(expected,float):
  if type(actual) not in (int,float) or not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('numeric summary differs')
 elif actual!=expected or type(actual)!=type(expected):raise ValueError('record differs')

def audit(root,source):
 root=Path(root);source=Path(source);selection=root/'selection';models=root/'models';evaluation=root/'evaluation';bench=root/'benchmark';data=root/'datasets'
 manifest=read(data/'manifest.json');original=read(models/'FINAL_LOCK.json');opening=read(evaluation/'EVALUATION_OPENING.json')
 if opening['final_lock_sha256']!=sha(models/'FINAL_LOCK.json'):raise ValueError('evaluation not bound to final model lock')
 for name,h in original['files'].items():
  if sha(models/name)!=h:raise ValueError('frozen model changed')
 for name,h in original['source'].items():
  if sha(source/'experiments/learned_metric'/name)!=h:raise ValueError('training source changed')
 for name,h in opening['data'].items():
  if sha(data/name)!=h:raise ValueError('data identity changed')
 selected=read(selection/'selected.json');records=[decode(l) for l in (selection/'selection.jsonl').read_text().splitlines()]
 expectedkeys={(t,s,a,c,g) for t in TASKS for s in SEEDS for a in LEARNERS for c in (1.,10.,100.) for g in (.5,2.,8.)}
 seen=set();allpreds={};ground={}
 for task in TASKS:
  train=arrays(data/(task+'-train.npz'));qtrain=rows(train['q']);ytrain=train['y']['values']
  for seed in SEEDS:
   split=arrays(selection/f'{task}-{seed}-split.npz');fit=split['fit']['values'];val=split['validation']['values']
   if set(fit)&set(val) or {qtrain[i] for i in fit}&{qtrain[i] for i in val}:raise ValueError('training/validation leakage')
   for r in records:
    if r['task']!=task or r['seed']!=seed:continue
    key=task,seed,r['arm'],r['C'],r['gamma']
    if key in seen or key not in expectedkeys:raise ValueError('selection inventory')
    seen.add(key);tag=f"{task}-{seed}-{r['arm']}-C{r['C']:g}-g{r['gamma']:g}"
    pred=arrays(selection/(tag+'-predictions.npz'));y=[ytrain[i] for i in val]
    if pred['expected']['values']!=y:raise ValueError('validation labels differ')
    score=outcome(pred['prediction']['values'],y)
    compare(r,{'correct':score['correct'],'accuracy':score['accuracy'],'fit_rows':len(fit),'validation_rows':len(val)})
  for arm in LEARNERS:
   choices=[]
   for j,(c,g) in enumerate(( (c,g) for c in (1.,10.,100.) for g in (.5,2.,8.))):
    rr=[r for r in records if r['task']==task and r['arm']==arm and r['C']==c and r['gamma']==g]
    if len(rr)!=3:raise ValueError('missing tuning cell')
    choices.append((sum(r['correct'] for r in rr)/sum(r['validation_rows'] for r in rr),-sum(r['support_vectors'] for r in rr),-j,c,g))
   best=max(choices);compare(selected[task][arm],{'C':best[3],'gamma':best[4],'validation_accuracy':best[0]})
  test=arrays(data/(task+'-test.npz'));y=test['y']['values'];qtest=rows(test['q']);ground[task]=y;training_keys=set(qtrain)
  mask=[r not in training_keys for r in qtest];report=read(evaluation/'quality.json')[task]
  for arm in ('uniform','variance','nca','linear','mlp'):
   arr=arrays(evaluation/task/(arm+'-predictions.npz'));pred=arr['prediction']['values'];allpreds[task,arm]=pred
   if arr['expected']['values']!=y or arr['nonoverlap']['values']!=mask:raise ValueError('evaluation label/group identity')
   compare(report[arm],outcome(pred,y));compare(report[arm]['nonoverlap'],outcome([v for v,m in zip(pred,mask) if m],[v for v,m in zip(y,mask) if m]))
  wins=sum(a==t and b!=t for a,b,t in zip(allpreds[task,'nca'],allpreds[task,'uniform'],y));loss=sum(a!=t and b==t for a,b,t in zip(allpreds[task,'nca'],allpreds[task,'uniform'],y))
  n=wins+loss;k=min(wins,loss);pv=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
  compare(report['difference'],{'difference_points':100*(wins-loss)/len(y),'candidate_only_correct':wins,'baseline_only_correct':loss,'exact_discordance_pvalue':pv})
 if seen!=expectedkeys:raise ValueError('incomplete model-selection grid')
 protocol=read(bench/'protocol.json');expected=(list(ARMS),[1,32,256],7,2026092806)
 if (protocol['arms'],protocol['chunks'],protocol['repeats'],protocol['seed'])!=expected:raise ValueError('benchmark protocol changed')
 for name,h in protocol['source'].items():
  if sha(source/name)!=h:raise ValueError('timed source differs')
 roots={'models':models,'data':data,'evaluation':evaluation,'controls':root/'controls'}
 for name,h in protocol['files'].items():
  prefix,suffix=name.split('/',1)
  if sha(roots[prefix]/suffix)!=h:raise ValueError('timed artifact changed')
 observed=[decode(l) for l in (bench/'rows.jsonl').read_text().splitlines()];schedule=[];rng=random.Random(2026092806)
 for task in TASKS:
  for repeat in range(7):
   order=[(c,a) for c in (1,32,256) for a in ARMS];rng.shuffle(order)
   for chunk,arm in order:schedule.append((task,repeat,chunk,arm))
 if len(schedule)!=len(observed):raise ValueError('timing inventory missing')
 for r,key in zip(observed,schedule):
  if tuple(r[n] for n in ('task','repeat','chunk','arm'))!=key or type(r['ns']) is not int or r['ns']<=0:raise ValueError('timing identity/order')
  task,repeat,chunk,arm=key
  pred=read(root/'historical'/task/'expected.json') if arm=='historical_finite' else allpreds[task,arm.split('_')[0]]
  packed=struct.pack('<'+str(len(pred))+'q',*pred)
  if r['prediction_sha256']!=hashlib.sha256(packed).hexdigest() or r['rows']!=len(pred):raise ValueError('timed prediction mismatch')
 summary=read(bench/'summary.json');decisions={}
 for task in TASKS:
  costs={str(c):{a:statistics.median(r['ns'] for r in observed if r['task']==task and r['chunk']==c and r['arm']==a) for a in ARMS} for c in (1,32,256)}
  ratio=costs['32']['nca_compiled']/costs['32']['uniform_compiled'];rr=summary['tasks'][task];compare(rr['batch_median_ns'],costs)
  gain=read(evaluation/'quality.json')[task]['difference']['difference_points'];gate=gain>=.5 and ratio<=1.25
  compare(rr,{'nca_over_uniform':ratio,'quality_gain_points':gain,'joint_gate':gate})
  compare(rr['batch32_us_per_row'],{a:v/len(ground[task])/1000 for a,v in costs['32'].items()})
  decisions[task]={'accuracy_gain_points':gain,'cost_ratio':ratio,'joint_gate':gate}
 return {'status':'PASS','selection_cells':len(records),'timing_cells':len(observed),'model_input_pairs':sum(len(v) for v in ground.values()),
  'checked_timing_predictions':sum(r['rows'] for r in observed),'tasks':decisions,
  'scope':'independent split, byte identity, outcome and arithmetic audit; bootstrap intervals and clock authenticity are not independently certified'}

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('root','source','out'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();result=audit(a.root,a.source)
 with a.out.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result,indent=2))
