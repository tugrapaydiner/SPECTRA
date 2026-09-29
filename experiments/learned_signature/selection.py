"""Choose fixed final configurations using only the three recorded validation splits."""
from __future__ import annotations
import argparse,hashlib,json,statistics
from pathlib import Path
SEEDS=(1401,2402,3403)
FAMILIES=('rbf','metric','alignment','uniform','metric_alignment','linear','mlp')

def select(root,out):
 result={'format':'spectra.learned-signature.selection.v1','scope':'training/validation only; final evaluation historically consumed but unopened in this continuation','tasks':{},'sources':{}}
 for task in ('letter','pendigits'):
  trials=[]
  for seed in SEEDS:
   path=root/f'{task}-{seed}'/'rows.jsonl';result['sources'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
   rows=[json.loads(x) for x in path.read_text().splitlines()]
   if len(rows)!=109:raise ValueError('incomplete development grid')
   trials.extend(rows)
  families={}
  for family in FAMILIES:
   configs={}
   for r in trials:
    if r['family']!=family:continue
    k=(r.get('C'),r.get('multiplier'));configs.setdefault(k,[]).append(r)
   values=[]
   for (C,mult),rs in configs.items():
    if sorted(r['seed'] for r in rs)!=list(SEEDS):raise ValueError('missing/duplicate split')
    values.append({'C':C,'multiplier':mult,'mean_accuracy':statistics.mean(r['accuracy'] for r in rs),'mean_supports':statistics.mean(r.get('supports',0) for r in rs),'grid_index':rs[0].get('grid_index',0),'split_accuracy':{str(r['seed']):r['accuracy'] for r in rs}})
   winner=max(values,key=lambda v:(v['mean_accuracy'],-v['mean_supports'],-v['grid_index']))
   families[family]={'chosen':winner,'all_configurations':values,'total_fit_cpu':sum(r['fit_cpu'] for r in trials if r['family']==family),'total_kernel_learning_cpu':sum(r.get('kernel_learning_cpu',0) for r in trials if r['family']==family)}
  ps=[]
  for seed in SEEDS:
   p=root/f'{task}-{seed}-pairs/summary.json'
   if not p.exists():raise ValueError('pair development unfinished')
   r=json.loads(p.read_text());ps.append(r)
   result['sources'][str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
  families['pair_experts']={'chosen':{'mean_accuracy':statistics.mean(p['accuracy'] for p in ps),'split_accuracy':{str(s):p['accuracy'] for s,p in zip(SEEDS,ps)},'mean_supports':statistics.mean(p['model']['supports'] for p in ps)},'total_fit_cpu':sum(p['fit_cpu'] for p in ps),'total_development_cpu':sum(p['cpu_seconds'] for p in ps),'fits':sum(p['fits'] for p in ps)}
  base=families['rbf']['chosen']['mean_accuracy']
  for f,v in families.items():v['validation_gain_pp']=100*(v['chosen']['mean_accuracy']-base)
  result['tasks'][task]=families
 learned=('metric','alignment','metric_alignment')
 result['primary_candidates_passing_validation']=[f for f in learned if all(result['tasks'][task][f]['validation_gain_pp']>0 for task in result['tasks'])]
 result['primary_validation_gate']=bool(result['primary_candidates_passing_validation'])
 with out.open('x') as f:json.dump(result,f,indent=2)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=select(a.root,a.out)
 for task,fs in r['tasks'].items():
  print(task)
  for family,v in fs.items():print(family,v['chosen'],v['validation_gain_pp'])
 print('Primary validation:',r['primary_validation_gate'])
