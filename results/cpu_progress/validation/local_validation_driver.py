from pathlib import Path
import subprocess,os,json,time,sys
root=Path('/mnt/data/SPECTRA_upstream_3ec');out=Path('/mnt/data/integrated_validation_a55');out.mkdir(exist_ok=False)
sys.path.insert(0,str(root))
from scripts.m16_cpu_experiment import source_identity
source=source_identity()
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MAX_JOBS='1',CUDA_VISIBLE_DEVICES='')
commands=[
('pytest_fast',[sys.executable,'-m','pytest','-m','not slow','-ra',f'--junitxml={out}/pytest_fast.xml']),
('stored_original',[sys.executable,'scripts/verify_retained_results.py','--out',str(out/'stored_original.json')]),
('checkpoint_original',[sys.executable,'scripts/verify_checkpoint_replay.py','--out',str(out/'checkpoint_original')]),
('ordered_fixed_pool',[sys.executable,'scripts/verify_fixed_pool_replay.py','--cpu-profile','historical-ordered','--out',str(out/'ordered_fixed_pool')]),
('symmetry_audit',[sys.executable,'scripts/audit_sudoku_symmetry.py','--out',str(out/'symmetry_audit'),'--verify-report','results/reliability/sudoku4_symmetry_audit.json']),
('stored_new',[sys.executable,'scripts/verify_cpu_progress.py','--out',str(out/'stored_new.json')]),
('checkpoint_new',[sys.executable,'scripts/verify_symmetry_checkpoint_replay.py','--out',str(out/'checkpoint_new')]),
]
commands=[(name,[cmd[0],'-S','/mnt/data/research_work/clean_python.py',*cmd[1:]]) for name,cmd in commands]
records=[]
for name,cmd in commands:
 start=time.perf_counter()
 with (out/(name+'.log')).open('w') as log:
  try:r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200);code=r.returncode
  except subprocess.TimeoutExpired:code=124
 records.append({'name':name,'command':cmd,'exit_code':code,'elapsed_seconds':time.perf_counter()-start})
 (out/'progress.json').write_text(json.dumps(records,indent=2)+'\n')
 if code!=0:break
report={'status':'PASS' if len(records)==len(commands) and all(r['exit_code']==0 for r in records) and source_identity()==source else 'FAIL',
 'source':source,'source_unchanged':source_identity()==source,'commands':records,'gpu_used':False,
 'interpreter_bootstrap':'python -S with original site-packages; skips host NumPy-preloading sitecustomize, not project checks',
 'scope':'local integrated source at upstream a55bb754 plus the delivered patch; not remote CI'}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'environment.lock.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
