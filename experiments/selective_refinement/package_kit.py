from pathlib import Path
import json,shutil,hashlib
import argparse
parser=argparse.ArgumentParser(description='Package frozen refinement models and matching native libraries, without fitting')
parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();r=args.evidence.resolve();repo=Path(__file__).resolve().parents[2];kit=args.out.resolve();kit.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for path in (repo/'spectra').rglob('*'):
    if path.is_file() and (path.suffix in ('.py','.cpp','.hpp') or path.name=='LICENSE'):
        out=kit/path.relative_to(repo);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
for name in ('budgeted_prototypes','finite_kernel','adaptive_kernel','selective_refinement','source_receipts','conditioned_heads'):
    for path in (repo/'experiments'/name).glob('*'):
        if path.is_file() and path.suffix in ('.py','.cpp','.md'):
            out=kit/path.relative_to(repo);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
shutil.copy2(repo/'LICENSE',kit/'LICENSE')
policies=json.loads((r/'refinement_calibration/POLICIES.json').read_text());results=json.loads((r/'refinement_evaluation/RESULTS.json').read_text())
manifest={'format':'spectra.refinement.kit.v1','parent_source_commit':'c5af3032e1f9dd8908e922836dd7c06d4222b876',
          'experimental':True,'scope':'same-environment classifier replay, not an IID risk/shift guarantee or production release',
          'tasks':{},'libraries':{},'source_files':{}}
import numpy as np
for task in ('letter','pendigits','satellite','optdigits'):
    folder=kit/'models'/task;folder.mkdir(parents=True)
    shutil.copy2(r/f'refinement_fit/{task}/fast/model.spp',folder/'fast.spp');shutil.copy2(r/f'refinement_fit/{task}/strong/model.srt',folder/'strong.srt')
    doc={'format':'spectra.refinement.policy.v1','experimental':True,'task':task,
         'models':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in folder.iterdir()},
         'policies':{n:policies[task][key] for n,key in (('primary','0.01'),('strict','0.005'),('loose','0.02'))},
         'scope':'calibration-conditioned routing; shifted/grouped evaluation does not meet an IID guarantee; never interprets 1% as total error',
         'calibration_lock_sha256':sha(r/'refinement_calibration/CALIBRATION_LOCK.json')}
    (folder/'policy.json').write_text(json.dumps(doc,indent=2,allow_nan=False))
    inp=kit/'inputs'/f'{task}.u8';inp.parent.mkdir(exist_ok=True);shutil.copy2(r/f'refinement_evaluation/{task}/input.u8',inp)
    exp=kit/'expected'/f'{task}.json';exp.parent.mkdir(exist_ok=True)
    with np.load(r/f'refinement_evaluation/{task}/predictions.npz') as x:
        pred={n:x[n].tolist() for n in ('fast','strong','primary','strict','loose','blind')}
    exp.write_text(json.dumps(pred,separators=(',',':')))
    spec=json.loads((r/f'refinement_fit/{task}/fast/fit.json').read_text())
    manifest['tasks'][task]={'features':spec['features'],'maximum':spec['maximum'],
        'rows':results['tasks'][task]['quality']['fast']['rows'],'policy_sha256':sha(folder/'policy.json'),
        'input_sha256':sha(inp),'expected_sha256':sha(exp),'observed':results['tasks'][task]}
for target,origin in (('avx2','refinement-native-final'),('portable','refinement-portable')):
    dest=kit/'native'/target;dest.mkdir(parents=True)
    for name in ('refinement.so','build.json'):shutil.copy2(r/origin/name,dest/name)
    manifest['libraries'][target]={'sha256':sha(dest/'refinement.so'),'bytes':(dest/'refinement.so').stat().st_size}
shutil.copy2(Path(__file__).with_name('kit_selftest.py'),kit/'selftest.py')
shutil.copy2(Path(__file__).with_name('kit_demo.py'),kit/'demo.py')
shutil.copy2(Path(__file__).with_name('RESULTS.md'),kit/'RESULTS.md')
shutil.copy2(Path(__file__).with_name('DATA_SOURCES.md'),kit/'DATA_SOURCES.md')
(kit/'README.md').write_text('''# Experimental selective refinement kit\n\nRun `python -I -S selftest.py --target portable` to replay all supplied policies.\nRun `python -I -S demo.py --task letter` for one included sample, or pass a JSON\ninteger array with --row. AVX2 is explicit and requires compatible hardware.\n\nThe libraries are for Linux x86-64 only. Python and compatible system C++/math\nlibraries must already be installed. No numerical Python framework is required.\n\nBoth component models remain in memory. The statistical calibration is NOT a\ndistribution-shift/individual-correctness guarantee; Pendigits exceeded its nominal\nadded-harm budget on the recorded test. The original two-task scientific gate failed.\nRead RESULTS.md and experiments/selective_refinement/CONTRACT.md before use.\nThis is a research artifact, not a production release or replacement default.\n\nSource hashes and model/policy identities are in MANIFEST.json. Verify the outer\nfile-closure seal against the separately supplied trusted manifest digest. The\nFP32-MLP benchmark is not reproduced by this kit: its external OpenBLAS dependency\nand model files are documented in the separate full evidence. Dataset attribution\nand limits are in DATA_SOURCES.md.\n''')
for p in sorted(kit.rglob('*')):
    if p.is_file() and p.suffix in ('.py','.cpp','.hpp'):
        manifest['source_files'][p.relative_to(kit).as_posix()]=sha(p)
(kit/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
