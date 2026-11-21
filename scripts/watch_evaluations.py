"""Wait for all three fixed checkpoints and evaluate each once on a spare GPU."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from scripts.evaluate_optimized import audit_results,cases,validate_selection
from scripts.plot_optimized import load_results
from scripts.prepare_optimized import ROBOTS,sha


def ready_checkpoints(run_root,iteration):
    paths={r:Path(run_root)/r/f'ckpt_{iteration}.pt' for r in ROBOTS}
    for path in paths.values():
        if path.is_symlink():raise ValueError(f'Checkpoint must not be a symlink: {path}')
    if any(not p.is_file() for p in paths.values()):return None
    for path in paths.values():
        if path.absolute()!=path.resolve():raise ValueError(f'Checkpoint path is not fixed: {path}')
    return {r:dict(path=str(p.resolve()),sha256=sha(p)) for r,p in paths.items()}


def verify_result(output,manifest_sha,iteration,manifest,checkpoints):
    rows=load_results([output],manifest_sha)
    expected=[(iteration,r,d,i) for r in ROBOTS for d,i in cases(manifest['clips'],r)]
    if not expected:raise ValueError('No validation cases in manifest')
    audited=audit_results(rows,expected)
    if any(r['status']!='ok' for r in audited):raise ValueError('Evaluation has missing or failed cases')
    if any(row['checkpoint_sha256']!=checkpoints[row['robot']]['sha256'] for row in audited):
        raise ValueError('Evaluation checkpoint identity differs from watcher')
    return audited


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','data-root','out-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations',nargs='+',type=int);p.add_argument('--gpu',type=int,default=1)
    p.add_argument('--num-envs',type=int,default=256);p.add_argument('--poll-seconds',type=float,default=5.)
    a=p.parse_args()
    try:iterations,_=validate_selection(a.iterations,None)
    except ValueError as error:p.error(str(error))
    if not math.isfinite(a.poll_seconds) or not 0<a.poll_seconds<=60 or a.num_envs<1:p.error('Invalid poll interval or env count')
    data_path=a.data_root/'manifest.json';manifest_sha=sha(data_path);manifest=json.loads(data_path.read_text())
    if not manifest.get('complete'):raise ValueError('Incomplete data manifest')
    a.out_root.mkdir(parents=True,exist_ok=True)
    for iteration in iterations:
        output=a.out_root/f'iteration_{iteration:05d}'
        while True:
            checkpoints=ready_checkpoints(a.run_root,iteration)
            if checkpoints is not None:break
            time.sleep(a.poll_seconds)
        if sha(data_path)!=manifest_sha:raise ValueError('Prepared manifest changed while waiting')
        if not output.exists():
            command=[sys.executable,'-m','scripts.evaluate_optimized','--run-root',str(a.run_root),
                '--data-root',str(a.data_root),'--out',str(output),'--gpu',str(a.gpu),
                '--num-envs',str(a.num_envs),'--iterations',str(iteration)]
            print('Evaluating fixed checkpoint',iteration,flush=True)
            completed=subprocess.run(command,check=False)
            if completed.returncode:raise RuntimeError(f'Evaluation {iteration} failed ({completed.returncode}); inspect {output}')
        verify_result(output,manifest_sha,iteration,manifest,checkpoints)
        after=ready_checkpoints(a.run_root,iteration)
        if after!=checkpoints or sha(data_path)!=manifest_sha:raise ValueError('Checkpoint or data changed during evaluation')
        receipt=output/'watch-receipt.json'
        if not receipt.exists():
            with receipt.open('x') as stream:
                json.dump(dict(iteration=iteration,manifest_sha256=manifest_sha,checkpoints=checkpoints,
                    results_sha256=sha(output/'results.json')),stream,indent=2)
        print('Completed fixed checkpoint',iteration,flush=True)

if __name__=='__main__':main()
