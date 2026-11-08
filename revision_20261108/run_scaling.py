"""Small exact-solver runtime benchmark; finite support growth remains exponential."""
from pathlib import Path
import sys,time,json,platform
import numpy as np,pandas as pd,scipy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision_20261031/revision_code'))
from completion import solve,weights,envelope
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)

def main():
    rows=[]
    for n in [5,6,7,8,9,10]:
        for rep in range(3):
            rng=np.random.default_rng(20261108+100*n+rep)
            # Four uniformly sampled distinct nonempty supports. Sparse stress
            # testing only; this does not characterize dense hard instances.
            masks=rng.choice(np.arange(1,1<<n),4,replace=False)
            q={0:0.,**{int(s):float(x) for s,x in zip(masks,rng.uniform(.2,.8,4))}}
            t=time.perf_counter();result=solve(n,q,1.,0,'min',contrast=1)
            elapsed=time.perf_counter()-t
            rows.append(dict(players=n,support_variables=1<<n,replicate=rep,
                observed_nonempty=4,selector_binaries=result['selector_binaries'],
                seconds=elapsed,worst_pair_margin=result['value'],gap=result['gap'],
                envelope_residual=result['envelope_residual']))
            print(n,rep,elapsed,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'scaling.csv',index=False)
    (OUT/'scaling_environment.json').write_text(json.dumps({'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
        'scope':'one pairwise worst-margin MILP per instance; not all allocation/rank optimizations',
        'random_seed_rule':'20261108+100*p+replicate','replicates':3,'time_limit_per_solve_s':90},indent=2))
if __name__=='__main__':main()
