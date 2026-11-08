"""Coverage-sensitive runtime checks, including positive ranking certificates.

These checks supplement four-support timings; they are not worst-case bounds.
Observed support counts include every singleton and random larger supports.
"""
from pathlib import Path
import sys,time,json,platform
import numpy as np,pandas as pd,scipy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision_20261031/revision_code'))
from completion import solve
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)

def main():
 rows=[]
 for n in [5,6,8]:
  for fraction in [.25,.5]:
   for kind in ['random_scores','planted_leader']:
    for rep in range(2):
     rng=np.random.default_rng(20261109+1000*n+100*int(fraction*100)+rep)
     singles=[1<<i for i in range(n)]
     k=int(np.ceil(fraction*((1<<n)-1)))
     rest=np.setdiff1d(np.arange(1,1<<n),singles)
     masks=np.r_[singles,rng.choice(rest,k-n,replace=False)]
     observed={0:0.,**{int(s):float(x) for s,x in zip(masks,rng.uniform(.2,.8,k))}}
     if kind=='planted_leader':observed[1]=1.
     start=time.perf_counter()
     # One worst margin against player 1, matching the sparse runtime scope.
     result=solve(n,observed,1.,0,'min',contrast=1)
     elapsed=time.perf_counter()-start
     if kind=='planted_leader':assert result['value']>0
     rows.append(dict(players=n,coverage_target=fraction,observed_nonempty=k,
       nonempty_supports=(1<<n)-1,coverage_actual=k/((1<<n)-1),instance_type=kind,
       replicate=rep,seconds=elapsed,worst_pair_margin=result['value'],
       selector_binaries=result['selector_binaries'],gap=result['gap'],
       envelope_residual=result['envelope_residual']))
     pd.DataFrame(rows).to_csv(OUT/'dense_scaling.csv',index=False)
     print(n,fraction,kind,rep,round(elapsed,4),round(result['value'],6),flush=True)
 (OUT/'dense_scaling_environment.json').write_text(json.dumps({'python':platform.python_version(),
 'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
 'scope':'one pairwise margin per instance; 2 replicates per cell; all singletons observed',
 'planted_leader':'player 0 singleton score 1 at the unit cap; all other observed scores below 0.8',
 'timing_limit_per_solve_s':90,'seed_rule':'20261109+1000*p+100*int(100*coverage)+replicate'},indent=2))
if __name__=='__main__':main()
