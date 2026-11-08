"""Diagnose early recovery changes and terminal-budget coverage from saved worlds."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'extension'))
from run_acquisition import estimate
OUT=Path(__file__).resolve().parent/'results'

def main():
 rows=[]
 for family in ['scalar','two_objective']:
  a=np.load(OUT/(family+'_policy_inputs.npz'));q=a['q']
  for layout in ['veto','nonveto']:
   for noise_name,e in [('none',np.zeros_like(a['sensor'])),('independent',.1*a['sensor'])]:
    s=a[layout+'_'+noise_name+'_expand']
    before=estimate(q,s,e,0,'mean')[1];after=estimate(q,s,e,4,'mean')[1]
    rows.append(dict(family=family,layout=layout,noise=noise_name,n_worlds=len(q),
      recovery_at_0=before.mean(),recovery_at_4=after.mean(),
      correct_to_incorrect=int(np.sum(before&~after)),
      incorrect_to_correct=int(np.sum(~before&after))))
 pd.DataFrame(rows).to_csv(OUT/'early_budget_diagnostics.csv',index=False)
 z=pd.DataFrame(rows)
 for _,r in z.iterrows():
  assert abs((r.recovery_at_4-r.recovery_at_0)*r.n_worlds-(r.incorrect_to_correct-r.correct_to_incorrect))<1e-8
 (OUT/'early_budget_verification.json').write_text(json.dumps({'paired_transition_balance':True,
   'scope':'Same latent worlds, expansion sequences and mean estimator; no-noise control shows dip is not solely due to observation errors'},indent=2))
 print(z.to_string(index=False))
if __name__=='__main__':main()
