"""Exact conditional Co certificate under additional, explicitly hypothetical evidence."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision_20261031/revision_code'))
from run_magnetic import normalize,P
from completion import solve,verification_tests
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)

def margins(q,upper=None):
    return [solve(5,q,1.,1,'min',contrast=j,missing_upper=upper) for j in [0,2,3,4]]

def main():
    verification_tests()
    _,_,f,_,_,_=normalize();q=f.groupby('mask').payoff.max().to_dict()
    threshold=q[2];rows=[];witnesses=[]
    for u in [0.,.3,.35,threshold-1e-5,threshold,threshold+1e-5,.38,.4,.5,1.]:
        rr=margins(q,u)
        for p,r in zip(['H','Mn','C','Ce'],rr):
            rows.append(dict(missing_score_cap=u,competitor=p,Co_worst_margin=r['value']))
            for s in range(32):witnesses.append(dict(missing_score_cap=u,competitor=p,mask=s,q=r['q'][s],v=r['v'][s]))
        if u < threshold-1e-8:assert min(r['value'] for r in rr)>1e-8
        if u==threshold:assert abs(min(r['value'] for r in rr))<1e-8
        if u>threshold:assert min(r['value'] for r in rr)<0
    pd.DataFrame(rows).to_csv(OUT/'nims_uniform_missing_cap.csv',index=False)
    pd.DataFrame(witnesses).to_csv(OUT/'nims_cap_witnesses.csv',index=False)
    # Reveal a single absent support hypothetically, testing low/intermediate/high values.
    # Failure even at endpoints alone does not exclude an interior certificate;
    # a separate obstruction witness below establishes the single-measurement result.
    one=[]
    missing=[s for s in range(1,32) if s not in q]
    for s in missing:
        for x in [0.,threshold,.5,1.]:
            rr=margins({**q,s:x})
            one.append(dict(mask=s,support=';'.join(P[i] for i in range(5) if s>>i&1),
                            hypothetical_score=x,Co_worst_margin=min(r['value'] for r in rr)))
        print('single support',s,'complete',flush=True)
    pd.DataFrame(one).to_csv(OUT/'nims_single_revelation_grid.csv',index=False)
    # Analytic obstruction covering every possible single revelation x in [0,1].
    # If Mn-only remains unknown, set it to 1 and every other missing support to 0.
    # Co cannot uniquely lead, including when a revealed support has value x.
    # Search a fixed competitor witness and verify
    # piecewise-linearity in the arbitrary revealed value using all score knots.
    from completion import envelope,weights
    W=weights(5);obstruction=[]
    for s in missing:
        knots=sorted(set([0.,1.,*q.values()]))
        candidates=[]
        for high in missing:
            if high==s:continue
            differences=[]
            for x in knots:
                a=np.zeros(32)
                for k,z in q.items():a[k]=z
                a[high]=1.;a[s]=x
                phi=W@envelope(a)
                differences.append(phi[1]-phi)
            worst_over_x=np.max(differences,axis=0)
            rivals=[j for j in [0,2,3,4] if worst_over_x[j]<=1e-9]
            if rivals:candidates.append((high,rivals[0],worst_over_x[rivals[0]]))
        # This constructive all-x check avoids inferring impossibility from a grid.
        assert candidates,(s,'no fixed one-high witness')
        high,rival,gap=candidates[0]
        obstruction.append(dict(revealed_mask=s,unrevealed_high_mask=high,
            fixed_competitor=P[rival],
            largest_Co_margin_at_all_breakpoints=gap,breakpoints=knots))
    (OUT/'nims_single_revelation_obstructions.json').write_text(json.dumps(obstruction,indent=2))
    (OUT/'nims_threshold_summary.json').write_text(json.dumps({'Co_singleton_score':threshold,
        'uniform_missing_cap_certificate':'necessary unique Co leader iff U < Co singleton score',
        'binding_competitor':'Mn','unknown_Mn_singleton_mask':4,
        'single_support_revelation_sufficient':False,
        'single_support_obstruction':'one unrevealed support at 1, other unrevealed scores zero; verified at every max-envelope breakpoint',
        'evidence_status':'hypothetical extra score upper bounds/revelations; no new alloy measurements'},indent=2))
    print('NIMS threshold',threshold,flush=True)

if __name__=='__main__':main()
