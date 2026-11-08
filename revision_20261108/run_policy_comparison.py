"""Paired support acquisition with veto and nonveto starting coverage.

The adaptive comparator is a deliberately specified Gaussian-process UCB rule
for utility acquisition, not an optimal Shapley-ranking acquisition algorithm.
Its posterior is updated only with observations available at the decision time.
All arms are evaluated using the same observed-support mean/max estimator.
"""
from pathlib import Path
import sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'extension'))
from run_acquisition import make_truth, estimate, BITS, PAIR

OUT=Path(__file__).resolve().parent/'results'
LAYOUTS={'veto':[6,22,11,15], 'nonveto':[6,21,11,29]}
COUNTS=[1,3,7,6]
BUDGETS=[0,4,8,16,27]
# Main and pair effects give a fixed covariance on binary supports.
FEATURES=np.column_stack([np.ones(32),2*BITS-1,2*PAIR-1])
K=.25**2*(FEATURES@FEATURES.T)/FEATURES.shape[1]


def adaptive(q, initial, noise, beta=1.96):
    """Sequential GP-UCB; q is used only by the observation interface.

    A fixed working noise SD .10 is used in every regime. Posterior means are
    not clipped; actual observed scalar scores and scoring estimates are.
    """
    n=len(q); ids=np.arange(n)
    mean=np.full((n,32),.5)
    cov=np.broadcast_to(K,(n,32,32)).copy()
    plan=np.empty((n,44),int);plan[:,:17]=initial
    observed=np.empty((n,44))
    for t in range(44):
        if t<17:s=plan[:,t]
        else:
            std=np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2),0))
            score=mean+beta*std;score[:,0]=-np.inf
            s=score.argmax(1);plan[:,t]=s
        y=np.clip(q[ids,s]+noise[:,t],0,1);observed[:,t]=y
        column=cov[ids,:,s].copy();den=cov[ids,s,s]+.1**2
        mean+=column*((y-mean[ids,s])/den)[:,None]
        cov-=column[:,:,None]*column[:,None,:]/den[:,None,None]
    return plan,observed


def fixed_plans(rng,n,masks):
    initial=np.repeat(masks,COUNTS)
    repeat=np.empty((n,27),int);expand=np.empty((n,27),int)
    missing=np.setdiff1d(np.arange(1,32),masks)
    for j in range(n):
        counts=np.array(COUNTS)
        for t in range(27):
            i=rng.choice(np.flatnonzero(counts==counts.min()))
            repeat[j,t]=masks[i];counts[i]+=1
        expand[j]=rng.permutation(missing)
    base=np.broadcast_to(initial,(n,17))
    return {a:np.concatenate([base,z],axis=1) for a,z in [('repeat',repeat),('expand',expand)]}


def verification():
    a=BITS[LAYOUTS['nonveto']]
    assert not a.all(0).any()
    assert len(set(map(tuple,a.T)))==5
    assert [int(x).bit_count() for x in LAYOUTS['veto']]==[int(x).bit_count() for x in LAYOUTS['nonveto']]
    rng=np.random.default_rng(91);q=make_truth(rng,12,'scalar')
    init=np.broadcast_to(np.repeat(LAYOUTS['veto'],COUNTS),(12,17))
    noise=rng.normal(0,.1,(12,44))
    plan,y=adaptive(q,init,noise)
    # Change only utilities hidden at the first decision. Its choice must agree.
    altered=q.copy();missing=np.setdiff1d(np.arange(1,32),LAYOUTS['veto'])
    altered[:,missing]=1-altered[:,missing]
    alternative,_=adaptive(altered,init,noise)
    np.testing.assert_array_equal(plan[:,17],alternative[:,17])
    assert len(np.unique(plan[:,17]))>1 # observed-data-dependent decisions
    assert np.all(plan[:,17:]>0)
    plans=fixed_plans(rng,12,LAYOUTS['nonveto'])
    for arm,z in plans.items():
        assert z.shape==(12,44)
        if arm=='expand':
            assert all(len(set(row))==31 for row in z)
            np.testing.assert_allclose(estimate(q,z,np.zeros_like(z),27,'mean')[0],0,atol=1e-12)
        else:assert all(set(row)==set(LAYOUTS['nonveto']) for row in z)
    return {'nonveto_no_veto_or_equivalent_pair':True,'support_sizes_and_counts_matched':True,
            'adaptive_first_choice_hidden_truth_invariance':True,'adaptive_choices_depend_on_observations':True,
            'budget_and_full_coverage_recovery':True}


def run(n=2000,bootstrap=1000,seed=20261108):
    OUT.mkdir(exist_ok=True);checks=verification()
    summary=[];contrasts=[];records=[]
    config={'seed':seed,'worlds_per_family':n,'layouts':LAYOUTS,'counts':COUNTS,
            'budgets':BUDGETS,'policies':['repeat','expand','gp_ucb'],
            'working_GP_mean':.5,'kernel':'0.25^2 * (1, centered binary main and pair features) inner product / 16',
            'working_noise_sd':.1,'UCB_beta':1.96,'adaptive_allows_revisiting':True,
            'adaptive_criterion':'posterior utility mean + 1.96 SD; utility criterion, not ranking certificate',
            'all_arms_ranking_estimator':'observed support mean; max pooling sensitivity; missing scores zero',
            'true_payoff_status':'synthetic fixed utilities, no physical feasibility claim',
            'bootstrap_resamples':bootstrap,'noise':['none','independent','batch'],
            'retrospective_design':True}
    for fi,family in enumerate(['scalar','two_objective']):
        rng=np.random.default_rng(np.random.SeedSequence([seed,fi]))
        q=make_truth(rng,n,family);sensor=rng.normal(size=(n,44))
        bix=np.r_[np.arange(17)//3,6+np.arange(27)//3]
        offsets=rng.normal(size=(n,bix.max()+1))[:,bix]
        boots=np.random.default_rng(seed+fi+700).integers(0,n,(bootstrap,n))
        saved={'q':q,'sensor':sensor,'offsets':offsets}
        for layout,masks in LAYOUTS.items():
            plans=fixed_plans(np.random.default_rng(seed+fi+200),n,masks)
            initial=plans['repeat'][:,:17]
            for regime,(sd,batchsd) in {'none':(0,0),'independent':(.1,0),'batch':(.1,.1)}.items():
                noise=sd*sensor+batchsd*offsets
                plans['gp_ucb'],_=adaptive(q,initial,noise)
                for arm,s in plans.items(): saved[layout+'_'+regime+'_'+arm]=s
                for pool in ['mean','max']:
                    for budget in BUDGETS:
                        values={arm:estimate(q,s,noise,budget,pool) for arm,s in plans.items()}
                        if budget==0:
                            for arm in ['expand','gp_ucb']:
                                for k in range(5):np.testing.assert_array_equal(values[arm][k],values['repeat'][k])
                        for arm,r in values.items():
                            valid=r[4];s=plans[arm][:,:17+budget]
                            coverage=np.array([len(set(row)) for row in s])
                            summary.append(dict(family=family,layout=layout,noise=regime,pooling=pool,budget=budget,
                                arm=arm,n_worlds=n,n_unique_truth=int(valid.sum()),recovery=float(r[1][valid].mean()),
                                mean_l1=float(r[0].mean()),tie_rate=float(r[2].mean()),regret=float(r[3].mean()),
                                mean_observed_nonempty_supports=float(coverage.mean())))
                            records.append(pd.DataFrame(dict(family=family,layout=layout,noise=regime,pooling=pool,
                                budget=budget,arm=arm,world=np.arange(n),correct_top=r[1],unique_true_top=valid,
                                l1=r[0],regret=r[3],coverage=coverage)))
                        for arm,ref in [('expand','repeat'),('gp_ucb','repeat'),('gp_ucb','expand')]:
                            for metric,k in [('recovery',1),('l1',0),('regret',3)]:
                                d=values[arm][k].astype(float)-values[ref][k].astype(float)
                                if metric=='recovery':d=np.where(values[ref][4],d,np.nan)
                                lo,hi=np.nanquantile(np.nanmean(d[boots],axis=1),[.025,.975])
                                contrasts.append(dict(family=family,layout=layout,noise=regime,pooling=pool,budget=budget,
                                    arm=arm,reference=ref,metric=metric,difference=float(np.nanmean(d)),ci_low=lo,ci_high=hi))
                print(f'{family} {layout} {regime} complete',flush=True)
        np.savez_compressed(OUT/(family+'_policy_inputs.npz'),**saved)
    s=pd.DataFrame(summary);s.to_csv(OUT/'policy_summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'policy_paired_differences.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(OUT/'policy_world_results.csv.gz',index=False)
    (OUT/'policy_config.json').write_text(json.dumps(config,indent=2)+'\n')
    (OUT/'policy_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(8.5,6.3),sharex=True,sharey=True)
    for i,family in enumerate(['scalar','two_objective']):
        for j,layout in enumerate(LAYOUTS):
            ax=axes[i,j]
            for arm,color,label in [('repeat','#a75237','Repeat covered'),('expand','#267a80','Random expansion'),('gp_ucb','#655386','Adaptive GP-UCB')]:
                z=s[(s.family==family)&(s.layout==layout)&(s.noise=='independent')&(s.pooling=='mean')&(s.arm==arm)]
                ax.plot(z.budget,z.recovery,'o-',color=color,label=label,markersize=4)
            ax.set_title(('Cr-veto start' if j==0 else 'No-veto start')+' | '+('scalar' if i==0 else 'two objectives'),fontsize=10)
            ax.set_ylim(0,1.02);ax.grid(alpha=.2)
            if j==0:ax.set_ylabel('True-leader recovery')
            if i==1:ax.set_xlabel('Additional observations')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'policy_comparison.png',dpi=350);fig.savefig(OUT/'policy_comparison.pdf')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worlds',type=int,default=2000)
    p.add_argument('--bootstrap',type=int,default=1000);a=p.parse_args();run(a.worlds,a.bootstrap)
