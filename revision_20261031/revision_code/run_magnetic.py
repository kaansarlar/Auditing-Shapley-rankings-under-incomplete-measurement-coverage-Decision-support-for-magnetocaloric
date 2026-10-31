"""Magnetocaloric transfer audit using the independently curated NIMS La-Fe-Si data.

No fitted field normalization or invented alloy measurements are used. Blank
article metadata inherited from merged workbook blocks is forward-filled;
physical measurements, processing entries and compositions are never filled
from adjacent rows. H/D isotope-labelled rows are excluded when identified.
"""
from pathlib import Path
from itertools import combinations
import ast, json, re, hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from completion import solve, weights, envelope, verification_tests
from audit_tools import audit_scores, all_bounds, dump

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'magnetic_data';DATA.mkdir(exist_ok=True)
OUT=ROOT/'magnetic_results';OUT.mkdir(exist_ok=True)
P=['H','Co','Mn','C','Ce']
SEED=20261031


def parse_numbers(x):
    if pd.isna(x):return None
    try:
        z=ast.literal_eval(str(x))
        z=z if isinstance(z,(list,tuple)) else[z]
        vals=[float(a) for a in z]
        return vals if np.isfinite(vals).all() else None
    except (ValueError,SyntaxError,TypeError):return None


def extract2(row):
    h=parse_numbers(row['HS (T)']);s=parse_numbers(row['|ΔSM|max (J/kg K) '])
    if h is None or s is None:return np.nan,'missing_or_non_numeric_field_entropy_pair'
    if len(h)!=len(s):return np.nan,'field_entropy_array_length_mismatch'
    values=[v for f,v in zip(h,s) if abs(f-2.0)<1e-8]
    if not values:return np.nan,'no_exact_2T_value'
    if not np.allclose(values,values[0],atol=1e-10,rtol=0):return np.nan,'multiple_disagreeing_2T_values'
    if values[0]<=0:return np.nan,'nonpositive_entropy'
    return values[0],'matched_2T'


def doi_list(text):
    # DOI blocks sometimes contain literal backslash-n rather than newline.
    return sorted(set(s.strip().rstrip('.,;').lower() for s in
               re.findall(r'10\.\d{4,9}/[^\s]+',str(text).replace('\\n','\n'))))


def source_groups(frame):
    parents={}
    def find(x):
        parents.setdefault(x,x)
        if parents[x]!=x:parents[x]=find(parents[x])
        return parents[x]
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    lists=frame.source_dois.apply(doi_list).tolist()
    for ds in lists:
        assert ds,'A source DOI is required for grouping.'
        for d in ds[1:]:union(ds[0],d)
        find(ds[0])
    # Join duplicate scientific records across DOI blocks, conservatively
    # preventing an exactly repeated predictor/target record crossing folds.
    keys=['Fe','Si','La']+P+['Ttr_K','Htr_T','DS2','method','anneal_h','anneal_K']
    seen={}
    for (_,r),ds in zip(frame.iterrows(),lists):
        k=tuple('NA' if pd.isna(r[c]) else (round(float(r[c]),6) if c!='method' else str(r[c])) for c in keys)
        if k in seen:union(ds[0],seen[k])
        else:seen[k]=ds[0]
    return np.array([find(ds[0]) for ds in lists]),len(set(d for ds in lists for d in ds))


def normalize():
    csv=DATA/'nims_lafesi_source_1113.csv'
    if csv.exists():raw=pd.read_csv(csv)
    else:
        raw=pd.read_excel(ROOT/'external_data/nims_lafesi/1-13_dataset.xlsx')
        raw.to_csv(csv,index=False,float_format='%.15g')
    assert len(raw)==1113
    a=raw.copy()
    a['source_dois']=a['Article DOI'].ffill()
    a['source_article_id']=a['Article ID'].ffill()
    a['formula_label']=a['Composition in the article'].ffill()
    a['row_id']=a['ID'].astype(int)
    a['H']=a['H [or D]'];a['Ttr_K']=a['Ttr (K)'];a['Htr_T']=a['Htr (T)']
    a['method']=a['Method'].fillna('unreported')
    a['anneal_h']=a['Ann. Time (h)'];a['anneal_K']=a['Ann. Temp. (K)']
    pairs=a.apply(extract2,axis=1)
    a['DS2']=[z[0] for z in pairs];a['entropy_status']=[z[1] for z in pairs]
    host=list(raw.columns[5:29]);inter=['H [or D]','C','N','B']
    excluded=[c for c in host+inter if c not in ['La','Fe','Si','H [or D]','Co','Mn','C','Ce']]
    a['base_present']=a[['La','Fe','Si']].gt(0).all(axis=1)
    a['outside_elements_absent']=a[excluded].eq(0).all(axis=1)
    a['explicit_deuterium']=a.formula_label.astype(str).str.contains(r'D(?![a-z])',regex=True)
    a['transition_valid']=pd.to_numeric(a.Ttr_K,errors='coerce').gt(0)
    a['transition_field_valid']=pd.to_numeric(a.Htr_T,errors='coerce').between(0,.1)
    a['eligible']=a.base_present&a.outside_elements_absent&(~a.explicit_deuterium)&a.transition_valid&a.transition_field_valid&a.DS2.gt(0)
    ledger=a[['row_id','source_dois','source_article_id','formula_label','base_present','outside_elements_absent',
              'explicit_deuterium','transition_valid','transition_field_valid','entropy_status','eligible']]
    ledger.to_csv(OUT/'nims_row_eligibility_ledger.csv',index=False)
    count=np.ones(len(a),dtype=bool);flow=[['Source records',len(a)]]
    for label,condition in [
        ('Positive La-Fe-Si host',a.base_present),
        ('Only H, Co, Mn, C, Ce as additions',a.outside_elements_absent),
        ('Exclude explicitly deuterated records',~a.explicit_deuterium),
        ('Positive reported Ttr',a.transition_valid),
        ('Reported Htr between 0 and 0.1 T',a.transition_field_valid),
        ('Unambiguous positive entropy maximum at 2 T',a.DS2.gt(0))]:
        count &= condition.to_numpy()
        flow.append([label,int(count.sum())])
    pd.DataFrame(flow,columns=['restriction','records']).to_csv(OUT/'nims_eligibility_flow.csv',index=False)

    selected=a[a.eligible].copy()
    keys=['source_dois','Fe','Si','La']+P+['Ttr_K','Htr_T','DS2','method','anneal_h','anneal_K']
    dup=selected.duplicated(keys,keep='first')
    selected.loc[dup].to_csv(OUT/'nims_duplicate_records_excluded.csv',index=False)
    selected=selected.loc[~dup].copy()
    selected['mask']=sum((2**i)*selected[p].gt(0).astype(int) for i,p in enumerate(P))
    # Fixed scientific scales: room-temperature window 250--350 K, 50 K decay,
    # and 100 J kg^-1 K^-1 entropy scale. No empirical clipping quantiles.
    distance=np.maximum.reduce([250-selected.Ttr_K.to_numpy(),np.zeros(len(selected)),selected.Ttr_K.to_numpy()-350])
    selected['u_T']=np.exp(-distance/50)
    assert selected.DS2.le(100).all()
    selected['payoff']=np.sqrt(selected.u_T*selected.DS2/100)
    selected['source_group'],ndoi=source_groups(selected)
    selected.to_csv(DATA/'nims_primary_2T_lowfield.csv',index=False)
    hc=selected[selected[['Co','Mn','Ce']].eq(0).all(axis=1)].copy()
    hc['mask']=hc.H.gt(0).astype(int)+2*hc.C.gt(0).astype(int)
    hc.to_csv(DATA/'nims_HC_subportfolio.csv',index=False)
    return raw,a,selected,hc,ndoi,int(dup.sum())


def tabulate(frame,players,label):
    report,q,v=audit_scores(frame,players)
    report['source_dois']=len(set(d for t in frame.source_dois for d in doi_list(t)))
    report['source_groups']=frame.source_group.nunique()
    z=frame.groupby('mask').agg(records=('row_id','size'),max_payoff=('payoff','max'),
            min_Ttr_K=('Ttr_K','min'),max_Ttr_K=('Ttr_K','max'),min_DS2=('DS2','min'),max_DS2=('DS2','max')).reset_index()
    z['optional_players']=z['mask'].apply(lambda s:';'.join(p for i,p in enumerate(players) if s>>i&1))
    z.to_csv(OUT/(label+'_supports.csv'),index=False)
    dump(OUT/(label+'_audit.json'),report)
    print(label,report,flush=True)
    return report,q,v


def modeling(frame):
    # Both representations receive identical recorded process and measurement
    # covariates. This is an information-loss diagnostic, not a new predictor.
    groups=frame.source_group.to_numpy()
    assert len(set(groups))>=5
    split=list(GroupKFold(5).split(frame,groups=groups))
    output=frame[['row_id','source_group','Ttr_K']].copy();records=[]
    for name in ['binary','amounts']:
        X=frame[['Fe','Si','La']+P+['anneal_h','anneal_K','Htr_T','method']].copy()
        if name=='binary':
            for p in P:X[p]=(X[p]>0).astype(int)
            X=X.drop(columns=['Fe','Si','La'])
        numeric=[c for c in X if c!='method']
        pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),numeric),
                              ('method',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['method'])])
        pred=np.empty(len(frame));fold=np.empty(len(frame),int)
        for k,(tr,te) in enumerate(split):
            assert not set(groups[tr])&set(groups[te])
            model=make_pipeline(pre,RandomForestRegressor(n_estimators=300,min_samples_leaf=3,
                     max_features=1.0,random_state=SEED,n_jobs=1))
            model.fit(X.iloc[tr],frame.Ttr_K.iloc[tr]);pred[te]=model.predict(X.iloc[te]);fold[te]=k
        output[name+'_prediction_K']=pred;output['fold']=fold
        records.append({'representation':name,'records':len(frame),'source_groups':len(set(groups)),
                        'mae_K':mean_absolute_error(frame.Ttr_K,pred),
                        'rmse_K':mean_squared_error(frame.Ttr_K,pred)**.5,'r2':r2_score(frame.Ttr_K,pred)})
    output.to_csv(OUT/'nims_source_grouped_oof_predictions.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'nims_representation_comparison.csv',index=False)
    return records


def masked_hc(q):
    true=weights(2)@envelope([q[s] for s in range(4)])
    rows=[]
    for k in range(4):
        for known in combinations(range(1,4),k):
            obs={s:q[s] for s in (0,)+known};widths=[];vals={}
            for i,p in enumerate(['H','C']):
                lo=solve(2,obs,1.,i,'min')['value'];hi=solve(2,obs,1.,i,'max')['value']
                assert lo-1e-8<=true[i]<=hi+1e-8
                vals[p+'_min']=lo;vals[p+'_max']=hi;widths.append(hi-lo)
            certified=[]
            for i,p in enumerate(['H','C']):
                margin=solve(2,obs,1.,i,'min',contrast=1-i)['value']
                vals[p+'_worst_margin']=margin
                if margin>1e-8:certified.append(p)
            assert len(certified)<=1
            rows.append({'known_nonempty':';'.join(map(str,known)), 'nonempty_count':k,
                         'mean_bound_width':np.mean(widths),'certified_leader':';'.join(certified),**vals})
    pd.DataFrame(rows).to_csv(OUT/'nims_HC_all_information_states.csv',index=False)
    return rows


def omission(frame,players):
    rows=[]
    for group in sorted(frame.source_group.unique()):
        f=frame[frame.source_group.ne(group)]
        r,_,_=audit_scores(f,players)
        rows.append({'omitted_source_group':group,'retained_records':len(f),'retained_supports':r['supports'],
                     **r['shapley']})
    pd.DataFrame(rows).to_csv(OUT/'nims_HC_source_omission.csv',index=False)


def main():
    tests=verification_tests();raw,ledger,frame,hc,ndoi,ndup=normalize()
    broad,qb,vb=tabulate(frame,P,'nims_broad')
    narrow,qn,vn=tabulate(hc,['H','C'],'nims_HC')
    # Main transfer comparison uses both the broader five-dopant chemistry and
    # its chemically defined H--C subportfolio; these are nested, not independent datasets.
    bounds,witness=all_bounds('NIMS_5dopant',P,qb,{'unit_cap':1.0},modes=('score',))
    bounds.to_csv(OUT/'nims_broad_completion_bounds.csv',index=False);witness.to_csv(OUT/'nims_broad_completion_witnesses.csv',index=False)
    bounds,witness=all_bounds('NIMS_HC',['H','C'],qn,{'unit_cap':1.0})
    bounds.to_csv(OUT/'nims_HC_completion_bounds.csv',index=False);witness.to_csv(OUT/'nims_HC_completion_witnesses.csv',index=False)
    assert np.max(bounds['max']-bounds['min'])<1e-8
    states=masked_hc(qn);omission(hc,['H','C']);models=modeling(frame)
    # Retain the original HEA evidence and recompute both information classes.
    p=DATA/'hea_empirical_strategy_portfolio.csv'
    if not p.exists():raise FileNotFoundError('Required HEA input is missing: '+str(p))
    hea=pd.read_csv(p);hea=hea[hea.game_eligible.astype(str).str.lower().isin(['true','1'])].copy()
    hp=['Mn','Cr','Ni','Al','Cu'];hea['mask']=hea.optional_players.apply(lambda s:sum(1<<hp.index(p) for p in s.split(';')));hea['payoff']=hea.nash_geometric_mean
    hr,qh,vh=audit_scores(hea,hp);qh[0]=0
    b,w=all_bounds('HEA',hp,qh,{'unit_cap':1.,'measured_cap':max(qh.values())})
    b.to_csv(OUT/'hea_completion_bounds.csv',index=False);w.to_csv(OUT/'hea_completion_witnesses.csv',index=False)
    dump(OUT/'hea_audit.json',hr)
    summary={'source':'NIMS DOI10.48505/nims.4909','source_rows':len(raw),'duplicate_rows_removed_within_source':ndup,
             'primary':broad,'HC_subportfolio':narrow,'models':models,'tests':tests,
             'raw_source_sha256':hashlib.sha256((DATA/'nims_lafesi_source_1113.csv').read_bytes()).hexdigest(),
             'explicit_deuterium_ids':ledger.loc[ledger.explicit_deuterium,'row_id'].tolist(),
             'new_data_status':'Independent secondary-data reanalysis; no new materials experiments and no claim of causal dopant identification.',
             'selection':'Revision-stage analysis; fixed 2T entropy, reported transition field 0--0.1T, La-Fe-Si host, allowed H/Co/Mn/C/Ce, no explicit D, no adjacent-row imputation of physical quantities.'}
    dump(OUT/'magnetic_revision_summary.json',summary)
    print('MAGNETOCALORIC TRANSFER ANALYSES COMPLETE',flush=True)


if __name__=='__main__':main()
