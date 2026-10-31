"""Predeclared composition representations, source-grouped validation and block ablations.

Host-only Magpie descriptors keep interstitial coordinates separate; no structure,
magnetic state, band structure or missing support payoff is inferred here.
"""
from pathlib import Path
import json,sys,importlib.metadata
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'matminer_results';OUT.mkdir(exist_ok=True)
HOST=['Fe','Si','La','Co','Mn','Ce'];P=['H','Co','Mn','C','Ce'];PROCESS=['anneal_h','anneal_K','Htr_T','method']
BLOCKS={'d_valence':'NdValence','d_unfilled':'NdUnfilled','radius':'CovalentRadius','electronegativity':'Electronegativity'}
CONFIG={'target':'Ttr_K','primary_split':'original GroupKFold(5), shuffle=False',
 'sensitivity_splits':'5 additional GroupKFold(5), shuffle=True, seeds 20261101--20261105',
 'rf':{'n_estimators':300,'min_samples_leaf':3,'max_features':1.0,'random_state':20261031,'n_jobs':1},
 'descriptors':list(BLOCKS.values()),'statistics':['mean','avg_dev'],'descriptor_scope':'Six host constituents only; H and C coordinates retained separately. Host weights normalized internally by Composition.',
 'feature_selection':'None; all eight host descriptors fixed before seeing results.',
 'hypothesis_gate':'Call gain consistent only if combined-versus-amounts MAE improves on all six partitions and at least four of five primary test folds.',
 'interpretation':'Exploratory revision-stage representation diagnostic; partition variation and correlated-feature ablations do not identify causal mechanisms.'}
(OUT/'configuration.json').write_text(json.dumps(CONFIG,indent=2))

def features(d):
 f=ElementProperty('magpie',list(BLOCKS.values()),['mean','avg_dev'],impute_nan=False)
 comps=[Composition({k:float(r[k]) for k in HOST if float(r[k])>0}) for _,r in d.iterrows()]
 values=np.array([f.featurize(c) for c in comps]);assert values.shape==(len(d),8) and np.isfinite(values).all()
 cols=[f'host_{b}_{s}' for b in BLOCKS for s in ['mean','avg_dev']]
 x=pd.DataFrame(values,columns=cols,index=d.index)
 pd.concat([d[['row_id','source_group']],x],axis=1).to_csv(OUT/'host_descriptors.csv',index=False)
 (OUT/'descriptor_definitions.json').write_text(json.dumps(dict(zip(cols,f.feature_labels())),indent=2))
 return x

def main():
 d=pd.read_csv(ROOT/'magnetic_data/nims_primary_2T_lowfield.csv');d=d.reset_index(drop=True)
 assert len(d)==140 and d.source_group.nunique()==53
 hostsum=d[HOST].sum(axis=1);assert np.max(np.abs(hostsum-100))<.01
 desc=features(d);base=d[PROCESS].copy()
 X={}
 X['binary']=pd.concat([d[P].gt(0).astype(int),base],axis=1)
 X['amounts']=pd.concat([d[['Fe','Si','La']+P],base],axis=1)
 X['matminer_host']=pd.concat([desc,d[['H','C']],base],axis=1)
 X['combined']=pd.concat([X['amounts'],desc],axis=1)
 for block in BLOCKS:X['drop_'+block]=X['combined'].drop(columns=[c for c in desc if c.startswith('host_'+block+'_')])
 group=d.source_group.to_numpy();y=d.Ttr_K.to_numpy()
 configs=[('primary',GroupKFold(5))]+[(f'shuffle_{s}',GroupKFold(5,shuffle=True,random_state=s)) for s in range(20261101,20261106)]
 summaries=[];foldrows=[];predictions=[]
 for partition,cv in configs:
  split=list(cv.split(d,groups=group))
  for name,x in X.items():
   pred=np.empty(len(d));fold=np.empty(len(d),int)
   for k,(tr,te) in enumerate(split):
    assert not set(group[tr])&set(group[te])
    num=[c for c in x if c!='method']
    pre=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),num),('method',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['method'])])
    model=make_pipeline(pre,RandomForestRegressor(**CONFIG['rf']))
    model.fit(x.iloc[tr],y[tr]);pred[te]=model.predict(x.iloc[te]);fold[te]=k
    foldrows.append({'partition':partition,'representation':name,'fold':k,'test_records':len(te),'mae_K':mean_absolute_error(y[te],pred[te])})
   err=np.abs(y-pred)
   summaries.append({'partition':partition,'representation':name,'mae_K':err.mean(),'rmse_K':mean_squared_error(y,pred)**.5,'r2':r2_score(y,pred),'group_balanced_mae_K':pd.Series(err).groupby(group).mean().mean()})
   predictions.append(pd.DataFrame({'partition':partition,'representation':name,'row_id':d.row_id,'source_group':group,'fold':fold,'observed_Ttr_K':y,'prediction_K':pred,'absolute_error_K':err}))
  print(partition,'complete',flush=True)
 a=pd.DataFrame(summaries);folds=pd.DataFrame(foldrows);oof=pd.concat(predictions,ignore_index=True)
 a.to_csv(OUT/'representation_metrics.csv',index=False);folds.to_csv(OUT/'fold_metrics.csv',index=False);oof.to_csv(OUT/'oof_predictions.csv',index=False)
 pivot=a.pivot(index='partition',columns='representation',values='mae_K')
 comparisons=pd.DataFrame({'combined_gain_vs_amounts_K':pivot.amounts-pivot.combined,'matminer_gain_vs_amounts_K':pivot.amounts-pivot.matminer_host})
 for b in BLOCKS:comparisons['drop_'+b+'_penalty_K']=pivot['drop_'+b]-pivot.combined
 comparisons.to_csv(OUT/'partition_comparisons.csv')
 f=folds[folds.partition.eq('primary')].pivot(index='fold',columns='representation',values='mae_K')
 primary=a[a.partition.eq('primary')].set_index('representation')
 old=pd.read_csv(ROOT/'magnetic_results/nims_representation_comparison.csv').set_index('representation')
 for name in ['binary','amounts']:
  assert abs(primary.loc[name,'mae_K']-old.loc[name,'mae_K'])<1e-9
 consistent=bool(comparisons.combined_gain_vs_amounts_K.gt(0).all() and ((f.amounts-f.combined)>0).sum()>=4)
 report={'status':'PASS','records':len(d),'groups':len(set(group)),'host_totals_min_max':[hostsum.min(),hostsum.max()],
 'descriptor_missing_values':0,'source_group_overlap':False,'baseline_matches_previous':True,
 'combined_improves_partitions':int(comparisons.combined_gain_vs_amounts_K.gt(0).sum()),'total_partitions':6,
 'combined_improves_primary_folds':int(((f.amounts-f.combined)>0).sum()),'consistent_gain_gate_passed':consistent,
 'versions':{k:importlib.metadata.version(k) for k in ['matminer','pymatgen','numpy','pandas','scikit-learn']}}
 (OUT/'verification.json').write_text(json.dumps(report,indent=2))
 print(a[a.partition.eq('primary')].to_string(index=False));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
