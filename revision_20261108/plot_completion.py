"""Consistent information-class colors in the empirical completion figure."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'revision_20261108/results';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
h=pd.read_csv(ROOT/'revision_20261031/magnetic_results/hea_completion_bounds.csv');h=h[h.cap_class=='unit_cap']
b=pd.read_csv(ROOT/'revision_20261031/magnetic_results/nims_broad_completion_bounds.csv')
n=pd.read_csv(ROOT/'revision_20261031/magnetic_results/nims_HC_completion_bounds.csv');n=n[n.completion_class=='score']
fig,axs=plt.subplots(1,3,figsize=(10.2,3.7),gridspec_kw={'width_ratios':[1.4,1.2,1]})
players=['Mn','Cr','Ni','Al','Cu']
for mode,offset,color,label in [('score',.12,'#a75237','Measured scores fixed'),('access',-.12,'#267a80','Scores and access fixed')]:
 z=h[h.completion_class==mode].set_index('player').loc[players];y=np.arange(5)+offset
 axs[0].hlines(y,z['min'],z['max'],lw=2.4,color=color,label=label);axs[0].scatter(z['max'],y,c=color,s=10)
axs[0].set_yticks(range(5),players);axs[0].invert_yaxis();axs[0].set_xlim(-.02,1.03)
fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,fontsize=8,loc='upper center',ncol=2)
axs[0].set_title('a  HEA information classes',loc='left')
for i,row in b.iterrows():
 axs[1].hlines(i,row['min'],row['max'],color='#a75237',lw=3);axs[1].scatter([row['min'],row['max']],[i,i],c='#a75237',s=12)
axs[1].set_yticks(range(5),b.player);axs[1].invert_yaxis();axs[1].set_xlim(-.02,.7)
axs[1].set_title('b  NIMS measured scores fixed',loc='left')
axs[2].scatter(n['min'],[0,1],c='#655386',s=40);axs[2].set_yticks([0,1],n.player);axs[2].invert_yaxis();axs[2].set_xlim(0,.075)
axs[2].set_title('c  Complete H-C catalog',loc='left')
for i,row in n.iterrows():axs[2].text(row['min']+.002,i,f"{row['min']:.4f}",va='center',fontsize=8)
for ax in axs:ax.set_xlabel('Conditional Shapley allocation');ax.grid(axis='x',alpha=.15)
fig.tight_layout(rect=[0,0,1,.90]);fig.savefig(OUT/'completion_classes.png',dpi=350);fig.savefig(OUT/'completion_classes.pdf');plt.close(fig)
