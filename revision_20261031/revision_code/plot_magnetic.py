"""Publication figures from verified magnetocaloric audit outputs."""
from pathlib import Path
import json, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'magnetic_results';FIG=ROOT/'magnetic_figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42})
C1='#a75237';C2='#267a80';C3='#655386'


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=350,bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def main():
    fig,ax=plt.subplots(figsize=(9.5,3.4));ax.axis('off')
    boxes=[(.02,.64,.25,.22,'Audit material records\nComposition, endpoint and field'),
           (.37,.64,.25,.22,'Define the finite support game\nUtility and background score'),
           (.73,.64,.25,.22,'Declare known information\nMeasured scores or access values'),
           (.08,.13,.36,.24,'Calculate admissible conclusions\nShapley ranges and ranking certificates'),
           (.59,.13,.36,.24,'Specify the next evidence\nMissing supports, quantities or repeatability')]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',fc='#e9f1f1',ec=C2,lw=1.2))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)
    for start,end in [((.27,.75),(.37,.75)),((.62,.75),(.73,.75)),((.855,.64),(.26,.37)),((.44,.25),(.59,.25))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#435964','lw':1.3})
    fig.tight_layout();save(fig,'Fig1_audit_workflow')

    d=pd.read_csv(ROOT/'magnetic_data/nims_primary_2T_lowfield.csv')
    s=pd.read_csv(OUT/'nims_broad_supports.csv')
    fig,axs=plt.subplots(1,2,figsize=(9.5,4),gridspec_kw={'width_ratios':[1,1.6]})
    vals=[4/32,14/32,1]
    axs[0].bar(['HEA\n5 players','La-Fe-Si\n5 dopants','La-Fe-Si\nH-C subset'],vals,color=[C1,C2,C3])
    for i,(v,t) in enumerate(zip(vals,['4 / 32','14 / 32','4 / 4'])):axs[0].text(i,v+.03,t,ha='center')
    axs[0].set_ylim(0,1.12);axs[0].set_ylabel('Observed support coverage')
    axs[0].set_title('a  Actual magnetocaloric portfolios',loc='left')
    for i,row in s.iterrows():
        axs[1].hlines(i,row.min_Ttr_K,row.max_Ttr_K,color=C2,lw=2)
        part=d[d['mask'].eq(row['mask'])]
        axs[1].scatter(part.Ttr_K,np.repeat(i,len(part)),s=12,c=C2,alpha=.6)
    labels=[format(int(m),'05b')[::-1] for m in s['mask']]
    axs[1].set_yticks(range(len(s)),labels);axs[1].invert_yaxis()
    axs[1].set_ylabel('Presence bits: H, Co, Mn, C, Ce')
    axs[1].set_xlabel('Reported transition temperature (K)')
    axs[1].set_title('b  Temperature variation inside binary supports',loc='left')
    fig.tight_layout();save(fig,'Fig2_magnetocaloric_coverage')

    h=pd.read_csv(OUT/'hea_completion_bounds.csv');h=h[h.cap_class.eq('unit_cap')]
    b=pd.read_csv(OUT/'nims_broad_completion_bounds.csv')
    n=pd.read_csv(OUT/'nims_HC_completion_bounds.csv');n=n[n.completion_class.eq('score')]
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.7),gridspec_kw={'width_ratios':[1.4,1.2,1]})
    players=['Mn','Cr','Ni','Al','Cu']
    for mode,offset,color,label in [('score',.12,C1,'Scores fixed'),('access',-.12,C2,'Access values fixed')]:
        z=h[h.completion_class.eq(mode)].set_index('player').loc[players]
        y=np.arange(5)+offset
        axs[0].hlines(y,z['min'],z['max'],lw=2.4,color=color,label=label)
        axs[0].scatter(z['max'],y,c=color,s=10)
    axs[0].set_yticks(range(5),players);axs[0].invert_yaxis();axs[0].set_xlim(-.02,1.03)
    axs[0].legend(frameon=False,fontsize=7.5,loc='lower right');axs[0].set_title('a  HEA information classes',loc='left')
    for i,row in b.iterrows():
        axs[1].hlines(i,row['min'],row['max'],color=C2,lw=3);axs[1].scatter([row['min'],row['max']],[i,i],c=C2,s=12)
    axs[1].set_yticks(range(5),b.player);axs[1].invert_yaxis();axs[1].set_xlim(-.02,.7)
    axs[1].set_title('b  La-Fe-Si five dopants',loc='left')
    axs[2].scatter(n['min'],[0,1],c=C3,s=40)
    axs[2].set_yticks([0,1],n.player);axs[2].invert_yaxis();axs[2].set_xlim(0,.075)
    axs[2].set_title('c  Complete H-C catalog',loc='left')
    for i,row in n.iterrows():axs[2].text(row['min']+.002,i,f"{row['min']:.4f}",va='center',fontsize=8)
    for ax in axs:ax.set_xlabel('Conditional Shapley allocation');ax.grid(axis='x',alpha=.15)
    fig.tight_layout();save(fig,'Fig3_completion_bounds')

    q=pd.read_csv(OUT/'nims_HC_all_information_states.csv').fillna({'known_nonempty':'','certified_leader':''})
    rep=pd.read_csv(OUT/'nims_representation_comparison.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4.3),gridspec_kw={'width_ratios':[1.7,1]})
    labels=[]
    for i,row in q.iterrows():
        maps={'1':'H','2':'C','3':'HC'}
        label='empty' + (', '+', '.join(maps[s] for s in row.known_nonempty.split(';')) if row.known_nonempty else ' only')
        labels.append(label)
        for p,off,color in [('H',-.12,C2),('C',.12,C1)]:
            axs[0].hlines(i+off,row[p+'_min'],row[p+'_max'],color=color,lw=2.2,label=p if i==0 else None)
            axs[0].scatter([row[p+'_min'],row[p+'_max']],[i+off]*2,c=color,s=10)
        if row.certified_leader=='H':axs[0].text(.73,i,'H',va='center',ha='center',color=C3,fontweight='bold')
    axs[0].set_yticks(range(8),labels);axs[0].invert_yaxis();axs[0].set_xlim(-.01,.77)
    axs[0].set_xlabel('Marginal Shapley bounds');axs[0].legend(frameon=False,ncol=2,fontsize=8,loc='lower right')
    axs[0].set_title('a  Which H-C support scores are revealed?',loc='left')
    axs[0].text(.73,-.65,'Certified\nleader',ha='center',fontsize=7.5,color=C3)
    axs[1].bar(['Binary\npresence','Composition\namounts'],rep.mae_K,color=[C1,C2])
    axs[1].set_ylim(0,23);axs[1].set_ylabel('Source-grouped out-of-fold MAE (K)')
    axs[1].set_title('b  Representation diagnostic',loc='left')
    for i,x in enumerate(rep.mae_K):axs[1].text(i,x+.4,f'{x:.2f}',ha='center')
    fig.tight_layout();save(fig,'Fig4_ranking_and_amounts')

    a=pd.read_csv(ROOT/'revision_code/results/summary.csv')
    fig,axs=plt.subplots(2,3,figsize=(9.5,5.5),sharex=True,sharey=True)
    for i,family in enumerate(['scalar','two_objective']):
        for j,regime in enumerate(['none','independent','batch']):
            ax=axs[i,j]
            for arm,color,label in [('repeat',C1,'Repeat covered supports'),('expand',C2,'Cover missing supports')]:
                z=a[a.family.eq(family)&a.noise.eq(regime)&a.pooling.eq('mean')&a.arm.eq(arm)]
                ax.plot(z.budget,z.recovery,'o-',color=color,label=label,lw=1.6,markersize=3.5)
            ax.set_ylim(0,1.03);ax.grid(alpha=.15)
            ax.set_title(['No observation noise','Independent noise','Shared batch offset'][j],fontsize=9)
            if j==0:ax.set_ylabel(('Scalar utility' if i==0 else 'Two latent objectives')+'\nTop-player recovery')
            if i==1:ax.set_xlabel('Additional observations')
    axs[0,1].legend(frameon=False,ncol=2,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.4))
    fig.tight_layout(rect=[0,0,1,.95]);save(fig,'Fig5_measurement_coverage')

    hc=pd.read_csv(ROOT/'magnetic_data/nims_HC_subportfolio.csv')
    fig,ax=plt.subplots(figsize=(6.7,4.1));ax.axvspan(250,350,color='#edf2f6')
    for mask,color,label in [(0,'#747b82','La-Fe-Si background'),(1,C2,'H present'),(2,C1,'C present'),(3,C3,'H and C present')]:
        z=hc[hc['mask'].eq(mask)];ax.scatter(z.Ttr_K,z.DS2,s=24,c=color,label=label,alpha=.8)
    ax.set_xlabel('Reported transition temperature (K)');ax.set_ylabel('Peak magnetic entropy change at 2 T (J kg$^{-1}$ K$^{-1}$)')
    ax.legend(frameon=False,fontsize=8);fig.tight_layout();save(fig,'FigS1_HC_measured_properties')


if __name__=='__main__':main()
