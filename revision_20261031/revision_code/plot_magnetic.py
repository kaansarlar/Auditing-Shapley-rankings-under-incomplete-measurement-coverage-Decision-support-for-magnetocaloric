"""Publication figures from verified magnetocaloric audit outputs."""
from pathlib import Path
import json, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MplPath

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'magnetic_results';FIG=ROOT/'magnetic_figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42})
C1='#a75237';C2='#267a80';C3='#655386'


def save(fig,name):
    fig.savefig(FIG/(name+'.png'),dpi=(600 if name=='Fig1_audit_workflow' else 350),bbox_inches='tight')
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    if name=='Fig1_audit_workflow':
        with plt.rc_context({'svg.fonttype':'none'}):fig.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def draw_audit_workflow():
    """Vector workflow; all connectors terminate outside card boundaries."""
    fig,ax=plt.subplots(figsize=(10.8,5.45))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set_xlim(0,10.8);ax.set_ylim(0,5.45);ax.axis('off')
    navy='#173e50';teal='#17777b';purple='#675488';orange='#b56b3c'
    ink='#203845';muted='#546672';line='#718792'
    # Group backgrounds make the two information assumptions visibly parallel.
    for x,w,color in [(.08,3.12,'#f3f7fa'),(3.89,3.16,'#f0f7f6'),(7.79,2.93,'#f7f5f9')]:
        ax.add_patch(FancyBboxPatch((x,.24),w,4.76,boxstyle='round,pad=0,rounding_size=.14',
                                  facecolor=color,edgecolor='none',zorder=0))
    for x,label,c in [(.25,'PORTFOLIO DEFINITION',navy),(4.07,'KNOWN INFORMATION',teal),(7.98,'AUDIT AND NEXT STEPS',purple)]:
        ax.text(x,5.22,label,color=c,fontsize=12.5,fontweight='bold',va='center')

    def card(x,y,w,h,c,num,title,body,formula=None):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.10',
                         facecolor='white',edgecolor=c,linewidth=1.2,zorder=3))
        # Small numbered badges and a colored title supply the visual hierarchy.
        ax.add_patch(Circle((x+.26,y+h-.27),.135,facecolor=c,edgecolor='none',zorder=4))
        ax.text(x+.26,y+h-.27,num,ha='center',va='center',fontsize=9.4,fontweight='bold',color='white',zorder=5)
        ax.text(x+.49,y+h-.16,title,ha='left',va='top',fontsize=13.8,fontweight='bold',
                color=c,linespacing=1.17,zorder=5)
        ax.text(x+.20,y+.58 if formula else y+(.73 if num=='4' else .55),body,ha='left',va='center',
                fontsize=13.5,color=ink,linespacing=1.40,zorder=5)
        if formula:ax.text(x+.20,y+.20,formula,ha='left',va='center',fontsize=12.4,color=muted,zorder=5)

    leftx=.25;leftw=2.78;midx=4.07;midw=2.80;rightx=7.98;rightw=2.54
    topy=2.97;lowy=.60;h=1.70
    card(leftx,topy,leftw,h,navy,'1','Audit the alloy\nrecords',
         'Composition and process\nEndpoint and field')
    card(leftx,lowy,leftw,h,navy,'2','Define the finite\ngame',
         'Choose utility / baseline\nMap supports to scores')
    ax.text(4.11,4.88,'3  Declare the information class',fontsize=11.2,fontweight='bold',color=teal)
    # Both alternatives retain the measured support scores.
    card(midx,topy,midw,h,teal,'A','Measured scores\nonly',
         'Fix measured scores\nUnknown scores in [0, M]',
         r'$q(S)=q_{\mathrm{obs}}(S),\quad S\ \mathrm{observed}$')
    card(midx,lowy,midw,h,teal,'B','Scores and access\nvalues fixed',
         'Keep measured scores\nAdd access equalities',
         r'$v_q(S)=v_{\mathrm{obs}}(S),\quad S\ \mathrm{observed}$')
    card(rightx,2.90,rightw,1.77,purple,'4','Compute bounds\nand certificates',
         'Allocation bounds\nRanking certificates')
    ax.text(rightx+.20,3.10,'Identified  /  unresolved',color=muted,fontsize=12.1,zorder=5)
    card(rightx,lowy,rightw,h,orange,'5','Specify further\nevidence',
         'Coverage / amounts\nReplication / process')

    # Ports are separated from cards by 0.08 coordinate units, even at arrow tips.
    # Only orthogonal connectors are used; none pass over a card or its text.
    def arrow(points):
        path=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=13,
                         linewidth=1.6,color=line,zorder=2,capstyle='round',joinstyle='round'))
    def wire(points):
        xs,ys=zip(*points);ax.plot(xs,ys,color=line,lw=1.6,zorder=2,solid_capstyle='round')
    lc=leftx+leftw/2;tc=topy+h/2;bc=lowy+h/2
    arrow([(lc,topy-.08),(lc,lowy+h+.08)])
    split=3.53
    wire([(leftx+leftw+.08,bc),(split,bc),(split,tc)])
    ax.add_patch(Circle((split,bc),.036,facecolor=line,edgecolor='none',zorder=2))
    arrow([(split,tc),(midx-.08,tc)])
    arrow([(split,bc),(midx-.08,bc)])
    merge=7.42;entry=3.53
    wire([(midx+midw+.08,tc),(merge,tc),(merge,bc),(midx+midw+.08,bc)])
    ax.add_patch(Circle((merge,entry),.036,facecolor=line,edgecolor='none',zorder=2))
    arrow([(merge,entry),(rightx-.08,entry)])
    rc=rightx+rightw/2
    arrow([(rc,2.90-.08),(rc,lowy+h+.08)])
    save(fig,'Fig1_audit_workflow')


def main():
    draw_audit_workflow()

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
    axs[0].legend(frameon=False,fontsize=7.5,loc='lower left',bbox_to_anchor=(0,1.10),ncol=2,borderaxespad=0);axs[0].set_title('a  HEA information classes',loc='left')
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
    axs[0].set_xlabel('Marginal Shapley bounds');axs[0].legend(frameon=False,ncol=2,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.10),borderaxespad=0)
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
