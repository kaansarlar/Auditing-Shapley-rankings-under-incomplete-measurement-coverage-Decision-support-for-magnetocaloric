"""Plot exploratory representation sensitivity and correlated-block ablations."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'magnetic_figures';O.mkdir(exist_ok=True)
a=pd.read_csv(R/'matminer_results/representation_metrics.csv');b=pd.read_csv(R/'matminer_results/partition_comparisons.csv')
fig,axs=plt.subplots(1,2,figsize=(10,4))
for name,label,col in [('binary','Binary','#a75237'),('amounts','Amounts','#267a80'),('matminer_host','Host descriptors + H,C','#655386'),('combined','Combined','#b49237')]:
 z=a[a.representation.eq(name)];axs[0].plot(range(6),z.mae_K,marker='o',label=label,c=col,lw=1.5,ms=4)
axs[0].set_xticks(range(6),['Primary','S1','S2','S3','S4','S5']);axs[0].set_ylabel('Out-of-fold MAE (K)');axs[0].set_xlabel('Source-grouped partition');axs[0].legend(frameon=False,fontsize=8)
axs[0].set_title('a  Fixed representations across partitions',loc='left',fontsize=10)
cols=[c for c in b if c.startswith('drop_')]
for i,c in enumerate(cols):
 vals=b[c].to_numpy();axs[1].scatter(np.repeat(i,6)+np.linspace(-.12,.12,6),vals,s=24,c='#655386');axs[1].plot([i-.25,i+.25],[vals.mean()]*2,c='black',lw=2)
axs[1].axhline(0,color='gray',ls='--',lw=1)
axs[1].set_xticks(range(4),['d valence','d unfilled','Radius','Electro-\nnegativity']);axs[1].set_ylabel('MAE after block removal minus combined (K)')
axs[1].set_title('b  Correlated descriptor block ablations',loc='left',fontsize=10)
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(O/('FigS2_matminer_sensitivity.'+ext),dpi=350,bbox_inches='tight')
plt.close(fig)
