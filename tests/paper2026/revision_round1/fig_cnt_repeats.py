"""Individual +20 e CNT repeats and the exact statistics used in Figure 7."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'cnt_repeat_results'
a=json.loads((DATA/'replica-summary.json').read_text())
provenance=json.loads((DATA/'provenance.json').read_text())
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(6.8,2.8),layout='constrained')
colors={'CPM':'#1864AB','uniform':'0.35'}
styles=['-',(0,(5,2)),(0,(2,1)),(0,(5,1,1,1)),(0,(1,1))]
for model in colors:
 for r in sorted([r for r in a['individual_gaps'] if r['model']==model],key=lambda r:r['replica']):
  p=DATA/(r['case']+'.dat');assert hashlib.sha256(p.read_bytes()).hexdigest()==provenance[r['case']]['analysis_sha256']
  d=np.loadtxt(p);gap=d[:,1]-1.356
  assert np.isclose(gap[(d[:,0]>=800)&(d[:,0]<=1000)].mean(),r['mean_gap_nm'])
  for ax in axes:ax.plot(d[:,0],gap,color=colors[model],ls=styles[r['replica']-1],lw=.6,alpha=.7)
axes[0].axvspan(800,1000,color='0.7',alpha=.18)
axes[0].set(xlim=(0,1000),ylim=(.28,.69),xlabel='Time (ps)',ylabel=r'Surface gap $d-2R$ (nm)',title='(a) Full trajectories')
axes[1].set(xlim=(0,20),ylim=(.28,.69),xlabel='Time (ps)',title='(b) Initial separation')
for ax in axes:
 ax.text(.97,.89,'Uniform charge',color=colors['uniform'],ha='right',transform=ax.transAxes)
 ax.text(.97,.11,'Self-consistent',color=colors['CPM'],ha='right',transform=ax.transAxes)
from matplotlib.lines import Line2D
handles=[Line2D([],[],color='black',ls=s,lw=.8,label=str(i+1)) for i,s in enumerate(styles)]
axes[0].legend(handles=handles,title='Replica',ncol=5,fontsize=7,title_fontsize=8,loc='center',frameon=False,columnspacing=.8,handlelength=1.5)
for ext in ['pdf','png']:fig.savefig(ROOT/('fig_cnt_repeats.'+ext),dpi=300)
rows=[]
for i in range(1,6):
 vals=[]
 for model in ['CPM','uniform']:
  vals.append(next(r['mean_gap_nm'] for r in a['individual_gaps'] if r['model']==model and r['replica']==i))
 for model in ['CPM','uniform']:
  vals.append(-next(r['repulsive_force_nN'] for r in a['individual_contact_forces'] if r['model']==model and r['replica']==i))
 rows.append(str(i)+' & '+' & '.join(f'{v:.4f}' for v in vals)+r' \\')
(DATA/'replica-table-rows.tex').write_text('\n'.join(rows)+'\n')
print('\n'.join(rows))
