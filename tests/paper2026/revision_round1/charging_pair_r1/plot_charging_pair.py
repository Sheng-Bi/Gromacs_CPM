"""Plot the paired 10 ns charge-ordering control from 100 ps bin means."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
x = np.loadtxt(HERE / 'charge_100ps_means.dat')
s = json.loads((HERE / 'window_statistics.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), layout='constrained')
colors = ['#1864AB', '#cf6924']
for col, (mode, label, color) in enumerate(zip(['predicted','fresh'], ['Predicted charges','Freshly solved charges'], colors), 1):
    for ax in axes:
        ax.plot(x[:,0], x[:,col], color=color, lw=1.2, label=label)
    axes[1].axhline(s[mode]['8-10ns'], color=color, ls='--', lw=.75, alpha=.7)
axes[0].axvspan(8, 10, color='0.5', alpha=.12)
axes[0].set(xlim=(0,10), ylim=(0,19), ylabel='Electrode charge (e)')
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
axes[1].set(xlim=(8,10), ylim=(16.5,18), xticks=[8,8.5,9,9.5,10])
for ax, letter in zip(axes,['a','b']):
    ax.set_xlabel('Time (ns)')
    ax.set_title('('+letter+')', loc='left', fontsize=10, fontweight='bold')
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)
    for spine in ax.spines.values(): spine.set_linewidth(.7)
for ext in ['pdf','png']:
    fig.savefig(HERE / ('fig_charging_order_pair.'+ext), dpi=300)
