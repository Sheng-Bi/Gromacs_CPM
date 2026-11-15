"""Plot the full paired 10 ns charge trajectories from 100 ps bin means."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
x = np.loadtxt(HERE / 'charge_100ps_means.dat')
fig, ax = plt.subplots(figsize=(3.5, 3.5), layout='constrained')
ax.set_box_aspect(1)
for col, label, color in [(1, 'Predicted charges', '#1864AB'),
                          (2, 'Freshly solved charges', '#cf6924')]:
    ax.plot(x[:, 0], x[:, col], color=color, lw=1.2, label=label)
ax.axvspan(8, 10, color='0.5', alpha=.12)
ax.set(xlim=(0, 10), ylim=(0, 19), xlabel='Time (ns)',
       ylabel='Electrode charge (e)')
ax.legend(frameon=False, fontsize=8, loc='lower right')
ax.tick_params(direction='in', top=True, right=True, labelsize=8)
for spine in ax.spines.values():
    spine.set_linewidth(.7)
for ext in ['pdf', 'png']:
    fig.savefig(HERE / ('fig_charging_order_pair.' + ext), dpi=300)
