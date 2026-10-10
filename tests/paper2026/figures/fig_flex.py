#!/usr/bin/env python3
"""Flexible electrodes: induced charge vs electrode separation at constant 4 V
in the pure-electrode capacitor, matrix-free CG against per-geometry matrix
inversion. Standalone figure (was panel c of the applications figure), drawn at the
half text width the SI places it at (0.5\\textwidth = 3.185 in), so a point here is a
point on the page."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.normpath(os.path.join(HERE, "..", "flex_static", "matinv_extended.txt"))
d = np.loadtxt(E1, comments="#")
c = np.loadtxt(E1.replace("matinv_extended.txt", "matinv_comparison.txt"), comments="#", skiprows=1)[:, 1:4]
d = np.vstack([d, c]); d = d[np.argsort(d[:, 0])]
static = d[:, 0] < 8.5
print(f"  {len(d)} geometries, d = {d[0,0]}-{d[-1,0]} nm; Q*d spread = {np.ptp(d[:,1]*d[:,0])/np.mean(d[:,1]*d[:,0])*100:.2f}%")

fig, ax = plt.subplots(figsize=(3.185, 2.55),
                       gridspec_kw=dict(left=0.165, right=0.965, top=0.965, bottom=0.175))
ax.plot(d[:, 0], d[:, 1], "o", ms=4.2, mfc="none", mec="#212529", mew=0.9, label="matrix inversion", zorder=4)
ax.plot(d[static, 0], d[static, 2], "s", ms=4.2, mfc="none", mec="#087F5B", mew=0.9, label="matrix-free CG, static", zorder=3)
ax.plot(d[~static, 0], d[~static, 2], "-", lw=1.3, color="#087F5B", label="matrix-free CG, pull", zorder=2)
ax.set_xlabel("electrode separation $d$ (nm)", fontsize=9.0)
ax.set_ylabel(r"induced charge $Q$ (e)", fontsize=9.0)
ax.tick_params(direction="in", top=True, right=True, labelsize=8.0, length=3)
for s in ax.spines.values(): s.set_linewidth(0.8)
lg = ax.legend(fontsize=8.0, frameon=True, loc="upper right", handlelength=1.4, borderpad=0.4, labelspacing=0.3)
lg.get_frame().set_linewidth(0.6); lg.get_frame().set_edgecolor("0.6")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_flex." + ext), dpi=300)
print("  wrote fig_flex.pdf/.png")
