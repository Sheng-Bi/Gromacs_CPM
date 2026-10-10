#!/usr/bin/env python
"""fig_gpu_validation: CPU/GPU electrode-charge validation figure.

Panel (a): per-atom electrode charges, CPU vs full-GPU, first common step
           of the F8 slit configuration, with y=x identity line and RMSD.
Panel (b): per-atom charge RMSD (log x) for the nine validated
           configurations, all from paper2026/gpuval3 (2026-09-28): the same
           tprs and step counts as the original F8 (13 Aug) and gpuval2
           (25 Aug) runs, rerun on the current 2020 binary on the local
           RTX 3090, because those predated the device charge solve and the
           potential-only psr kernel.

Outputs fig_gpu_validation.pdf (vector) and fig_gpu_validation.png (300 dpi)
alongside this script.
"""

import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpuval3"))
CPU_FILE = os.path.join(BASE, "slit", "cpu", "per_atom_charges.dat")
GPU_FILE = os.path.join(BASE, "slit", "gpu", "per_atom_charges.dat")
SUMMARY_FILES = [os.path.join(BASE, "gpuval3_summary.txt")]

BLUE = "#0072B2"  # single-series color (colorblind-safe palette, slot 1)
GRID = "#CCCCCC"

# ----------------------------------------------------------------- rcParams
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.6,
    "axes.labelsize": 9.6,
    "axes.titlesize": 9.6,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ------------------------------------------------------------- data loading
def load_charges(path):
    """Return dict step -> array of charges ordered by atom_id."""
    data = np.loadtxt(path, comments="#")
    steps = data[:, 0].astype(int)
    out = {}
    for s in np.unique(steps):
        sel = data[steps == s]
        sel = sel[np.argsort(sel[:, 1])]
        out[int(s)] = sel[:, 2]
    return out


cpu = load_charges(CPU_FILE)
gpu = load_charges(GPU_FILE)
common_steps = sorted(set(cpu) & set(gpu))
if not common_steps:
    raise SystemExit("No common step between CPU and GPU per-atom charge files")
step = common_steps[0]
qc, qg = cpu[step], gpu[step]
if qc.shape != qg.shape:
    raise SystemExit("CPU/GPU atom counts differ at step %d" % step)
rmsd_scatter = float(np.sqrt(np.mean((qc - qg) ** 2)))

# Parse the seven-configuration summary: lines like
#   slit  PASS  per-atom step 1: RMSD=1.996e-06 e  max|diff|=...
LABELS = {
    "slit": "MatInv slit",
    "slab3dc": "MatInv slab+3DC",
    "d1cv": "V(t)",
    "d2gcd": "Q(t)",
    "densecg": "Dense-matrix CG",
    "mfcg": "Matrix-free CG",
    "extlag": "Ext. Lagrangian",
    "e1flex": "Flexible electrodes",
    "single": "Single electrode",
}
ORDER = ["slit", "slab3dc", "d1cv", "d2gcd", "densecg", "mfcg", "extlag", "e1flex", "single"]
rmsd = {}
pat = re.compile(r"^(\w+)\s+PASS\s+per-atom step \d+:\s*RMSD=([0-9.]+e[+-]?\d+)\s*e")
for summary_file in SUMMARY_FILES:
    with open(summary_file) as fh:
        for line in fh:
            m = pat.match(line.strip())
            if m:
                rmsd[m.group(1)] = float(m.group(2))
missing = [k for k in ORDER if k not in rmsd]
if missing:
    raise SystemExit("Missing configurations in summary: %s" % missing)
values = [rmsd[k] for k in ORDER]
labels = [LABELS[k] for k in ORDER]


def sci_label(v):
    """Format value as m x 10^e mathtext string."""
    e = int(np.floor(np.log10(v)))
    m = v / 10.0 ** e
    return r"$%.2f\times10^{%d}$" % (m, e)


# ------------------------------------------------------------------- figure
fig = plt.figure(figsize=(6.5, 2.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35],
                      left=0.12, right=0.975, top=0.93, bottom=0.17,
                      wspace=0.62)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

# ------------------------------------------------------------- panel (a)
lim = 1.05 * float(np.max(np.abs(np.concatenate([qc, qg]))))
ax_a.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, zorder=1)
ax_a.scatter(qc, qg, s=5, color=BLUE, alpha=0.45, linewidths=0, zorder=2)
ax_a.set_xlim(-lim, lim)
ax_a.set_ylim(-lim, lim)
ax_a.set_aspect("equal")
ax_a.grid(True, color=GRID, lw=0.4, zorder=0)
ax_a.set_axisbelow(True)
ax_a.set_xlabel(r"CPU charge $q_i$ (e)")
ax_a.set_ylabel(r"GPU charge $q_i$ (e)")
# the empty lower-right triangle: at 9 pt the block ran into the parity line in the upper-left
ax_a.text(0.95, 0.05,
          "step %d,  $N$ = %d\nRMSD = %s e" % (step, len(qc),
                                               sci_label(rmsd_scatter)),
          transform=ax_a.transAxes, ha="right", va="bottom", fontsize=9.0)

# ------------------------------------------------------------- panel (b)
xmin, xmax = 1e-10, 1e-4
y = np.arange(len(values))
ax_b.barh(y, np.array(values) - xmin, left=xmin, height=0.62,
          color=BLUE, edgecolor="none", zorder=2)
ax_b.set_xscale("log")
ax_b.set_xlim(xmin, xmax)
ax_b.set_yticks(y)
ax_b.set_yticklabels(labels)
ax_b.invert_yaxis()  # first configuration on top
ax_b.grid(True, axis="x", color=GRID, lw=0.4, zorder=0)
ax_b.set_axisbelow(True)
ax_b.set_xlabel("Per-atom charge RMSD (e), log scale")
ax_b.tick_params(axis="y", length=0)
for yi, v in zip(y, values):
    ax_b.text(v * 1.6, yi, sci_label(v), va="center", ha="left",
              fontsize=8.4, color="black")
for spine in ("top", "right"):
    ax_a.spines[spine].set_visible(False)
    ax_b.spines[spine].set_visible(False)
# Panel letters top-left of each panel, in figure coordinates so they are
# never clipped by the y-axis label gutter.
for ax, letter, xoff in ((ax_a, "(a)", 0.10), (ax_b, "(b)", 0.125)):
    pos = ax.get_position()
    fig.text(pos.x0 - xoff, pos.y1 + 0.012, letter,
             ha="left", va="bottom", fontsize=10.0, fontweight="bold")

# ------------------------------------------------------------------- output
pdf = os.path.join(HERE, "fig_gpu_validation_paper.pdf")
png = os.path.join(HERE, "fig_gpu_validation_paper.png")
fig.savefig(pdf)
fig.savefig(png, dpi=300)
print("scatter step=%d  N=%d  RMSD=%.3e e" % (step, len(qc), rmsd_scatter))
for lab, v in zip(labels, values):
    print("%-22s %.3e e" % (lab, v))
print("wrote", pdf)
print("wrote", png)
