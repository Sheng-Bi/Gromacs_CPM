#!/usr/bin/env python3
"""Impedance figure, one column: (a) the sinusoidal drive and the charge
response at the fastest (200 GHz) and slowest (0.5 GHz) frequencies, time in
units of the period, with the fitted sinusoids; (b) the Nyquist plot."""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paperstyle as PS

# ---- system cross-section inset, as in the throughput figure ------------------
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from render_atoms import read_gro, STYLE, elem_of, sphere

def system_inset(ax, gro, elec_res, ysel_lo, ysel_hi, box_frac, elec_col="#3A3A3A", elec_rad=0.075, scale=1.5):
    """Draw a thin slab of the simulation cell (z horizontal, x vertical) inside ax."""
    axi = inset_axes(ax, width="100%", height="100%", loc="upper left",
                     bbox_to_anchor=box_frac, bbox_transform=ax.transAxes, borderpad=0)
    r, nm_, x, box = read_gro(gro)
    idx = np.where((x[:, 1] > ysel_lo) & (x[:, 1] < ysel_hi))[0]
    idx = idx[np.argsort(-x[idx, 1])]
    ELEC = np.isin(r, elec_res)
    for k in idx:
        e = elem_of(nm_[k])
        if e in ("H", "MW"): continue
        col, rad = STYLE.get(e, ("#909090", 0.06))
        if ELEC[k]:                      col, rad = elec_col, elec_rad
        elif e == "O" and r[k] == "SOL": col = "#7FA8C9"
        d = 1.0 - (x[k, 1] - ysel_lo) / (ysel_hi - ysel_lo)
        sphere(axi, x[k, 2], x[k, 0], rad * scale, col, depth=d, zorder=2 + d, lw=0.06)
    axi.set_xlim(0, box[2]); axi.set_ylim(0, box[0]); axi.set_aspect("equal")
    axi.set_xticks([]); axi.set_yticks([])
    for s_ in axi.spines.values(): s_.set_color("#B8B8B8"); s_.set_linewidth(0.5)
    return axi
HERE = os.path.dirname(os.path.abspath(__file__)); EIS = os.path.normpath(os.path.join(HERE, "..", "eis"))
sys.path.insert(0, EIS); import analyze_eis as E
DT = 0.002; V0 = 0.5

def load(run, T):
    p = os.path.join(EIS, run, "electrode_charge.dat"); p = p if os.path.exists(p) else p + ".gz"
    d = np.loadtxt(p, skiprows=1); t, Q = d[:, 0] * DT, d[:, 1]
    ncyc = int(t[-1] // T); m = (t >= T) & (t < ncyc * T)
    w = 2 * np.pi / T; tc = t[m] - t[m][0]
    A = np.vstack([np.sin(w * t[m]), np.cos(w * t[m]), np.ones(m.sum()), tc / tc[-1], (tc / tc[-1]) ** 2]).T
    c = np.linalg.lstsq(A, Q[m], rcond=None)[0]
    return t, Q, c, w

W_IN, H_IN = 6.5, 3.25; L, R = 0.12, 0.475                              # (a) occupies the left column
_bot, _topE = 0.13, 0.93
PH = (_topE - _bot) * H_IN                                              # (b): square, full column height (in)
ax2 = fig_ax2 = None
fig = plt.figure(figsize=(W_IN, H_IN))
ax2 = fig.add_axes([0.985 - PH / W_IN, _bot, PH / W_IN, _topE - _bot])   # (b): square, right column
axU = fig.add_axes([L, _topE - 0.13, R - L, 0.13])                       # (a) upper: the drive
ax1 = fig.add_axes([L, _bot, R - L, _topE - 0.13 - 0.045 - _bot])        # (a) lower: the response
# ---- (a) drive and response, two cycles, t in periods --------------------------
# five frequencies spanning the range; folded mean (circles) and fitted sinusoid (line)
SEL = (("eis_p2000", 2000.0, 1, "0.5 GHz"), ("eis_p500", 500.0, 1, "2 GHz"), ("eis_p100", 100.0, 5, "10 GHz"), ("eis_p20", 20.0, 20, "50 GHz"), ("eis_p5", 5.0, 50, "200 GHz"))
cmap = plt.get_cmap("viridis"); cols = [cmap(v) for v in np.linspace(0.0, 0.92, len(SEL))]
xx0 = np.linspace(0, 2, 400)
axU.plot(xx0, V0 * np.sin(2 * np.pi * xx0), "-", color="0.35", lw=1.1)
axU.set_xlim(0, 2); axU.set_ylim(-0.78, 0.78); axU.set_xticks([0, 0.5, 1, 1.5, 2]); axU.set_xticklabels([])
axU.set_yticks([-0.5, 0, 0.5]); axU.set_ylabel("$U$ (V)", fontsize=9.75, labelpad=1)
axU.tick_params(direction="in", top=True, right=True, labelsize=9.1, length=3)
axU.axhline(0, color="0.85", lw=0.5, zorder=0)
for (run, T, c0, lab), col in zip(SEL, cols):
    t, Q, c, w = load(run, T)
    ncyc = int(t[-1] // T); nph = 100; ph = (np.arange(nph) + 0.5) / nph
    tw = t[(t >= T) & (t < ncyc * T)]; tau = (tw - tw[0]) / (tw[-1] - tw[0])
    Qd = Q[(t >= T) & (t < ncyc * T)] - (c[2] + c[3] * tau + c[4] * tau ** 2)
    idx = np.minimum((np.mod(tw, T) / T * nph).astype(int), nph - 1)
    fold = np.bincount(idx, Qd, nph) / np.bincount(idx, None, nph)
    sel = slice(1, None, 4)
    ax1.plot(np.r_[ph, ph + 1][sel], np.r_[fold, fold][sel], "o", ms=3.0, mfc="none", mec=col, mew=0.7, alpha=0.75, zorder=5)
    xx = np.linspace(0, 2, 400)
    ax1.plot(xx, c[0] * np.sin(2 * np.pi * xx) + c[1] * np.cos(2 * np.pi * xx), "-", color=col, lw=1.3, zorder=4, label=lab)
    amp = np.hypot(c[0], c[1]); phs = np.degrees(np.arctan2(c[1], c[0]))
    print(f"  {run}: {lab}: amplitude {amp:.3f} e, phase {phs:+.1f} deg, folded over {ncyc - 1} cycles")
ax1.set_xlim(0, 2); ax1.set_ylim(-9.4, 7.4); ax1.set_xlabel("time (periods)", fontsize=10.0); ax1.set_ylabel("charge response $Q(t) - Q_0$ (e)", fontsize=10.0)
ax1.axhline(0, color="0.8", lw=0.5, zorder=0)
ax1.legend(fontsize=8.0, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3,
           handlelength=1.1, handletextpad=0.4, columnspacing=0.7, borderpad=0.2)
# ---- (b) Nyquist ------------------------------------------------------------------
rows = [l.split() for l in open(os.path.join(EIS, "eis_results.tsv")) if l.startswith("eis_p")]
main = sorted([r for r in rows if "a025" not in r[0]], key=lambda r: float(r[1]))
f = np.array([float(r[1]) for r in main]); ReZ = np.array([float(r[9]) for r in main]); nImZ = np.array([float(r[10]) for r in main])
ax2.plot([0, 30], [0, 30], "--", color="0.6", lw=0.7, zorder=1)
ax2.plot(ReZ, nImZ, "o-", ms=3.6, lw=1.0, color="#1864AB", zorder=3)
ax2.annotate("0.5 GHz", (ReZ[0], nImZ[0]), fontsize=8.06, xytext=(6, -1), textcoords="offset points", color="0.2")
ax2.annotate("200 GHz", (ReZ[-1], nImZ[-1]), fontsize=8.06, xytext=(20, 10), textcoords="offset points", color="0.2",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45", shrinkA=0, shrinkB=2))
ax2.set_xlim(0, 30); ax2.set_ylim(0, 30); ax2.set_aspect("equal"); ax2.set_xticks([0, 10, 20, 30]); ax2.set_yticks([0, 10, 20, 30])
ax2.set_xlabel(r"Re $Z$ (V ps e$^{-1}$)", fontsize=10.0); ax2.set_ylabel(r"$-$Im $Z$ (V ps e$^{-1}$)", fontsize=10.0)
for ax, tag in ((ax1, "a"), (ax2, "b")):
    ax.tick_params(direction="in", top=True, labelsize=9.1, length=3)
    for s in ax.spines.values(): s.set_linewidth(0.8)
PS.tag(axU, "a", dx=-0.50); PS.tag(ax2, "b", dx=-0.50)
SLIT = os.path.normpath(os.path.join(HERE, "..", "inputs", "slit", "run1step.gro"))
# below the 45 deg line, clear of it: at the inset's top edge (y = 0.215) the line sits at x = 0.215
system_inset(ax2, SLIT, ["WAL1", "WAL2", "GRA"], 1.34, 1.66, (0.44, 0.085, 0.54, 0.13))
ax2.text(0.985, 0.235, "slit cell: Na$_2$SO$_4$(aq) / graphite", transform=ax2.transAxes, fontsize=8.06, color="0.3", ha="right")
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_eis_main_paper." + ext), dpi=300)
print("  wrote fig_eis_main_paper.pdf/.png")
