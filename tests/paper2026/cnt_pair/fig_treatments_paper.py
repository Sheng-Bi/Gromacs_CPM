#!/usr/bin/env python3
"""Constant potential vs fixed uniform charge on the two-tube bundle.
The CNT_DATA environment variable selects the input data directory.
(a) surface gap (inter-axis distance minus 2R) at +20/+20 e for both treatments, 0-500 ps with a 0-10 ps inset from a 20 fs rerun;
(b) surface gap after equilibration (last 200 ps of each free run) against charge per tube;
(c) mean contact force, same contact-start protocol, with the fixed/CPM ratio;
(d) end views of the final frames, both treatments at +10, +20, +30 e."""
import os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.environ.get("CNT_DATA", HERE)   # run data (CNT_DATA overrides)
T = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "traces")   # COM traces (gmx traj -com on CNT1/CNT2) and az/ frames
box = np.array([8.0, 6.0, 4.91902]); R = 0.678

def trace(r):
    a = np.atleast_2d(np.loadtxt(f"{T}/{r}_tube1.xvg", comments=("#", "@"))); b = np.atleast_2d(np.loadtxt(f"{T}/{r}_tube2.xvg", comments=("#", "@")))
    n = min(len(a), len(b)); u = a[:n, 1:3] - b[:n, 1:3]; u -= box[:2] * np.round(u / box[:2]); return a[:n, 0], np.linalg.norm(u, axis=1)

from matplotlib.transforms import ScaledTranslation
fig = plt.figure(figsize=(6.5, 6.05))
L_, R_, WS = 0.100, 0.990, 0.33
H = 0.245                                   # one cell, as a fraction of the figure: every cell is this tall
TOP, GAP = 0.950, 0.085                     # gap between the (a-c) row and the snapshot block
gs_top = fig.add_gridspec(1, 3, left=L_, right=R_, top=TOP, bottom=TOP - H, wspace=WS)
_b = TOP - H - GAP
gs_bot = fig.add_gridspec(2, 3, left=L_, right=R_, top=_b, bottom=_b - 2 * H - 0.020, wspace=WS, hspace=0.020 / H)
ax = [fig.add_subplot(gs_top[0, k]) for k in range(3)]
axs = [fig.add_subplot(gs_bot[rr, k]) for rr in range(2) for k in range(3)]

def tag(a, letter, dx=-0.55, dy=0.11):
    """panel letter outside the axes, clear of the y label"""
    a.text(0.0, 1.0, f"({letter})", transform=a.transAxes + ScaledTranslation(dx, dy, fig.dpi_scale_trans),
           fontsize=10.0, fontweight="bold", ha="left", va="bottom")
# ---- (a) surface gap at +20/+20 e, both treatments: 0-500 ps (1 ps frames) with a 0-10 ps inset (20 fs rerun) ----
CP, FX = "#1864AB", "0.35"
t, d = trace("q20_free"); m = t <= 500; ax[0].plot(t[m], d[m] - 2 * R, "-", color=CP, lw=1.0, label="constant potential")
t, d = trace("fq20_free"); m = t <= 500; ax[0].plot(t[m], d[m] - 2 * R, "-", color=FX, lw=1.0, label="non-polarizable")
ax[0].set_xlabel("time (ps)", fontsize=9.0); ax[0].set_ylabel("surface gap $d - 2R$ (nm)", fontsize=9.0); GAPLIM = (0.15, 0.95)
ax[0].set_ylim(0.15, 1.06); ax[0].set_xlim(0, 500)
ax[0].legend(fontsize=8.0, frameon=False, loc="center left", bbox_to_anchor=(0.03, 0.375), handlelength=1.8, labelspacing=0.3)
ax[0].text(0.97, 0.06, "$+20\\,e$ per tube", transform=ax[0].transAxes, fontsize=8.25, ha="right", va="bottom")
axi = ax[0].inset_axes([0.36, 0.615, 0.60, 0.33])
for r, col in (("q20_dense", CP), ("fq20_dense", FX)):
    t, d = trace(r); m = t <= 10; axi.plot(t[m], d[m] - 2 * R, "-", color=col, lw=0.8)
axi.axhline(1.6172 - 2 * R, color="0.6", lw=0.5, ls=":"); axi.axhline(1.96 - 2 * R, color="0.6", lw=0.5, ls=":")
axi.set_xlim(0, 10); axi.set_ylim(0.10, 0.70); axi.set_xticks([0, 5, 10]); axi.set_yticks([0.2, 0.4, 0.6]); axi.tick_params(labelsize=8.0, direction="in", length=2.5, pad=1.5)
axi.text(0.96, 0.03, "20 fs frames", transform=axi.transAxes, fontsize=8.0, color="0.35", va="bottom", ha="right")
for sp in axi.spines.values(): sp.set_linewidth(0.6)
# ---- (b) equilibrated surface gap against charge --------------------------
def gap_eq(run, window=200.0):
    p = os.path.join(DATA, run, "pair_distance.dat")
    if os.path.exists(p):
        a = np.loadtxt(p, comments="#"); t, r = a[:, 0], a[:, 1]
    else:
        t, r = trace(run)
    m = t >= t[-1] - window; g = r[m] - 2 * R; return g.mean(), g.std()
Qc = [0, 10, 20, 30]; gc = np.array([gap_eq(f"q{q}_free") for q in Qc])   # +5 has no fixed-charge counterpart; +40 never equilibrates within 1 ns -- both left out
Qf = [10, 20, 30]; gf = np.array([gap_eq(f"fq{q}_free") for q in Qf])
ax[1].axhline(1.6172 - 2 * R, color="0.6", lw=0.6, ls=":"); ax[1].text(33, 1.6172 - 2 * R - 0.018, "van der Waals contact", fontsize=8.0, color="0.4", va="top", ha="right"); ax[1].axhline(1.96 - 2 * R, color="0.6", lw=0.6, ls=":"); ax[1].text(-2.2, 1.96 - 2 * R + 0.018, "solvent-separated", fontsize=8.0, color="0.4", va="bottom", ha="left"); ax[1].errorbar(Qc, gc[:, 0], yerr=gc[:, 1], fmt="o-", color=CP, ms=4.5, lw=1.2, capsize=2, label="constant potential", zorder=3)
ax[1].errorbar(Qf, gf[:, 0], yerr=gf[:, 1], fmt="s--", color=FX, ms=4.5, lw=1.2, capsize=2, label="non-polarizable", zorder=3)
ax[1].set_xlabel("charge per tube (e)", fontsize=9.0); ax[1].set_ylabel("mean surface gap (nm)", fontsize=9.0)
ax[1].set_xlim(-3, 34); ax[1].set_xticks([0, 10, 20, 30]); ax[1].set_ylim(*GAPLIM)
ax[1].legend(fontsize=8.0, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
print("  (b) gaps (last 200 ps): CPM " + ", ".join(f"+{q}: {g:.3f}" for q, (g, e) in zip(Qc, gc)) + "; fixed " + ", ".join(f"+{q}: {g:.3f}" for q, (g, e) in zip(Qf, gf)))
# ---- (c) mean force at contact against charge ---------------------------
# Q = 0 is a CPM reference, not a measured uniform-zero-charge control.
# +10/+20: average 20-120 ps; +30: average 200-500 ps; Q=0: average 5-100 ps.
import json
with open(os.path.join(DATA, "force_summary.json")) as stream:
    force_summary = json.load(stream)
Q = np.array([0, 10, 20, 30])
cpm = np.array([force_summary[k]["mean"] for k in ["q0_force", "cq10_force", "cq20_force", "cq30_force"]])
cpm_e = np.array([force_summary[k]["block_sem"] for k in ["q0_force", "cq10_force", "cq20_force", "cq30_force"]])
fix = np.array([force_summary[k]["mean"] for k in ["fq10_force", "fq20_force", "fq30_force"]])
fix_e = np.array([force_summary[k]["block_sem"] for k in ["fq10_force", "fq20_force", "fq30_force"]])
FORCE_TO_NN = 0.0016605390671738467
ax[2].errorbar(Q, cpm * FORCE_TO_NN, yerr=cpm_e * FORCE_TO_NN, fmt="o-", color="#1864AB", ms=6, lw=1.4, capsize=2.5, label="constant potential")
ax[2].errorbar(Q[1:], fix * FORCE_TO_NN, yerr=fix_e * FORCE_TO_NN, fmt="s--", color="0.35", ms=6, lw=1.4, capsize=2.5, label="non-polarizable")
ax[2].axhline(0, color="0.6", lw=0.6); ax[2].set_ylim(-8.0, 0.55); ax[2].set_xlim(-3, 34); ax[2].set_xticks([0, 10, 20, 30])
ax[2].set_xlabel("charge per tube (e)", fontsize=9.0); ax[2].set_ylabel("force at contact (nN)", fontsize=9.0)
# ---- (d-i) end-view snapshots of the final frames: constant potential (top) and fixed charge (bottom) at +10, +20, +30 e ----
def frame(run, fixed_q=None):
    L = open(os.path.join(DATA, run, "run.gro")).read().splitlines(); n = int(L[1]); bx = np.array([float(x) for x in L[2 + n].split()[:3]])
    res = np.array([l[5:10].strip() for l in L[2:2 + n]]); nm = np.array([l[10:15].strip() for l in L[2:2 + n]])
    X = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in L[2:2 + n]])
    if fixed_q is None:
        dd = np.loadtxt(os.path.join(DATA, run, "per_atom_charges.dat"), comments="#"); st = dd[:, 0].max(); g = dd[dd[:, 0] == st]; q = g[np.argsort(g[:, 1])][:, 2]
    else:
        q = np.full(1600, fixed_q)
    c1 = X[:800].mean(0)
    for k in (0, 1): X[:, k] -= bx[k] * np.round((X[:, k] - c1[k]) / bx[k])
    c = X[:1600, :2].mean(0)
    for k in (0, 1): X[:, k] -= bx[k] * np.round((X[:, k] - c[k]) / bx[k])
    ow = np.where((res == "SOL") & (nm == "OW"))[0]          # water: put each H in the minimum image of its own O
    for h in (ow + 1, ow + 2):
        d2 = X[h, :2] - X[ow, :2]; d2 -= bx[:2] * np.round(d2 / bx[:2]); X[h, :2] = X[ow, :2] + d2
    # rotate about the pair centre so the inter-axis vector lies along +x in every panel
    u = X[800:1600, :2].mean(0) - X[:800, :2].mean(0); dist = np.linalg.norm(u)
    th = -np.arctan2(u[1], u[0]); ct, st_ = np.cos(th), np.sin(th)
    v = X[:, :2] - c; X[:, 0] = c[0] + ct * v[:, 0] - st_ * v[:, 1]; X[:, 1] = c[1] + st_ * v[:, 0] + ct * v[:, 1]
    return X, res, nm, q, c, dist
snaps = [("q10_free", None), ("q20_free", None), ("q30_free", None),            # row 1: constant potential
         ("fq10_free", 10 / 800), ("fq20_free", 20 / 800), ("fq30_free", 30 / 800)]  # row 2: fixed uniform charge
QDEV = 0.02            # colour half-range of the per-carbon charge deviation (e); see the comment at the scatter
for j, (a, (run, fq)) in enumerate(zip(axs, snaps)):
    X, res, nm, q, c, dist = frame(run, fq)
    owi = np.where((res == "SOL") & (nm == "OW"))[0]; na = res == "NA"; cl = res == "CL"
    seg = np.concatenate([np.stack([X[owi, :2], X[owi + 1, :2]], 1), np.stack([X[owi, :2], X[owi + 2, :2]], 1)])
    keep = (np.abs(seg[:, 0, 0] - c[0]) < 2.7) & (np.abs(seg[:, 0, 1] - c[1]) < 2.4)
    a.add_collection(LineCollection(seg[keep], colors="#9EC5DE", linewidths=0.35, zorder=1, rasterized=True))
    # Colour the carbons by their charge MINUS the tube's own mean Q/N, not by the charge itself.
    # The absolute scale could not show the effect: the three columns sit at 0.0125, 0.025 and
    # 0.0375 e, so on one fixed +-0.05 e range each column occupied a different narrow slice of
    # the map and the polarization inside it was a few pale tones. The deviation is the quantity
    # that differs between the treatments and it happens to be the same size at every charge
    # (rms 0.008-0.010 e, extremes -0.040/+0.027), so one symmetric range serves all six panels:
    # the non-polarizable rows are flat at zero by construction and the constant-potential rows
    # show depletion on the facing sides (blue, down to -0.04 e) and accumulation on the far
    # sides (red). Carbons keep a faint outline so the flat rows still read as tubes.
    qd = (q.reshape(2, 800) - q.reshape(2, 800).mean(axis=1, keepdims=True)).ravel()
    sc = a.scatter(X[:1600, 0], X[:1600, 1], s=5, c=qd, cmap="coolwarm", vmin=-QDEV, vmax=QDEV,
                   edgecolors="0.55", linewidths=0.12, zorder=4)
    a.scatter(X[na, 0], X[na, 1], s=14, c="#1864AB", edgecolors="k", linewidths=0.25, zorder=5)
    a.scatter(X[cl, 0], X[cl, 1], s=14, c="#2B8A3E", edgecolors="k", linewidths=0.25, zorder=5)
    a.set_xlim(c[0] - 2.35, c[0] + 2.35); a.set_ylim(c[1] - 2.20, c[1] + 2.20); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    x1, x2 = c[0] - dist / 2 + R, c[0] + dist / 2 - R          # ruler across the gap between the two surfaces
    # Use an arrow without end caps to keep the facing carbon atoms visible.
    a.annotate("", xy=(x1, c[1]), xytext=(x2, c[1]), zorder=6,
               arrowprops=dict(arrowstyle="<|-|>", color="#212529", lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=6))
    print(f"    {run}: carbon charge {q.min():+.4f} .. {q.max():+.4f} e (mean {q.mean():+.4f}), "
          f"deviation {qd.min():+.4f} .. {qd.max():+.4f} (rms {qd.std():.4f})")
    a.text(c[0], c[1] + R + 0.10, f"{dist - 2 * R:.2f} nm", fontsize=8.44, ha="center", va="bottom", zorder=7,   # above the tubes
           bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none", alpha=0.6))   # faint background for legibility over the water lines
    if j < 3: a.set_title(f"$+{(10, 20, 30)[j]}\\,e$ per tube", fontsize=9.56, pad=3)
    if j % 3 == 0: a.set_ylabel("constant potential" if j == 0 else "non-polarizable", fontsize=8.25)
    for sp in a.spines.values(): sp.set_linewidth(0.6)
axs[0].scatter([], [], s=14, c="#1864AB", edgecolors="k", linewidths=0.25, label="Na$^+$"); axs[0].scatter([], [], s=14, c="#2B8A3E", edgecolors="k", linewidths=0.25, label="Cl$^-$")
axs[0].legend(fontsize=8.0, loc="lower left", frameon=True, handletextpad=0.2, borderpad=0.3).get_frame().set_linewidth(0.5)
cax = fig.add_axes([0.365, 0.075, 0.27, 0.013]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
cb.set_label("carbon charge $-$ tube mean $Q/N$ (e)", fontsize=8.62, labelpad=3); cb.set_ticks([-0.02, 0, 0.02])
cb.ax.set_xticklabels([r"$-0.02$", "0", r"$+0.02$"]); cb.ax.tick_params(labelsize=8.0, pad=2); cb.outline.set_linewidth(0.6)
for a, t in zip(ax, "abc"):
    a.tick_params(direction="in", top=True, right=True, labelsize=8.25, length=3.5, width=0.8)
    for sp in a.spines.values(): sp.set_linewidth(0.8)
    tag(a, t)
tag(axs[0], "d")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, "fig_treatments_paper." + ext), dpi=300)
print("  wrote fig_treatments_paper.png/pdf")
