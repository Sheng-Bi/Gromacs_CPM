#!/usr/bin/env python3
"""Thomas-Fermi electrode model, one column, three panels.
(a) charge penetration into a three-layer graphene electrode vs l_TF, CPM-GROMACS
    (filled) against MetalWalls (open) on identical coordinates;
(b) single-layer capacitor: 1/C against l_TF^2 with the analytical slope;
(c) per-atom parity CPM-GROMACS vs MetalWalls at l_TF = 0.3 nm, both electrodes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paperstyle as PS

# ---- system cross-section inset, as in the throughput figure ------------------
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from render_atoms import read_gro, STYLE, elem_of, sphere

def system_inset(ax, gro, elec_res, ysel_lo, ysel_hi, box_frac, elec_col="#3A3A3A", elec_rad=0.075, scale=1.5, zlim=None):
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
    axi.set_xlim(*(zlim if zlim else (0, box[2]))); axi.set_ylim(0, box[0]); axi.set_aspect("equal")
    axi.set_xticks([]); axi.set_yticks([])
    for s_ in axi.spines.values(): s_.set_color("#B8B8B8"); s_.set_linewidth(0.5)
    return axi

HERE = os.path.dirname(os.path.abspath(__file__))
TF = os.path.normpath(os.path.join(HERE, "..", "tf"))
N = 1144; LAY = ["surface", "middle", "inner"]
KE = 1.43996              # e^2/(V nm) -> k_e in V nm / e ; slope = 2 k_e 4 pi / (N V_at)
VAT = 0.00878
def gmx(l):
    d = np.loadtxt(f"{TF}/multilayer/gmx_l{l}/per_atom_charges.dat", comments="#")
    s0 = d[:, 0].min(); q = d[d[:, 0] == s0]; return q[np.argsort(q[:, 1])][:, 2]
def mw(l):
    q = [float(x) for x in open(f"{TF}/multilayer/mw_l{l}/charges.out") if not x.startswith("#")]
    return np.array(q[:6864])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                               gridspec_kw=dict(left=0.085, right=0.985, top=0.93, bottom=0.155, wspace=0.34))
# ---- (a) layer penetration --------------------------------------------------
Ls = ["0.00", "0.10", "0.30"]; cols = ["#495057", "#1864AB", "#C92A2A"]
x = np.arange(3); w = 0.26
for j, (l, c) in enumerate(zip(Ls, cols)):
    g, m = gmx(l), mw(l)
    Qg, Qm = g[:3 * N].sum(), m[:3 * N].sum()
    sg = [100 * g[k * N:(k + 1) * N].sum() / Qg for k in range(3)]
    sm = [100 * m[k * N:(k + 1) * N].sum() / Qm for k in range(3)]
    a1.bar(x + (j - 1) * w, sg, w, color=c, alpha=0.85, label=rf"$l_\mathrm{{TF}}$ = {float(l):.1f} nm", zorder=2)
    a1.plot(x + (j - 1) * w, sm, "o", ms=4.5, mfc="white", mec="k", mew=0.8, zorder=3)
    print(f"  (a) l={l}: gmx {['%.1f'%v for v in sg]}  mw {['%.1f'%v for v in sm]}")
a1.plot([], [], "o", ms=4.5, mfc="white", mec="k", mew=0.8, label="MetalWalls")
a1.axhline(0, color="0.5", lw=0.6); a1.set_xticks(x); a1.set_xticklabels(LAY)
a1.set_ylabel("share of electrode charge (%)", fontsize=10.0)
a1.legend(fontsize=8.32, frameon=False, loc="upper right", ncol=1, handlelength=1.4, labelspacing=0.25)

# three-layer electrode the bars are measured on
ML = os.path.join(TF, "multilayer", "ml.gro")
system_inset(a1, ML, ["WaL", "WaU"], 2.60, 2.90, (0.44, 0.40, 0.34, 0.30), scale=1.9, zlim=(10.3, 19.9))
a1.text(0.61, 0.375, "three-layer electrodes", transform=a1.transAxes, fontsize=8.06, color="0.3", ha="center", va="top")

# ---- (b) capacitance vs l^2 -----------------------------------------------
ls = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.50]); Q = []
for l in ls:
    p = f"{TF}/capacitor/mfcg_l{l:.2f}/electrode_charge.dat"
    Q.append(float([ln for ln in open(p) if not ln.startswith(("#", "Step"))][-1].split()[1]))
Q = np.array(Q); invC = 4.0 / Q                                  # 4 V applied
slope = 2 * KE * 4 * np.pi / (N * VAT)
a2.plot(ls**2, invC, "o", ms=5, mfc="#E8590C", mec="none", zorder=3, label="CPM-GROMACS")
xx = np.linspace(0, (ls**2).max() * 1.05, 50)
a2.plot(xx, invC[0] + slope * xx, "-", color="0.25", lw=1.0, zorder=2,
        label="theoretical prediction")
a2.set_xlabel(r"$l_\mathrm{TF}^2$ (nm$^2$)", fontsize=10.0); a2.set_ylabel(r"$1/C$ (V e$^{-1}$)", fontsize=10.0)
a2.legend(fontsize=8.32, frameon=False, loc="upper left")
CAP = os.path.join(TF, "capacitor", "start.gro")
# below the fitted line, clear of it
system_inset(a2, CAP, ["WaL", "WaU"], 2.60, 2.90, (0.56, 0.085, 0.40, 0.20), scale=2.6, zlim=(10.2, 20.0))
a2.text(0.985, 0.305, "single-layer capacitor", transform=a2.transAxes, fontsize=8.06, color="0.3", ha="right")
print(f"  (b) slope predicted {slope:.4f}, fitted {np.polyfit(ls**2, invC, 1)[0]:.4f}")
g, m = gmx("0.30"), mw("0.30"); d = m - g
print(f"  per-atom at l=0.30: r={np.corrcoef(g, m)[0,1]:.8f} rms dq={np.sqrt((d**2).mean()):.1e} e")
for ax, t in ((a1, "a"), (a2, "b")):
    ax.tick_params(direction="in", top=True, right=True, labelsize=9.1, length=3)
    PS.tag(ax, t, dx=-0.50)
    for s in ax.spines.values(): s.set_linewidth(0.8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_tf_paper." + ext), dpi=300)
print("  wrote fig_tf_paper.pdf/.png")
