#!/usr/bin/env python3
"""Fig. 2 -- validation, one column, three panels:
(a) per-atom electrode charges, CPM-LAMMPS against CPM-GROMACS, on one
    equilibrated charged frame;
(b) 10 ns charging of the slab at 4 V, 336 K: matrix inversion, matrix-free CG,
    three extended-Lagrangian and three CPM-LAMMPS replicas;
(c, d) double-layer structure: anion (B) density at the positive electrode and
    cation (ring-centre) density at the negative one, from the last 1 ns of the 4 V slab
    trajectories of every approach (matrix inversion, matrix-free CG, extended
    Lagrangian x3, CPM-LAMMPS x3), profiles from ../edl/profiles/."""
import os, glob, numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paperstyle as PS
HERE = os.path.dirname(os.path.abspath(__file__)); P26 = os.path.normpath(os.path.join(HERE, ".."))
ELC = os.path.join(P26, "elcons"); RS = os.path.join(ELC, "remote_snap"); F0 = os.path.join(P26, "atomwise", "frame0")
NAT = 28944; FIT0, FIT1 = 200.0, 500.0
fig = plt.figure(figsize=(6.5, 5.35))
gs = gridspec.GridSpec(2, 2, figure=fig, left=0.115, right=0.980, top=0.945, bottom=0.088,
                       wspace=0.30, hspace=0.38)
ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])]
axR = fig.add_subplot(gs[1, 1])
# ---- (a) per-atom parity with LAMMPS -----------------------------------------
mi = np.loadtxt(os.path.join(F0, "q_matinv.dat")); mf = np.loadtxt(os.path.join(F0, "q_matfree.dat")); lm = np.loadtxt(os.path.join(F0, "q_lammps.dat")); NL = 1144
c = ax[0]; lo, hi = mi.min() * 1.08, mi.max() * 1.08; c.plot([lo, hi], [lo, hi], "-", color="0.35", lw=0.6, zorder=1)
c.plot(mi[:NL], lm[:NL], "o", ms=1.8, mfc="#1864AB", mec="none", alpha=0.7, zorder=3, rasterized=True, label="positive electrode")
c.plot(mi[NL:], lm[NL:], "s", ms=1.8, mfc="#74C0FC", mec="none", alpha=0.7, zorder=3, rasterized=True, label="negative electrode")
c.set_xlim(lo, hi); c.set_ylim(lo, hi); c.set_xlabel(r"$q_i$, CPM-GROMACS (e)", fontsize=10.0); c.set_ylabel(r"$q_i$, CPM-LAMMPS (e)", fontsize=10.0)
d = lm - mi
c.text(0.97, 0.30, f"$r$ = {np.corrcoef(mi, lm)[0,1]:.6f}\nRMS $\\Delta q_i$ = {np.sqrt((d**2).mean()):.1e} e", transform=c.transAxes, fontsize=8.0, va="bottom", ha="right")
c.legend(fontsize=8.0, frameon=False, loc="lower right", handletextpad=0.2, borderpad=0.3, markerscale=2.5)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from render_atoms import read_gro, STYLE, elem_of, sphere
SLAB = os.path.normpath(os.path.join(P26, "inputs", "slab", "start.gro"))
axi = inset_axes(c, width="100%", height="100%", loc="upper left", bbox_to_anchor=(0.055, 0.685, 0.46, 0.28), bbox_transform=c.transAxes, borderpad=0)
_r, _nm, _x, _box = read_gro(SLAB)
_i = np.where((_x[:, 1] > 2.58) & (_x[:, 1] < 2.90))[0]; _i = _i[np.argsort(-_x[_i, 1])]
_E = np.isin(_r, ["WaL", "WaU", "GrO"])
for _k in _i:
    _e = elem_of(_nm[_k])
    if _e in ("H", "MW"): continue
    _c, _rad = STYLE.get(_e, ("#909090", 0.06))
    if _E[_k]: _c, _rad = "#3A3A3A", 0.075
    axi.add_patch(plt.Circle((0, 0), 0, fc="none", ec="none")) if False else None
    sphere(axi, _x[_k, 2], _x[_k, 0], _rad * 1.5, _c, depth=1.0 - (_x[_k, 1] - 2.58) / 0.32, zorder=2, lw=0.06)
axi.set_xlim(10.15, 20.05); axi.set_ylim(0, _box[0]); axi.set_aspect("equal"); axi.set_xticks([]); axi.set_yticks([])   # crop to the capacitor, the rest of the 30 nm cell is vacuum
for _s in axi.spines.values(): _s.set_color("#B8B8B8"); _s.set_linewidth(0.5)
c.text(0.285, 0.665, "slab cell: BMIM-BF$_4$ / graphite", ha="center", va="top", transform=c.transAxes, fontsize=8.12, color="0.3")
# ---- (b) charging curves -----------------------------------------------------
def gmx(p):
    q = []
    for ln in open(p):
        s_ = ln.replace(",", " ").split()
        try: q.append((float(s_[0]) * 1e-6, float(s_[1])))
        except Exception: continue
    z = np.array(q); return z[:, 0], z[:, 1]
def lmp(p):
    q = []
    for ln in open(p):
        s_ = ln.split()
        if len(s_) >= 4 and s_[0].isdigit():
            try: q.append((int(s_[0]) * 1e-6, float(s_[3])))
            except Exception: continue
    z = np.array(q); k = np.r_[True, np.diff(z[:, 0]) > 0]; return z[k, 0], z[k, 1]
def smooth(t, y, w):
    m = np.convolve(y, np.ones(w) / w, mode="valid"); tc = t[w // 2: w // 2 + len(m)]; return np.r_[0.0, tc], np.r_[0.0, m]
b = ax[1]; W = 100; C = {"matinv": "#E8590C", "matfree": "#087F5B", "extlag": "#9C36B5", "lammps": "#1864AB"}
for key, path, lab in (("matinv", os.path.join(ELC, "long_matinv", "electrode_charge.dat"), "matrix inversion"), ("matfree", os.path.join(RS, "matfree_electrode_charge.dat"), "matrix-free CG")):
    t, Q = gmx(path); ts, ys = smooth(t, Q, W); b.plot(ts, ys, "-", color=C[key], lw=1.1, label=lab, zorder=4)
def band(series, key, lab, cadence):
    tend = min(t[-1] for t, _ in series); grid = np.arange(0.0, tend, cadence); M = np.vstack([np.interp(grid, t, Q) for t, Q in series])
    g, mu = smooth(grid, M.mean(0), W); _, sd = smooth(grid, M.std(0, ddof=1), W)
    b.fill_between(g, mu - sd, mu + sd, color=C[key], alpha=0.22, lw=0, zorder=3); b.plot(g, mu, "-", color=C[key], lw=1.1, zorder=4, label=lab)
band([gmx(os.path.join(ELC, d, "electrode_charge.dat")) for d in ("long_extlag", "long_extlag_r2", "long_extlag_r3")], "extlag", "ext. Lagrangian (3 rep.)", 1e-4)
band([lmp(f) for f in sorted(glob.glob(os.path.join(RS, "lmp_rep*.log")))], "lammps", "CPM-LAMMPS (3 rep.)", 1e-4)
b.set_xlim(0, 10); b.set_ylim(0, 18.5); b.set_xlabel("time (ns)", fontsize=10.0); b.set_ylabel("electrode charge $Q$ (e)", fontsize=10.0)
b.legend(fontsize=8.0, frameon=False, loc="lower right", handlelength=1.4, labelspacing=0.3)
# ---- (c) double-layer structure from the last 1 ns of every approach --------
# profiles: the final symmetric campaign if present, else the running one's first window
_cand = [os.path.join(P26, "edl", "sym", "profiles"), os.path.join(P26, "edl", "prelim200ps", "profiles"), os.path.join(P26, "edl", "profiles")]
EDL = os.environ.get("EDL_PROFILES") or next(d for d in _cand if os.path.isdir(d))
ZL, ZU = 10.976, 19.208   # electrode planes (nm): lower +2 V, upper -2 V
CM = {"matinv": ("#E8590C", "matrix inversion"), "matfree": ("#087F5B", "matrix-free CG"), "extlag": ("#9C36B5", "ext. Lagrangian"), "lammps": ("#1864AB", "CPM-LAMMPS")}
def method_of(name):
    n = os.path.basename(name).lower()
    for k in ("matinv", "matfree", "extlag"):
        if k in n: return k
    return "lammps" if ("lmp" in n or "lammps" in n) else None
prof = {}
for f in sorted(glob.glob(os.path.join(EDL, "*.tsv"))):
    k = method_of(f)
    if k is None: continue
    d = np.loadtxt(f, skiprows=1); prof.setdefault(k, []).append(d)
cL, cR = ax[2], axR; XMAX = 1.5      # upper: anions at the positive electrode; lower: cations at the negative
for k, runs in prof.items():
    z = runs[0][:, 0]; rc = np.mean([r[:, 1] for r in runs], 0); ra = np.mean([r[:, 2] for r in runs], 0)
    col, lab = CM[k]; nrep = f" ({len(runs)} rep.)" if len(runs) > 1 else ""
    if len(runs) > 1:   # replica spread: the scale on which two trajectories of one method differ
        sc_ = np.std([r[:, 2] for r in runs], 0, ddof=1); ss = np.std([r[:, 1] for r in runs], 0, ddof=1)
        cL.fill_between(z - ZL, ra - sc_, ra + sc_, color=col, alpha=0.25, lw=0, zorder=2)
        cR.fill_between(ZU - z, rc - ss, rc + ss, color=col, alpha=0.25, lw=0, zorder=2)
    cL.plot(z - ZL, ra, "-", color=col, lw=1.0, label=lab + nrep, zorder=3)
    cR.plot(ZU - z, rc, "-", color=col, lw=1.0, zorder=3)
    print(f"  (c) {k}: {len(runs)} run(s), anion peak at + {z[np.argmax(ra*(z>ZL)*(z<ZL+0.6))]-ZL:.3f} nm / {ra.max():.1f} nm^-3, cation peak at - {ZU-z[np.argmax(rc*(z<ZU)*(z>ZU-0.6))]:.3f} nm")
for c_, side, ymax in ((cL, "anions at the positive electrode (+2 V)", 53.0), (cR, "cations at the negative electrode ($-$2 V)", 26.0)):
    c_.set_xlim(0, XMAX); c_.set_ylim(0, ymax); c_.set_xticks([0, 0.5, 1.0, 1.5])
    c_.text(0.97, 0.93, side, transform=c_.transAxes, fontsize=8.0, color="0.3", ha="right", va="top")
    c_.set_ylabel(r"number density (nm$^{-3}$)", fontsize=10.0)
cL.set_xlabel("distance from the surface (nm)", fontsize=10.0); cR.set_xlabel("distance from the surface (nm)", fontsize=10.0)
if prof:
    cL.legend(fontsize=8.0, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.52), handlelength=1.4, labelspacing=0.3, borderpad=0.2)
else:
    cL.text(0.5, 0.5, "profiles pending", transform=cL.transAxes, ha="center", fontsize=8.75, color="0.5")
for a_, tag in zip(ax + [axR], "abcd"):
    a_.tick_params(direction="in", top=True, right=True, labelsize=8.75, length=3)
    for sp in a_.spines.values(): sp.set_linewidth(0.7)
    if tag.strip(): PS.tag(a_, tag, dx=-0.46)          # 10 pt bold, as in every other figure
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_validation_paper." + ext), dpi=300)
print("  wrote fig_validation_paper.pdf/.png")
