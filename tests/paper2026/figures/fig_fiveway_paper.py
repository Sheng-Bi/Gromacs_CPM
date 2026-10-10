"""Three-panel throughput figure on a 7-inch-wide canvas.
Optional --preview5 adds a conventional-GROMACS A100 bar and writes a separate preview."""
"""Constant-potential throughput, GROMACS vs CPM-LAMMPS (single column)."""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paperstyle as PS
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_atoms import read_gro, STYLE, elem_of, sphere

HERE = os.path.dirname(os.path.abspath(__file__))
PREVIEW5 = "--preview5" in sys.argv
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
# A100 campaign (tests/paper2026/fairbench/RESULTS_a100.md): every bar from the 10x window (_long keys)
VALS = argv[0] if len(argv) > 0 else os.path.join(HERE, "fiveway_values_a100.json")
VALS_SLAB = argv[1] if len(argv) > 1 else os.path.join(HERE, "fiveway_values_slab_a100.json")
WIN = "_long"   # timing window: "_long" = 20000 steps reset at 10000 (LAMMPS 10000+10000); "" = the 1000-step protocol
GRO  = os.path.normpath(os.path.join(HERE, "..", "inputs", "slit", "run1step.gro"))
GRO_SLAB = os.path.normpath(os.path.join(HERE, "..", "inputs", "slab", "start.gro"))
matplotlib.rcParams["hatch.linewidth"] = 0.5
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.36,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.top": True, "ytick.right": True, "savefig.dpi": 600})
BLUE, GREY, INK = "#3B7EA1", "#A8A8A8", "#1a1a1a"
D = json.load(open(VALS))
DS = json.load(open(VALS_SLAB)) if os.path.exists(VALS_SLAB) else None

BARS = [  # key, colour, hatch, label, configuration
    ("gmx_conv",    BLUE, None,   "GROMACS",     ""),
    ("gmx_cpm",     BLUE, None,   "CPM-GROMACS", ""),
    ("lmp_conp",    GREY, None,   "CPM-LAMMPS",  ""),
    ("gmx_cpm_gpu", BLUE, "/////","CPM-GROMACS", "A100"),
]
if PREVIEW5:
    BARS.insert(3, ("gmx_conv_gpu", BLUE, "/////", "GROMACS", "A100"))

# One-row layout for the single-column manuscript page: (a) and (b) nearly square,
# (c) wider on the right.  Fixed 7.0 x 2.9 in canvas (no tight bbox).
W, H = 6.5, 3.14
fig = plt.figure(figsize=(W, H))
def axes_in(left, bottom, width, height):
    return fig.add_axes([left / W, bottom / H, width / W, height / H])
PW, PH, PB = 1.62, 1.85, 0.97
ax  = axes_in(0.52, PB, PW, PH)
axs = axes_in(0.52 + PW + 0.50, PB, PW, PH)
axb = axes_in(0.52 + 2 * (PW + 0.50), PB, PW, PH)

def throughput_panel(ax, D, gro, tag, elec_res, ysel_lo, ysel_hi, ymax, inset_box, title):
    pos  = list(range(len(BARS)))
    vals = [D[k + WIN] for k, *_ in BARS]
    errs = [D.get(k + WIN + "_sd", 0.0) for k, *_ in BARS]
    for p, v, e, (key, col, hat, lab, cfg) in zip(pos, vals, errs, BARS):
        ax.bar(p, v, 0.58, color=("white" if hat else col), yerr=e, capsize=1.8, zorder=3,
               error_kw=dict(lw=0.6, ecolor="0.3", capthick=0.6), hatch=hat, edgecolor=col, linewidth=0.8)
        ax.text(p, v + e + ymax * 0.018, f"{v:.1f}", ha="center", va="bottom", fontsize=8.58, color=INK)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{lab} {cfg}".strip() for *_, lab, cfg in BARS], rotation=38, ha="right", rotation_mode="anchor", fontsize=8.06)
    for t in ax.get_xticklabels(): t.set_linespacing(1.25)
    ax.set_ylabel("performance  (ns/day)", fontsize=9.62)
    ax.set_xlim(-0.68, len(BARS) - 0.32); ax.set_ylim(0, ymax)
    ax.tick_params(axis="x", length=0)
    axi = inset_axes(ax, width="100%", height="100%", loc="upper center", bbox_to_anchor=inset_box, bbox_transform=ax.transAxes, borderpad=0)
    r, nm_, x, box = read_gro(gro)
    ysel = (x[:, 1] > ysel_lo) & (x[:, 1] < ysel_hi)
    idx = np.where(ysel)[0]; idx = idx[np.argsort(-x[idx, 1])]
    ELEC = np.isin(r, elec_res)
    for k in idx:
        e = elem_of(nm_[k])
        if e in ("H", "MW"): continue
        col, rad = STYLE.get(e, ("#909090", 0.06))
        if ELEC[k]:                      col, rad = "#3A3A3A", 0.075
        elif e == "O" and r[k] == "SOL": col = "#7FA8C9"
        d = 1.0 - (x[k, 1] - ysel_lo) / (ysel_hi - ysel_lo)
        sphere(axi, x[k, 2], x[k, 0], rad * 1.5, col, depth=d, zorder=2 + d, lw=0.06)
    axi.set_xlim(0, box[2]); axi.set_ylim(0, box[0]); axi.set_aspect("equal")
    axi.set_xticks([]); axi.set_yticks([])
    for s_ in axi.spines.values(): s_.set_color("#B8B8B8"); s_.set_linewidth(0.5)
    PS.tag(ax, tag.strip("()"), dx=-0.44)
    ax.text(0.5, 0.985, title, transform=ax.transAxes, fontsize=8.06, ha="center", va="top", color="0.35")

YF = 1.52   # head-room above the tallest bar for the inset and its title
ymax_a = max(D[k + WIN] + D.get(k + WIN + "_sd", 0) for k, *_ in BARS) * YF
# (a) slab first (the capacitor of the rest of the paper), (b) the slit cell
if DS is not None:
    ymax_s = max(DS[k + WIN] + DS.get(k + WIN + "_sd", 0) for k, *_ in BARS) * YF
    throughput_panel(ax, DS, GRO_SLAB, "(a)", ["WaL", "WaU", "GrO"], 2.58, 2.90, ymax_s, (0.08, 0.76, 0.84, 0.15), "slab cell, 28 944 atoms")
else:
    ax.text(0.5, 0.5, "slab values pending", transform=ax.transAxes, ha="center", fontsize=10.0, color="0.5"); ax.set_xticks([]); ax.set_yticks([])
throughput_panel(axs, D, GRO, "(b)", ["WAL1", "WAL2", "GRA"], 1.34, 1.66, ymax_a, (0.08, 0.76, 0.84, 0.15), "slit cell, 82 916 atoms")

# ================= panel (c): where the added cost goes ==================
# A100 node (cpmtiming/RESULTS_a100.txt): ms per step from dedicated wallcycle
# sub-counters (GROMACS) and the ELECTRODE fix timers (LAMMPS), CPM step minus
# the conventional step on the same hardware. RAW counters:
#   GROMACS CPU 64 thr.        : (1.227, 0.780, 0.549, 0.000, 2.412)  sum 2.555   conv step 3.993
#   GROMACS GPU 16 thr. (A100) : (0.251, 0.240, 0.074, 0.300, 0.517)  sum 0.866   conv step 1.556  (psr-only kernel, device solve)
#   LAMMPS 64 MPI              : (5.029, 0.908, 2.123, 0.000, 8.068)  sum 8.060   conv step 10.793
# The raw counters over-count where a region overlaps idle time the conventional
# step already had; the residual is folded into data handling so the segments sum
# exactly to the measured net. On the GPU the 0.30 ms wait for the electrode
# potentials is smaller than the conventional step's own idle GPU waits, so no
# wait is attributed (the row plots (0.251, 0.240, 0.026, 0, 0.517)).
ADDED = {   # net-attributed ms/step: (potential, solve, data handling, net GPU wait, net, conventional step)
    "CPU":               (1.227, 0.780, 0.406, 0.000, 2.412, 3.993),
    "A100":              (0.251, 0.240, 0.026, 0.000, 0.517, 1.556),
    "LAMMPS":              (5.029, 0.908, 2.131, 0.000, 8.068, 10.793),
}
SEG = [("electrode potential", "#3B7EA1"),
       ("charge solve",        "#D08770"),
       ("data handling",       "#C6C6C6")]
xs = np.arange(len(ADDED))
for x, (lab, vals) in zip(xs, ADDED.items()):
    bottom = 0.0
    for (segname, segcol), v in zip(SEG, vals[:4]):
        if v <= 0:
            continue
        axb.bar(x, v, 0.58, bottom=bottom, color=segcol, edgecolor="white",
                linewidth=0.5, zorder=3)
        bottom += v
    axb.plot([x - 0.34, x + 0.34], [vals[4], vals[4]], color="#1a1a1a", lw=1.2, zorder=5)
    axb.text(x, vals[4] + 0.25, f"{vals[4]:.1f}\n(conv. {vals[5]:.1f})", va="bottom", ha="center",
             fontsize=8.19, color=INK, linespacing=1.15)
axb.set_xticks(xs)
axb.set_xticklabels(list(ADDED), fontsize=8.06)
for t in axb.get_xticklabels(): t.set_linespacing(1.25)
axb.set_ylabel("added wall time  (ms/step)", fontsize=9.62)
axb.set_xlim(-0.68, len(ADDED) - 0.32); axb.set_ylim(0, 12.0)
axb.tick_params(axis="x", length=0)
PS.tag(axb, "c", dx=-0.44)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="white", lw=0.5) for _, c in SEG]
axb.legend(handles, [n for n, _ in SEG], fontsize=8.0, frameon=False, ncol=1,
           loc="upper left", bbox_to_anchor=(0.01, 0.99), handlelength=1.1,
           handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)

if PREVIEW5:
    out = os.path.join(HERE, "fig_fiveway_paper_preview5.png")
    fig.savefig(out, dpi=260); print("  wrote", out, "-- 5-bar preview")
else:
    out = os.path.join(HERE, "fig_fiveway_paper.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=260)
    print("  wrote", out, "-- single-row manuscript layout")
