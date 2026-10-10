#!/usr/bin/env python3
"""Fig. 4 -- time-dependent protocols in the slit cell (2 x 2 panels).
(a) cyclic voltammetry: current I = dQ/dt against U(t) for symmetric -2..+2 V
    triangular sweeps at periods 20, 100 and 500 ps, cycle-averaged
    (tests/paper2026/timedep/cv5, 6 ns per run; the first 1 ns is discarded and every later
    cycle averaged: 250, 50 and 10 cycles);
(b) galvanostatic charge-discharge with voltage cut-offs (0 and 2 V) at |I| = 0.04,
    0.02 and 0.01 e/ps as time traces: the cell voltage the cycler compares with the
    cut-offs (its moving average, time constant 10, 20 and 50 ps) over two cycles of
    0.01 e/ps, each trace from its first stationary reversal to charging, offset
    vertically, with the 0 V and 2 V cut-offs of each labelled on the right;
(c) CV b-value analysis: |I| at U = 0 (mean of the two branches) against the scan rate for
    the CV sweeps in cv5 (the three of (a) and further periods down to 10 ps; the short 5 ps
    sweep is left out), with the standard error over
    cycles and the slopes b = 1 (capacitive) and b = 0.5 (transport-limited);
(d) GCD rate capability: capacitance from the discharge slope and from the discharge time
    against the current for every finished cut-off run, with the quasi-static capacitance;
    the currents of (b) in their colours (tests/paper2026/timedep/gcd_cutoff,
    analyze_gcdc.py).
Prints the numbers quoted in the text and writes RESULTS.md next to the CV data."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import paperstyle as PS

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.normpath(os.path.join(HERE, "..", "timedep", "cv5"))
DT = 0.002            # ps per step
T_SKIP = 1000.0       # ps of cycling discarded before averaging
PERIODS = ((20.0, "#C92A2A"), (100.0, "#1864AB"), (500.0, "#2B8A3E"))
V0, QMAX = 2.0, 24.0
C_QS = 12.1           # e/V, quasi-static capacitance Q(2 V)/2 V of the slit cell
lines = []


def fold(t, y, per, t_skip=T_SKIP):
    """Cycle-average y(t) over every whole period after t_skip; returns phase, mean, n."""
    npts = int(round(per / DT / 10))                       # one point per 10-step sample
    ph = np.linspace(0, per, npts, endpoint=False)
    c0, c1 = int(np.ceil(t_skip / per)), int(t[-1] // per)
    acc = []
    for c in range(c0, c1):
        m = (t >= c * per) & (t < (c + 1) * per)
        if m.sum() >= npts // 2:
            acc.append(np.interp(ph, t[m] - c * per, y[m]))
    return ph, np.mean(acc, axis=0), np.std(acc, axis=0), len(acc)


def load(run, name="electrode_charge.dat"):
    d = np.loadtxt(os.path.join(TD, run, name), comments="#", skiprows=(1 if name.startswith("electrode_charge") else 0))
    return d[:, 0] * DT, d[:, 1]


fig, axs = plt.subplots(2, 2, figsize=(6.5, 5.3),
                        gridspec_kw=dict(left=0.09, right=0.925, top=0.965,
                                         bottom=0.08, wspace=0.42, hspace=0.36))
(ax1, ax2), (ax3, ax4) = axs

# ---- (a) voltammograms -------------------------------------------------------
for per, col in PERIODS:
    t, Q = load(f"cv_T{int(per)}")
    ph, Qm, Qs, nc = fold(t, Q, per)
    U = V0 * (1 - np.abs(ph - per / 2) * 4 / per)
    w = max(5, len(ph) // 40)
    Qsm = np.convolve(np.r_[Qm[-w:], Qm, Qm[:w]], np.ones(w) / w, mode="same")[w:-w]
    I = np.gradient(Qsm, ph)
    rate = 4.0 / (per / 2)                                  # V/ps
    ax1.plot(np.r_[U, U[:1]], np.r_[I, I[:1]], "-", color=col, lw=1.2, label=f"{per:g} ps")
    # capacitance from the branches over |U| < 1 V: charging branch (first half) and discharging (second half)
    half = len(ph) // 2
    up = (np.abs(U[:half]) < 1.0); dn = (np.abs(U[half:]) < 1.0)
    Iu, Id = I[:half][up].mean(), I[half:][dn].mean()
    Cu, Cd = Iu / rate, -Id / rate
    lines.append(f"CV T={per:g} ps: {nc} cycles averaged; scan rate {rate:.3f} V/ps; mean branch currents {Iu:+.3f} / {Id:+.3f} e/ps over |U|<1 V; C = {Cu:.2f} / {Cd:.2f} e/V (mean {0.5*(Cu+Cd):.2f}); I range {I.min():+.2f}..{I.max():+.2f} e/ps; Q range {Qm.min():+.2f}..{Qm.max():+.2f} e; cycle-to-cycle sd of Q {Qs.mean():.3f} e")
ax1.axhline(0, color="0.6", lw=0.6, zorder=0); ax1.axvline(0, color="0.6", lw=0.6, zorder=0)
ax1.set_xlabel(r"$U(t)$ (V)", fontsize=10.0)
ax1.set_ylabel(r"current (e ps$^{-1}$)", fontsize=10.0)
ax1.legend(fontsize=8.25, frameon=False, loc="upper left", handlelength=1.6,
           labelspacing=0.28, borderpad=0.3, handletextpad=0.6, title="period", title_fontsize=8.25)

# ---- (b) GCD with voltage cut-offs: time traces at 0.04, 0.02 and 0.01 e/ps ------
# Each trace is the voltage the cycler compares with the cut-offs (moving average of the
# realized voltage, time constant 10, 20 and 50 ps), from the run's first stationary reversal
# to charging (analyze_gcdc.start_time). The window is two cycles of 0.01 e/ps; the faster
# currents fill it with consecutive cycles. The traces are offset vertically.
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "timedep", "gcd_cutoff")))
import analyze_gcdc as GA
TRACES = ((0.04, "gcdc_I0.04", "#C92A2A"), (0.02, "gcdc_I0.02", "#1864AB"), (0.01, "gcdc_I0.01", "#2B8A3E"))
OFFSET, N_SLOW = 3.0, 2
tr = {}
for I_, run, col in TRACES:
    t, Q, U, sw = GA.load(run)
    t0 = GA.start_time(run, sw)
    tr[run] = (t, GA.ema(t, U, GA.tau_of(run)), sw[(sw[:, 5] > 0) & (sw[:, 2] >= t0 - 1e-6), 2])
window = tr["gcdc_I0.01"][2][N_SLOW] - tr["gcdc_I0.01"][2][0]          # ps
yt, yl = [], []
for k, (I_, run, col) in enumerate(TRACES):
    t, Uf, ups = tr[run]
    base = (len(TRACES) - 1 - k) * OFFSET
    a = ups[0]
    m = (t >= a) & (t <= a + window)
    assert t[m][-1] >= a + window - 1.0, f"{run}: data end before the window does"
    stp = max(1, int(round(0.5 / (t[1] - t[0]))))                       # one point per 0.5 ps
    ax2.plot((t[m][::stp] - a) / 1000.0, base + Uf[m][::stp], "-", color=col, lw=0.9)
    for v in (0.0, 2.0):
        ax2.axhline(base + v, color="0.6", lw=0.6, zorder=0)
        yt.append(base + v); yl.append(f"{v:g} V")
    ax2.text(0.015, base + 2.1, f"{I_:g} e ps$^{{-1}}$", transform=ax2.get_yaxis_transform(),
             fontsize=8.25, color=col, va="bottom", ha="left")
    ncyc = int(np.sum((ups > a) & (ups <= a + window + 1e-6)))
    lines.append(f"GCD trace I={I_:g} e/ps (Utau {GA.tau_of(run):g} ps): from {a:.0f} ps, {ncyc} complete cycles in {window:.0f} ps")
ax2.set_xlim(0, window / 1000)
ax2.set_ylim(-0.3, (len(TRACES) - 1) * OFFSET + 2.9)
ax2.set_yticks(yt); ax2.set_yticklabels(yl)
ax2.set_xlabel("time (ns)", fontsize=10.0)
ax2.set_ylabel(r"cell voltage $U$", fontsize=10.0)

# ---- (c) CV b-value: |I| at U = 0 against the scan rate -------------------------
# Every CV sweep in cv5: the three periods of (a) (ikkem CPU build) and the further periods run on a
# local GPU with the same input. Each cycle is smoothed and differentiated as in (a), so the mean over
# cycles is the value of the cycle-averaged curve; the error bar is the standard error over cycles
# (neighbouring cycles are nearly uncorrelated, lag-1 autocorrelation <= 0.32, so the error bar is the
# block-averaged standard error below).
# One series (U = 0, mean of both branches); the +-1 V values go to RESULTS.md and the SI.
RATE_PERIODS = (10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)   # the 5 ps sweep (2 ns) is left out: too short
B_SPLIT = 100.0       # ps: b is fitted separately to the sweeps with T >= and T <= this period
U_AT = (0.0, 1.0, -1.0)
MIN_BLOCKS = 5        # shortest block series kept when blocking the cycles


def block_sem(x):
    """Standard error of the mean of x allowing for correlation between neighbouring cycles: the
    largest standard error over block lengths 1, 2, 4, ... that still leave MIN_BLOCKS blocks."""
    sem = x.std(ddof=1) / np.sqrt(len(x))
    b = 2
    while len(x) // b >= MIN_BLOCKS:
        n = len(x) // b
        sem = max(sem, x[:n * b].reshape(n, b).mean(axis=1).std(ddof=1) / np.sqrt(n))
        b *= 2
    return sem


def cycle_currents(t, Q, per, t_skip=T_SKIP):
    """Per cycle, the mean of the charging and discharging |I| at each U of U_AT (n_cycles x 3)."""
    npts = int(round(per / DT / 10)); ph = np.linspace(0, per, npts, endpoint=False)
    U = V0 * (1 - np.abs(ph - per / 2) * 4 / per); half = npts // 2; w = max(5, npts // 40)
    rows = []
    for c in range(int(np.ceil(t_skip / per)), int(t[-1] // per)):
        m = (t >= c * per) & (t < (c + 1) * per)
        if m.sum() < npts // 2:
            continue
        q = np.interp(ph, t[m] - c * per, Q[m])
        I = np.gradient(np.convolve(np.r_[q[-w:], q, q[:w]], np.ones(w) / w, mode="same")[w:-w], ph)
        rows.append([0.5 * (np.interp(u, U[:half], I[:half]) - np.interp(u, U[half:][::-1], I[half:][::-1]))
                     for u in U_AT])
    return np.array(rows)


def fit_b(idx):
    """Weighted fit of ln|I| against ln v at U = 0; the uncertainty is scaled up if chi2/dof > 1."""
    lv, li, sl = np.log(v[idx]), np.log(i0[idx]), s0[idx] / i0[idx]
    p, cov = np.polyfit(lv, li, 1, w=1 / sl, cov="unscaled")
    dof = len(idx) - 2
    chi2 = float(np.sum(((li - np.polyval(p, lv)) / sl) ** 2))
    return p[0], np.sqrt(cov[0, 0] * max(1.0, chi2 / dof if dof > 0 else 1.0)), chi2, dof


pts = []
for per in RATE_PERIODS:
    if os.path.isfile(os.path.join(TD, f"cv_T{int(per)}", "electrode_charge.dat")):
        x = cycle_currents(*load(f"cv_T{int(per)}"), per)
        pts.append((8.0 / per, per, len(x), x.mean(axis=0),
                    np.array([block_sem(x[:, j]) for j in range(x.shape[1])])))
pts.sort()
v = np.array([p[0] for p in pts]); T_of = np.array([p[1] for p in pts])
i0 = np.array([p[3][0] for p in pts]); s0 = np.array([p[4][0] for p in pts])
for vk, per, n, m, s in pts:
    lines.append(f"CV rate T={per:g} ps (v = {vk:.4f} V/ps, {n} cycles): |I| at U = 0, +1, -1 V = "
                 + ", ".join(f"{a:.4f} +- {e:.4f}" for a, e in zip(m, s))
                 + " e/ps (block-averaged standard error over cycles)")
for j, u0 in enumerate(U_AT):
    iq = np.array([p[3][j] for p in pts]); sq = np.array([p[4][j] for p in pts])
    bq = np.diff(np.log(iq)) / np.diff(np.log(v))
    eq = np.sqrt((sq[1:] / iq[1:]) ** 2 + (sq[:-1] / iq[:-1]) ** 2) / np.diff(np.log(v))
    lines.append(f"CV b-value U={u0:+.0f} V, neighbouring sweeps: "
                 + ", ".join(f"{T_of[k]:g}->{T_of[k + 1]:g} ps {bq[k]:.2f} +- {eq[k]:.2f}" for k in range(len(bq))))
slow = np.flatnonzero(T_of >= B_SPLIT); fast = np.flatnonzero(T_of <= B_SPLIT)
fits = [(idx, fit_b(idx)) for idx in (slow, fast) if len(idx) >= 3]
for idx, (bb, eb, chi2, dof) in fits:
    lines.append(f"CV b-value U=0 V, weighted fit over T = {T_of[idx].max():g}-{T_of[idx].min():g} ps: "
                 f"b = {bb:.3f} +- {eb:.3f} (chi2 {chi2:.2f} for {dof} dof)")
if os.path.isfile(os.path.join(TD, "cv_T20x", "electrode_charge.dat")):        # local GPU rerun of the CPU 20 ps sweep
    x = cycle_currents(*load("cv_T20x"), 20.0)
    lines.append(f"CV cross-check T=20 ps on the local GPU build ({len(x)} cycles): |I| at U = 0 = "
                 f"{x[:, 0].mean():.4f} +- {block_sem(x[:, 0]):.4f} e/ps (CPU build: "
                 f"{i0[T_of == 20.0][0]:.4f} +- {s0[T_of == 20.0][0]:.4f})")
xlo, xhi = v.min() / 1.6, v.max() * 1.6
ylo, yhi = 0.05, 3.0
vg = np.array([xlo, xhi])
ax3.plot(vg, i0[0] * (vg / v[0]), "--", color="0.55", lw=0.7, zorder=0)
ax3.plot(vg, i0[0] * np.sqrt(vg / v[0]), ":", color="0.55", lw=0.9, zorder=0)
# guide labels just clear of their lines: above-left of the dashed one, below the dotted one
xl1 = v[0] * (yhi / 7.5) / i0[0]   # low enough on the line to clear the fitted-exponent block
ax3.text(xl1, i0[0] * (xl1 / v[0]), "$b$ = 1", fontsize=8.25, color="0.45", ha="right", va="bottom")
xl2 = float(np.exp(np.mean(np.log(v))))                     # middle of the rate range, clear of the points
ax3.text(xl2, i0[0] * np.sqrt(xl2 / v[0]) * 0.72, "$b$ = 0.5", fontsize=8.25, color="0.45", ha="center", va="top")
ax3.errorbar(v, i0, yerr=s0, fmt="o-", color="k", ms=4.2, lw=0.9, elinewidth=0.9, capsize=2.2, zorder=3)
# measured exponents in one block, top left: fitted over the slow and the fast sweeps when there are
# enough of them, otherwise between neighbouring sweeps
if len(fits) == 2:
    blk = [f"$b$ = {bb:.2f} $\\pm$ {eb:.2f}  ({T_of[idx].max():g}–{T_of[idx].min():g} ps)" for idx, (bb, eb, _, _) in fits]
else:
    b = np.diff(np.log(i0)) / np.diff(np.log(v))
    blk = [f"$b$ = {b[k]:.2f}  ({T_of[k]:g}–{T_of[k + 1]:g} ps)" for k in range(len(b))]
ax3.text(0.04, 0.96, "\n".join(blk), transform=ax3.transAxes, fontsize=8.25, ha="left", va="top", linespacing=1.4)
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlim(xlo, xhi); ax3.set_ylim(ylo, yhi)
ax3.set_xlabel(r"scan rate $v$ (V ps$^{-1}$)", fontsize=10.0)
ax3.set_ylabel(r"$|I|$ at $U = 0$ (e ps$^{-1}$)", fontsize=10.0)

# ---- (d) GCD rate capability: discharge-time capacitance against current ---------
# C = I t_dis / dU over the 0-2 V window, the standard GCD capacitance; error bars are the standard
# error of the per-cycle values.
rc = [GA.fold(run, I_) for I_, run in sorted(GA.RUNS)
      if os.path.isfile(os.path.join(GA.HERE, run, "gcd_switches.dat"))]
cur = np.array([r["I"] for r in rc])
ax4.errorbar(cur, [r["C_tdis_plain"] for r in rc], yerr=[r["C_tdis_plain_sem"] for r in rc], fmt="o-",
             color="k", ms=4.2, lw=0.9, elinewidth=0.9, capsize=2.2, zorder=3)
for r in rc:
    lines.append(f"GCD rate I={r['I']:g} e/ps: {r['n']} cycles, period {r['T']:.0f} ps; C = I t_dis/(2 V) "
                 f"{r['C_tdis_plain']:.2f} +- {r['C_tdis_plain_sem']:.2f} e/V (standard error); "
                 f"C from discharge slope {r['C_dis_slope']:.2f} e/V")
ax4.axhline(C_QS, color="0.35", lw=0.7, ls=":", zorder=0)
ax4.text(0.97, C_QS + 0.3, "quasi-static", transform=ax4.get_yaxis_transform(), fontsize=8.25,
         color="0.35", va="bottom", ha="right")
ax4.set_xscale("log"); ax4.set_xlim(0.003, 0.7); ax4.set_ylim(0, 14.5)
ax4.set_xlabel(r"current $I$ (e ps$^{-1}$)", fontsize=10.0)
ax4.set_ylabel(r"capacitance $I\,t_\mathrm{dis}/\Delta U$ (e V$^{-1}$)", fontsize=10.0)

from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
plain = FuncFormatter(lambda x, _: f"{x:g}")
# decade-and-a-half spacing: at 8.75 pt the full 1-2-5 ladder ran the labels into each other
for ax, ticks in ((ax3, [0.01, 0.03, 0.1, 0.3, 1]), (ax4, [0.005, 0.02, 0.1, 0.5])):
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(plain)
    ax.xaxis.set_minor_formatter(NullFormatter())
ax3.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2])); ax3.yaxis.set_major_formatter(plain)
ax3.yaxis.set_minor_formatter(NullFormatter())
for ax in (ax1, ax2, ax3, ax4):
    ax.tick_params(direction="in", top=True, labelsize=8.75, length=3)
    ax.tick_params(which="minor", direction="in", top=True, length=1.8)
    for s in ax.spines.values(): s.set_linewidth(0.8)
for ax in (ax1, ax3, ax4):
    ax.tick_params(which="both", right=True)
ax2.tick_params(axis="y", left=True, right=True, labelleft=False, labelright=True)   # 0 V / 2 V of each trace on the right
for ax, letter in ((ax1, "a"), (ax2, "b"), (ax3, "c"), (ax4, "d")):
    PS.tag(ax, letter, dx=-0.50)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "fig_apps_v2_paper." + ext), dpi=300)
print("\n".join("  " + l for l in lines))
with open(os.path.join(TD, "RESULTS.md"), "w") as f:
    f.write("# Fig. 4 campaign (cv5), analysed by figures/fig_apps_v2_paper.py\n\n" + "\n".join("- " + l for l in lines) + "\n")
print("  wrote fig_apps_v2_paper.pdf/.png and cv5/RESULTS.md")
