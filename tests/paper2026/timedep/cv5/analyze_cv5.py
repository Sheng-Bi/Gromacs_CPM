#!/usr/bin/env python3
"""Standard electrochemical analyses of the cv5 campaign (2026-09-14).
CV : integral capacitance C = (oint |I| dU) / (2 v dU_window), and the power-law
     exponent b in I = a v^b at fixed potential (b = 1 capacitive, 0.5 diffusive).
GCD: capacitance from the discharge branch, C = I / (dU/dt) (linear fit over the
     middle 60 % of the branch) and C = I dt / dU (branch after the reversal jump),
     and the equivalent series resistance R = dU_jump / (2 I) from the voltage step
     at the current reversals.  Same folding as fig_apps_v2_paper.py (t > 1 ns).
Writes RESULTS_analysis.md next to the data."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DT, T_SKIP, V0, QMAX = 0.002, 1000.0, 2.0, 24.0
OHM = 6.2415   # 1 V ps / e = 6.2415 MOhm
out = []


def fold(t, y, per):
    npts = int(round(per / DT / 10)); ph = np.linspace(0, per, npts, endpoint=False)
    c0, c1 = int(np.ceil(T_SKIP / per)), int(t[-1] // per); acc = []
    for c in range(c0, c1):
        m = (t >= c * per) & (t < (c + 1) * per)
        if m.sum() >= npts // 2: acc.append(np.interp(ph, t[m] - c * per, y[m]))
    return ph, np.mean(acc, axis=0), len(acc)


def load(run, name="electrode_charge.dat"):
    d = np.loadtxt(os.path.join(HERE, run, name), comments="#", skiprows=(1 if name.startswith("electrode_charge") else 0))
    return d[:, 0] * DT, d[:, 1]


def cycle_currents(t, Q, per):
    """Per cycle, the mean of the charging and discharging |I| at U = 0, +1, -1 V, smoothed and
    differentiated as the folded curve (same as fig_apps_v2_paper.py): the mean over cycles is the
    value of the cycle-averaged curve, and the spread gives the standard error."""
    npts = int(round(per / DT / 10)); ph = np.linspace(0, per, npts, endpoint=False)
    U = V0 * (1 - np.abs(ph - per / 2) * 4 / per); half = npts // 2; w = max(5, npts // 40)
    rows = []
    for c in range(int(np.ceil(T_SKIP / per)), int(t[-1] // per)):
        m = (t >= c * per) & (t < (c + 1) * per)
        if m.sum() < npts // 2: continue
        q = np.interp(ph, t[m] - c * per, Q[m])
        I = np.gradient(np.convolve(np.r_[q[-w:], q, q[:w]], np.ones(w) / w, mode="same")[w:-w], ph)
        rows.append([0.5 * (np.interp(u, U[:half], I[:half]) - np.interp(u, U[half:][::-1], I[half:][::-1]))
                     for u in (0.0, 1.0, -1.0)])
    return np.array(rows)


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


# ------------------------------- CV --------------------------------------
# Every sweep present: 500, 100 and 20 ps (Fig. 4a, ikkem CPU build) and the further periods of
# Fig. 4c (local RTX 3090, CUDA build, same input).
CV_PERIODS = (1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0)   # the 5 ps sweep (2 ns) is left out: too short
B_SPLIT = 100.0       # ps: b fitted separately over T >= and T <= this period
rates, T_used, I_at, S_at = [], [], {0.0: [], 1.0: [], -1.0: []}, {0.0: [], 1.0: [], -1.0: []}
out.append("## CV\n")
for per in CV_PERIODS:
    if not os.path.isfile(os.path.join(HERE, f"cv_T{int(per)}", "electrode_charge.dat")): continue
    t, Q = load(f"cv_T{int(per)}"); ph, Qm, nc = fold(t, Q, per)
    U = V0 * (1 - np.abs(ph - per / 2) * 4 / per); v = 4.0 / (per / 2)
    w = max(5, len(ph) // 40); Qs = np.convolve(np.r_[Qm[-w:], Qm, Qm[:w]], np.ones(w) / w, mode="same")[w:-w]
    I = np.gradient(Qs, ph)
    dU = np.abs(np.gradient(U, ph)) * (ph[1] - ph[0])
    C_int = np.sum(np.abs(I) * dU) / (2 * v * 4.0)            # oint |I| dU / (2 v dU_window)
    half = len(ph) // 2
    up, dn = slice(0, half), slice(half, None)
    Iu = {u: np.interp(u, U[up], I[up]) for u in (-1.0, 0.0, 1.0)}          # charging branch (U rising)
    Id = {u: np.interp(u, U[dn][::-1], I[dn][::-1]) for u in (-1.0, 0.0, 1.0)}  # discharging branch
    C_pl = 0.5 * (I[up][np.abs(U[up]) < 1.0].mean() - I[dn][np.abs(U[dn]) < 1.0].mean()) / v   # plateau I/v over |U| < 1 V
    x = cycle_currents(t, Q, per); sem = np.array([block_sem(x[:, j]) for j in range(x.shape[1])])
    rates.append(v); T_used.append(per)
    for j, u in enumerate((0.0, 1.0, -1.0)): I_at[u].append(x[:, j].mean()); S_at[u].append(sem[j])
    out.append(f"- T = {per:g} ps (v = {v:.3f} V/ps, {nc} cycles): plateau capacitance {C_pl:.2f} e/V; integral capacitance {C_int:.2f} e/V; "
               f"|I| at U = -1, 0, +1 V: charging {abs(Iu[-1.0]):.3f} / {abs(Iu[0.0]):.3f} / {abs(Iu[1.0]):.3f}, "
               f"discharging {abs(Id[-1.0]):.3f} / {abs(Id[0.0]):.3f} / {abs(Id[1.0]):.3f} e/ps; "
               f"mean of both branches with the block-averaged standard error over cycles {x[:, 2].mean():.4f} +- {sem[2]:.4f} / "
               f"{x[:, 0].mean():.4f} +- {sem[0]:.4f} / {x[:, 1].mean():.4f} +- {sem[1]:.4f} e/ps")
lv, Tu = np.log(rates), np.array(T_used)
for u in (0.0, 1.0, -1.0):
    li, sl = np.log(I_at[u]), np.array(S_at[u]) / np.array(I_at[u])
    nb = [f"{Tu[k]:g}->{Tu[k + 1]:g} ps {(li[k + 1] - li[k]) / (lv[k + 1] - lv[k]):.2f} +- "
          f"{np.hypot(sl[k], sl[k + 1]) / abs(lv[k + 1] - lv[k]):.2f}" for k in range(len(lv) - 1)]
    fits = []
    for sel in (Tu >= B_SPLIT, Tu <= B_SPLIT):
        if sel.sum() < 3: continue
        p, cov = np.polyfit(lv[sel], li[sel], 1, w=1 / sl[sel], cov="unscaled")
        dof = int(sel.sum()) - 2; chi2 = float(np.sum(((li[sel] - np.polyval(p, lv[sel])) / sl[sel]) ** 2))
        fits.append(f"T = {Tu[sel].max():g}-{Tu[sel].min():g} ps b = {p[0]:.3f} +- "
                    f"{np.sqrt(cov[0, 0] * max(1.0, chi2 / dof if dof else 1.0)):.3f} (chi2 {chi2:.2f}, {dof} dof)")
    out.append(f"- b-value at U = {u:+.0f} V (mean of both branches): weighted fits " + ("; ".join(fits) or "(too few sweeps)")
               + "; neighbouring sweeps " + ", ".join(nb))

# ------------------------------- GCD -------------------------------------
out.append("\n## GCD\n")
for per in (500.0, 100.0, 20.0):
    t, Q = load(f"gcd_T{int(per)}"); tu, U = load(f"gcd_T{int(per)}", "electrode_potential.dat")
    n = min(len(t), len(tu)); ph, Um, nc = fold(t[:n], U[:n], per); _, Qm, _ = fold(t[:n], Q[:n], per)
    I = QMAX / (per / 2); half = len(ph) // 2; k = max(2, len(ph) // 50)
    # linear fits over the middle 60 % of each branch
    sc = slice(half // 5, half - half // 5); sd = slice(half + half // 5, len(ph) - half // 5)
    dUdt_c = np.polyfit(ph[sc], Um[sc], 1)[0]; dUdt_d = np.polyfit(ph[sd], Um[sd], 1)[0]
    C_c, C_d = I / dUdt_c, -I / dUdt_d
    # C = I dt / dU on the discharge branch after the jump (2 % of the period after the reversal to 2 % before the end)
    U1, U2 = Um[half + k:half + 2 * k].mean(), Um[-2 * k:-k].mean(); dt = ph[-k] - ph[half + k]
    C_dis = I * dt / (U1 - U2)
    jump_half = Um[half - 1 - k:half - 1].mean() - Um[half + 1:half + 1 + k].mean()
    jump_zero = Um[-k:].mean() - Um[1:1 + k].mean()
    R = 0.5 * (jump_half + abs(jump_zero)) / (2 * I)
    out.append(f"- T = {per:g} ps (I = {I:.3f} e/ps, {nc} cycles): C = I/(dU/dt) charging {C_c:.2f}, discharging {C_d:.2f} e/V; "
               f"C = I dt/dU over the discharge branch {C_dis:.2f} e/V; reversal jumps {jump_half:.2f} (T/2) and {abs(jump_zero):.2f} V (T); "
               f"R = dU/(2I) = {R:.2f} V ps/e = {R * OHM:.1f} MOhm; U at Qmax {Um[half]:+.2f} V, U at end {Um[-1]:+.2f} V")
txt = "# Standard CV / GCD analyses of the cv5 campaign (analyze_cv5.py)\n\n" + "\n".join(out) + "\n"
open(os.path.join(HERE, "RESULTS_analysis.md"), "w").write(txt)
print(txt)
