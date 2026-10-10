#!/usr/bin/env python3
"""Analysis of the voltage-cut-off galvanostatic runs of Fig. 4b (gcdc_I*, 2026-09-14).

Runs: I = 0.004, 0.01, 0.02, 0.04, 0.096 and 0.48 e/ps between 0 and 2 V from the 0 V slit
frame (Fig. 4b shows 0.02, 0.096 and 0.48; all six form the SI rate series). The
cycler compares the cut-offs with an exponential moving average of the realized cell voltage
(time constant Utau). Utau has to scale with the half-cycle: too long and the reversal lags,
too short and the thermal voltage noise that survives the average (about 0.15 V at 0.5 ps,
0.03 V at 20 ps) triggers it early on a slow branch. The 0.02, 0.04 and 0.096 e/ps runs were
started with 0.5 ps and continued from a checkpoint with 20, 10 and 5 ps (RESTART); only cycles
after the continuation are used, from the second switch to charging after it. The 0.48 e/ps
run kept 0.5 ps (branch at 2.03 / -0.03 V at its switches) and uses every cycle after 1 ns.
The 2.4 e/ps runs (gcdc_I2.4, gcdc_I2.4_tau0.1) are not in the paper: at that current no
filter time works (with 0.1 ps the branch stood at 2.29 / -0.30 V at the switches).

A cycle runs from one switch to charging to the next one. Each cycle is folded on its own
switches (charging half to phase 0..0.5, discharging half to 0.5..1) and U, the cycler's
filtered voltage and Q are averaged on a common phase grid. Per current: cycle period, charge
per half-cycle, coulombic efficiency, C = I/(dU/dt) over the middle 60 % of each branch of the
mean cycle, the voltage of the branch at the switches and the voltage step at the reversals
(straight lines fitted to the per-step voltage over W_STEP ps before and after each switch,
both evaluated at the switch), R = step/2I, C = I t_dis/(2 V - step), and the overshoot of the
mean cycle past the cut-offs. Writes RESULTS.md and folded_I*.npz; imported by
figures/fig_apps_v2_paper.py."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DT, T_SKIP, NPH, W_STEP = 0.002, 1000.0, 400, 2.0
RUNS = ((0.004, "gcdc_I0.004"), (0.01, "gcdc_I0.01"), (0.02, "gcdc_I0.02"), (0.04, "gcdc_I0.04"),
        (0.096, "gcdc_I0.096"), (0.48, "gcdc_I0.48"))
# Fig. 4b shows three of them (the rest are the SI rate series)
FIG_RUNS = ((0.48, "gcdc_I0.48"), (0.096, "gcdc_I0.096"), (0.02, "gcdc_I0.02"))
# Utau (ps) in force for the cycles that are analysed
TAU = {"gcdc_I0.004": 100.0, "gcdc_I0.01": 50.0, "gcdc_I0.02": 20.0, "gcdc_I0.04": 10.0,
       "gcdc_I0.096": 5.0, "gcdc_I0.48": 0.5, "gcdc_I2.4": 0.5, "gcdc_I2.4_tau0.1": 0.1}
# time (ps) of the checkpoint from which a run was continued with its final Utau
RESTART = {"gcdc_I0.02": 3785.4, "gcdc_I0.04": 2590.4, "gcdc_I0.096": 6000.0}
# impedance real part of the slit cell (tests/paper2026/eis/eis_results.txt, U0 = 0.5 V)
EIS_F_GHZ = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
EIS_REZ = np.array([7.3702, 6.5309, 4.2609, 2.4603, 1.5770, 1.2219, 0.8923, 0.7273, 0.5232])


def rez_at(f_ghz):
    """Re Z (V ps/e) at f, log-log interpolated; below 0.5 GHz the 0.5 GHz value (flagged by the caller)."""
    f = min(max(f_ghz, EIS_F_GHZ[0]), EIS_F_GHZ[-1])
    return float(np.exp(np.interp(np.log(f), np.log(EIS_F_GHZ), np.log(EIS_REZ))))


def tau_of(run):
    return TAU.get(run, 0.5)


def _last_per_step(a):
    """Keep the last row of every MD step: a checkpoint restart may repeat a few rows."""
    s = a[:, 0][::-1]
    _, idx = np.unique(s, return_index=True)
    return a[np.sort(len(a) - 1 - idx)]


def load(run):
    q = _last_per_step(np.loadtxt(os.path.join(HERE, run, "electrode_charge.dat"), comments="#", skiprows=1))
    u = _last_per_step(np.loadtxt(os.path.join(HERE, run, "electrode_potential.dat"), comments="#"))
    steps, iq, iu = np.intersect1d(q[:, 0], u[:, 0], return_indices=True)
    t, Q, U = steps * DT, q[iq, 1], u[iu, 1]
    sw = np.atleast_2d(np.loadtxt(os.path.join(HERE, run, "gcd_switches.dat"), comments="#"))
    return t, Q, U, sw            # sw columns: switch, step, time, Q_set, U, new_direction, U_filtered


def ema(t, U, tau):
    """Exponential moving average of U on the sampled grid, the cycler's own signal."""
    Uf = np.empty_like(U); Uf[0] = U[0]
    a = 1.0 - np.exp(-(t[1] - t[0]) / tau)
    for i in range(1, len(U)):
        Uf[i] = Uf[i - 1] + a * (U[i] - Uf[i - 1])
    return Uf


def start_time(run, sw):
    """First time of the averaged cycles."""
    if run not in RESTART:
        return T_SKIP
    ups = sw[(sw[:, 5] > 0) & (sw[:, 2] > RESTART[run]), 2]
    return ups[1] if len(ups) > 1 else np.inf


def _line_at(t, U, a, b, x):
    m = (t >= a) & (t <= b)
    return np.polyval(np.polyfit(t[m], U[m], 1), x)


def fold(run, I, tau=None, umax=2.0, umin=0.0):
    tau = tau_of(run) if tau is None else tau
    t, Q, U, sw = load(run)
    Uf = ema(t, U, tau)          # what the cycler compares with the cut-offs
    t0 = start_time(run, sw)
    ups = sw[(sw[:, 5] > 0) & (sw[:, 2] >= t0 - 1e-6), 2]     # times of switches to charging
    downs = sw[(sw[:, 5] < 0), 2]
    ph = np.linspace(0.0, 1.0, NPH, endpoint=False)
    Us, Qs, Ufs, periods, frac, swing, eff, tdis_c = [], [], [], [], [], [], [], []
    for a, b in zip(ups[:-1], ups[1:]):
        d = downs[(downs > a) & (downs < b)]
        m = (t >= a) & (t < b)
        if m.sum() < 20 or len(d) == 0:
            continue
        d = d[0]
        tm = t[m]
        x = np.where(tm < d, 0.5 * (tm - a) / (d - a), 0.5 + 0.5 * (tm - d) / (b - d))
        Us.append(np.interp(ph, x, U[m])); Qs.append(np.interp(ph, x, Q[m])); Ufs.append(np.interp(ph, x, Uf[m]))
        periods.append(b - a)
        tdis_c.append(b - d)                                   # discharge time of this cycle
        frac.append((d - a) / (b - a))
        qa, qd, qb = np.interp(a, t, Q), np.interp(d, t, Q), np.interp(b, t, Q)
        swing.append(qd - qa); eff.append((qd - qb) / (qd - qa))
    Us, Qs, Ufs = np.array(Us), np.array(Qs), np.array(Ufs)
    res = dict(I=I, run=run, tau=tau, t0=t0, n=len(periods), ph=ph, U=Us.mean(0), Usd=Us.std(0),
               Q=Qs.mean(0), Uf=Ufs.mean(0), T=np.mean(periods), Tsd=np.std(periods),
               frac=np.mean(frac), swing=np.mean(swing), swingsd=np.std(swing),
               eff=np.mean(eff), effsd=np.std(eff))
    Um, f = res["U"], res["frac"]
    itop = NPH // 2
    chg, dis = slice(0, itop), slice(itop, NPH)
    def mid(s):
        n = s.stop - s.start; return slice(s.start + n // 5, s.stop - n // 5)
    tt = np.where(ph < 0.5, ph * 2 * f * res["T"], f * res["T"] + (ph - 0.5) * 2 * (1 - f) * res["T"])
    res["C_chg_slope"] = I / np.polyfit(tt[mid(chg)], Um[mid(chg)], 1)[0]
    res["C_dis_slope"] = -I / np.polyfit(tt[mid(dis)], Um[mid(dis)], 1)[0]
    # branch voltage and voltage step at every switch of the averaged cycles
    # The line before the switch spans a tenth of the half-cycle (at least W_STEP): the switch
    # fires when noise has pushed the recent voltage up (or down), so a short window before it
    # is biased by the same selection; the line after it spans W_STEP ps (fast response).
    w_pre = max(W_STEP, 0.1 * 0.5 * res["T"])
    sel = sw[(sw[:, 2] >= t0 - 1e-6) & (sw[:, 2] <= ups[-1] + 1e-6)]
    top, bot, ubt, ubb = [], [], [], []
    for s in sel:
        ts = s[2]
        if ts + W_STEP > t[-1]:
            continue
        pre, post = _line_at(t, U, ts - w_pre, ts, ts), _line_at(t, U, ts, ts + W_STEP, ts)
        if s[5] < 0:
            top.append(pre - post); ubt.append(pre)
        else:
            bot.append(post - pre); ubb.append(pre)
    res["step_top"], res["step_bottom"] = np.mean(top), np.mean(bot)
    res["step"] = 0.5 * (res["step_top"] + res["step_bottom"])
    res["U_branch_top"], res["U_branch_bottom"] = np.mean(ubt), np.mean(ubb)
    res["R"] = res["step"] / (2 * I)
    res["t_dis"] = (1.0 - f) * res["T"]
    res["C_tdis"] = I * res["t_dis"] / (umax - umin - res["step"])
    # Ion-transport resistance without a step fit: the voltage gap between the charging and the
    # discharging branch at equal charge (U-Q hysteresis), averaged over the middle half of the
    # common charge window, is 2 I R. It matches the impedance Re Z at f = 1/T.
    wsm = max(3, NPH // 40)
    sm = lambda y: np.convolve(np.r_[y[:wsm][::-1], y, y[-wsm:][::-1]], np.ones(wsm) / wsm, mode="same")[wsm:-wsm]
    Uc, Qc = sm(Um[:itop]), res["Q"][:itop]
    Ud, Qd = sm(Um[itop:][::-1]), res["Q"][itop:][::-1]
    lo, hi = max(Qc.min(), Qd.min()), min(Qc.max(), Qd.max())
    qs = np.linspace(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo), 50)
    oc, od = np.argsort(Qc), np.argsort(Qd)
    res["gap"] = float(np.mean(np.interp(qs, Qc[oc], Uc[oc]) - np.interp(qs, Qd[od], Ud[od])))
    res["R_gap"] = res["gap"] / (2 * I)
    # capacitance from the discharge time over the full cut-off window (no IR correction: the
    # model has no lumped series resistance, see the branch gap above)
    cdis = I * np.array(tdis_c) / (umax - umin)                 # per-cycle discharge-time capacitance
    res["C_tdis_plain"] = float(cdis.mean())
    res["C_tdis_plain_sem"] = float(cdis.std(ddof=1) / np.sqrt(len(cdis))) if len(cdis) > 1 else float("nan")
    res["ReZ_1T"] = rez_at(1000.0 / res["T"])
    res["Q_at_0V"], res["Q_at_2V"] = res["Q"][0], res["Q"][itop]
    res["Umax_reached"], res["Umin_reached"] = Um.max(), Um.min()
    res["over_top_raw"], res["over_bottom_raw"] = Um.max() - umax, umin - Um.min()
    res["over_top_filt"], res["over_bottom_filt"] = res["Uf"].max() - umax, umin - res["Uf"].min()
    return res


if __name__ == "__main__":
    lines = []
    q = ("\n## Numbers quoted in the manuscript\n\n| I (e/ps) | run | Utau (ps) | from (ps) | cycles | "
         "period (ps) | charge per half-cycle (e) | Q at 0 V / 2 V (e) | branch U at the switches (V) | "
         "step (V) | R = step/2I (V ps/e) | C = I t_dis/(2 V - step) (e/V) | C slope, charge / discharge (e/V) | "
         "coulombic eff. | overshoot of the mean cycle, per step / averaged (V) | branch gap (V) | R = gap/2I (V ps/e) | "
         "C = I t_dis/2 V (e/V) | Re Z at 1/T (V ps/e) |\n|" + "---|" * 19 + "\n")
    for I, run in RUNS:
        if not os.path.isfile(os.path.join(HERE, run, "gcd_switches.dat")):
            print(f"{run}: no data here, skipped")
            continue
        r = fold(run, I)
        np.savez(os.path.join(HERE, f"folded_I{I:g}.npz"), **{k: v for k, v in r.items() if isinstance(v, np.ndarray) or np.isscalar(v)})
        lines.append(f"- I = {I:g} e/ps ({run}, Utau {r['tau']:g} ps, cycles from {r['t0']:.0f} ps): {r['n']} cycles; "
                     f"period {r['T']:.1f} +- {r['Tsd']:.1f} ps; charging fraction {r['frac']:.3f}; "
                     f"charge per half-cycle {r['swing']:.2f} +- {r['swingsd']:.2f} e")
        q += (f"| {I:g} | {run} | {r['tau']:g} | {r['t0']:.0f} | {r['n']} | {r['T']:.1f} | {r['swing']:.2f} | "
              f"{r['Q_at_0V']:.2f} / {r['Q_at_2V']:.2f} | {r['U_branch_bottom']:+.2f} / {r['U_branch_top']:+.2f} | "
              f"{r['step']:.3f} | {r['R']:.3f} | {r['C_tdis']:.2f} | {r['C_chg_slope']:.2f} / {r['C_dis_slope']:.2f} | "
              f"{r['eff']:.3f} +- {r['effsd']:.3f} | {max(r['over_top_raw'], r['over_bottom_raw']):.2f} / "
              f"{max(r['over_top_filt'], r['over_bottom_filt']):.2f} | {r['gap']:.3f} | {r['R_gap']:.2f} | "
              f"{r['C_tdis_plain']:.2f} | {r['ReZ_1T']:.2f}{' (0.5 GHz)' if 1000.0 / r['T'] < 0.5 else ''} |\n")
    txt = "# Voltage-cut-off GCD (gcdc runs), analyze_gcdc.py\n\n" + "\n".join(lines) + "\n" + q
    open(os.path.join(HERE, "RESULTS.md"), "w").write(txt)
    print(txt)
