#!/usr/bin/env python3
"""Simulated electrochemical impedance spectroscopy.

Each run drives U(t) = V0 sin(2 pi t / T) about 0 V and logs the total
electrode charge Q(t). With the phasor convention x(t) = Im[X e^{i w t}]:
    U -> V0,   Q -> a + i b  (from Q = a sin + b cos + baseline),
    I = dQ/dt -> i w (a + i b),   Z = V0 / (i w (a + i b)).
The first cycle is discarded (transient), the remaining integer number of
cycles is fitted with a quadratic baseline for the slowly relaxing DC offset.
Per-cycle fits give the scatter. Units: V, e, ps -> Z in V ps / e;
C' = a/V0 and C'' = -b/V0 in e/V (complex capacitance, C = C' - i C'').
Usage: analyze_eis.py [--data DIR] [--min-cycles N]
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.002


def read_runs(tsv):
    R = []
    for ln in open(tsv):
        s = ln.split()
        if len(s) == 4:
            R.append((s[0], float(s[1]), float(s[2]), int(s[3])))
    return R


def fit(t, Q, T, V0):
    w = 2 * np.pi / T
    tc = t - t[0]
    A = np.vstack([np.sin(w * t), np.cos(w * t), np.ones_like(t), tc / tc[-1], (tc / tc[-1]) ** 2]).T
    c = np.linalg.lstsq(A, Q, rcond=None)[0]
    return c[0], c[1]


def analyze(run, T, V0, data, min_cycles):
    p = os.path.join(data, run, "electrode_charge.dat")
    if not os.path.exists(p):
        p += ".gz"
        if not os.path.exists(p):
            return None
    d = np.loadtxt(p, skiprows=1)
    if d.ndim < 2 or len(d) < 10:
        return None
    t, Q = d[:, 0] * DT, d[:, 1]
    ncyc = int(t[-1] // T)
    if ncyc < min_cycles + 1:
        return None
    m = (t >= T) & (t < ncyc * T)                       # drop cycle 0, keep whole cycles
    a, b = fit(t[m], Q[m], T, V0)
    per = []
    for c in range(1, ncyc):
        mc = (t >= c * T) & (t < (c + 1) * T)
        if mc.sum() > 20:
            per.append(fit(t[mc], Q[mc], T, V0))
    per = np.array(per)
    w = 2 * np.pi / T
    Z = V0 / (1j * w * (a + 1j * b))
    Zc = V0 / (1j * w * (per[:, 0] + 1j * per[:, 1]))
    return dict(run=run, T=T, V0=V0, f_GHz=1e3 / T, ncyc=ncyc - 1, t_end=t[-1],
                a=a, b=b, Z=Z, Zsd=Zc.std(ddof=1) if len(Zc) > 1 else np.nan,
                Cp=a / V0, Cpp=-b / V0, Cp_sd=per[:, 0].std(ddof=1) / V0 if len(per) > 1 else np.nan,
                amp=np.hypot(a, b), phase=np.degrees(np.arctan2(b, a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=HERE)
    ap.add_argument("--min-cycles", type=int, default=2)
    a = ap.parse_args()
    res = [r for r in (analyze(n, T, V0, a.data, a.min_cycles) for n, T, V0, _ in read_runs(os.path.join(HERE, "runs.tsv"))) if r]
    if not res:
        print("  no run has enough cycles yet"); return
    print(f"  {'run':14s} {'f(GHz)':>7s} {'cyc':>4s} {'t_end':>7s} {'|Q|(e)':>7s} {'phase':>7s} {'C_p(e/V)':>9s} {'+-':>6s} {'C_pp':>7s} {'ReZ':>8s} {'-ImZ':>8s} {'|Z| +-':>7s}  (Z in V ps/e; +- = per-cycle sd/sqrt(n))")
    for r in sorted(res, key=lambda r: -r["f_GHz"]):
        n = max(r["ncyc"], 1)
        print(f"  {r['run']:14s} {r['f_GHz']:7.1f} {r['ncyc']:4d} {r['t_end']:7.0f} {r['amp']:7.3f} {r['phase']:+7.1f} {r['Cp']:9.3f} {r['Cp_sd']/np.sqrt(n):6.3f} {r['Cpp']:7.3f} {r['Z'].real:8.4f} {-r['Z'].imag:8.4f} {abs(r['Zsd'])/np.sqrt(n):7.4f}")
    main_ = [r for r in res if "a025" not in r["run"]]
    chk = [r for r in res if "a025" in r["run"]]
    ref = [r for r in main_ if abs(r["T"] - 100) < 1e-6]
    if chk and ref:
        z1, z2 = ref[0]["Z"], chk[0]["Z"]
        print(f"\n  linearity at {ref[0]['f_GHz']:.0f} GHz: Z(0.5 V) = {z1.real:.4f}{z1.imag:+.4f}i, Z(0.25 V) = {z2.real:.4f}{z2.imag:+.4f}i  ->  |dZ|/|Z| = {abs(z1-z2)/abs(z1):.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(left=0.07, right=0.985, top=0.92, bottom=0.2, wspace=0.62))
    m = sorted(main_, key=lambda r: r["f_GHz"])
    f = np.array([r["f_GHz"] for r in m]); Z = np.array([r["Z"] for r in m])
    Cp = np.array([r["Cp"] for r in m]); Cpp = np.array([r["Cpp"] for r in m])
    ax[0].plot(Z.real, -Z.imag, "o-", ms=4, lw=1, color="#1864AB")
    for r in m:
        ax[0].annotate(f"{r['f_GHz']:g}", (r["Z"].real, -r["Z"].imag), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel(r"Re $Z$ (V ps e$^{-1}$)", fontsize=8); ax[0].set_ylabel(r"$-$Im $Z$ (V ps e$^{-1}$)", fontsize=8)
    ax[0].set_aspect("equal", adjustable="datalim"); ax[0].set_title("Nyquist (labels: GHz)", fontsize=7.5)
    ax[1].loglog(f, abs(Z), "o-", ms=4, lw=1, color="#1864AB"); ax[1].set_xlabel("f (GHz)", fontsize=8); ax[1].set_ylabel(r"$|Z|$ (V ps e$^{-1}$)", fontsize=8)
    axp = ax[1].twinx(); axp.semilogx(f, np.degrees(np.angle(Z)), "s--", ms=3, lw=0.8, color="#C92A2A"); axp.set_ylabel("phase (deg)", fontsize=8, color="#C92A2A"); axp.tick_params(labelsize=7, colors="#C92A2A")
    ax[2].semilogx(f, Cp, "o-", ms=4, lw=1, color="#1864AB", label="$C'$"); ax[2].semilogx(f, Cpp, "s-", ms=4, lw=1, color="#C92A2A", label="$C''$")
    ax[2].set_xlabel("f (GHz)", fontsize=8); ax[2].set_ylabel("capacitance (e/V)", fontsize=8)
    if chk:
        ax[2].plot(chk[0]["f_GHz"], chk[0]["Cp"], "o", ms=8, mfc="none", mec="0.3", mew=1.0, zorder=5)
        ax[2].plot(chk[0]["f_GHz"], chk[0]["Cpp"], "s", ms=8, mfc="none", mec="0.3", mew=1.0, zorder=5)
        ax[2].plot([], [], "o", ms=6, mfc="none", mec="0.3", label="0.25 V drive")
    ax[2].legend(fontsize=6.5, frameon=False, loc="upper right")
    for x, tag in zip(ax, "abc"):
        x.tick_params(direction="in", labelsize=7, length=3); x.text(0.04, 0.9, f"({tag})", transform=x.transAxes, fontsize=8.5, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "fig_eis." + ext), dpi=300)
    print(f"  wrote fig_eis.pdf/.png with {len(m)} frequencies")


if __name__ == "__main__":
    main()
