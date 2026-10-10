#!/usr/bin/env python3
"""What sits between the two tubes: over the last <window> ps of a run (frames
dumped from the xtc every <every> ps), counts per frame of (i) water O and (ii)
Cl- in the slot -- the region between the two tube surfaces along the
inter-axis line (strictly between the surfaces; the tubes are water-filled), within 0.25 nm laterally of that line (the groove where the cylinders diverge is excluded), any z -- and (iii)
"bridging" Cl-: within 0.45 nm of a carbon of BOTH tubes. Also the inter-axis
distance and min C-C per analysed frame.
Usage: analyze_slot.py <run dir> [--window 100] [--every 10]"""
import os, sys, subprocess, argparse, numpy as np
GMX = os.environ.get("GMX_BIN", "gmx"); N = 800; R = 0.678
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--window", type=float, default=100); ap.add_argument("--every", type=float, default=10); a = ap.parse_args()
run = a.run.rstrip("/")
def read_gro(p):
    L = open(p).read().splitlines(); n = int(L[1])
    res = np.array([l[5:10].strip() for l in L[2:2 + n]]); name = np.array([l[10:15].strip() for l in L[2:2 + n]])
    X = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in L[2:2 + n]]); box = np.array([float(v) for v in L[2 + n].split()[:3]])
    return res, name, X, box
# last frame time from the xtc
chk = subprocess.run([GMX, "check", "-f", os.path.join(run, "run.xtc")], capture_output=True, text=True)
found = __import__("re").findall(r"Last frame\s+\d+\s+time\s+([\d.]+)", chk.stdout + chk.stderr)
if found: tlast = max(float(t) for t in found)
else:      tlast = float(np.loadtxt(os.path.join(run, "pair_distance.dat"))[-1, 0])   # snapshot with a partial last frame
times = np.arange(tlast - a.window + a.every, tlast + 1e-6, a.every)
rows = []
for t in times:
    out = os.path.join(run, f"slot_t{t:.0f}.gro")
    if not os.path.exists(out):
        subprocess.run([GMX, "trjconv", "-s", os.path.join(run, "run.tpr"), "-f", os.path.join(run, "run.xtc"), "-o", out, "-dump", str(t)], input="System\n", text=True, capture_output=True)
    if not os.path.exists(out): continue
    res, name, X, box = read_gro(out)
    def mi(v): return v - box * np.round(v / box)
    def axis(A):
        ref = A[0]; B = A - box * np.round((A - ref) / box); return B[:, :2].mean(0)
    c1, c2 = axis(X[:N]), axis(X[N:2 * N]); dvec = mi(np.r_[c2 - c1, 0.0])[:2]; d = np.linalg.norm(dvec); u = dvec / d; w = np.array([-u[1], u[0]])
    def slot(P):
        r = mi(np.c_[P[:, :2] - c1, np.zeros(len(P))])[:, :2]; s = r @ u; lat = np.abs(r @ w)
        return (s > R + 0.05) & (s < d - R - 0.05) & (lat < 0.25)   # strictly between the two tube surfaces, in the narrow band where they face each other (the tubes are water-filled)
    OW = (res == "SOL") & (name == "OW"); CL = res == "CL"
    nO = slot(X[OW]).sum(); nCl = slot(X[CL]).sum()
    C1, C2 = X[:N], X[N:2 * N]; P = X[CL]
    def mind(P, C):
        out = np.empty(len(P))
        for i in range(0, len(P), 500):
            dd = mi(P[i:i + 500, None, :] - C[None, :, :]); out[i:i + 500] = np.linalg.norm(dd, axis=2).min(1)
        return out
    br = ((mind(P, C1) < 0.45) & (mind(P, C2) < 0.45)).sum()
    dd = mi(C1[:, None, :] - C2[None, :, :]); mcc = np.linalg.norm(dd, axis=2).min()
    rows.append((t, d, mcc, nO, nCl, br))
rows = np.array(rows); np.savetxt(os.path.join(run, "slot.dat"), rows, header="t  inter-axis  minCC  nOW_slot  nCl_slot  nCl_bridging(<0.45 nm of both tubes)", fmt="%.3f")
print(f"  {os.path.basename(run)}: last {a.window:.0f} ps, {len(rows)} frames: inter-axis {rows[:,1].mean():.3f} nm, min C-C {rows[:,2].mean():.3f}; in the slot: water O {rows[:,3].mean():.1f}, Cl- {rows[:,4].mean():.1f}; bridging Cl- {rows[:,5].mean():.1f} (range {rows[:,5].min():.0f}-{rows[:,5].max():.0f})")
