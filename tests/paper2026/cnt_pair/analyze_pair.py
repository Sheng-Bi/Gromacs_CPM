#!/usr/bin/env python3
"""Two-tube bundle analysis: inter-axis distance and minimum inter-tube C-C
distance vs time (from the xtc), per-tube total charge vs time, and the
per-atom charge pattern on each tube resolved by azimuth about its own axis
(0 deg = pointing at the other tube), from per_atom_charges.dat.
Usage: analyze_pair.py <run dir> [--dt-ps 0.002]"""
import os, sys, subprocess, numpy as np
run = sys.argv[1]; DT = 0.002; N = 800
GMX = os.environ.get("GMX_BIN", "gmx")

def read_gro(p):
    L = open(p).read().splitlines(); n = int(L[1])
    X = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in L[2:2 + n]])
    return X, np.array([float(v) for v in L[2 + n].split()[:3]])

def axes_from_frame(X, box):
    """Axis (x,y) of each tube = mean over its atoms after unwrapping about the first atom."""
    out = []
    for t in range(2):
        A = X[t * N:(t + 1) * N, :2]; ref = A[0]
        A = A - box[:2] * np.round((A - ref) / box[:2])
        out.append(A.mean(0))
    return np.array(out)

def trajectory(run):
    """Dump the CNT atoms of every xtc frame to a gro stream via trjconv."""
    tpr, xtc = os.path.join(run, "run.tpr"), os.path.join(run, "run.xtc")
    ndx = os.path.join(os.path.dirname(run.rstrip("/")), "index_q0.ndx")
    out = os.path.join(run, "cnt_traj.gro")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(xtc):
        r = subprocess.run([GMX, "trjconv", "-s", tpr, "-f", xtc, "-n", ndx, "-o", out], input="CNT\n", text=True, capture_output=True)
        if not os.path.exists(out):
            raise SystemExit("trjconv failed:\n" + r.stderr[-1500:])
    L = open(out).read().splitlines(); frames = []; i = 0
    while i < len(L) - 2:
        n = int(L[i + 1]); t = float(L[i].split("t=")[1].split()[0]) if "t=" in L[i] else len(frames)
        X = np.array([[float(l[20:28]), float(l[28:36]), float(l[36:44])] for l in L[i + 2:i + 2 + n]])
        box = np.array([float(v) for v in L[i + 2 + n].split()[:3]]); frames.append((t, X, box)); i += n + 3
    return frames

def main():
    fr = trajectory(run)
    rows = []
    for t, X, box in fr:
        ax = axes_from_frame(X, box); d = ax[1] - ax[0]; d -= box[:2] * np.round(d / box[:2])
        A, B = X[:N], X[N:]
        dd = A[:, None, :] - B[None, :, :]; dd -= box * np.round(dd / box); rmin = np.linalg.norm(dd, axis=2).min()
        rows.append((t, np.hypot(*d), rmin))
    rows = np.array(rows); np.savetxt(os.path.join(run, "pair_distance.dat"), rows, header="t(ps)  inter-axis(nm)  min_CC(nm)", fmt="%.4f")
    print(f"  {len(rows)} frames, t = {rows[0,0]:.0f}-{rows[-1,0]:.0f} ps")
    print(f"  inter-axis distance: start {rows[0,1]:.3f}, mean {rows[:,1].mean():.3f} +- {rows[:,1].std():.3f}, min {rows[:,1].min():.3f}, max {rows[:,1].max():.3f} nm   (vdW contact = 1.696)")
    print(f"  min inter-tube C-C:  mean {rows[:,2].mean():.3f}, min {rows[:,2].min():.3f}, max {rows[:,2].max():.3f} nm")
    # separation criterion: min C-C > 0.6 nm sustained for > 50 ps
    sep = rows[:, 2] > 0.6; dt = np.median(np.diff(rows[:, 0])) if len(rows) > 1 else 1.0
    best = 0; cur = 0; t_on = None
    for k, v in enumerate(sep):
        cur = cur + 1 if v else 0
        if cur > best: best = cur; t_on = rows[k - cur + 1, 0]
    print(f"  separation (min C-C > 0.6 nm for > 50 ps): {'YES, from t = %.0f ps, longest stretch %.0f ps' % (t_on, best * dt) if best * dt > 50 else 'NO (longest stretch above 0.6 nm: %.0f ps)' % (best * dt)}")
    # charges
    p = os.path.join(run, "per_atom_charges.dat")
    if os.path.exists(p):
        d = np.loadtxt(p, comments="#")
        nf = len(d) // (2 * N); d = d[:nf * 2 * N]; Q = d[:, 2].reshape(nf, 2 * N); steps = d[::2 * N, 0]
        q1, q2 = Q[:, :N].sum(1), Q[:, N:].sum(1)
        print(f"  per-tube charge over {nf} frames: tube1 {q1.mean():+.6f} +- {q1.std():.2e}, tube2 {q2.mean():+.6f} +- {q2.std():.2e} e")
        # azimuth about each tube's own axis from the LAST frame geometry (contact side = 0 deg)
        t, X, box = fr[-1]; ax = axes_from_frame(X, box)
        res = {}
        for tb in range(2):
            A = X[tb * N:(tb + 1) * N, :2]; c = ax[tb]; v = A - c; v -= box[:2] * np.round(v / box[:2])
            other = ax[1 - tb] - c; other -= box[:2] * np.round(other / box[:2]); phi0 = np.arctan2(other[1], other[0])
            phi = (np.degrees(np.arctan2(v[:, 1], v[:, 0]) - phi0) + 180) % 360 - 180
            qm = Q[:, tb * N:(tb + 1) * N].mean(0); qs = Q[:, tb * N:(tb + 1) * N].std(0)
            face = np.abs(phi) < 30; back = np.abs(phi) > 150
            res[tb] = (phi, qm, qs)
            print(f"  tube {tb+1}: <q_i> facing the other tube {qm[face].mean():+.5f} e, outer side {qm[back].mean():+.5f} e; sd over atoms {qm.std():.2e}; sigma_i mean {qs.mean():.2e}")
        np.savez(os.path.join(run, "azimuth.npz"), phi1=res[0][0], q1=res[0][1], s1=res[0][2], phi2=res[1][0], q2=res[1][1], s2=res[1][2], steps=steps, Q1=q1, Q2=q2)
main()
