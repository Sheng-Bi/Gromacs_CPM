#!/usr/bin/env python3
"""Per-atom and per-layer comparison of CPM-GROMACS (matrix-free CG, 3DC) against
MetalWalls (CG, exact 2D Ewald) on the three-layer graphene capacitor, at each l_TF."""
import numpy as np, sys, os
N=1144; layers=["surface","middle","inner"]
def gmx(l):
    d=np.loadtxt(f"gmx_l{l}/per_atom_charges.dat",comments="#"); s0=d[:,0].min(); q=d[d[:,0]==s0]; return q[np.argsort(q[:,1])][:,2]
def mw(l):
    q=[float(x) for x in open(f"mw_l{l}/charges.out") if not x.startswith("#")]; return np.array(q[:6864])
res={}
for l in ("0.00","0.10","0.30"):
    if not os.path.exists(f"mw_l{l}/charges.out") or os.path.getsize(f"mw_l{l}/charges.out")<1000: continue
    g,m=gmx(l),mw(l); d=m-g
    print(f"=== l_TF = {l} nm ===")
    print(f"  electrode-1 total:  GROMACS {g[:3*N].sum():.6f}   MetalWalls {m[:3*N].sum():.6f}   rel diff {(m[:3*N].sum()-g[:3*N].sum())/g[:3*N].sum():+.2e}")
    print(f"  per-atom: r = {np.corrcoef(g,m)[0,1]:.8f}   rms|dq| = {np.sqrt((d**2).mean()):.2e} e   max|dq| = {np.abs(d).max():.2e} e   sd(q) = {g.std():.3e}   rms/sd = {np.sqrt((d**2).mean())/g.std():.2e}")
    print(f"  {'layer':8s} {'GROMACS':>10s} {'MetalWalls':>11s} {'rel':>9s}")
    for k in range(3):
        a,b=g[k*N:(k+1)*N].sum(),m[k*N:(k+1)*N].sum(); print(f"  {layers[k]:8s} {a:10.6f} {b:11.6f} {(b-a)/a:+9.2e}")
    res[l]=(g,m)
if "0.00" in res and len(res)>1:
    print("\n=== TF response, per layer: q(l)/q(0) -- the Ewald-treatment difference cancels ===")
    g0,m0=res["0.00"]
    for l in res:
        if l=="0.00": continue
        g,m=res[l]; print(f"  l={l}: " + "  ".join(f"{layers[k]}: gmx {g[k*N:(k+1)*N].sum()/g0[k*N:(k+1)*N].sum():.5f} mw {m[k*N:(k+1)*N].sum()/m0[k*N:(k+1)*N].sum():.5f}" for k in range(3)))
        dg=g-g0; dm=m-m0; print(f"         per-atom TF shift dq(l)-dq(0): r = {np.corrcoef(dg,dm)[0,1]:.6f}, rms|gmx-mw| = {np.sqrt(((dg-dm)**2).mean()):.2e} e vs rms shift {np.sqrt((dg**2).mean()):.2e} e")
