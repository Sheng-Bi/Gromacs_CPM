#!/usr/bin/env python3
"""Assertions for the CPM feature test suite.

Every check prints one line and exits non-zero on failure, so the runner can
report PASS/FAIL. The thresholds below apply to the supplied test inputs.
Agreement depends on solver tolerance, precision and electrostatic settings.
"""
import sys
import numpy as np

SOLVER_TOL = 1.0e-5   # e per atom, cross-solver / CPU-vs-GPU agreement
NEUTRAL_TOL = 1.0e-4  # e, total system charge
CHARGE_TOL = 5.0e-3   # e, realised vs requested constrained charge


def per_atom(path, step=None):
    """step -> array of per-atom electrode charges, ordered by atom id."""
    d = {}
    for line in open(path):
        if line.startswith('#'):
            continue
        f = line.split()
        if len(f) >= 3:
            d.setdefault(int(f[0]), {})[int(f[1])] = float(f[2])
    if not d:
        raise SystemExit(f"FAIL  no per-atom charges in {path} "
                         "(was GMX_CPM_PER_ATOM_CHARGES=1 set?)")
    s = step if step in d else max(d)
    return np.array([d[s][k] for k in sorted(d[s])]), s


def total_charge(path):
    """Last (step, total electrode charge) from electrode_charge.dat."""
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith(('#', 'Step'))]
    if not rows:
        raise SystemExit(f"FAIL  no rows in {path}")
    return int(rows[-1][0]), float(rows[-1][1])


def _steps(path):
    d = {}
    for line in open(path):
        if line.startswith('#'):
            continue
        f = line.split()
        if len(f) >= 3:
            d.setdefault(int(f[0]), {})[int(f[1])] = float(f[2])
    if not d:
        raise SystemExit(f"FAIL  no per-atom charges in {path} "
                         "(was GMX_CPM_PER_ATOM_CHARGES=1 set?)")
    return d


def common_step(a, b):
    """Latest step present in BOTH files.

    Comparing each file at its own last step silently compares different
    physical times -- two runs need not log the same number of steps. Always
    intersect first.
    """
    da, db = _steps(a), _steps(b)
    shared = sorted(set(da) & set(db))
    if not shared:
        raise SystemExit(f"FAIL  no step common to {a} and {b} "
                         f"({sorted(da)} vs {sorted(db)})")
    s = shared[-1]
    return (np.array([da[s][k] for k in sorted(da[s])]),
            np.array([db[s][k] for k in sorted(db[s])]), s)


def cmp_solvers(a, b, label, tol=SOLVER_TOL):
    x, y, s = common_step(a, b)
    if x.shape != y.shape:
        print(f"FAIL  {label}: {x.size} vs {y.size} atoms"); return 1
    d = np.sqrt(((y - x) ** 2).mean())
    m = np.abs(y - x).max()
    ok = d < tol
    print(f"{'PASS' if ok else 'FAIL'}  {label}: RMSD {d:.3e} e, max {m:.3e} e "
          f"(tol {tol:.0e}, step {s})")
    return 0 if ok else 1


def check_neutral(path, label):
    q, s = per_atom(path)
    tot = q.sum()
    ok = abs(tot) < NEUTRAL_TOL
    print(f"{'PASS' if ok else 'FAIL'}  {label}: total electrode charge {tot:+.2e} e "
          f"(tol {NEUTRAL_TOL:.0e})")
    return 0 if ok else 1


def check_constrained(path, target, label, half=True):
    """Electrode 1 (first half) must carry `target` electrons."""
    q, s = per_atom(path)
    got = q[:len(q) // 2].sum() if half else q.sum()
    ok = abs(got - target) < CHARGE_TOL
    print(f"{'PASS' if ok else 'FAIL'}  {label}: realised {got:+.4f} e vs "
          f"requested {target:+.4f} e (tol {CHARGE_TOL:.0e})")
    return 0 if ok else 1


def check_constrained2(path, q1, q2, label):
    """Electrode 1 (first half) must carry q1 and electrode 2 (second half) q2."""
    q, s = per_atom(path)
    n = len(q) // 2
    g1, g2 = q[:n].sum(), q[n:].sum()
    ok = abs(g1 - q1) < CHARGE_TOL and abs(g2 - q2) < CHARGE_TOL
    print(f"{'PASS' if ok else 'FAIL'}  {label}: realised {g1:+.4f} / {g2:+.4f} e vs "
          f"requested {q1:+.4f} / {q2:+.4f} e (tol {CHARGE_TOL:.0e})")
    return 0 if ok else 1


def check_tracks(path, expr, label, dt=0.002, tol=0.05):
    """Total charge must follow an analytic Q(t), evaluated at each logged step."""
    import math
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith(('#', 'Step'))]
    worst, worst_t = 0.0, 0.0
    for r in rows:
        t = int(r[0]) * dt
        want = eval(expr, {"t": t, "sin": math.sin, "cos": math.cos, "pi": math.pi})
        worst = max(worst, abs(float(r[1]) - want))
        if worst == abs(float(r[1]) - want):
            worst_t = t
    ok = worst < tol
    print(f"{'PASS' if ok else 'FAIL'}  {label}: max deviation from {expr} is "
          f"{worst:.4f} e at t={worst_t:.3f} ps (tol {tol})")
    return 0 if ok else 1


def check_differs(a, b, label, minrmsd=1e-4):
    """Guard against a test that would pass by being insensitive."""
    x, y, _ = common_step(a, b)
    d = np.sqrt(((y - x) ** 2).mean())
    ok = d > minrmsd
    print(f"{'PASS' if ok else 'FAIL'}  {label}: RMSD {d:.3e} e "
          f"(must exceed {minrmsd:.0e}, else the test is insensitive)")
    return 0 if ok else 1


def check_restart(cont, split, restart_step, label, first_max=3e-4):
    """A restart is not bit-exact, and should not be asserted to be.

    Re-entering the solver from a checkpoint reintroduces a small transient:
    steps up to the restart reproduce exactly, the first step after it differs
    at the 1e-4 level, and the difference then decays as the solver re-converges.
    That decay is the real claim, so that is what this checks.
    """
    da, db = _steps(cont), _steps(split)
    shared = sorted(set(da) & set(db))
    pre = [s for s in shared if s <= restart_step]
    post = [s for s in shared if s > restart_step]
    if not post:
        print(f"FAIL  {label}: no steps after the restart"); return 1
    def rms(s):
        x = np.array([da[s][k] for k in sorted(da[s])])
        y = np.array([db[s][k] for k in sorted(db[s])])
        return float(np.sqrt(((y - x) ** 2).mean()))
    bad_pre = [s for s in pre if rms(s) > 1e-12]
    devs = [rms(s) for s in post]
    decaying = all(b <= a * 1.05 for a, b in zip(devs, devs[1:]))
    ok = not bad_pre and devs[0] < first_max and decaying
    print(f"{'PASS' if ok else 'FAIL'}  {label}: "
          f"pre-restart {'identical' if not bad_pre else 'DIFFERS at ' + str(bad_pre)}, "
          f"post-restart {' -> '.join(f'{d:.2e}' for d in devs)} e"
          f"{'' if decaying else ' (NOT decaying)'}")
    return 0 if ok else 1


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'solvers':    sys.exit(cmp_solvers(sys.argv[2], sys.argv[3], sys.argv[4]))
    if cmd == 'neutral':    sys.exit(check_neutral(sys.argv[2], sys.argv[3]))
    if cmd == 'constrained':sys.exit(check_constrained(sys.argv[2], float(sys.argv[3]), sys.argv[4]))
    if cmd == 'single':     sys.exit(check_constrained(sys.argv[2], float(sys.argv[3]), sys.argv[4], half=False))
    if cmd == 'constrained2': sys.exit(check_constrained2(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]))
    if cmd == 'tracks':     sys.exit(check_tracks(sys.argv[2], sys.argv[3], sys.argv[4]))
    if cmd == 'differs':    sys.exit(check_differs(sys.argv[2], sys.argv[3], sys.argv[4]))
    if cmd == 'restart':    sys.exit(check_restart(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]))
    raise SystemExit(f"unknown check '{cmd}'")
