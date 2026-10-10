"""gpuval3: per-atom CPU vs GPU electrode charges for the nine Fig. S2 configurations.
Reported step = the one the original F8/gpuval2 summaries used (step 1; step 10 for the two
time-dependent cases, whose first recorded step is 10; the last step for dense CG); the worst
step over the run is given too. Writes gpuval3_summary.txt in the format fig_gpu_validation reads."""
import os
import sys

import numpy as np

O = os.path.dirname(os.path.abspath(__file__))
REPORT = {"slit": 1, "slab3dc": 1, "d1cv": 10, "d2gcd": 10, "mfcg": 1, "extlag": 1, "e1flex": 1,
          "densecg": "last", "single": 1}


def load(p):
    d = {}
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        s, a, q = l.split()[:3]
        d.setdefault(int(s), {})[int(a)] = float(q)
    return {s: np.array([v[k] for k in sorted(v)]) for s, v in d.items()}


lines = []
for c, rep in REPORT.items():
    pc, pg = f"{O}/{c}/cpu/per_atom_charges.dat", f"{O}/{c}/gpu/per_atom_charges.dat"
    if not (os.path.exists(pc) and os.path.exists(pg)):
        lines.append(f"{c:8s} MISSING")
        continue
    A, B = load(pc), load(pg)
    steps = sorted(set(A) & set(B))
    s = steps[-1] if rep == "last" else (rep if rep in steps else steps[0])
    d = A[s] - B[s]
    worst = max(steps, key=lambda k: np.sqrt(((A[k] - B[k]) ** 2).mean()))
    dw = A[worst] - B[worst]
    rmsd, mx = np.sqrt((d ** 2).mean()), np.abs(d).max()
    ok = "PASS" if mx < 1e-4 else "CHECK"
    lines.append(f"{c:8s} {ok}  per-atom step {s}: RMSD={rmsd:.3e} e  max|diff|={mx:.3e} e  "
                 f"(worst of {len(steps)} steps: step {worst}, RMSD={np.sqrt((dw**2).mean()):.3e} e; "
                 f"q rms {A[s].std():.3e} e)")
out = "\n".join(lines)
print(out)
if len(sys.argv) < 2 or sys.argv[1] != "--no-write":
    open(f"{O}/gpuval3_summary.txt", "w").write(out + "\n")
