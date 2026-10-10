# CPM feature test suite

Feature regression checks on a 28,944-atom slab supercapacitor with 2,288
electrode atoms. Runtime and the number of executed checks depend on hardware,
selected cases and available matrix files. Python with NumPy is required.

```bash
export GMX_BIN=/path/to/your/gmx     # defaults to `gmx` on PATH
./run_tests.sh setup                 # once: builds the elastance matrix
./run_tests.sh                       # all CPU cases
./run_tests.sh --gpu                 # also compare the GPU path to the CPU path
./run_tests.sh 01 03 04 05           # selected cases with their matrix references
```

`setup` is needed only for the cases that use a precomputed matrix (01, 02,
04, 06, 17). The matrix-free solver never reads a matrix. Cases 01, 02, 04
and 06 explicitly skip when it is absent; case 17 invokes the matrix path and
may build its own matrix. Run `setup` before the full suite. It builds matrices
on the CPU even if the later suite is run with `--gpu`.

For a selected case, include its comparison prerequisites: 02/03 need 01;
05 needs 04; 07 needs 06; 09/10 need 03; and the optional 17-vs-16 comparison
needs both 16 and 17. An existing `work/` directory can otherwise supply stale
reference output. Use a fresh working copy for a clean validation.

`--gpu` adds GPU comparisons for 03, 05 and 08; it does not move every case to
the GPU. Set `GPU_ID` to choose a physical GPU. The mobile CPU case checks
charge behavior only, not production energy conservation. See the
[user guide](../../../docs/CPM_USER_GUIDE.md#flexible-mobile-electrodes).

## What each case checks

| case | feature | assertion |
|------|---------|-----------|
| 01 | matrix inversion, fixed potential | electrode charges are electroneutral |
| 02 | dense-matrix CG | agrees with 01 |
| 03 | matrix-free CG | agrees with 01 |
| 04 | matrix inversion, fixed charge | reference for 05 |
| 05 | matrix-free CG, fixed charge | agrees with 04, and realises the requested charge |
| 06 | single electrode, matrix inversion | reference for 07 |
| 07 | single electrode, matrix-free CG | agrees with 06 |
| 08 | extended Lagrangian | electroneutral; GPU matches CPU |
| 09 | per-atom Gaussian widths | a uniform `electrode_eta.dat` reproduces the global-eta path, and two different widths genuinely change the answer |
| 10 | time-dependent U(t) | the drive produces a different state than a static bias |
| 11 | time-dependent Q(t) | the realised charge follows the requested ramp |
| 12 | CPM disabled | the binary behaves as plain GROMACS, including under domain decomposition |
| 13 | flexible (mobile) electrodes | the matrix-free solver stays electroneutral with the electrode unfrozen, and moving it changes the charges |
| 14 | checkpoint restart | steps before the restart reproduce exactly, the transient after it decays, and `electrode_charge.dat` appends rather than rotating |
| 15 | extended Lagrangian, fixed charge | realises the requested electrode charge |
| 16 | independent Q2, matrix-free | both electrode charge targets are met |
| 17 | independent Q2, matrix inversion | both targets are met; agrees with 16 when its output exists |

## Historical reference output

The output below is from the earlier 20-check suite on an RTX 3090
workstation; it predates cases 15–17. Compare against the current assertions
and their tolerances, not the last digits of these historical values.

```
PASS  01 matrix inversion, fixed potential: electroneutral: total electrode charge +1.02e-08 e
PASS  02 dense-matrix CG == matrix inversion: RMSD 1.825e-07 e, max 8.326e-07 e
PASS  03 matrix-free CG == matrix inversion: RMSD 7.082e-07 e, max 3.450e-06 e
PASS  05 matrix-free CG realises the requested electrode charge: realised +4.0000 e vs requested +4.0000 e
PASS  05 matrix-free == matrix inversion (fixed charge): RMSD 7.715e-07 e, max 3.143e-06 e
PASS  07 single electrode: matrix-free == matrix inversion: RMSD 7.092e-07 e, max 3.533e-06 e
PASS  08 extended Lagrangian: electroneutral: total electrode charge -2.35e-08 e
PASS  09 uniform electrode_eta.dat reproduces the global-eta path: RMSD 0.000e+00 e
PASS  09 two different widths actually change the solution: RMSD 5.344e-04 e
PASS  10 U(t) drive gives a different state than a static bias: RMSD 7.244e-04 e
PASS  11 Q(t) drive: realised charge follows the requested ramp: max deviation 0.0080 e
PASS  12 CPM disabled: the binary runs as plain GROMACS
PASS  12 CPM disabled: domain decomposition still works (-ntmpi 2)
PASS  GPU == CPU for 03_mfcg_pot: RMSD 3.775e-07 e
PASS  GPU == CPU for 05_mfcg_charge: RMSD 4.689e-07 e
PASS  13 mobile electrodes: matrix-free solver stays electroneutral: +1.35e-07 e
PASS  13 letting the electrode move changes the charges: RMSD 2.105e-02 e
PASS  14 restart: identical before, transient decays after: 1.16e-04 -> 5.81e-05 -> 2.91e-05 e
PASS  14 electrode_charge.dat appends across the restart: 4 -> 10 lines, 1 header, 0 .bak files
PASS  GPU == CPU for 08_extlag_pot: RMSD 4.033e-10 e
passed 20   failed 0   skipped 0
```

## Under assertions

The earlier 20-check suite was also run against a `RelWithAssert` build (`-O3` with GROMACS's
internal assertions live), which catches bounds and invariant violations a
Release build silently tolerates. All 20 checks pass there with **no assertion
fired**, so the container and projection work in the solvers is clean on the
checks that build enables.

```bash
cmake .. -DCMAKE_BUILD_TYPE=RelWithAssert -DGMX_GPU=ON -DGMX_SIMD=AVX2_256
```

## Reading the numbers

The historical per-atom errors above apply to those inputs and settings.
Accuracy depends on CG tolerance, PME settings, precision and the system;
they do not establish a universal error floor. `check.py` defines the actual
thresholds for each assertion. Tighter tolerances can improve agreement until
numerical limitations prevent further convergence.

Design points worth knowing if you extend the suite:

* **Always compare at a step both runs logged.** Two runs need not write the
  same number of frames; taking each file's last step silently compares
  different physical times. `check.py:common_step()` intersects first.
* **Assert what the code actually claims.** A restart is not bit-exact: it
  reintroduces a small transient that then decays. Case 14 asserts that decay
  rather than demanding equality, which would only be passable by loosening a
  tolerance until it stopped meaning anything.
* **A test that cannot fail is not a test.** Case 09 pairs its agreement check
  with a `differs` check, so that "uniform eta reproduces global eta" cannot
  pass merely because the code ignores the file.

## Notes

Run directories are never deleted: `run_tests.sh` moves any previous contents
aside as `<dir>.prev.<timestamp>` before each run, so a re-run always starts
clean without destroying earlier output. Clear them yourself when you no
longer want them.

## Files

```
run_tests.sh   the runner
check.py       assertions (also usable standalone)
lib/           shared mdp header and the auxiliary inputs (eta, U(t)/Q(t))
mdp/           one CPM parameter block per case, each commented
work/          created on run: tprs, run directories, logs
matrix/        created by `setup`: matrix, inverse and cache metadata
```
