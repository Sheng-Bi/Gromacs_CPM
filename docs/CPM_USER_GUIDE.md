# Constant-Potential MD in GROMACS — user guide

This is a patched GROMACS 2020 that lets electrode charges fluctuate
self-consistently under an applied bias, instead of being fixed by the
topology. It is what you want for electrode–electrolyte interfaces:
supercapacitors and other electrochemical cells where the electrode responds
to the electrolyte.

Gaussian short-range terms are implemented in the SIMD and CUDA nonbonded
kernels. The CUDA implementation accelerates PME, the dense inverse-matrix
product and the matrix-free CG iterations. Performance depends on the system,
solver, hardware and run settings; see the paper dataset for measured timings.

**Contents**
[Install](#install) · [Quick start](#quick-start) ·
[Setting up a system](#setting-up-a-system) ·
[Parameter reference](#parameter-reference) ·
[Choosing a solver](#choosing-a-solver) ·
[Features](#features) · [The elastance matrix](#the-elastance-matrix) ·
[Output files](#output-files) ·
[Environment variables](#environment-variables) ·
[Limitations](#limitations) · [Troubleshooting](#troubleshooting) ·
[Testing your build](#testing-your-build)

---

## Install

Run these commands from the repository root. Use CMake 3.x, GCC 11 and a
compatible CUDA toolkit; the verified toolchain includes CMake 3.22.1, GCC 11,
CUDA 11.6 and AVX2. The build recipe is not compatible with CMake 4.x.
`GMX_BUILD_OWN_FFTW=ON` downloads and builds FFTW.

### CPU only

```bash
cmake -S gromacs -B build_cpu -DCMAKE_BUILD_TYPE=Release -DGMX_GPU=OFF \
  -DGMX_SIMD=AVX2_256 -DGMX_BUILD_OWN_FFTW=ON
cmake --build build_cpu --parallel
```

### With GPU support

```bash
cmake -S gromacs -B build -DCMAKE_BUILD_TYPE=Release -DGMX_GPU=ON \
  -DGMX_SIMD=AVX2_256 -DGMX_BUILD_OWN_FFTW=ON
cmake --build build --parallel
build/bin/gmx --version
```

Choose a SIMD target supported by the CPU where the binary will run. For a
specific GPU, set `GMX_CUDA_TARGET_SM` and `GMX_CUDA_TARGET_COMPUTE` to an
architecture supported by both that GPU and your CUDA toolkit. Check GPU and
SIMD support in `--version`, then run the [feature suite](#testing-your-build).

---

## Quick start

For a runnable example with bundled inputs, use the [feature suite](#testing-your-build).
The following is a parameter fragment to add to a complete, system-specific
`.mdp`, not a complete input file. For rigid-electrode CPM, the
electrode atoms come first in the coordinate file, they are frozen, and the
CPM parameters are set through the `userint`/`userreal` fields of the `.mdp`.

```
; --- CPM ---
userint1  = 2288        ; number of electrode atoms
userreal1 = 19.79       ; Gaussian width eta, nm^-1
userreal2 = 4.0         ; applied potential difference, V
userint2  = 2           ; solver: matrix-free CG (needs no matrix files)
userint3  = 0           ; two electrodes
userint4  = 0           ; fix the potential

; --- required alongside ---
coulombtype    = PME
ewald_geometry = 3dc            ; for slab cells with vacuum gaps
freezegrps     = WaL WaU GrO    ; the electrode and its support
freezedim      = Y Y Y Y Y Y Y Y Y
```

```bash
gmx grompp -f run.mdp -c start.gro -p topol.top -o run.tpr
gmx mdrun -deffnm run -ntmpi 1        # one rank: see Limitations
```

Resolve preprocessing warnings for your system before running.
Watch `electrode_charge.dat` to see the electrode charge respond.

---

## Setting up a system

Three requirements, all of which produce confusing results if missed:

**1. Electrode atoms must come first.** The code treats atoms
`0 … userint1-1` as the electrode. For two electrodes the first half is
electrode 1 and the second half is electrode 2. Two-electrode mode therefore
requires an even count and equally many charge sites in each electrode. The
coordinate and topology atom orders must agree.

**2. Electrode charges in the topology must be zero.** They are computed each
step; whatever you put in the `.itp` is overwritten.

**3. Electrode atoms must be frozen** for the matrix-based solvers
(`userint2 = 0` or `1`). Those solvers cache PME splines and reuse a fixed
elastance matrix, both of which assume the electrode has not moved. The code
does **not** check this — if you let the electrode move under a matrix solver
you get silently wrong charges. Use the matrix-free solver
(`userint2 = 2`) for mobile electrodes.

A support layer of ordinary fixed-charge atoms behind the electrode (graphite
backing, say) is fine; it just must not be counted in `userint1`.

---

## Parameter reference

| Parameter | Meaning | Values |
|---|---|---|
| `userint1` | number of constant-potential electrode atoms | 0 disables CPM entirely |
| `userint2` | solver | 0 matrix inversion · 1 legacy dense-matrix CG · 2 matrix-free CG |
| `userint3` | geometry | 0 two electrodes · 1 single electrode |
| `userint4` | constraint | 0 fixed potential · 1 fixed charge |
| `userreal1` | Gaussian width η (nm⁻¹) | > 0 uniform · < 0 read `electrode_eta.dat` |
| `userreal2` | applied value | potential (V) if `userint4=0`, charge (e) if `userint4=1` · < 0 read `potential_control.dat` |
| `userreal3` | extended-Lagrangian fictitious mass | 0 off · > 0 on (requires `userint2=2`) |

### Supported combinations

| `userint2` | `userint3` | `userint4` | `userreal3` | what it is |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | matrix inversion, two electrodes, fixed potential |
| 0 | 0 | 1 | 0 | matrix inversion, two electrodes, fixed charge |
| 0 | 1 | 1 | 0 | matrix inversion, single electrode |
| 1 | 0 | 0 | 0 | dense-matrix CG, two electrodes, fixed potential |
| 2 | 0 | 0 | 0 | matrix-free CG, two electrodes, fixed potential |
| 2 | 0 | 1 | 0 | matrix-free CG, two electrodes, fixed charge |
| 2 | 1 | 1 | 0 | matrix-free CG, single electrode |
| 2 | 0 | 0 | > 0 | extended Lagrangian, fixed potential |
| 2 | 0 | 1 | > 0 | extended Lagrangian, fixed charge |
| 2 | 1 | 1 | > 0 | extended Lagrangian, single electrode |

The legacy dense-matrix CG path falls back to matrix inversion for charge
constraints or a single electrode, with a warning:

```
WARNING: Matrix CG (userint2=1) only supports two electrodes (userint3=0) with
         potential constraint (userint4=0).
         Falling back to matrix inversion (userint2=0).
```

This implementation supports a single electrode only with a charge constraint:
`userint3=1` forces `userint4=1`. A positive `userreal3` with a solver other than 2 is disabled
with a warning. Use an explicitly supported combination and check startup output.

---

## Choosing a solver

| Solver | Matrix storage | Electrode motion | Main operation |
|---|---|---|---|
| Matrix inversion (`userint2=0`) | Dense matrix and inverse, each O(N²) | Frozen only | Inverse-matrix/vector product; CUDA acceleration available |
| Matrix-free CG (`userint2=2`) | No dense electrode matrix | Frozen or mobile | Iterative electrostatic response; CUDA CG available |

Use matrix inversion for rigid electrodes when the dense matrices fit in
memory. Use matrix-free CG for mobile electrodes or to avoid dense matrix
storage and construction. The legacy dense-matrix CG option (`userint2=1`)
is retained for comparison, but is deprecated for new simulations.

Extended-Lagrangian propagation (`userreal3>0`, `userint2=2`) avoids the
iterative solve after initialization. It introduces a fictitious charge mass
and must be checked separately against self-consistent charge solutions.

Agreement between self-consistent solvers depends on CG tolerance, PME
settings, precision and the system. Use identical coordinates for numerical
comparisons; the paper's measured errors are described in
[Reproducing the paper results](REPRODUCING.md).

---

## Features

### Fixed potential (the default)

`userint4 = 0`, `userreal2` = the potential difference in volts. The targets
are +U/2 and −U/2 with a common offset determined by the constraint that the total electrode
charge is zero. This constrains the potential difference, rather than two
independent absolute potentials. Use a neutral electrolyte for this mode.

### Fixed charge

`userint4 = 1`, `userreal2` = the charge on electrode 1 in elementary charges.
Electrode 2 takes the opposite charge by default, and the electrode potentials
float. Check the realized values in `electrode_charge.dat`.

To prescribe the two charges independently, set `userreal2 = -1` and supply
`potential_control.dat` in the run directory, for example:

```text
Q2 = 4.0
4.0
```

The expression on the last line sets Q1 = +4 e; `Q2` sets Q2 = +4 e. Provide
an electrolyte charge of −8 e for a neutral cell in this example. `Q2` is a
constant parameter, not a second time-dependent expression. Omitting it
restores Q2 = −Q1. The program warns about a non-neutral cell.

### Single electrode

`userint3 = 1` treats all `userint1` atoms as one electrode. The constraint
is on total charge (`userint4 = 1`).

One thing surprises people: **the electrode charge is set by overall system
neutrality, not by `userreal2`**. With a neutral electrolyte the electrode
charge comes out ≈ 0 no matter what you request, and the code says so:

```
WARNING: Total charge of single electrode system is not neutral. ...
Adjusting applied_total_charge from 4 to -3.76701e-05 to ensure neutrality.
```

This is the neutrality convention implemented here. To study a charged
single electrode, supply the compensating electrolyte charge.

### Time-dependent U(t) and Q(t)

Set `userreal2` negative and put an expression in `potential_control.dat` in
the working directory. Whether it means volts or electrons follows
`userint4`. Negative literal voltages or charges must also be supplied through
this file: a negative `userreal2` selects file input, not a static negative bias.

```
# potential_control.dat
U0 = 4.0
w  = 0.5
# t is the simulation time in ps
U0 * sin(w * t)
```

The last non-comment, non-assignment line is the expression. Available:
`+ - * / ^`, `sin cos tan asin acos atan`, `exp log log10 sqrt`,
`abs min max mod floor ceil`, `if(cond, a, b)`, and the constants `pi`, `e`.

Useful shapes:

| waveform | expression |
|---|---|
| sine | `U0 * sin(w * t)` |
| square | `U0 * if(sin(w * t) > 0, 1, -1)` |
| triangle | `U0 * asin(sin(w * t)) * 2 / pi` |
| linear ramp | `U0 * t / 1000` |
| step at 100 ps | `if(t < 100, 0, U0)` |
| cyclic voltammetry | `U0 * (1 - 2 * abs(mod(2*t/period, 2) - 1))` |

Here `period` is the full triangular-cycle duration in ps; define it in the
control file. Expression identifiers are case-insensitive; avoid `T`, which
collides with the time variable `t`. The control-file keywords (`mode`, `Q2`,
`I`, `Umax`, etc.) must retain the spelling shown here. Put comments on separate
lines: the loader does not strip trailing `#` comments from expressions.
Expressions containing `=` (including `<=`, `>=` and `==`) are interpreted as
assignments by the file loader; use strict `<` or `>` in conditional examples.

The current value is echoed every 1000 steps:

```
CPM Time-dep: t = 1.000 ps, V(t) = 3.999 V
```

### Galvanostatic cycling with voltage cut-offs

Constant-current charge–discharge between two voltage limits, the way a
battery cycler runs it. Use the charge constraint (`userint4 = 1`), a negative
`userreal2`, and a control file with `mode = cutoff` instead of an expression:

```
# potential_control.dat
mode = cutoff
I = 0.02          # |current|, e/ps
Umax = 2.0        # V: reverse to discharge when the voltage reaches it
Umin = 0.0        # V: reverse to charge when the voltage falls to it
Utau = 20         # ps: time constant of the voltage average the cycler tests (see below)
Q0 = 0.0          # optional: start charge of electrode 1, e (default 0)
direction = +1    # optional: +1 starts charging, -1 discharging (default +1)
```

The imposed charge of electrode 1 is integrated in time,
Q_set += direction · I · Δt, and the other electrode carries −Q_set (or `Q2`,
if given). The current reverses when the realized voltage reaches a cut-off:
the cell voltage U = V1 − V2 for two electrodes. Use two-electrode mode for
cycling: single-electrode neutrality fixes its total charge. `I`, `Umax` and
`Umin` are required. The run stops with an error if `I = 0` or `Umax ≤ Umin`;
negative `I` is converted to its magnitude. An expression line is ignored.

Every reversal is printed and appended to `gcd_switches.dat` (switch number,
step, time, Q_set, per-step voltage, new direction, averaged voltage). The
realized voltage is written to `electrode_potential.dat` and the charge to
`electrode_charge.dat`.

**Choosing `Utau`.** The cycler compares an exponential moving average of
the realized voltage with the limits. A short averaging time allows noise to
trigger reversals; a long one delays them. With `Utau = 0` (the default), it
uses the raw voltage. Choose the time constant relative to the cycle duration
and verify that the voltage branches reach the intended limits. Discard the
initial transient before calculating cycle averages.

### Per-atom Gaussian widths

For bimetallic or otherwise heterogeneous electrodes, set `userreal1` to a
negative number and supply `electrode_eta.dat`:

```
# atom_id  eta(nm^-1)      ids are 0-based and must cover 0 … userint1-1
0     19.79
1     19.79
...
1144  25.00
```

Widths combine pairwise as η_ij = η_i η_j / √(η_i² + η_j²). A uniform
width file is tested against the global-width path in feature case 09.

### Thomas–Fermi electrodes

Place `electrode_tf.dat` in the run directory to add finite screening. A global
line has two columns, `l_TF` in nm and atomic volume `V_at` in nm³; per-atom
lines have three columns, `atom_id l_TF V_at`, with zero-based IDs. Supply a
global line before any per-atom overrides, or specify every electrode atom.
Use non-negative screening lengths and positive volumes.

The matrix diagonal gains `4*pi*l_TF^2/V_at` in inverse nm. With no file the
model is a perfect metal. Use solver 0 or 2; the legacy dense-matrix CG path
rejects Thomas–Fermi input. Rebuild cached matrices after changing this file
and retain the same file for restarts. Paper examples are in
`tests/paper2026/tf/` in the companion dataset.

### Electron spillover (sCPM)

Displaced charge sites can represent electron spillover without a separate
solver option. The fluctuating Gaussian charges belong to the first
`userint1` sites; uncharged metal nuclei carry the Lennard-Jones interactions.
For frozen electrodes, construct the displaced sites and nuclei in the
coordinate/topology files, keeping the two charge-site blocks first. Count
only charge sites in `userint1`, and freeze both those sites and the nuclei.
Use `electrode_eta.dat` if different sites need different widths.

For moving electrodes, displaced sites must follow their nuclei through a
consistent virtual-site construction with force redistribution. Use the
matrix-free solver and the CUDA nonbonded path described below. A displaced
charge site is not automatically attached to a nucleus simply because its
coordinate is nearby.

The companion dataset contains the paper's prepared spillover inputs under
`tests/paper2026/spillover/`. It does not include a general structure-generation
helper. Displacements, widths and force fields are model choices, not
universal defaults. The Au(111) demonstration uses a reconstructed electrolyte
model; validate interfacial structure, equilibration and parameter sensitivity
before applying it to a new physical system.

### Flexible (mobile) electrodes

Use `userint2 = 2` and leave the moving electrode atoms out of `freezegrps`,
or freeze only selected dimensions. Matrix-based solvers reuse a fixed
response and must not be used for moving electrodes.

For production with mobile electrodes, use CUDA nonbonded kernels (`-nb gpu`).
The default mobile-electrode schedule solves charges at the current positions
before evaluating the CUDA force kernel. The CPU nonbonded path does not
provide this schedule and warns about charge-lag energy drift. Its short
feature-suite check tests charge neutrality and sensitivity to motion, not
production energy conservation. Check the selected force order in startup
output and validate your time step and energy behavior.

### Extended Lagrangian

`userreal3 > 0` (with `userint2 = 2`) turns the charges into dynamical
variables with a fictitious mass, propagated alongside the atoms instead of
being solved each step, which removes the iterative solve.

The fictitious mass is specified in the solver's internal units, not as an
atomic mass in daltons. Choose it together with the integration time step and
validate against the self-consistent solver. Finite-mass charge dynamics need
not reproduce instantaneous self-consistent charges or their fluctuations.
The extended energy includes fictitious charge kinetic energy; physical-energy
drift alone is not a measure of extended-energy conservation, and a thermostat
does not replace a stability check.

### GPU

```bash
gmx mdrun -deffnm run -ntmpi 1 -nb gpu -pme gpu
```

CUDA accelerates the short-range kernels and reciprocal-space response. With
GPU PME, the inverse-matrix product uses custom CUDA kernels (double precision
by default), and matrix-free CG keeps its iteration vectors and operations on
the device. Host work remains for setup, transfers, constraints/bookkeeping on
the matrix-inverse path and convergence control. Check startup output for the
selected solver backend.

### Restarts

`gmx mdrun -deffnm run -cpi run.cpt -ntmpi 1` resumes a checkpoint.
Warm starts, predictor history and
extended-Lagrangian charge velocities are all checkpointed via
`cpm_state.dat`, and time-dependent protocols read the true simulation time,
so the waveform uses the resumed simulation time.
Keep `cpm_state.dat` together with its matching GROMACS checkpoint and the
auxiliary control/model files. A missing or unreadable state causes a cold
start; a step mismatch warns but still loads the state. Restarted charges can
show a transient and are not guaranteed bit-identical to an uninterrupted run.
`electrode_charge.dat` is appended, not rotated.

The state of the cut-off cycler (direction, imposed charge, number of
reversals, averaged voltage) is part of `cpm_state.dat`, so `-cpi` continues
the same cycle. `potential_control.dat` is read again at every start, so the
current, the limits or `Utau` can be changed on a restart; the average then
continues from its saved value with the new time constant.

---

## The elastance matrix

Needed only for `userint2 = 0` and `1`.

**Let the code build it.** Without cached matrix files, the matrix-based path
constructs `MatrixA.dat` and `inv_MatrixA.dat`, and writes `MatrixA.meta` with
settings used for cache checks. Dense storage scales as O(N²); forming the
inverse scales as O(N³). Exclude first-run matrix construction when measuring
steady-state throughput.

Keep all three files together. Metadata checks detect several mismatches,
including atom count, geometry and Thomas–Fermi settings, but are not a full
identity check of every input. A cache without metadata is accepted with a
warning. Rebuild after changing atom order, geometry, box, Gaussian widths,
Thomas–Fermi parameters or Ewald settings.

Externally generated matrices must use this implementation's internal nm
convention and match its electrostatic model. An incompatible unit convention
can produce incorrect charges even if the files load. Prefer construction by
the executable that will run the simulation.

---

## Output files

| file | contents |
|---|---|
| `electrode_charge.dat` | step and total electrode charge, appended across restarts |
| `charge_prediction.dat` | predictor accuracy: actual vs predicted charge, RMSE, max error (solver modes only) |
| `cpm_state.dat` | solver state for checkpoint restarts, including the cut-off cycler |
| `electrode_potential.dat` | charge-constraint runs: step and realized voltage (cell voltage, or single-electrode potential) |
| `gcd_switches.dat` | cut-off cycler: one line per current reversal (switch, step, time, Q_set, voltage, new direction, averaged voltage) |
| `per_atom_charges.dat` | per-atom charges — only when `GMX_CPM_PER_ATOM_CHARGES=1` |
| `MatrixA.dat`, `inv_MatrixA.dat`, `MatrixA.meta` | elastance matrix, inverse and cache metadata, if built |

---

## Environment variables

| variable | effect |
|---|---|
| `GMX_CPM_PER_ATOM_CHARGES=1` | write `per_atom_charges.dat`; needed for any per-atom comparison |
| `GMX_CPM_CG_TOL=<x>` | Matrix-free CG stopping threshold on the sum of squared projected potential residuals in V² (default 1e-6), not a per-atom charge error |
| `GMX_CPM_CG_MAXITER=<n>` | Matrix-free CG iteration cap (default 1000) |
| `GMX_CPM_CG_TRACE=1` | host charge-constrained CG residual diagnostics; output depends on the backend |
| `GMX_CPM_DEBUG_AQ=1` | one-off breakdown of the host A·v components |
| `GMX_CPM_DEBUG_POTENTIALS=1` | write `debug_potentials_full.dat` |
| `GMX_CPM_DIRECT_KSPACE=1` | build the elastance matrix with a slow direct Ewald sum instead of the optimised one, to cross-check it |
| `GMX_CPM_ALLOW_UNVERIFIED_KERNEL=1` | bypass the nonbonded-kernel safeguard below. **Development only**; bypassing the guard permits kernels without CPM corrections |

For backend comparisons, `GMX_CPM_NO_GPU_SOLVE=1` disables the dense CUDA
product and `GMX_CPM_NO_GPU_CG=1` disables device CG iterations.
`GMX_CPM_GPU_SOLVE_SINGLE=1` selects a float matrix copy instead of the default
double copy. These switches change the numerical/performance path; leave them
unset for the default behavior. Presence-based flags must be **unset** to turn
them off; setting them to `0` still enables them.

---

## Limitations

* **CPM requires a SIMD or CUDA nonbonded kernel.** Gaussian short-range
  corrections are implemented in SIMD 4xN/2xNN and CUDA kernels. Plain-C
  reference kernels lack these corrections and are rejected at setup. Keep
  SIMD enabled for CPU runs and check the selected kernel in the log.

* **No domain decomposition.** Every electrode atom must live on one rank, so
  run `-ntmpi 1`. OpenMP threading (`-ntomp N`) is fully supported and is how
  you scale on a node. Non-CPM runs of the same binary are unaffected and may
  use domain decomposition normally.
* **Matrix solvers need frozen electrodes**, and this is not verified for you.
* **Numerical accuracy depends on the system and settings.** Tightening CG
  tolerance can improve agreement until arithmetic/PME error limits it; an
  excessively tight threshold can prevent convergence. There is no universal
  charge-error floor or tolerance that guarantees a given accuracy.
* **Pressure coupling** is rejected for matrix solvers and accepted with an
  unvalidated-use warning for matrix-free CG. Validate changing-box runs
  separately; the paper does not establish their accuracy.
* **Extended-Lagrangian results need separate validation**, including mass,
  time-step and observable dependence.

---

## Troubleshooting

| symptom | cause and fix |
|---|---|
| `CPM does not support domain decomposition` | you ran multiple ranks; use `-ntmpi 1 -ntomp N` |
| `inv_MatrixA file cannot be loaded` | matrix missing or truncated; delete both files and let the code rebuild |
| Run stalls for minutes before step 0 | matrix construction may be underway; verify log progress and CPU activity |
| Electrode charges ~10× too large | check external-matrix units and normalization; rebuild from the current inputs |
| `nEleAtoms is not even` | two-electrode mode needs an even `userint1` |
| Single-electrode charge is ~0 whatever you set | correct: neutrality fixes it; charge the electrolyte instead |
| `Failed to compile expression` | check `potential_control.dat` syntax; do not name a parameter `T` |
| Charges are NaN | check atom count/order, control expressions, width files and convergence logs |
| Non-CPM run behaves oddly | set `userint1 = 0` to disable CPM entirely |

---

## Testing your build

```bash
cd tests/paper2026/cpm_testsuite
export GMX_BIN=/path/to/your/gmx
./run_tests.sh setup      # build the shared matrices; time depends on hardware
./run_tests.sh --gpu
```

The suite requires Python with NumPy and reports individual deviations,
thresholds, failures and skips. `--gpu` adds CPU/GPU comparisons; set `GPU_ID`
to select the physical GPU (the runner sets `CUDA_VISIBLE_DEVICES`). Without
`--gpu`, it runs CPU checks. Matrix construction and test runtime depend on the
hardware. Check the final summary and logs, not just whether files appeared.

The suite includes solver, constraint, width, time-dependent drive, mobile
charge, restart and independent-Q2 cases. It is a feature regression suite,
not a validation of every material model or production ensemble. See
[`tests/paper2026/cpm_testsuite/README.md`](../tests/paper2026/cpm_testsuite/README.md)
for case details. Paper-specific numerical and figure reproduction additionally
requires the companion dataset; see [REPRODUCING.md](REPRODUCING.md).
