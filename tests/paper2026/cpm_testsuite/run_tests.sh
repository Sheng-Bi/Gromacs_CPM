#!/bin/bash
# CPM feature test suite.
#
#   ./run_tests.sh setup        one-time: build the elastance matrix (~4 min)
#   ./run_tests.sh              run every case on the CPU
#   ./run_tests.sh --gpu        also check the GPU path against the CPU path
#   ./run_tests.sh 03 05        run only the named cases
#
# Set GMX_BIN to choose the binary (default: the first gmx on PATH).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GMX="${GMX_BIN:-gmx}"
SLAB="${CPM_SLAB_DIR:-$HERE/../inputs/slab}"
WORK="$HERE/work"
MTX="$HERE/matrix"
NT="${NTOMP:-8}"
GPU_ID="${GPU_ID:-0}"
DO_GPU=0
CASES=()
for a in "$@"; do
  case "$a" in
    --gpu) DO_GPU=1 ;;
    setup) CASES=(setup) ;;
    *)     CASES+=("$a") ;;
  esac
done

command -v "$GMX" >/dev/null 2>&1 || { echo "ERROR: '$GMX' not found; set GMX_BIN"; exit 1; }
[ -f "$SLAB/start.gro" ] || { echo "ERROR: slab system not found at $SLAB"; exit 1; }

pass=0; fail=0; skip=0
declare -a RESULT

note()  { printf '  %s\n' "$*"; }
# Start a run directory from a guaranteed-clean state without deleting anything:
# any previous contents are moved aside, never removed.
fresh_dir() {
  local d=$1
  [ -e "$d" ] && mv "$d" "${d}.prev.$(date +%s%N)"
  mkdir -p "$d"
}
record(){ if [ "$1" -eq 0 ]; then pass=$((pass+1)); else fail=$((fail+1)); fi; }

build_tpr() { # build_tpr <case>
  local c=$1
  mkdir -p "$WORK"
  # a case that sets its own freeze groups replaces the shared ones, rather
  # than duplicating the key (which grompp rejects)
  if grep -q "^freezegrps" "$HERE/mdp/$c.mdp"; then
    grep -vE "^(freezegrps|freezedim)" "$HERE/lib/common.mdp" > "$WORK/$c.mdp"
  else
    cat "$HERE/lib/common.mdp" > "$WORK/$c.mdp"
  fi
  cat "$HERE/mdp/$c.mdp" >> "$WORK/$c.mdp"
  "$GMX" grompp -f "$WORK/$c.mdp" -c "$SLAB/start.gro" -p "$SLAB/topol.top" \
        -o "$WORK/$c.tpr" -maxwarn 6 > "$WORK/$c.grompp.log" 2>&1
}

run_case() { # run_case <case> <nsteps> <mode:cpu|gpu> [aux files...]
  local c=$1 n=$2 mode=$3; shift 3
  local d="$WORK/$c/$mode"
  fresh_dir "$d"; ( cd "$d" || exit 1
    case "$c" in
      01_*|02_*|04_*|06_*|17_*) ln -sf "$MTX/MatrixA.dat" .; ln -sf "$MTX/inv_MatrixA.dat" . ;;
    esac
    # aux files are given as "source:destination" so a case can supply e.g.
    # electrode_eta_uniform.dat under the name the code expects.
    for f in "$@"; do cp "$HERE/lib/${f%%:*}" "${f##*:}"; done
    # CPM_FORCE_MODE=gpu reruns EVERY case on the GPU path. It selects HARDWARE
    # only -- the directory still carries the caller's label, because some cases
    # use that argument as a variant name ("uniform" in case 09) rather than as a
    # hardware selector, and renaming it breaks the comparison that follows.
    local extra=(-nb cpu -pme cpu) cvd=""
    if [ "$mode" = gpu ] || [ "${CPM_FORCE_MODE:-}" = gpu ]; then
      extra=(-nb gpu -pme gpu); cvd="$GPU_ID"
    fi
    CUDA_VISIBLE_DEVICES="$cvd" GMX_CPM_PER_ATOM_CHARGES=1 \
      "$GMX" mdrun -deffnm r -s "$WORK/$c.tpr" -nsteps "$n" -ntmpi 1 -ntomp "$NT" \
      "${extra[@]}" > mdrun.out 2>&1 )
}

# ------------------------------------------------------------------ setup
if [ "${CASES[0]:-}" = setup ]; then
  echo "Building the elastance matrix for the slab system (2288 electrode atoms)."
  echo "This runs once and takes a few minutes; the result is reused by every"
  echo "matrix-based case (01, 02, 04, 06)."
  mkdir -p "$MTX"; build_tpr 01_matinv_pot
  ( cd "$MTX" && CUDA_VISIBLE_DEVICES="" "$GMX" mdrun -deffnm setup -s "$WORK/01_matinv_pot.tpr" \
      -nsteps 1 -ntmpi 1 -ntomp "$NT" -nb cpu -pme cpu > setup.log 2>&1 )
  if [ -s "$MTX/MatrixA.dat" ] && [ -s "$MTX/inv_MatrixA.dat" ]; then
    echo "OK: $(du -sh "$MTX" | cut -f1) of matrix files in $MTX"
  else
    echo "FAILED -- see $MTX/setup.log"; exit 1
  fi
  exit 0
fi

if [ ! -s "$MTX/inv_MatrixA.dat" ]; then
  echo "NOTE: no elastance matrix yet. Run './run_tests.sh setup' first to enable"
  echo "      the matrix-based cases (01, 02, 04, 06). Matrix-free cases run anyway."
  echo
fi

want() { # want <case>  -- true if the user asked for it (or asked for everything)
  [ ${#CASES[@]} -eq 0 ] && return 0
  # Keep the loop variable local to preserve the caller's selected case name.
  local _w
  for _w in "${CASES[@]}"; do [[ "$1" == "$_w"* ]] && return 0; done
  return 1
}
have_matrix() { [ -s "$MTX/inv_MatrixA.dat" ]; }

echo "=== CPM feature test suite ==================================="
echo "binary : $($GMX --version 2>/dev/null | grep -m1 'GROMACS version' | sed 's/.*: *//')"
echo "system : slab supercapacitor, 28944 atoms, 2288 electrode atoms"
echo

# ---------------------------------------------------------- solver family
for c in 01_matinv_pot 02_densecg_pot 03_mfcg_pot; do
  want "$c" || continue
  case "$c" in 01_*|02_*) have_matrix || { note "SKIP  $c (no matrix; run setup)"; skip=$((skip+1)); continue; };; esac
  build_tpr "$c"; run_case "$c" 3 cpu
  [ -s "$WORK/$c/cpu/per_atom_charges.dat" ] || { note "FAIL  $c did not produce charges"; fail=$((fail+1)); continue; }
done
if want 01_matinv_pot && have_matrix; then
  python3 "$HERE/check.py" neutral "$WORK/01_matinv_pot/cpu/per_atom_charges.dat" \
    "01 matrix inversion, fixed potential: electroneutral"; record $?
fi
if want 02_densecg_pot && have_matrix; then
  python3 "$HERE/check.py" solvers "$WORK/01_matinv_pot/cpu/per_atom_charges.dat" \
    "$WORK/02_densecg_pot/cpu/per_atom_charges.dat" "02 dense-matrix CG == matrix inversion"; record $?
fi
if want 03_mfcg_pot && have_matrix; then
  python3 "$HERE/check.py" solvers "$WORK/01_matinv_pot/cpu/per_atom_charges.dat" \
    "$WORK/03_mfcg_pot/cpu/per_atom_charges.dat" "03 matrix-free CG == matrix inversion"; record $?
fi

# ------------------------------------------------------- charge constraint
for c in 04_matinv_charge 05_mfcg_charge; do
  want "$c" || continue
  case "$c" in 04_*) have_matrix || { note "SKIP  $c (no matrix; run setup)"; skip=$((skip+1)); continue; };; esac
  build_tpr "$c"; run_case "$c" 3 cpu
done
if want 05_mfcg_charge; then
  python3 "$HERE/check.py" constrained "$WORK/05_mfcg_charge/cpu/per_atom_charges.dat" 4.0 \
    "05 matrix-free CG realises the requested electrode charge"; record $?
  have_matrix && { python3 "$HERE/check.py" solvers "$WORK/04_matinv_charge/cpu/per_atom_charges.dat" \
    "$WORK/05_mfcg_charge/cpu/per_atom_charges.dat" "05 matrix-free == matrix inversion (fixed charge)"; record $?; }
fi

# ------------------------------------------------------- single electrode
for c in 06_matinv_single 07_mfcg_single; do
  want "$c" || continue
  case "$c" in 06_*) have_matrix || { note "SKIP  $c (no matrix; run setup)"; skip=$((skip+1)); continue; };; esac
  build_tpr "$c"; run_case "$c" 3 cpu
done
if want 07_mfcg_single && have_matrix; then
  python3 "$HERE/check.py" solvers "$WORK/06_matinv_single/cpu/per_atom_charges.dat" \
    "$WORK/07_mfcg_single/cpu/per_atom_charges.dat" "07 single electrode: matrix-free == matrix inversion"; record $?
fi

# --------------------------------------------------- extended Lagrangian
if want 08_extlag_pot; then
  build_tpr 08_extlag_pot; run_case 08_extlag_pot 5 cpu
  python3 "$HERE/check.py" neutral "$WORK/08_extlag_pot/cpu/per_atom_charges.dat" \
    "08 extended Lagrangian: electroneutral"; record $?
fi

if want 15_extlag_charge; then
  build_tpr 15_extlag_charge; run_case 15_extlag_charge 5 cpu
  # The charge-constrained extended-Lagrangian path: Q_set must be imposed in e,
  # not on the solver's internal V*nm vector (which realised evscale times too
  # little charge). Tolerance is looser than case 05's because the charges here
  # are propagated rather than solved to convergence.
  python3 "$HERE/check.py" constrained "$WORK/15_extlag_charge/cpu/per_atom_charges.dat" 4.0 \
    "15 extended Lagrangian realises the requested electrode charge"; record $?
fi

# ------------------------------------------------------ per-atom eta
if want 09_hetero_eta; then
  build_tpr 09_hetero_eta
  run_case 09_hetero_eta 3 cpu electrode_eta.dat:electrode_eta.dat
  fresh_dir "$WORK/09_hetero_eta/hetero"; mv "$WORK/09_hetero_eta/cpu"/* "$WORK/09_hetero_eta/hetero/"
  run_case 09_hetero_eta 3 cpu electrode_eta_uniform.dat:electrode_eta.dat
  fresh_dir "$WORK/09_hetero_eta/uniform"; mv "$WORK/09_hetero_eta/cpu"/* "$WORK/09_hetero_eta/uniform/"
  python3 "$HERE/check.py" solvers "$WORK/03_mfcg_pot/cpu/per_atom_charges.dat" \
    "$WORK/09_hetero_eta/uniform/per_atom_charges.dat" \
    "09 uniform electrode_eta.dat reproduces the global-eta path"; record $?
  python3 "$HERE/check.py" differs "$WORK/09_hetero_eta/uniform/per_atom_charges.dat" \
    "$WORK/09_hetero_eta/hetero/per_atom_charges.dat" \
    "09 two different widths actually change the solution"; record $?
fi

# ----------------------------------------------------- time-dependent
if want 10_timedep_U; then
  build_tpr 10_timedep_U
  run_case 10_timedep_U 5 cpu potential_control_U.dat:potential_control.dat
  python3 "$HERE/check.py" differs "$WORK/03_mfcg_pot/cpu/per_atom_charges.dat" \
    "$WORK/10_timedep_U/cpu/per_atom_charges.dat" \
    "10 U(t) drive gives a different state than a static bias"; record $?
fi
if want 11_timedep_Q; then
  build_tpr 11_timedep_Q
  run_case 11_timedep_Q 10 cpu potential_control_Q.dat:potential_control.dat
  python3 "$HERE/check.py" tracks "$WORK/11_timedep_Q/cpu/electrode_charge.dat" "4.0*t" \
    "11 Q(t) drive: realised charge follows the requested ramp"; record $?
fi

# --------------------------------------- independent Q2 (two like-charged electrodes)
# bisheng 2026 - a "Q2 = ..." parameter in potential_control.dat sets electrode 2's
# charge independently of electrode 1 (default -Q1). Both solvers must hold +4/+4 e
# and agree per atom; the run warns about the +8 e net cell, which is intended here.
if want 16_q2_mfcg; then
  build_tpr 16_q2_mfcg
  run_case 16_q2_mfcg 6 cpu potential_control_Q2.dat:potential_control.dat
  python3 "$HERE/check.py" constrained2 "$WORK/16_q2_mfcg/cpu/per_atom_charges.dat" 4.0 4.0 \
    "16 Q2 = +Q1, matrix-free: both electrodes at the requested charge"; record $?
fi
if want 17_q2_matinv; then
  build_tpr 17_q2_matinv
  run_case 17_q2_matinv 6 cpu potential_control_Q2.dat:potential_control.dat
  python3 "$HERE/check.py" constrained2 "$WORK/17_q2_matinv/cpu/per_atom_charges.dat" 4.0 4.0 \
    "17 Q2 = +Q1, matrix inversion: both electrodes at the requested charge"; record $?
  if [ -f "$WORK/16_q2_mfcg/cpu/per_atom_charges.dat" ]; then
    python3 "$HERE/check.py" solvers "$WORK/17_q2_matinv/cpu/per_atom_charges.dat" \
      "$WORK/16_q2_mfcg/cpu/per_atom_charges.dat" "17 vs 16: matinv and matrix-free agree per atom with Q2 = +Q1"; record $?
  fi
fi

# ------------------------------------------------------------- non-CPM
if want 12_noncpm; then
  build_tpr 12_noncpm; run_case 12_noncpm 10 cpu
  if grep -q "Finished mdrun" "$WORK/12_noncpm/cpu/mdrun.out" 2>/dev/null || \
     grep -qE "^ *Time:" "$WORK/12_noncpm/cpu/r.log" 2>/dev/null; then
    echo "PASS  12 CPM disabled: the binary runs as plain GROMACS"; pass=$((pass+1))
  else
    echo "FAIL  12 CPM disabled: run did not complete"; fail=$((fail+1)); fi
  d="$WORK/12_noncpm/dd"; fresh_dir "$d"
  ( cd "$d" && CUDA_VISIBLE_DEVICES="" "$GMX" mdrun -deffnm r -s "$WORK/12_noncpm.tpr" -nsteps 10 \
      -ntmpi 2 -ntomp 2 -nb cpu -pme cpu > mdrun.out 2>&1 )
  if grep -qE "^ *Time:" "$d/r.log" 2>/dev/null; then
    echo "PASS  12 CPM disabled: domain decomposition still works (-ntmpi 2)"; pass=$((pass+1))
  else
    echo "FAIL  12 CPM disabled: -ntmpi 2 failed (see $d/mdrun.out)"; fail=$((fail+1)); fi
fi

# --------------------------------------------------- flexible electrodes
if want 13_flexible; then
  # the sensitivity check compares against the frozen-electrode run
  if [ ! -s "$WORK/03_mfcg_pot/cpu/per_atom_charges.dat" ]; then
    build_tpr 03_mfcg_pot; run_case 03_mfcg_pot 3 cpu
  fi
  build_tpr 13_flexible; run_case 13_flexible 3 cpu
  if [ -s "$WORK/13_flexible/cpu/per_atom_charges.dat" ]; then
    python3 "$HERE/check.py" neutral "$WORK/13_flexible/cpu/per_atom_charges.dat" \
      "13 mobile electrodes: matrix-free solver stays electroneutral"; record $?
    python3 "$HERE/check.py" differs "$WORK/03_mfcg_pot/cpu/per_atom_charges.dat" \
      "$WORK/13_flexible/cpu/per_atom_charges.dat" \
      "13 letting the electrode move changes the charges"; record $?
  else
    echo "FAIL  13 mobile electrodes produced no charges"; fail=$((fail+1))
  fi
fi

# ------------------------------------------------------ restart continuity
if want 14_restart; then
  build_tpr 03_mfcg_pot
  # (a) one continuous 6-step run, in its own directory so case 03 is untouched
  fresh_dir "$WORK/14_restart"
  c="$WORK/14_restart/continuous"; mkdir -p "$c"
  # -reprod: this case asserts that the pre-restart steps are BIT-IDENTICAL, so it
  # must run in GROMACS's reproducible mode. Without it, FFTW_MEASURE picks its FFT
  # plan by timing, so two runs of the same command differ at ~2e-08 per atom -- in
  # base GROMACS 2026, with CPM off entirely -- and no bit-identity assertion can
  # hold. (2020 happens to be reproducible here: it links the system FFTW rather
  # than the one GROMACS builds.) -reprod switches FFTW to FFTW_ESTIMATE, which is
  # deterministic; measured rms then drops from 2.1e-08 to exactly 0.
  ( cd "$c" && CUDA_VISIBLE_DEVICES="" GMX_CPM_PER_ATOM_CHARGES=1 \
      "$GMX" mdrun -deffnm r -s "$WORK/03_mfcg_pot.tpr" -nsteps 6 -ntmpi 1 -ntomp "$NT" \
      -nb cpu -pme cpu -reprod > mdrun.out 2>&1 )
  # (b) 3 steps, then restart from the checkpoint and finish the remaining 3
  d="$WORK/14_restart/split"; fresh_dir "$d"
  ( cd "$d" && CUDA_VISIBLE_DEVICES="" GMX_CPM_PER_ATOM_CHARGES=1 \
      "$GMX" mdrun -deffnm r -s "$WORK/03_mfcg_pot.tpr" -nsteps 3 -ntmpi 1 -ntomp "$NT" \
      -nb cpu -pme cpu -reprod > mdrun1.out 2>&1
    lines_before=$(grep -c . electrode_charge.dat 2>/dev/null || echo 0)
    CUDA_VISIBLE_DEVICES="" GMX_CPM_PER_ATOM_CHARGES=1 \
      "$GMX" mdrun -deffnm r -cpi r.cpt -s "$WORK/03_mfcg_pot.tpr" -nsteps 6 -ntmpi 1 -ntomp "$NT" \
      -nb cpu -pme cpu -reprod > mdrun2.out 2>&1
    echo "$lines_before" > .lines_before )
  if [ -s "$d/per_atom_charges.dat" ]; then
    python3 "$HERE/check.py" restart "$WORK/14_restart/continuous/per_atom_charges.dat" \
      "$d/per_atom_charges.dat" 3 "14 restart: identical before, transient decays after"; record $?
    before=$(cat "$d/.lines_before" 2>/dev/null || echo 0)
    after=$(grep -c . "$d/electrode_charge.dat" 2>/dev/null || echo 0)
    heads=$(grep -c "Total Charge" "$d/electrode_charge.dat" 2>/dev/null || echo 0)
    baks=$(ls "$d"/electrode_charge.dat.bak* 2>/dev/null | wc -l)
    if [ "$after" -gt "$before" ] && [ "$heads" -eq 1 ] && [ "$baks" -eq 0 ]; then
      echo "PASS  14 electrode_charge.dat appends across the restart: $before -> $after lines, 1 header, 0 .bak files"
      pass=$((pass+1))
    else
      echo "FAIL  14 electrode_charge.dat: $before -> $after lines, $heads headers, $baks .bak files"
      fail=$((fail+1))
    fi
  else
    echo "FAIL  14 restart produced no charges (see $d/mdrun2.out)"; fail=$((fail+1))
  fi
fi

# ----------------------------------------------------------- GPU checks
if [ "$DO_GPU" -eq 1 ]; then
  echo
  echo "--- GPU path versus CPU path ---"
  for c in 03_mfcg_pot 05_mfcg_charge 08_extlag_pot; do
    want "$c" || continue
    run_case "$c" 3 gpu
    if [ -s "$WORK/$c/gpu/per_atom_charges.dat" ]; then
      python3 "$HERE/check.py" solvers "$WORK/$c/cpu/per_atom_charges.dat" \
        "$WORK/$c/gpu/per_atom_charges.dat" "GPU == CPU for $c"; record $?
    else
      echo "FAIL  GPU run of $c produced no charges (see $WORK/$c/gpu/mdrun.out)"; fail=$((fail+1))
    fi
  done
fi

echo
echo "=============================================================="
printf 'passed %d   failed %d   skipped %d\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || echo "Failure details are in $WORK/<case>/<mode>/mdrun.out"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
