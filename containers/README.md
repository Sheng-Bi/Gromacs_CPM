# GROMACS-CPM container

This Apptainer recipe builds the source in this checkout with CUDA support and
includes two short simulations with numerical checks. It packages GROMACS-CPM,
FFTW, Python and NumPy; the host supplies Apptainer and, for GPU runs, the NVIDIA
driver. A CUDA toolkit installation on the host is not required.

## Build

Use a Linux x86-64 machine with AVX2 support and Apptainer. Run from the repository
root so the relative paths in the recipe refer to this checkout:

```bash
apptainer build --fakeroot gromacs-cpm.sif containers/Apptainer.def
apptainer run --cleanenv gromacs-cpm.sif --version
```

The recipe pins the NVIDIA CUDA 11.6.2 / Ubuntu 20.04 base image by its manifest
digest. It compiles the checked-out GROMACS source with four build workers,
AVX2 SIMD, OpenMP, thread-MPI, and CUDA targets 60, 70, 75, 80 and 86 plus PTX 86.
The Ubuntu package repositories supply build dependencies. These packages are
not snapshot-pinned, so rebuilding later need not produce a byte-identical
image. Retain the completed SIF image and its checksum when recording a run.

Building requires access to the container and Ubuntu package registries and a
host that permits [Apptainer's unprivileged build mode](https://apptainer.org/docs/user/main/fakeroot.html). On a host with configured
subordinate IDs but no `newuidmap`/`newgidmap` helpers, Apptainer's
`--ignore-subuid` build option can select its root-mapped fakeroot mode instead.
Consult your cluster's Apptainer instructions if unprivileged builds are disabled;
a built SIF can be transferred to the cluster for execution.

## Quick start: CPU

Choose a new output directory. The test copies the example inputs there and
retains all outputs and logs. It refuses to overwrite an existing directory.

```bash
apptainer exec --cleanenv --bind "$PWD:/work" gromacs-cpm.sif \
  smoke-test /work/container-cpu-test cpu
```

The two checks are:

- Fixed total charge: the matrix-free solver realises the requested electrode
  charge of 4 elementary charges.
- Time-dependent total charge: the electrode charge follows the prescribed
  ramp in the supplied control file.

A successful run ends with `passed 2   failed 0   skipped 0` and a container
smoke-test PASS message. These are short functional checks, not production
trajectories or performance benchmarks. The note about an absent elastance
matrix is expected: these two cases use the matrix-free solver.

## GPU

On an allocated NVIDIA GPU, use `--nv` to expose the host driver and GPU:

```bash
apptainer exec --cleanenv --nv --bind "$PWD:/work" gromacs-cpm.sif \
  smoke-test /work/container-gpu-test gpu
```

The example uses GPU index 0 by default. Respect the scheduler's GPU allocation;
where necessary pass the allocated visible device through `--env GPU_ID=...`.
The GPU must be compatible with the compiled CUDA targets and the host driver.
The CPU command does not need a GPU or `--nv`.

After running both examples, compare the per-atom fixed-charge results:

```bash
apptainer exec --cleanenv --bind "$PWD:/work" gromacs-cpm.sif \
  python3 /opt/cpm/examples/cpm_testsuite/check.py solvers \
  /work/container-cpu-test/cpm_testsuite/work/05_mfcg_charge/cpu/per_atom_charges.dat \
  /work/container-gpu-test/cpm_testsuite/work/05_mfcg_charge/cpu/per_atom_charges.dat \
  'CPU/GPU fixed-charge agreement'
```

The suite retains its internal `cpu/` case-directory name when its hardware
selection is overridden by the GPU smoke test; the run log records the actual GPU.

## Run your own system

```bash
apptainer exec --cleanenv --bind "$PWD:/work" --pwd /work gromacs-cpm.sif \
  gmx grompp -f run.mdp -c start.gro -p topol.top -o run.tpr
apptainer exec --cleanenv --nv --bind "$PWD:/work" --pwd /work gromacs-cpm.sif \
  gmx mdrun -deffnm run -ntmpi 1 -ntomp 8 -nb gpu -pme gpu
```

For CPU execution, omit `--nv` and select `-nb cpu -pme cpu`. CPM requires one
MPI rank. The container does not change the method's restrictions: matrix-based
solvers require frozen electrodes; mobile electrodes use the matrix-free
solver. See the [user guide](../docs/CPM_USER_GUIDE.md) and
[feature suite](../tests/paper2026/cpm_testsuite/README.md) for controls and
further examples.

## Record the build

The image retains the corresponding GROMACS source and license under
`/opt/source/`. `/opt/cpm/build-info/` contains the CMake cache, installed-package
versions, `gmx --version` output and source-file checksums. Record the repository
revision and SIF checksum alongside simulation results:

```bash
git rev-parse HEAD
sha256sum gromacs-cpm.sif
apptainer exec --cleanenv gromacs-cpm.sif cat /opt/cpm/build-info/gromacs-version.txt
```

Keep large SIF images and simulation outputs outside Git. The recipe, example
runner and instructions belong in the repository; built images can accompany a
versioned release.

## Validation

The recipe was built with Apptainer 1.5.5 on 15 November 2026 using the GROMACS
source at commit `67b33e93e1bddd39dc926182d06dadfb571ed8ad`. The resulting image
uses GCC 9.4.0, CUDA 11.6.124 and FFTW 3.3.8. Both smoke tests passed without
skips on the CPU (Xeon Platinum 8168, four threads) and on an NVIDIA RTX 3090
(driver 550.163.01). The fixed-charge test's CPU/GPU per-atom charge RMSD was
`5.018e-7 e` at step 3, below the existing test threshold of `1e-5 e`.

This validates the packaged build and these two examples on the stated hardware.
It does not constitute a benchmark or a test of every supported GPU and feature.
