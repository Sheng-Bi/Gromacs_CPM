# Constant-Potential Molecular Dynamics in GROMACS

GROMACS 2020 with self-consistent electrode charges, SIMD and CUDA short-range
kernels, and GPU reciprocal-space charge solving. This release contains the
implementation used for *Kernel-Level Constant-Potential Molecular Dynamics in
GROMACS* by Sheng Bi, Mathieu Salanne, Jun Cheng and Guang Feng.

## Build

Use CMake 3.x (3.22.1 was verified), GCC 11 and a compatible CUDA toolkit.
CMake 4.x is not supported by this build recipe.

```bash
cmake -S gromacs -B build -DCMAKE_BUILD_TYPE=Release -DGMX_GPU=ON \
  -DGMX_SIMD=AVX2_256 -DGMX_BUILD_OWN_FFTW=ON
cmake --build build --parallel
build/bin/gmx --version
```

Use `-DGMX_GPU=OFF` for a CPU build. Keep SIMD enabled. A verified toolchain is
GCC 11 with CUDA 11.6 and AVX2; choose CUDA architecture flags for your GPU.
See [the user guide](docs/CPM_USER_GUIDE.md) for parameters and worked examples.

## Run and verify

CPM runs require a single MPI rank (`-ntmpi 1`). Matrix-based solvers require
frozen electrodes; mobile electrodes use the matrix-free solver and charges
solved at the current positions. Runnable examples are provided by the
[feature test suite](tests/paper2026/cpm_testsuite/README.md), using the shared inputs in
`tests/paper2026/inputs/slab/`.

```bash
export GMX_BIN="$PWD/build/bin/gmx"
cd tests/paper2026/cpm_testsuite
./run_tests.sh setup
./run_tests.sh --gpu
```

The suite's results depend on the selected hardware and must be checked for
failures and skipped cases. Python analysis uses NumPy, SciPy and Matplotlib.

## Paper data and analysis

The companion data archive extracts into this same directory. It supplies the
saved measurements, simulation inputs and completed CNT trajectories used by
`tests/paper2026/`. See [reproduction instructions](docs/REPRODUCING.md).
The versioned code and companion data archives will accompany publication.

## License

GROMACS and this derived implementation are distributed under GNU LGPL 2.1 or
later, with bundled-component notices preserved in [COPYING](COPYING) and the
source tree. Upstream authorship and source notices are retained.
