/*
 * bisheng 2026 - Constant-potential charge solve on the GPU.
 *
 * The dense matrix-vector product with the cached inverse elastance
 * matrix, y = A^{-1} x, which is the whole O(N_el^2) cost of the
 * matrix-inversion solver, runs on the device. The matrix is uploaded once
 * (transposed, so that consecutive threads read consecutive addresses) and
 * kept resident; each step moves only two vectors of N_el doubles.
 *
 * Without a CUDA build every function is a stub that reports "unavailable"
 * and the solver keeps its host path.
 */
#ifndef GMX_EWALD_CPM_GPU_SOLVE_H
#define GMX_EWALD_CPM_GPU_SOLVE_H

#include "gromacs/gpu_utils/gpu_macros.h"

/*! \brief True if a CUDA device is usable from this process. */
GPU_FUNC_QUALIFIER bool cpm_gpu_solve_available() GPU_FUNC_TERM_WITH_RETURN(false);

/*! \brief Upload (or re-use) the n x n inverse matrix on the device.
 *
 * Keyed on the host pointer and n: a second call with the same matrix is a
 * no-op. \p useSingle stores a float copy and computes in single precision
 * (not the default; changes the answer at the 1e-7 e level).
 * \returns false if the device could not be set up (caller falls back). */
GPU_FUNC_QUALIFIER bool cpm_gpu_solve_init(const double* GPU_FUNC_ARGUMENT(invA),
                                           int           GPU_FUNC_ARGUMENT(n),
                                           bool          GPU_FUNC_ARGUMENT(useSingle))
        GPU_FUNC_TERM_WITH_RETURN(false);

/*! \brief y = A^{-1} x on the device (synchronous; x and y are host arrays of n doubles). */
GPU_FUNC_QUALIFIER bool cpm_gpu_solve_gemv(const double* GPU_FUNC_ARGUMENT(x),
                                           double*       GPU_FUNC_ARGUMENT(y),
                                           int           GPU_FUNC_ARGUMENT(n))
        GPU_FUNC_TERM_WITH_RETURN(false);

/*! \brief Release the device copy. */
GPU_FUNC_QUALIFIER void cpm_gpu_solve_free() GPU_FUNC_TERM;

/*! \brief Wall time spent inside cpm_gpu_solve_gemv so far, in ms (for the cost breakdown). */
GPU_FUNC_QUALIFIER double cpm_gpu_solve_elapsed_ms() GPU_FUNC_TERM_WITH_RETURN(0.0);

#endif
