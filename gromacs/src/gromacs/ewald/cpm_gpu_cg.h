/*
 * bisheng 2026 - device-resident matrix-free CG for the constant potential method.
 *
 * When PME runs on the GPU the whole conjugate
 * gradient solve of A q = b lives on the device: the k-space part of A q reuses
 * the PME GPU spread/FFT/solve/gather (restricted to the electrode atoms), the
 * real-space electrode-electrode term is a CSR sparse product from the host-built
 * neighbour list, the self/Thomas-Fermi diagonal and the 3DC dipole term are
 * fused into one assembly kernel, and the CG vector algebra, projections and
 * dot products run as small device kernels. Per iteration only a few doubles
 * cross the PCIe bus. The host keeps the boundaries: b comes in, q goes out.
 */
#ifndef GMX_EWALD_CPM_GPU_CG_H
#define GMX_EWALD_CPM_GPU_CG_H

#include "gromacs/gpu_utils/gpu_macros.h"
#include "gromacs/utility/real.h"

struct gmx_pme_t;

/*! \brief True in CUDA builds. */
GPU_FUNC_QUALIFIER bool cpm_gpu_cg_available() GPU_FUNC_TERM_WITH_RETURN(false);

/*! \brief Upload the electrode-electrode real-space neighbour list (CSR). Call after every rebuild. */
GPU_FUNC_QUALIFIER void cpm_gpu_cg_set_nlist(const int* GPU_FUNC_ARGUMENT(rowPtr),
                                             const int* GPU_FUNC_ARGUMENT(colIdx),
                                             const real* GPU_FUNC_ARGUMENT(coef),
                                             int GPU_FUNC_ARGUMENT(n),
                                             int GPU_FUNC_ARGUMENT(nnz)) GPU_FUNC_TERM;

/*! \brief Per-step data: z (box-centred, nm), self coefficient, TF diagonal (nullptr if none),
 *         3DC prefactor 4 pi / V (0 when not slab), and the k-space unit factors. */
GPU_FUNC_QUALIFIER void cpm_gpu_cg_set_step(const real* GPU_FUNC_ARGUMENT(z),
                                            const real* GPU_FUNC_ARGUMENT(selfCoef),
                                            const real* GPU_FUNC_ARGUMENT(tfDiag),
                                            int GPU_FUNC_ARGUMENT(n),
                                            real GPU_FUNC_ARGUMENT(prefactor3dc),
                                            real GPU_FUNC_ARGUMENT(evscale),
                                            real GPU_FUNC_ARGUMENT(potUnit)) GPU_FUNC_TERM;

/*! \brief y = A x for host vectors (used by the extended Lagrangian and the constrained
 *         solvers' potential read-out). */
GPU_FUNC_QUALIFIER void cpm_gpu_cg_Aq(gmx_pme_t* GPU_FUNC_ARGUMENT(pme),
                                      const real* GPU_FUNC_ARGUMENT(x),
                                      real* GPU_FUNC_ARGUMENT(y),
                                      int GPU_FUNC_ARGUMENT(n)) GPU_FUNC_TERM;

/*! \brief Solve A q = b on the device.
 *
 * mode 0: two electrodes, potential constraint (q and the CG vectors are kept neutral);
 * mode 1: two electrodes, charge constraint (halves shifted to Q1 and Q2; r, d zero-mean per half);
 * mode 2: single electrode, charge constraint (q shifted to Q1; r, d zero-mean).
 * q holds the initial guess on entry and the solution on return. Returns the iteration
 * count, negative when maxiter was hit without reaching tol. cellVoltage receives the
 * realised potential for modes 1 and 2 (the same read-out as the host solvers).
 */
GPU_FUNC_QUALIFIER int cpm_gpu_cg_solve(gmx_pme_t* GPU_FUNC_ARGUMENT(pme),
                                        const real* GPU_FUNC_ARGUMENT(b),
                                        real* GPU_FUNC_ARGUMENT(q),
                                        int GPU_FUNC_ARGUMENT(n),
                                        int GPU_FUNC_ARGUMENT(mode),
                                        real GPU_FUNC_ARGUMENT(Q1),
                                        real GPU_FUNC_ARGUMENT(Q2),
                                        real GPU_FUNC_ARGUMENT(tol),
                                        int GPU_FUNC_ARGUMENT(maxiter),
                                        double* GPU_FUNC_ARGUMENT(cellVoltage)) GPU_FUNC_TERM_WITH_RETURN(0);

/*! \brief Accumulated wall time spent inside the device solve, ms. */
GPU_FUNC_QUALIFIER double cpm_gpu_cg_elapsed_ms() GPU_FUNC_TERM_WITH_RETURN(0.0);

#endif
