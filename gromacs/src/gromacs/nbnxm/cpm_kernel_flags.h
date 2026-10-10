/* bisheng 2026 - CPM kernel diagnostic flags.
 *
 * GMX_CPM_NO_KERNEL_FORCE suppresses only the CPM force correction in the nbnxm
 * kernels, leaving the short-range potential psr_coul and solved charges unchanged.
 * This isolates the force correction during validation.
 */
#ifndef GMX_NBNXM_CPM_KERNEL_FLAGS_H
#define GMX_NBNXM_CPM_KERNEL_FLAGS_H

#include <cstdlib>

static inline bool cpmNoKernelForce()
{
    static const bool v = (std::getenv("GMX_CPM_NO_KERNEL_FORCE") != nullptr);
    return v;
}

/* bisheng - GMX_CPM_NO_PSR_J suppresses the j-side psr store. Cached the same way
 * as the flag above: the call site is the INNERMOST loop body (kernel_inner.h, one
 * evaluation per j-cluster), where an uncached std::getenv() hits libc on every
 * iteration and the compiler cannot hoist it. Same knob, same semantics, one
 * evaluation per run instead of ~1e5-1e6 per step. */
static inline bool cpmNoPsrJ()
{
    static const bool v = (std::getenv("GMX_CPM_NO_PSR_J") != nullptr);
    return v;
}

/* bisheng 2026 - mobile-electrode force ordering.
 *
 * Mobile electrodes use charges solved at the current positions in the CUDA
 * force evaluation. Rigid electrodes retain the predictor ordering. Select the
 * schedule from electrode mobility rather than from the charge-solver type.
 *
 * The flag is set once during setup, where the freeze groups are known, and read by
 * the CUDA nonbonded init when it picks the psr/force ordering.
 */
inline bool& cpmElectrodesMobileRef()
{
    static bool v = false;
    return v;
}
inline void cpmSetElectrodesAreMobile(bool mobile)
{
    cpmElectrodesMobileRef() = mobile;
}
inline bool cpmElectrodesAreMobile()
{
    return cpmElectrodesMobileRef();
}

/* Force the lagged (predictor) ordering even for mobile electrodes, for A/B work. */
static inline bool cpmNoForceAfterSolve()
{
    static const bool v = (std::getenv("GMX_CPM_NO_FORCE_AFTER_SOLVE") != nullptr);
    return v;
}

#endif
