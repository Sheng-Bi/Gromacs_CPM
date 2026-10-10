/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team.
 * Copyright (c) 2013,2014,2015,2016,2017,2018,2019, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \internal \file
 *
 * \brief This file contains function definitions necessary for
 * computing energies and forces for the PME long-ranged part (Coulomb
 * and LJ).
 *
 * \author Erik Lindahl <erik@kth.se>
 * \author Berk Hess <hess@kth.se>
 * \ingroup module_ewald
 */
/* IMPORTANT FOR DEVELOPERS:
 *
 * Triclinic pme stuff isn't entirely trivial, and we've experienced
 * some bugs during development (many of them due to me). To avoid
 * this in the future, please check the following things if you make
 * changes in this file:
 *
 * 1. You should obtain identical (at least to the PME precision)
 *    energies, forces, and virial for
 *    a rectangular box and a triclinic one where the z (or y) axis is
 *    tilted a whole box side. For instance you could use these boxes:
 *
 *    rectangular       triclinic
 *     2  0  0           2  0  0
 *     0  2  0           0  2  0
 *     0  0  6           2  2  6
 *
 * 2. You should check the energy conservation in a triclinic box.
 *
 * It might seem an overkill, but better safe than sorry.
 * /Erik 001109
 */

#include "gmxpre.h"

#include "pme.h"

#include "config.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <cinttypes> // bisheng 2026 - PRId64/SCNd64 for CPM restart state

// bisheng 2025 for CG
#include <vector>

#include <algorithm>
#include <list>

#include "gromacs/domdec/domdec.h"
#include "gromacs/ewald/ewald_utils.h"
#include "gromacs/fft/parallel_3dfft.h"
#include "gromacs/fileio/pdbio.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/hardware/hw_info.h"
#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/invertmatrix.h"
#include "gromacs/math/units.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/cyclecounter.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/timing/walltime_accounting.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/gmxomp.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/unique_cptr.h"

#include "calculate_spline_moduli.h"
#include "pme_gather.h"
#include "pme_gpu_internal.h"
#include "pme_grid.h"
#include "pme_internal.h"
#include "pme_redistribute.h"
#include "pme_solve.h"
#include "pme_spline_work.h"
#include "pme_spread.h"

/* bisheng 2026 - device-resident matrix-free CG (see cpm_gpu_cg.h) */
#include "cpm_gpu_cg.h"
static bool g_cpmUseGpuCg         = false;
static int  g_cpmNlistGen         = 0;  // bumped by build_electrode_nlist
static int  g_cpmNlistGenUploaded = -1; // generation resident on the device

/*! \brief Help build a descriptive message in \c error if there are
 * \c errorReasons why PME on GPU is not supported.
 *
 * \returns Whether the lack of errorReasons indicate there is support. */
static bool addMessageIfNotSupported(const std::list<std::string>& errorReasons, std::string* error)
{
    bool isSupported = errorReasons.empty();
    if (!isSupported && error)
    {
        std::string regressionTestMarker = "PME GPU does not support";
        // this prefix is tested for in the regression tests script gmxtest.pl
        *error = regressionTestMarker;
        if (errorReasons.size() == 1)
        {
            *error += " " + errorReasons.back();
        }
        else
        {
            *error += ": " + gmx::joinStrings(errorReasons, "; ");
        }
        *error += ".";
    }
    return isSupported;
}

bool pme_gpu_supports_build(std::string* error)
{
    std::list<std::string> errorReasons;
    if (GMX_DOUBLE)
    {
        errorReasons.emplace_back("a double-precision build");
    }
    if (GMX_GPU == GMX_GPU_NONE)
    {
        errorReasons.emplace_back("a non-GPU build");
    }
    return addMessageIfNotSupported(errorReasons, error);
}

bool pme_gpu_supports_hardware(const gmx_hw_info_t gmx_unused& hwinfo, std::string* error)
{
    std::list<std::string> errorReasons;

    if (GMX_GPU == GMX_GPU_OPENCL)
    {
#ifdef __APPLE__
        errorReasons.emplace_back("Apple OS X operating system");
#endif
    }
    return addMessageIfNotSupported(errorReasons, error);
}

bool pme_gpu_supports_input(const t_inputrec& ir, const gmx_mtop_t& mtop, std::string* error)
{
    std::list<std::string> errorReasons;
    if (!EEL_PME(ir.coulombtype))
    {
        errorReasons.emplace_back("systems that do not use PME for electrostatics");
    }
    if (ir.pme_order != 4)
    {
        errorReasons.emplace_back("interpolation orders other than 4");
    }
    if (ir.efep != efepNO)
    {
        if (gmx_mtop_has_perturbed_charges(mtop))
        {
            errorReasons.emplace_back(
                    "free energy calculations with perturbed charges (multiple grids)");
        }
    }
    if (EVDW_PME(ir.vdwtype))
    {
        errorReasons.emplace_back("Lennard-Jones PME");
    }
    if (!EI_DYNAMICS(ir.eI))
    {
        errorReasons.emplace_back("not a dynamical integrator");
    }
    return addMessageIfNotSupported(errorReasons, error);
}

/*! \brief \libinternal
 * Finds out if PME with given inputs is possible to run on GPU.
 * This function is an internal final check, validating the whole PME structure on creation,
 * but it still duplicates the preliminary checks from the above (externally exposed) pme_gpu_supports_input() - just in case.
 *
 * \param[in]  pme          The PME structure.
 * \param[out] error        The error message if the input is not supported on GPU.
 * \returns                 True if this PME input is possible to run on GPU, false otherwise.
 */
static bool pme_gpu_check_restrictions(const gmx_pme_t* pme, std::string* error)
{
    std::list<std::string> errorReasons;
    if (pme->nnodes != 1)
    {
        errorReasons.emplace_back("PME decomposition");
    }
    if (pme->pme_order != 4)
    {
        errorReasons.emplace_back("interpolation orders other than 4");
    }
    if (pme->bFEP)
    {
        errorReasons.emplace_back("free energy calculations (multiple grids)");
    }
    if (pme->doLJ)
    {
        errorReasons.emplace_back("Lennard-Jones PME");
    }
    if (GMX_DOUBLE)
    {
        errorReasons.emplace_back("double precision");
    }
    if (GMX_GPU == GMX_GPU_NONE)
    {
        errorReasons.emplace_back("non-GPU build of GROMACS");
    }

    return addMessageIfNotSupported(errorReasons, error);
}

PmeRunMode pme_run_mode(const gmx_pme_t* pme)
{
    GMX_ASSERT(pme != nullptr, "Expecting valid PME data pointer");
    return pme->runMode;
}

gmx::PinningPolicy pme_get_pinning_policy()
{
    return gmx::PinningPolicy::PinnedIfSupported;
}

/*! \brief Number of bytes in a cache line.
 *
 * Must also be a multiple of the SIMD and SIMD4 register size, to
 * preserve alignment.
 */
const int gmxCacheLineSize = 64;

//! Set up coordinate communication
static void setup_coordinate_communication(PmeAtomComm* atc)
{
    int nslab, n, i;
    int fw, bw;

    nslab = atc->nslab;

    n = 0;
    for (i = 1; i <= nslab / 2; i++)
    {
        fw = (atc->nodeid + i) % nslab;
        bw = (atc->nodeid - i + nslab) % nslab;
        if (n < nslab - 1)
        {
            atc->slabCommSetup[n].node_dest = fw;
            atc->slabCommSetup[n].node_src  = bw;
            n++;
        }
        if (n < nslab - 1)
        {
            atc->slabCommSetup[n].node_dest = bw;
            atc->slabCommSetup[n].node_src  = fw;
            n++;
        }
    }
}

/*! \brief Round \p n up to the next multiple of \p f */
static int mult_up(int n, int f)
{
    return ((n + f - 1) / f) * f;
}

/*! \brief Return estimate of the load imbalance from the PME grid not being a good match for the number of PME ranks */
static double estimate_pme_load_imbalance(struct gmx_pme_t* pme)
{
    int    nma, nmi;
    double n1, n2, n3;

    nma = pme->nnodes_major;
    nmi = pme->nnodes_minor;

    n1 = mult_up(pme->nkx, nma) * mult_up(pme->nky, nmi) * pme->nkz;
    n2 = mult_up(pme->nkx, nma) * mult_up(pme->nkz, nmi) * pme->nky;
    n3 = mult_up(pme->nky, nma) * mult_up(pme->nkz, nmi) * pme->nkx;

    /* pme_solve is roughly double the cost of an fft */

    return (n1 + n2 + 3 * n3) / static_cast<double>(6 * pme->nkx * pme->nky * pme->nkz);
}

#ifndef DOXYGEN

PmeAtomComm::PmeAtomComm(MPI_Comm   PmeMpiCommunicator,
                         const int  numThreads,
                         const int  pmeOrder,
                         const int  dimIndex,
                         const bool doSpread) :
    dimind(dimIndex),
    bSpread(doSpread),
    pme_order(pmeOrder),
    nthread(numThreads),
    spline(nthread)
{
    if (PmeMpiCommunicator != MPI_COMM_NULL)
    {
        mpi_comm = PmeMpiCommunicator;
#    if GMX_MPI
        MPI_Comm_size(mpi_comm, &nslab);
        MPI_Comm_rank(mpi_comm, &nodeid);
#    endif
    }
    if (debug)
    {
        fprintf(debug, "For PME atom communication in dimind %d: nslab %d rank %d\n", dimind, nslab, nodeid);
    }

    if (nslab > 1)
    {
        slabCommSetup.resize(nslab);
        setup_coordinate_communication(this);

        count_thread.resize(nthread);
        for (auto& countThread : count_thread)
        {
            countThread.resize(nslab);
        }
    }

    if (nthread > 1)
    {
        threadMap.resize(nthread);

#    pragma omp parallel for num_threads(nthread) schedule(static)
        for (int thread = 0; thread < nthread; thread++)
        {
            try
            {
                /* Allocate buffer with padding to avoid cache polution */
                threadMap[thread].nBuffer.resize(nthread + 2 * gmxCacheLineSize);
                threadMap[thread].n = threadMap[thread].nBuffer.data() + gmxCacheLineSize;
            }
            GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
        }
    }
}

#endif // !DOXYGEN

/*! \brief Initialize data structure for communication */
static void init_overlap_comm(pme_overlap_t* ol, int norder, MPI_Comm comm, int nnodes, int nodeid, int ndata, int commplainsize)
{
    gmx_bool bCont;

    ol->mpi_comm = comm;
    ol->nnodes   = nnodes;
    ol->nodeid   = nodeid;

    /* Linear translation of the PME grid won't affect reciprocal space
     * calculations, so to optimize we only interpolate "upwards",
     * which also means we only have to consider overlap in one direction.
     * I.e., particles on this node might also be spread to grid indices
     * that belong to higher nodes (modulo nnodes)
     */

    ol->s2g0.resize(ol->nnodes + 1);
    ol->s2g1.resize(ol->nnodes);
    if (debug)
    {
        fprintf(debug, "PME slab boundaries:");
    }
    for (int i = 0; i < nnodes; i++)
    {
        /* s2g0 the local interpolation grid start.
         * s2g1 the local interpolation grid end.
         * Since in calc_pidx we divide particles, and not grid lines,
         * spatially uniform along dimension x or y, we need to round
         * s2g0 down and s2g1 up.
         */
        ol->s2g0[i] = (i * ndata + 0) / nnodes;
        ol->s2g1[i] = ((i + 1) * ndata + nnodes - 1) / nnodes + norder - 1;

        if (debug)
        {
            fprintf(debug, "  %3d %3d", ol->s2g0[i], ol->s2g1[i]);
        }
    }
    ol->s2g0[nnodes] = ndata;
    if (debug)
    {
        fprintf(debug, "\n");
    }

    /* Determine with how many nodes we need to communicate the grid overlap */
    int testRankCount = 0;
    do
    {
        testRankCount++;
        bCont = FALSE;
        for (int i = 0; i < nnodes; i++)
        {
            if ((i + testRankCount < nnodes && ol->s2g1[i] > ol->s2g0[i + testRankCount])
                || (i + testRankCount >= nnodes && ol->s2g1[i] > ol->s2g0[i + testRankCount - nnodes] + ndata))
            {
                bCont = TRUE;
            }
        }
    } while (bCont && testRankCount < nnodes);

    ol->comm_data.resize(testRankCount - 1);
    ol->send_size = 0;

    for (size_t b = 0; b < ol->comm_data.size(); b++)
    {
        pme_grid_comm_t* pgc = &ol->comm_data[b];

        /* Send */
        pgc->send_id  = (ol->nodeid + (b + 1)) % ol->nnodes;
        int fft_start = ol->s2g0[pgc->send_id];
        int fft_end   = ol->s2g0[pgc->send_id + 1];
        if (pgc->send_id < nodeid)
        {
            fft_start += ndata;
            fft_end += ndata;
        }
        int send_index1  = ol->s2g1[nodeid];
        send_index1      = std::min(send_index1, fft_end);
        pgc->send_index0 = fft_start;
        pgc->send_nindex = std::max(0, send_index1 - pgc->send_index0);
        ol->send_size += pgc->send_nindex;

        /* We always start receiving to the first index of our slab */
        pgc->recv_id    = (ol->nodeid - (b + 1) + ol->nnodes) % ol->nnodes;
        fft_start       = ol->s2g0[ol->nodeid];
        fft_end         = ol->s2g0[ol->nodeid + 1];
        int recv_index1 = ol->s2g1[pgc->recv_id];
        if (pgc->recv_id > nodeid)
        {
            recv_index1 -= ndata;
        }
        recv_index1      = std::min(recv_index1, fft_end);
        pgc->recv_index0 = fft_start;
        pgc->recv_nindex = std::max(0, recv_index1 - pgc->recv_index0);
    }

#if GMX_MPI
    /* Communicate the buffer sizes to receive */
    MPI_Status stat;
    for (size_t b = 0; b < ol->comm_data.size(); b++)
    {
        MPI_Sendrecv(&ol->send_size, 1, MPI_INT, ol->comm_data[b].send_id, b, &ol->comm_data[b].recv_size,
                     1, MPI_INT, ol->comm_data[b].recv_id, b, ol->mpi_comm, &stat);
    }
#endif

    /* For non-divisible grid we need pme_order iso pme_order-1 */
    ol->sendbuf.resize(norder * commplainsize);
    ol->recvbuf.resize(norder * commplainsize);
}

int minimalPmeGridSize(int pmeOrder)
{
    /* The actual grid size limitations are:
     *   serial:        >= pme_order
     *   DD, no OpenMP: >= 2*(pme_order - 1)
     *   DD, OpenMP:    >= pme_order + 1
     * But we use the maximum for simplicity since in practice there is not
     * much performance difference between pme_order and 2*(pme_order -1).
     */
    int minimalSize = 2 * (pmeOrder - 1);

    GMX_RELEASE_ASSERT(pmeOrder >= 3, "pmeOrder has to be >= 3");
    GMX_RELEASE_ASSERT(minimalSize >= pmeOrder + 1, "The grid size should be >= pmeOrder + 1");

    return minimalSize;
}

bool gmx_pme_check_restrictions(int pme_order, int nkx, int nky, int nkz, int numPmeDomainsAlongX, bool useThreads, bool errorsAreFatal)
{
    if (pme_order > PME_ORDER_MAX)
    {
        if (!errorsAreFatal)
        {
            return false;
        }

        std::string message = gmx::formatString(
                "pme_order (%d) is larger than the maximum allowed value (%d). Modify and "
                "recompile the code if you really need such a high order.",
                pme_order, PME_ORDER_MAX);
        GMX_THROW(gmx::InconsistentInputError(message));
    }

    const int minGridSize = minimalPmeGridSize(pme_order);
    if (nkx < minGridSize || nky < minGridSize || nkz < minGridSize)
    {
        if (!errorsAreFatal)
        {
            return false;
        }
        std::string message =
                gmx::formatString("The PME grid sizes need to be >= 2*(pme_order-1) (%d)", minGridSize);
        GMX_THROW(gmx::InconsistentInputError(message));
    }

    /* Check for a limitation of the (current) sum_fftgrid_dd code.
     * We only allow multiple communication pulses in dim 1, not in dim 0.
     */
    if (useThreads
        && (nkx < numPmeDomainsAlongX * pme_order && nkx != numPmeDomainsAlongX * (pme_order - 1)))
    {
        if (!errorsAreFatal)
        {
            return false;
        }
        gmx_fatal(FARGS,
                  "The number of PME grid lines per rank along x is %g. But when using OpenMP "
                  "threads, the number of grid lines per rank along x should be >= pme_order (%d) "
                  "or = pmeorder-1. To resolve this issue, use fewer ranks along x (and possibly "
                  "more along y and/or z) by specifying -dd manually.",
                  nkx / static_cast<double>(numPmeDomainsAlongX), pme_order);
    }

    return true;
}

/*! \brief Round \p enumerator */
static int div_round_up(int enumerator, int denominator)
{
    return (enumerator + denominator - 1) / denominator;
}

gmx_pme_t* gmx_pme_init(const t_commrec*         cr,
                        const NumPmeDomains&     numPmeDomains,
                        const t_inputrec*        ir,
                        gmx_bool                 bFreeEnergy_q,
                        gmx_bool                 bFreeEnergy_lj,
                        gmx_bool                 bReproducible,
                        real                     ewaldcoeff_q,
                        real                     ewaldcoeff_lj,
                        int                      nthread,
                        PmeRunMode               runMode,
                        PmeGpu*                  pmeGpu,
                        const gmx_device_info_t* gpuInfo,
                        PmeGpuProgramHandle      pmeGpuProgram,
                        const gmx::MDLogger& /*mdlog*/)
{
    int  use_threads, sum_use_threads, i;
    ivec ndata;

    if (debug)
    {
        fprintf(debug, "Creating PME data structures.\n");
    }

    gmx::unique_cptr<gmx_pme_t, gmx_pme_destroy> pme(new gmx_pme_t());

    pme->sum_qgrid_tmp    = nullptr;
    pme->sum_qgrid_dd_tmp = nullptr;

    pme->buf_nalloc = 0;

    // bisheng 2025 - Initialize electrode neighbor list as invalid
    pme->sr_nlist_valid = FALSE;

    pme->nnodes  = 1;
    pme->bPPnode = TRUE;

    pme->nnodes_major = numPmeDomains.x;
    pme->nnodes_minor = numPmeDomains.y;

    if (numPmeDomains.x * numPmeDomains.y > 1)
    {
        pme->mpi_comm = cr->mpi_comm_mygroup;

#if GMX_MPI
        MPI_Comm_rank(pme->mpi_comm, &pme->nodeid);
        MPI_Comm_size(pme->mpi_comm, &pme->nnodes);
#endif
        if (pme->nnodes != numPmeDomains.x * numPmeDomains.y)
        {
            gmx_incons("PME rank count mismatch");
        }
    }
    else
    {
        pme->mpi_comm = MPI_COMM_NULL;
    }

    if (pme->nnodes == 1)
    {
        pme->mpi_comm_d[0] = MPI_COMM_NULL;
        pme->mpi_comm_d[1] = MPI_COMM_NULL;
        pme->ndecompdim    = 0;
        pme->nodeid_major  = 0;
        pme->nodeid_minor  = 0;
    }
    else
    {
        if (numPmeDomains.y == 1)
        {
            pme->mpi_comm_d[0] = pme->mpi_comm;
            pme->mpi_comm_d[1] = MPI_COMM_NULL;
            pme->ndecompdim    = 1;
            pme->nodeid_major  = pme->nodeid;
            pme->nodeid_minor  = 0;
        }
        else if (numPmeDomains.x == 1)
        {
            pme->mpi_comm_d[0] = MPI_COMM_NULL;
            pme->mpi_comm_d[1] = pme->mpi_comm;
            pme->ndecompdim    = 1;
            pme->nodeid_major  = 0;
            pme->nodeid_minor  = pme->nodeid;
        }
        else
        {
            if (pme->nnodes % numPmeDomains.x != 0)
            {
                gmx_incons(
                        "For 2D PME decomposition, #PME ranks must be divisible by the number of "
                        "domains along x");
            }
            pme->ndecompdim = 2;

#if GMX_MPI
            MPI_Comm_split(pme->mpi_comm, pme->nodeid % numPmeDomains.y, pme->nodeid,
                           &pme->mpi_comm_d[0]); /* My communicator along major dimension */
            MPI_Comm_split(pme->mpi_comm, pme->nodeid / numPmeDomains.y, pme->nodeid,
                           &pme->mpi_comm_d[1]); /* My communicator along minor dimension */

            MPI_Comm_rank(pme->mpi_comm_d[0], &pme->nodeid_major);
            MPI_Comm_size(pme->mpi_comm_d[0], &pme->nnodes_major);
            MPI_Comm_rank(pme->mpi_comm_d[1], &pme->nodeid_minor);
            MPI_Comm_size(pme->mpi_comm_d[1], &pme->nnodes_minor);
#endif
        }
    }
    // cr is always initialized if there is a a PP rank, so we can safely assume
    // that when it is not, like in ewald tests, we not on a PP rank.
    pme->bPPnode = ((cr != nullptr && cr->duty != 0) && thisRankHasDuty(cr, DUTY_PP));

    pme->nthread = nthread;

    /* Check if any of the PME MPI ranks uses threads */
    use_threads = (pme->nthread > 1 ? 1 : 0);
#if GMX_MPI
    if (pme->nnodes > 1)
    {
        MPI_Allreduce(&use_threads, &sum_use_threads, 1, MPI_INT, MPI_SUM, pme->mpi_comm);
    }
    else
#endif
    {
        sum_use_threads = use_threads;
    }
    pme->bUseThreads = (sum_use_threads > 0);

    if (ir->ePBC == epbcSCREW)
    {
        gmx_fatal(FARGS, "pme does not (yet) work with pbc = screw");
    }

    /* NOTE:
     * It is likely that the current gmx_pme_do() routine supports calculating
     * only Coulomb or LJ while gmx_pme_init() configures for both,
     * but that has never been tested.
     * It is likely that the current gmx_pme_do() routine supports calculating,
     * not calculating free-energy for Coulomb and/or LJ while gmx_pme_init()
     * configures with free-energy, but that has never been tested.
     */
    pme->doCoulomb     = EEL_PME(ir->coulombtype);
    pme->doLJ          = EVDW_PME(ir->vdwtype);
    pme->bFEP_q        = ((ir->efep != efepNO) && bFreeEnergy_q);
    pme->bFEP_lj       = ((ir->efep != efepNO) && bFreeEnergy_lj);
    pme->bFEP          = (pme->bFEP_q || pme->bFEP_lj);
    pme->nkx           = ir->nkx;
    pme->nky           = ir->nky;
    pme->nkz           = ir->nkz;
    pme->bP3M          = (ir->coulombtype == eelP3M_AD || getenv("GMX_PME_P3M") != nullptr);
    pme->pme_order     = ir->pme_order;
    pme->ewaldcoeff_q  = ewaldcoeff_q;
    pme->ewaldcoeff_lj = ewaldcoeff_lj;

    /* Always constant electrostatics coefficients */
    pme->epsilon_r = ir->epsilon_r;

    /* Always constant LJ coefficients */
    pme->ljpme_combination_rule = ir->ljpme_combination_rule;

    // The box requires scaling with nwalls = 2, we store that condition as well
    // as the scaling factor
    delete pme->boxScaler;
    pme->boxScaler = new EwaldBoxZScaler(*ir);

    // bisheng 2025 - Store 3DC flag for matrix-free CG solver
    pme->bEwald3DC = (ir->ewald_geometry == eewg3DC);

    // bisheng 2025 - Store nstenergy for debug print frequency
    pme->nstenergy = ir->nstenergy;

    /* If we violate restrictions, generate a fatal error here */
    gmx_pme_check_restrictions(pme->pme_order, pme->nkx, pme->nky, pme->nkz, pme->nnodes_major,
                               pme->bUseThreads, true);

    if (pme->nnodes > 1)
    {
        double imbal;

#if GMX_MPI
        MPI_Type_contiguous(DIM, GMX_MPI_REAL, &(pme->rvec_mpi));
        MPI_Type_commit(&(pme->rvec_mpi));
#endif

        /* Note that the coefficient spreading and force gathering, which usually
         * takes about the same amount of time as FFT+solve_pme,
         * is always fully load balanced
         * (unless the coefficient distribution is inhomogeneous).
         */

        imbal = estimate_pme_load_imbalance(pme.get());
        if (imbal >= 1.2 && pme->nodeid_major == 0 && pme->nodeid_minor == 0)
        {
            fprintf(stderr,
                    "\n"
                    "NOTE: The load imbalance in PME FFT and solve is %d%%.\n"
                    "      For optimal PME load balancing\n"
                    "      PME grid_x (%d) and grid_y (%d) should be divisible by #PME_ranks_x "
                    "(%d)\n"
                    "      and PME grid_y (%d) and grid_z (%d) should be divisible by #PME_ranks_y "
                    "(%d)\n"
                    "\n",
                    gmx::roundToInt((imbal - 1) * 100), pme->nkx, pme->nky, pme->nnodes_major,
                    pme->nky, pme->nkz, pme->nnodes_minor);
        }
    }

    /* For non-divisible grid we need pme_order iso pme_order-1 */
    /* In sum_qgrid_dd x overlap is copied in place: take padding into account.
     * y is always copied through a buffer: we don't need padding in z,
     * but we do need the overlap in x because of the communication order.
     */
    init_overlap_comm(&pme->overlap[0], pme->pme_order, pme->mpi_comm_d[0], pme->nnodes_major,
                      pme->nodeid_major, pme->nkx,
                      (div_round_up(pme->nky, pme->nnodes_minor) + pme->pme_order)
                              * (pme->nkz + pme->pme_order - 1));

    /* Along overlap dim 1 we can send in multiple pulses in sum_fftgrid_dd.
     * We do this with an offset buffer of equal size, so we need to allocate
     * extra for the offset. That's what the (+1)*pme->nkz is for.
     */
    init_overlap_comm(&pme->overlap[1], pme->pme_order, pme->mpi_comm_d[1], pme->nnodes_minor,
                      pme->nodeid_minor, pme->nky,
                      (div_round_up(pme->nkx, pme->nnodes_major) + pme->pme_order + 1) * pme->nkz);

    /* Double-check for a limitation of the (current) sum_fftgrid_dd code.
     * Note that gmx_pme_check_restrictions checked for this already.
     */
    if (pme->bUseThreads && (pme->overlap[0].comm_data.size() > 1))
    {
        gmx_incons(
                "More than one communication pulse required for grid overlap communication along "
                "the major dimension while using threads");
    }

    snew(pme->bsp_mod[XX], pme->nkx);
    snew(pme->bsp_mod[YY], pme->nky);
    snew(pme->bsp_mod[ZZ], pme->nkz);

    pme->gpu     = pmeGpu; /* Carrying over the single GPU structure */
    pme->runMode = runMode;

    /* The required size of the interpolation grid, including overlap.
     * The allocated size (pmegrid_n?) might be slightly larger.
     */
    pme->pmegrid_nx = pme->overlap[0].s2g1[pme->nodeid_major] - pme->overlap[0].s2g0[pme->nodeid_major];
    pme->pmegrid_ny = pme->overlap[1].s2g1[pme->nodeid_minor] - pme->overlap[1].s2g0[pme->nodeid_minor];
    pme->pmegrid_nz_base = pme->nkz;
    pme->pmegrid_nz      = pme->pmegrid_nz_base + pme->pme_order - 1;
    set_grid_alignment(&pme->pmegrid_nz, pme->pme_order);
    pme->pmegrid_start_ix = pme->overlap[0].s2g0[pme->nodeid_major];
    pme->pmegrid_start_iy = pme->overlap[1].s2g0[pme->nodeid_minor];
    pme->pmegrid_start_iz = 0;

    make_gridindex_to_localindex(pme->nkx, pme->pmegrid_start_ix,
                                 pme->pmegrid_nx - (pme->pme_order - 1), &pme->nnx, &pme->fshx);
    make_gridindex_to_localindex(pme->nky, pme->pmegrid_start_iy,
                                 pme->pmegrid_ny - (pme->pme_order - 1), &pme->nny, &pme->fshy);
    make_gridindex_to_localindex(pme->nkz, pme->pmegrid_start_iz, pme->pmegrid_nz_base, &pme->nnz,
                                 &pme->fshz);

    pme->spline_work = make_pme_spline_work(pme->pme_order);

    ndata[0] = pme->nkx;
    ndata[1] = pme->nky;
    ndata[2] = pme->nkz;
    /* It doesn't matter if we allocate too many grids here,
     * we only allocate and use the ones we need.
     */
    if (pme->doLJ)
    {
        pme->ngrids = ((ir->ljpme_combination_rule == eljpmeLB) ? DO_Q_AND_LJ_LB : DO_Q_AND_LJ);
    }
    else
    {
        pme->ngrids = DO_Q;
    }
    snew(pme->fftgrid, pme->ngrids);
    snew(pme->cfftgrid, pme->ngrids);
    snew(pme->pfft_setup, pme->ngrids);

    for (i = 0; i < pme->ngrids; ++i)
    {
        if ((i < DO_Q && pme->doCoulomb && (i == 0 || bFreeEnergy_q))
            || (i >= DO_Q && pme->doLJ
                && (i == 2 || bFreeEnergy_lj || ir->ljpme_combination_rule == eljpmeLB)))
        {
            pmegrids_init(&pme->pmegrid[i], pme->pmegrid_nx, pme->pmegrid_ny, pme->pmegrid_nz,
                          pme->pmegrid_nz_base, pme->pme_order, pme->bUseThreads, pme->nthread,
                          pme->overlap[0].s2g1[pme->nodeid_major]
                                  - pme->overlap[0].s2g0[pme->nodeid_major + 1],
                          pme->overlap[1].s2g1[pme->nodeid_minor]
                                  - pme->overlap[1].s2g0[pme->nodeid_minor + 1]);
            /* This routine will allocate the grid data to fit the FFTs */
            const auto allocateRealGridForGpu = (pme->runMode == PmeRunMode::Mixed)
                                                        ? gmx::PinningPolicy::PinnedIfSupported
                                                        : gmx::PinningPolicy::CannotBePinned;
            gmx_parallel_3dfft_init(&pme->pfft_setup[i], ndata, &pme->fftgrid[i], &pme->cfftgrid[i],
                                    pme->mpi_comm_d, bReproducible, pme->nthread, allocateRealGridForGpu);
        }
    }

    if (!pme->bP3M)
    {
        /* Use plain SPME B-spline interpolation */
        make_bspline_moduli(pme->bsp_mod, pme->nkx, pme->nky, pme->nkz, pme->pme_order);
    }
    else
    {
        /* Use the P3M grid-optimized influence function */
        make_p3m_bspline_moduli(pme->bsp_mod, pme->nkx, pme->nky, pme->nkz, pme->pme_order);
    }

    /* Use atc[0] for spreading */
    const int firstDimIndex   = (numPmeDomains.x > 1 ? 0 : 1);
    MPI_Comm  mpiCommFirstDim = (pme->nnodes > 1 ? pme->mpi_comm_d[firstDimIndex] : MPI_COMM_NULL);
    bool      doSpread        = true;
    pme->atc.emplace_back(mpiCommFirstDim, pme->nthread, pme->pme_order, firstDimIndex, doSpread);
    if (pme->ndecompdim >= 2)
    {
        const int secondDimIndex = 1;
        doSpread                 = false;
        pme->atc.emplace_back(pme->mpi_comm_d[1], pme->nthread, pme->pme_order, secondDimIndex, doSpread);
    }

    if (pme_gpu_active(pme.get()))
    {
        if (!pme->gpu)
        {
            // Initial check of validity of the data
            std::string errorString;
            bool        canRunOnGpu = pme_gpu_check_restrictions(pme.get(), &errorString);
            if (!canRunOnGpu)
            {
                GMX_THROW(gmx::NotImplementedError(errorString));
            }
        }

        pme_gpu_reinit(pme.get(), gpuInfo, pmeGpuProgram);
    }

    pme_init_all_work(&pme->solve_work, pme->nthread, pme->nkx);

    // no exception was thrown during the init, so we hand over the PME structure handle
    return pme.release();
}

void gmx_pme_reinit(struct gmx_pme_t** pmedata,
                    const t_commrec*   cr,
                    struct gmx_pme_t*  pme_src,
                    const t_inputrec*  ir,
                    const ivec         grid_size,
                    real               ewaldcoeff_q,
                    real               ewaldcoeff_lj)
{
    // Create a copy of t_inputrec fields that are used in gmx_pme_init().
    // TODO: This would be better as just copying a sub-structure that contains
    // all the PME parameters and nothing else.
    t_inputrec irc;
    irc.ePBC                   = ir->ePBC;
    irc.coulombtype            = ir->coulombtype;
    irc.vdwtype                = ir->vdwtype;
    irc.efep                   = ir->efep;
    irc.pme_order              = ir->pme_order;
    irc.epsilon_r              = ir->epsilon_r;
    irc.ljpme_combination_rule = ir->ljpme_combination_rule;
    irc.nkx                    = grid_size[XX];
    irc.nky                    = grid_size[YY];
    irc.nkz                    = grid_size[ZZ];

    try
    {
        const gmx::MDLogger dummyLogger;
        // This is reinit which is currently only changing grid size/coefficients,
        // so we don't expect the actual logging.
        // TODO: when PME is an object, it should take reference to mdlog on construction and save it.
        GMX_ASSERT(pmedata, "Invalid PME pointer");
        NumPmeDomains numPmeDomains = { pme_src->nnodes_major, pme_src->nnodes_minor };
        *pmedata = gmx_pme_init(cr, numPmeDomains, &irc, pme_src->bFEP_q, pme_src->bFEP_lj, FALSE,
                                ewaldcoeff_q, ewaldcoeff_lj, pme_src->nthread, pme_src->runMode,
                                pme_src->gpu, nullptr, nullptr, dummyLogger);
        /* When running PME on the CPU not using domain decomposition,
         * the atom data is allocated once only in gmx_pme_(re)init().
         */
        if (!pme_src->gpu && pme_src->nnodes == 1)
        {
            gmx_pme_reinit_atoms(*pmedata, pme_src->atc[0].numAtoms(), nullptr);
        }
        // TODO this is mostly passing around current values
    }
    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR

    /* We can easily reuse the allocated pme grids in pme_src */
    reuse_pmegrids(&pme_src->pmegrid[PME_GRID_QA], &(*pmedata)->pmegrid[PME_GRID_QA]);
    /* We would like to reuse the fft grids, but that's harder */
}

void gmx_pme_calc_energy(gmx_pme_t* pme, gmx::ArrayRef<const gmx::RVec> x, gmx::ArrayRef<const real> q, real* V)
{
    pmegrids_t* grid;

    if (pme->nnodes > 1)
    {
        gmx_incons("gmx_pme_calc_energy called in parallel");
    }
    if (pme->bFEP_q)
    {
        gmx_incons("gmx_pme_calc_energy with free energy");
    }

    if (!pme->atc_energy)
    {
        pme->atc_energy = std::make_unique<PmeAtomComm>(MPI_COMM_NULL, 1, pme->pme_order, 0, true);
    }
    PmeAtomComm* atc = pme->atc_energy.get();
    atc->setNumAtoms(x.ssize());
    atc->x           = x;
    atc->coefficient = q;

    /* We only use the A-charges grid */
    grid = &pme->pmegrid[PME_GRID_QA];

    /* Only calculate the spline coefficients, don't actually spread */
    spread_on_grid(pme, atc, nullptr, TRUE, FALSE, pme->fftgrid[PME_GRID_QA], FALSE, PME_GRID_QA);

    *V = gather_energy_bsplines(pme, grid->grid.grid, atc);
}

// bisheng 2025
void gmx_pme_calc_phik(gmx_pme_t* pme, gmx::ArrayRef<const gmx::RVec> x, gmx::ArrayRef<const real> q, real* Pot, gmx_bool bFreeze)
{
    if (pme->nnodes > 1)
    {
        gmx_incons("gmx_pme_calc_phik called in parallel");
    }
    if (pme->bFEP_q)
    {
        gmx_incons("gmx_pme_calc_phik with free energy");
    }

    // Use pme->nthread for proper multi-threading support
    const int nthread = pme->nthread;

    // Create atc_phik with proper thread count for distributed splines
    if (!pme->atc_phik)
    {
        pme->atc_phik = std::make_unique<PmeAtomComm>(MPI_COMM_NULL, nthread, pme->pme_order, 0, true);
    }
    PmeAtomComm* atc = pme->atc_phik.get();

    pmegrids_t* grids = &pme->pmegrid[PME_GRID_QA];

    if (bFreeze)
    {
        // Only compute spline coefficients if electrodes are not already frozen
        if (!pme->cpm_splineFrozen)
        {
            atc->setNumAtoms(x.ssize());
            atc->x           = x;
            atc->coefficient = q;
            // Calculate and cache spline coefficients with proper multi-threading
            // bSpread=FALSE means we only compute splines, grid is not modified
            spread_on_grid(pme, atc, grids, TRUE, FALSE, pme->fftgrid[PME_GRID_QA], FALSE, PME_GRID_QA);
            pme->cpm_splineFrozen = true;
            printf("The electrodes are fixed, therefore the spline coefficients are frozen.\n");
        }
        // else: keep cached spline coefficients, do NOT call spread_on_grid again
    }
    else
    {
        atc->setNumAtoms(x.ssize());
        atc->x           = x;
        atc->coefficient = q;
        // Always recalculate for non-frozen case with proper multi-threading
        spread_on_grid(pme, atc, grids, TRUE, FALSE, pme->fftgrid[PME_GRID_QA], FALSE, PME_GRID_QA);
        pme->cpm_splineFrozen = false;
    }

    // Gather potential at electrode positions with multi-threading
    gather_energy_bsplines_peratom(pme, grids->grid.grid, atc, Pot, nthread);
}

// bisheng 2025 - Matrix-free CG: compute A*q (electrode-electrode k-space potential)
// This function reuses splines computed by gmx_pme_calc_phik (same electrode positions)
void gmx_pme_calc_Aq_kspace(gmx_pme_t*                     pme,
                            gmx::ArrayRef<const gmx::RVec> x_electrode,
                            gmx::ArrayRef<const real>      q_electrode,
                            int                            nEleAtoms,
                            const matrix                   box,
                            real*                          phi_kspace)
{
    if (pme->nnodes > 1)
    {
        gmx_incons("gmx_pme_calc_Aq_kspace called in parallel");
    }

    // bisheng 2026 - GPU PME: run the whole electrode-only k-space cycle on the
    // GPU. Spreading a full-size charge array with zeros on the electrolyte
    // produces the same grid the CPU electrode-only spread does, and the GPU
    // spread recomputes splines each call, which also covers flexible electrodes.
    if (pme->runMode == PmeRunMode::GPU)
    {
        static std::vector<real> fullCharges;
        const int                nAtomsTotal = pme_gpu_cpm_num_atoms(pme);
        GMX_RELEASE_ASSERT(nAtomsTotal >= nEleAtoms, "CPM: electrode count exceeds PME atom count");
        // assign() every call: a stale tail (atom count shrank, or leftovers from a
        // previous PME instance) must never be spread as phantom charges
        fullCharges.assign(nAtomsTotal, 0._real);
        std::copy(q_electrode.begin(), q_electrode.begin() + nEleAtoms, fullCharges.begin());
        pme_gpu_cpm_run_kspace_cycle(pme, fullCharges.data());
        pme_gpu_cpm_gather_potential(pme, phi_kspace, nEleAtoms);
        return;
    }

    // Use pme->nthread for proper multi-threading support
    const int nthread = pme->nthread;

    // Reuse atc_phik - splines were already computed by gmx_pme_calc_phik
    // Both functions use the same electrode positions, and splines only depend on positions
    GMX_ASSERT(pme->atc_phik != nullptr, "gmx_pme_calc_phik must be called before gmx_pme_calc_Aq_kspace");
    PmeAtomComm* atc = pme->atc_phik.get();

    // Update charges only (positions and splines remain the same)
    atc->coefficient = q_electrode;

    pmegrids_t* pmegrid  = &pme->pmegrid[PME_GRID_QA];
    real*       grid     = pmegrid->grid.grid;  // Actual grid data
    real*       fftgrid  = pme->fftgrid[PME_GRID_QA];
    t_complex*  cfftgrid = pme->cfftgrid[PME_GRID_QA];

    // Step 1: Spread charges on grid (reuse splines from gmx_pme_calc_phik)
    // bCalcSplines=FALSE: reuse cached splines, only do spreading
    // bSpread=TRUE: spread charges to grid
    spread_on_grid(pme, atc, pmegrid, FALSE, TRUE, fftgrid, FALSE, PME_GRID_QA);

    // When bUseThreads=true, spread_on_grid already handles wrap and copy internally
    // Only need manual wrap/copy when bUseThreads=false
    if (!pme->bUseThreads)
    {
        wrap_periodic_pmegrid(pme, grid);
        copy_pmegrid_to_fftgrid(pme, grid, fftgrid, PME_GRID_QA);
    }

    // Steps 2-4: FFT forward, solve, FFT backward, copy back
    // These must be done in a parallel region with ALL pme->nthread threads,
    // because the FFT was initialized for parallel execution with that many threads.
    real vol = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];

    #pragma omp parallel num_threads(nthread)
    {
        int thread = gmx_omp_get_thread_num();

        // FFT forward (real to complex)
        gmx_parallel_3dfft_execute(pme->pfft_setup[PME_GRID_QA], GMX_FFT_REAL_TO_COMPLEX, thread, nullptr);

        // Solve in k-space (multiply by Green's function)
        solve_pme_yzx(pme, cfftgrid, vol, FALSE, nthread, thread);

        // FFT backward (complex to real)
        gmx_parallel_3dfft_execute(pme->pfft_setup[PME_GRID_QA], GMX_FFT_COMPLEX_TO_REAL, thread, nullptr);

        // Copy fftgrid back to pmegrid
        copy_fftgrid_to_pmegrid(pme, fftgrid, grid, PME_GRID_QA, nthread, thread);
    }

    unwrap_periodic_pmegrid(pme, grid);

    // Step 5: Gather potential at electrode positions with multi-threading
    gather_energy_bsplines_peratom(pme, grid, atc, phi_kspace, nthread);
}

// bisheng 2025 - Build neighbor list for electrode short-range interactions
// Uses buffer distance so list can be reused for many MD steps
// bisheng 2025 - Modified to use per-atom eta
static void build_electrode_nlist(gmx_pme_t*                     pme,
                                  gmx::ArrayRef<const gmx::RVec> x_electrode,
                                  int                            nEleAtoms,
                                  const matrix                   box,
                                  real                           beta,
                                  real                           rcoulomb,
                                  gmx_bool                       b3dc,
                                  int                            nthreads)
{
    g_cpmNlistGen++; // bisheng 2026 - the device copy of the CSR list is now stale
    // Uses pme->eta_per_atom for per-atom eta values
    const real buffer   = 0.2;  // Buffer distance (nm) - list valid until atoms move this far
    const real rc_list  = rcoulomb + buffer;  // Neighbor list cutoff
    const real rc_list2 = rc_list * rc_list;
    const real rc2      = rcoulomb * rcoulomb;

    const real boxx = box[XX][XX];
    const real boxy = box[YY][YY];
    const real boxz = box[ZZ][ZZ];

    // Store reference positions and buffer
    pme->sr_nlist_x_ref.resize(nEleAtoms);
    for (int i = 0; i < nEleAtoms; i++)
    {
        pme->sr_nlist_x_ref[i] = x_electrode[i];
    }
    pme->sr_nlist_buffer = buffer;

    // First pass: count neighbors per atom (parallel)
    std::vector<int> neighbor_count(nEleAtoms, 0);

    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; i++)
    {
        int count = 0;
        const real xi = x_electrode[i][XX];
        const real yi = x_electrode[i][YY];
        const real zi = x_electrode[i][ZZ];

        for (int j = 0; j < nEleAtoms; j++)
        {
            if (i == j) continue;

            real dx = x_electrode[j][XX] - xi;
            real dy = x_electrode[j][YY] - yi;
            real dz = x_electrode[j][ZZ] - zi;

            dx -= boxx * std::round(dx / boxx);
            dy -= boxy * std::round(dy / boxy);
            if (!b3dc)
            {
                dz -= boxz * std::round(dz / boxz);
            }

            real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rc_list2 && r2 > 1e-14)
            {
                count++;
            }
        }
        neighbor_count[i] = count;
    }

    // Build row pointers
    pme->sr_nlist_row_ptr.resize(nEleAtoms + 1);
    pme->sr_nlist_row_ptr[0] = 0;
    for (int i = 0; i < nEleAtoms; i++)
    {
        pme->sr_nlist_row_ptr[i + 1] = pme->sr_nlist_row_ptr[i] + neighbor_count[i];
    }

    int total_pairs = pme->sr_nlist_row_ptr[nEleAtoms];
    pme->sr_nlist_col_idx.resize(total_pairs);
    pme->sr_nlist_coeff.resize(total_pairs);

    // Second pass: fill neighbor list and precompute coefficients (parallel)
    // bisheng 2025 - Use per-atom eta to compute eta_ij for each pair
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; i++)
    {
        int idx = pme->sr_nlist_row_ptr[i];
        const real xi = x_electrode[i][XX];
        const real yi = x_electrode[i][YY];
        const real zi = x_electrode[i][ZZ];
        const real eta_i  = pme->eta_per_atom[i];
        const real eta_i2 = eta_i * eta_i;

        for (int j = 0; j < nEleAtoms; j++)
        {
            if (i == j) continue;

            real dx = x_electrode[j][XX] - xi;
            real dy = x_electrode[j][YY] - yi;
            real dz = x_electrode[j][ZZ] - zi;

            dx -= boxx * std::round(dx / boxx);
            dy -= boxy * std::round(dy / boxy);
            if (!b3dc)
            {
                dz -= boxz * std::round(dz / boxz);
            }

            real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 < rc_list2 && r2 > 1e-14)
            {
                pme->sr_nlist_col_idx[idx] = j;

                // Precompute coefficient only if within actual cutoff
                if (r2 < rc2)
                {
                    real r     = std::sqrt(r2);
                    real inv_r = 1.0 / r;
                    real term1 = std::erfc(beta * r) * inv_r;

                    // Compute combined eta_ij = eta_i * eta_j / sqrt(eta_i^2 + eta_j^2)
                    // For same eta: eta_ij = eta/sqrt(2), matching MetalWalls
                    const real eta_j  = pme->eta_per_atom[j];
                    const real eta_j2 = eta_j * eta_j;
                    const real eta_ij = (eta_i * eta_j) / std::sqrt(eta_i2 + eta_j2);

                    real term2 = std::erfc(eta_ij * r) * inv_r;
                    pme->sr_nlist_coeff[idx] = term1 - term2;
                }
                else
                {
                    // Pair is in buffer zone - coefficient is 0 (beyond cutoff)
                    pme->sr_nlist_coeff[idx] = 0.0;
                }
                idx++;
            }
        }
    }

    pme->sr_nlist_valid = TRUE;
    printf("Built electrode neighbor list: %d atoms, %d pairs (%.1f pairs/atom avg)\n",
           nEleAtoms, total_pairs, (real)total_pairs / nEleAtoms);
}

// bisheng 2025 - Check if neighbor list needs rebuild
static gmx_bool check_nlist_needs_rebuild(gmx_pme_t*                     pme,
                                          gmx::ArrayRef<const gmx::RVec> x_electrode,
                                          int                            nEleAtoms)
{
    if (!pme->sr_nlist_valid || pme->sr_nlist_x_ref.empty())
    {
        return TRUE;
    }

    // Check max displacement - if any atom moved more than buffer/2, rebuild
    const real threshold2 = (pme->sr_nlist_buffer * 0.5) * (pme->sr_nlist_buffer * 0.5);

    for (int i = 0; i < nEleAtoms; i++)
    {
        real dx = x_electrode[i][XX] - pme->sr_nlist_x_ref[i][XX];
        real dy = x_electrode[i][YY] - pme->sr_nlist_x_ref[i][YY];
        real dz = x_electrode[i][ZZ] - pme->sr_nlist_x_ref[i][ZZ];
        real d2 = dx * dx + dy * dy + dz * dz;

        if (d2 > threshold2)
        {
            return TRUE;
        }
    }
    return FALSE;
}

// bisheng 2025 - Matrix-free CG: compute short-range electrode-electrode potential
// Uses neighbor list for efficiency - list is built once and reused for many MD steps
// Uses pme->eta_per_atom for per-atom eta values
void gmx_pme_calc_Aq_sr(gmx_pme_t*                     pme,
                        gmx::ArrayRef<const gmx::RVec> x_electrode,
                        gmx::ArrayRef<const real>      q_electrode,
                        int                            nEleAtoms,
                        const matrix                   box,
                        real                           beta,
                        real                           rcoulomb,
                        real                           epsilon_r,
                        real*                          phi_sr,
                        int                            nthreads,
                        gmx_bool                       b3dc)
{
    (void)epsilon_r;  // Unused - kept for API compatibility

    // Check if neighbor list needs rebuild
    if (check_nlist_needs_rebuild(pme, x_electrode, nEleAtoms))
    {
        build_electrode_nlist(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads);
    }

    // Use neighbor list for efficient A*q computation
    // phi_sr[i] = sum_j coeff[i,j] * q[j]
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; i++)
    {
        real phi_i = 0.0;
        const int start = pme->sr_nlist_row_ptr[i];
        const int end   = pme->sr_nlist_row_ptr[i + 1];

        for (int idx = start; idx < end; idx++)
        {
            const int j     = pme->sr_nlist_col_idx[idx];
            const real coef = pme->sr_nlist_coeff[idx];
            phi_i += coef * q_electrode[j];
        }

        phi_sr[i] = phi_i;
    }
}

// bisheng 2025 - Compute short-range electrode-electrode forces for flexible electrodes
// SIMD kernel + PME gives 1/r for electrode-electrode, but we need erf(eta_ij*r)/r
// This adds the correction: -erfc(eta_ij*r)/r so total becomes erf(eta_ij*r)/r
// Force expression: F_i = sum_j q_i * q_j * f(r_ij) * r_hat_ij
// where f(r) = -d/dr[-erfc(eta_ij*r)/r] = -2*eta_ij/sqrt(pi)*exp(-eta_ij^2*r^2)/r - erfc(eta_ij*r)/r^2
// Note: Assumes Aq_sr has been called first so neighbor list is already built
// Uses pme->eta_per_atom for per-atom eta values
void gmx_pme_calc_Aq_sr_force(gmx_pme_t*                     pme,
                              gmx::ArrayRef<const gmx::RVec> x_electrode,
                              gmx::ArrayRef<const real>      q_electrode,
                              int                            nEleAtoms,
                              const matrix                   box,
                              real                           rcoulomb,
                              real                           coulomb_factor,
                              gmx::ArrayRef<gmx::RVec>       f_electrode,
                              int                            nthreads,
                              gmx_bool                       b3dc)
{
    const real inv_sqrt_pi = 1.0 / std::sqrt(M_PI);
    const real rc2         = rcoulomb * rcoulomb;

    const real boxx = box[XX][XX];
    const real boxy = box[YY][YY];
    const real boxz = box[ZZ][ZZ];

    // bisheng 2026 - the Extended-Lagrangian ordering computes forces BEFORE the
    // first Aq_sr call of the run, so the neighbor list may not exist yet.
    // Apply the same rebuild check as gmx_pme_calc_Aq_sr.
    if (check_nlist_needs_rebuild(pme, x_electrode, nEleAtoms))
    {
        build_electrode_nlist(pme, x_electrode, nEleAtoms, box, pme->ewaldcoeff_q, rcoulomb,
                              b3dc, nthreads);
    }

    // Use thread-local force buffers to avoid race conditions
    std::vector<std::vector<gmx::RVec>> f_local(nthreads);
    for (int t = 0; t < nthreads; t++)
    {
        f_local[t].resize(nEleAtoms, {0.0, 0.0, 0.0});
    }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = gmx_omp_get_thread_num();
        gmx::RVec* f_thread = f_local[tid].data();

        #pragma omp for
        for (int i = 0; i < nEleAtoms; i++)
        {
            const real qi = q_electrode[i];
            const real xi = x_electrode[i][XX];
            const real yi = x_electrode[i][YY];
            const real zi = x_electrode[i][ZZ];
            const real eta_i  = pme->eta_per_atom[i];
            const real eta_i2 = eta_i * eta_i;

            const int start = pme->sr_nlist_row_ptr[i];
            const int end   = pme->sr_nlist_row_ptr[i + 1];

            for (int idx = start; idx < end; idx++)
            {
                const int  j  = pme->sr_nlist_col_idx[idx];
                const real qj = q_electrode[j];

                // Compute distance vector with PBC
                real dx = x_electrode[j][XX] - xi;
                real dy = x_electrode[j][YY] - yi;
                real dz = x_electrode[j][ZZ] - zi;

                dx -= boxx * std::round(dx / boxx);
                dy -= boxy * std::round(dy / boxy);
                if (!b3dc)
                {
                    dz -= boxz * std::round(dz / boxz);
                }

                real r2 = dx * dx + dy * dy + dz * dz;

                // Only compute for pairs within actual cutoff
                if (r2 < rc2 && r2 > 1e-14)
                {
                    real r      = std::sqrt(r2);
                    real inv_r  = 1.0 / r;
                    real inv_r2 = inv_r * inv_r;

                    // Compute combined eta_ij = eta_i * eta_j / sqrt(eta_i^2 + eta_j^2)
                    // For same eta: eta_ij = eta/sqrt(2), matching MetalWalls
                    const real eta_j      = pme->eta_per_atom[j];
                    const real eta_j2     = eta_j * eta_j;
                    const real eta_ij     = (eta_i * eta_j) / std::sqrt(eta_i2 + eta_j2);
                    const real two_eta_ij = 2.0 * eta_ij;
                    const real eta_ij_sq  = eta_ij * eta_ij;

                    // Compute force magnitude: -d/dr[-erfc(eta_ij*r)/r]
                    // SIMD kernel + PME already gives 1/r, we need to correct to erf(eta_ij*r)/r
                    // Correction potential: -erfc(eta_ij*r)/r
                    // Force = -d/dr[-erfc(eta_ij*r)/r] = -2*eta_ij/sqrt(pi)*exp(-eta_ij^2*r^2)/r - erfc(eta_ij*r)/r^2
                    real exp_eta_ij2 = std::exp(-eta_ij_sq * r2);
                    real erfc_eta_ij = std::erfc(eta_ij * r);

                    real f_mag = - two_eta_ij * inv_sqrt_pi * exp_eta_ij2 * inv_r - erfc_eta_ij * inv_r2;

                    // Force on i from j: F_i = q_i * q_j * f_mag * r_hat
                    // r_hat = (r_j - r_i) / r = (dx, dy, dz) / r
                    real fscal = coulomb_factor * qi * qj * f_mag * inv_r;

                    // Add force to i
                    f_thread[i][XX] += fscal * dx;
                    f_thread[i][YY] += fscal * dy;
                    f_thread[i][ZZ] += fscal * dz;
                }
            }
        }
    }

    // Reduce thread-local forces to output
    for (int i = 0; i < nEleAtoms; i++)
    {
        for (int t = 0; t < nthreads; t++)
        {
            f_electrode[i][XX] += f_local[t][i][XX];
            f_electrode[i][YY] += f_local[t][i][YY];
            f_electrode[i][ZZ] += f_local[t][i][ZZ];
        }
    }
}

// bisheng 2025 - Matrix-free CG: compute self-term for electrode atoms
// Modified to support per-atom eta
void gmx_pme_calc_Aq_self(gmx::ArrayRef<const real> q_electrode,
                          gmx::ArrayRef<const real> eta_per_atom,
                          int                       nEleAtoms,
                          real                      beta,
                          real                      epsilon_r,
                          real*                     phi_self)
{
    // Self-term: phi_self(i) = (sqrt(2)*eta[i] - 2*beta) / sqrt(pi) * q_i
    // This matches cpm_matrix.cpp calc_matrix_self with per-atom eta
    // No Coulomb constant - matrix is in 1/nm units
    (void)epsilon_r;  // Unused - kept for API compatibility
    const real inv_sqrt_pi = 1.0 / std::sqrt(M_PI);

    for (int i = 0; i < nEleAtoms; i++)
    {
        const real self_coeff = (std::sqrt(2.0) * eta_per_atom[i] - 2.0 * beta) * inv_sqrt_pi;
        phi_self[i] = self_coeff * q_electrode[i];
    }
}

// bisheng 2025 - Matrix-free CG: compute 3DC correction for electrode atoms
void gmx_pme_calc_Aq_3dc(gmx::ArrayRef<const gmx::RVec> x_electrode,
                         gmx::ArrayRef<const real>      q_electrode,
                         int                            nEleAtoms,
                         const matrix                   box,
                         real                           epsilon_r,
                         real*                          phi_3dc)
{
    // 3DC correction: phi_3dc(i) = (4*pi/V) * z_i * sum_j(z_j * q_j)
    // This matches cpm_matrix.cpp line 240: factor = 4.0 * c_pi / volume
    // Z-coordinates are shifted so center of box is at z=0 (line 239, 247-248)
    // No Coulomb constant - matrix is in 1/nm units
    (void)epsilon_r;  // Unused - kept for API compatibility

    real vol       = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];
    real boxz      = box[ZZ][ZZ];
    real zShift    = boxz / 2.0;  // Shift z so center of box is at z=0
    real prefactor = 4.0 * M_PI / vol;

    // Compute dipole moment sum: Σ_j (z_j - zShift) * q_j
    real dipole_sum = 0.0;
    for (int j = 0; j < nEleAtoms; j++)
    {
        real zj = x_electrode[j][ZZ] - zShift;
        dipole_sum += zj * q_electrode[j];
    }

    // Apply correction to each electrode atom
    for (int i = 0; i < nEleAtoms; i++)
    {
        real zi = x_electrode[i][ZZ] - zShift;
        phi_3dc[i] = prefactor * zi * dipole_sum;
    }
}

// bisheng 2025 - Debug function to compare matrix-free A*q with matrix A*q
static void debug_compare_Aq_components(gmx_pme_t*                     pme,
                                        gmx::ArrayRef<const gmx::RVec> x_electrode,
                                        int                            nEleAtoms,
                                        const matrix                   box,
                                        real                           beta,
                                        real                           rcoulomb,
                                        gmx_bool                       b3dc,
                                        int                            nthreads)
{
    printf("\n========== DEBUG: Comparing A*q components ==========\n");

    // Create random charge vector that sums to zero
    std::vector<real> q_test(nEleAtoms);
    // bisheng 2026: use a local generator -- std::srand would reseed the
    // process-wide C rand() state that other libraries in this process may rely on.
    std::mt19937                          rng(12345);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> qdist(-1.0, 1.0);
    real sum = 0.0;
    for (int i = 0; i < nEleAtoms - 1; i++)
    {
        // Random values between -1 and 1
        q_test[i] = static_cast<real>(qdist(rng));
        sum += q_test[i];
    }
    // Last charge ensures sum is zero
    q_test[nEleAtoms - 1] = -sum;

    // Verify sum is zero
    real total_charge = 0.0;
    for (int i = 0; i < nEleAtoms; i++) total_charge += q_test[i];
    printf("Test charge vector: random values, sum = %.6e\n", total_charge);
    printf("First few charges: q[0]=%.4f, q[1]=%.4f, q[2]=%.4f, q[3]=%.4f, q[4]=%.4f\n",
           q_test[0], q_test[1], q_test[2], q_test[3], q_test[4]);

    gmx::ArrayRef<const real> q_ref(q_test);

    // Compute each component using matrix-free method
    std::vector<real> phi_kspace(nEleAtoms, 0.0);
    std::vector<real> phi_sr(nEleAtoms, 0.0);
    std::vector<real> phi_self(nEleAtoms, 0.0);
    std::vector<real> phi_3dc(nEleAtoms, 0.0);

    // K-space (reuses splines from gmx_pme_calc_phik)
    gmx_pme_calc_Aq_kspace(pme, x_electrode, q_ref, nEleAtoms, box, phi_kspace.data());

    // Short-range
    gmx_pme_calc_Aq_sr(pme, x_electrode, q_ref, nEleAtoms, box, beta, rcoulomb,
                       pme->epsilon_r, phi_sr.data(), nthreads, b3dc);

    // Self-term - use per-atom eta
    gmx_pme_calc_Aq_self(q_ref, pme->eta_per_atom, nEleAtoms, beta, pme->epsilon_r, phi_self.data());

    // 3DC correction
    if (b3dc)
    {
        gmx_pme_calc_Aq_3dc(x_electrode, q_ref, nEleAtoms, box, pme->epsilon_r, phi_3dc.data());
    }

    // Convert k-space from kJ/mol/e to 1/nm
    constexpr real potUnit = 1.0 / ELECTRONVOLT;
    constexpr real evscale = ELECTRONVOLT / ONE_4PI_EPS0;
    std::vector<real> phi_kspace_nm(nEleAtoms);
    for (int i = 0; i < nEleAtoms; i++)
    {
        phi_kspace_nm[i] = phi_kspace[i] * evscale * potUnit;
    }

    // Print first few values from matrix-free
    printf("\nMatrix-free A*q (random q, sum=0):\n");
    printf("  i    phi_kspace(kJ/mol)  phi_kspace(1/nm)    phi_sr(1/nm)    phi_self(1/nm)   phi_3dc(1/nm)      Total(1/nm)\n");
    for (int i = 0; i < std::min(10, nEleAtoms); i++)
    {
        real total = phi_kspace_nm[i] + phi_sr[i] + phi_self[i];
        if (b3dc) total += phi_3dc[i];
        printf("  %d    %14.8f    %14.8f    %14.8f    %14.8f    %14.8f    %14.8f\n",
               i, phi_kspace[i], phi_kspace_nm[i], phi_sr[i], phi_self[i], phi_3dc[i], total);
    }

    // Load matrix and compute A*q for comparison
    printf("\nLoading matrix and computing A*q for comparison...\n");

    // Lambda to compute A*q from matrix file
    auto compute_Aq_from_matrix = [nEleAtoms, &q_test](const char* filename, std::vector<double>& result) -> bool {
        FILE* fp = fopen(filename, "r");
        if (!fp)
        {
            printf("  Warning: Cannot open %s\n", filename);
            return false;
        }
        result.resize(nEleAtoms, 0.0);
        for (int i = 0; i < nEleAtoms; i++)
        {
            double row_sum = 0.0;
            for (int j = 0; j < nEleAtoms; j++)
            {
                double val;
                if (fscanf(fp, "%lf", &val) != 1)
                {
                    fclose(fp);
                    return false;
                }
                row_sum += val * q_test[j];  // A[i][j] * q[j]
            }
            result[i] = row_sum;
        }
        fclose(fp);
        return true;
    };

    std::vector<double> mat_sr, mat_self, mat_3dc, mat_kspace, mat_total;
    bool have_sr = compute_Aq_from_matrix("MatrixA_sr.dat", mat_sr);
    bool have_self = compute_Aq_from_matrix("MatrixA_self.dat", mat_self);
    bool have_3dc = compute_Aq_from_matrix("MatrixA_3dc.dat", mat_3dc);
    bool have_kspace = compute_Aq_from_matrix("MatrixA_kspace.dat", mat_kspace);
    bool have_total = compute_Aq_from_matrix("MatrixA.dat", mat_total);

    if (have_sr && have_self && have_kspace)
    {
        printf("\nMatrix A*q (full matrix-vector product):\n");
        printf("  i    mat_kspace(1/nm)    mat_sr(1/nm)    mat_self(1/nm)    mat_3dc(1/nm)      mat_total(1/nm)\n");
        for (int i = 0; i < std::min(10, nEleAtoms); i++)
        {
            printf("  %d    %14.8f    %14.8f    %14.8f    %14.8f    %14.8f\n",
                   i, mat_kspace[i], mat_sr[i], mat_self[i], mat_3dc[i], mat_total[i]);
        }

        // Compare differences
        printf("\nDifference (matrix-free - matrix):\n");
        printf("  i    diff_kspace         diff_sr             diff_self           diff_3dc            diff_total\n");
        double sum_diff_ks = 0, sum_diff_sr = 0, sum_diff_self = 0, sum_diff_3dc = 0, sum_diff_total = 0;
        for (int i = 0; i < std::min(10, nEleAtoms); i++)
        {
            real total_mf = phi_kspace_nm[i] + phi_sr[i] + phi_self[i];
            if (b3dc) total_mf += phi_3dc[i];
            double diff_ks = phi_kspace_nm[i] - mat_kspace[i];
            double diff_sr = phi_sr[i] - mat_sr[i];
            double diff_self = phi_self[i] - mat_self[i];
            double diff_3dc = phi_3dc[i] - mat_3dc[i];
            double diff_total = total_mf - mat_total[i];
            printf("  %d    %14.8e    %14.8e    %14.8e    %14.8e    %14.8e\n",
                   i, diff_ks, diff_sr, diff_self, diff_3dc, diff_total);
            sum_diff_ks += std::abs(diff_ks);
            sum_diff_sr += std::abs(diff_sr);
            sum_diff_self += std::abs(diff_self);
            sum_diff_3dc += std::abs(diff_3dc);
            sum_diff_total += std::abs(diff_total);
        }
        printf("\nSum of absolute differences (first 10 atoms):\n");
        printf("  kspace: %g, sr: %g, self: %g, 3dc: %g, total: %g\n",
               sum_diff_ks, sum_diff_sr, sum_diff_self, sum_diff_3dc, sum_diff_total);

        // Also compute relative error
        double max_rel_err = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            real total_mf = phi_kspace_nm[i] + phi_sr[i] + phi_self[i];
            if (b3dc) total_mf += phi_3dc[i];
            if (std::abs(mat_total[i]) > 1e-10)
            {
                double rel_err = std::abs((total_mf - mat_total[i]) / mat_total[i]);
                if (rel_err > max_rel_err) max_rel_err = rel_err;
            }
        }
        printf("Max relative error (total): %g\n", max_rel_err);
    }
    else
    {
        printf("  Could not load all matrix component files.\n");
        printf("  Make sure MatrixA_sr.dat, MatrixA_self.dat, MatrixA_3dc.dat, MatrixA_kspace.dat exist.\n");
    }

    printf("========== END DEBUG ==========\n\n");
}

// bisheng 2025 - Matrix-free CG: compute A*q (combines all components)
// NOTE: gmx_pme_calc_phik must be called before this function to compute splines
// Uses pme->eta_per_atom for per-atom eta values
/* bisheng 2026 - hand the device CG everything it needs for this step.
 * Rebuilds the electrode neighbour list on the host when the electrode moved,
 * uploads it when it changed, and uploads the per-step z / self / TF data.
 * Returns false when the device path is not in use. */
static bool cpm_gpu_cg_prepare(gmx_pme_t*                     pme,
                               gmx::ArrayRef<const gmx::RVec> x_electrode,
                               int                            nEleAtoms,
                               const matrix                   box,
                               real                           beta,
                               real                           rcoulomb,
                               gmx_bool                       b3dc,
                               int                            nthreads)
{
    static bool bInit = false;
    if (!bInit)
    {
        bInit         = true;
        g_cpmUseGpuCg = cpm_gpu_cg_available() && pme->runMode == PmeRunMode::GPU
                        && getenv("GMX_CPM_NO_GPU_CG") == nullptr;
        printf("CPM: matrix-free CG %s\n",
               g_cpmUseGpuCg ? "on the GPU (device-resident A q, projections and vector algebra)" : "on the CPU");
    }
    if (!g_cpmUseGpuCg)
    {
        return false;
    }
    if (check_nlist_needs_rebuild(pme, x_electrode, nEleAtoms))
    {
        build_electrode_nlist(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads);
    }
    if (g_cpmNlistGenUploaded != g_cpmNlistGen)
    {
        cpm_gpu_cg_set_nlist(pme->sr_nlist_row_ptr.data(), pme->sr_nlist_col_idx.data(),
                             pme->sr_nlist_coeff.data(), nEleAtoms, pme->sr_nlist_row_ptr[nEleAtoms]);
        g_cpmNlistGenUploaded = g_cpmNlistGen;
    }
    static std::vector<real> z, selfCoef;
    z.resize(nEleAtoms);
    selfCoef.resize(nEleAtoms);
    const real zShift      = box[ZZ][ZZ] / 2.0;
    const real inv_sqrt_pi = 1.0 / std::sqrt(M_PI);
    for (int i = 0; i < nEleAtoms; i++)
    {
        z[i]        = x_electrode[i][ZZ] - zShift;
        selfCoef[i] = (std::sqrt(2.0) * pme->eta_per_atom[i] - 2.0 * beta) * inv_sqrt_pi;
    }
    const real     vol     = box[XX][XX] * box[YY][YY] * box[ZZ][ZZ];
    const real     pref    = b3dc ? 4.0 * M_PI / vol : 0.0;
    constexpr real potUnit = 1.0 / ELECTRONVOLT;
    constexpr real evscale = ELECTRONVOLT / ONE_4PI_EPS0;
    cpm_gpu_cg_set_step(z.data(), selfCoef.data(), pme->bThomasFermi ? pme->tf_diag.data() : nullptr,
                        nEleAtoms, pref, evscale, potUnit);
    return true;
}

void gmx_pme_calc_Aq(gmx_pme_t*                     pme,
                     gmx::ArrayRef<const gmx::RVec> x_electrode,
                     gmx::ArrayRef<const real>      q_electrode,
                     int                            nEleAtoms,
                     const matrix                   box,
                     real                           beta,
                     real                           rcoulomb,
                     gmx_bool                       b3dc,
                     real*                          Aq,
                     int                            nthreads)
{
    // Debug: compare components on first call.
    // bisheng 2026: developer diagnostic -- off unless explicitly asked for,
    // it prints ~60 lines and does a throwaway four-component A*q on step 0.
    static bool bDebugOnce = (getenv("GMX_CPM_DEBUG_AQ") != nullptr);
    if (bDebugOnce)
    {
        debug_compare_Aq_components(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads);
        bDebugOnce = false;
    }
    // A*q = φ_kspace + φ_sr + φ_self + φ_3dc (if enabled)

    // Allocate temporary arrays (static to avoid repeated allocation)
    if (cpm_gpu_cg_prepare(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads))
    {
        cpm_gpu_cg_Aq(pme, q_electrode.data(), Aq, nEleAtoms); // bisheng 2026 - device A q
        return;
    }
    static std::vector<real> phi_kspace;
    static std::vector<real> phi_sr;
    static std::vector<real> phi_self;
    static std::vector<real> phi_3dc;

    if (static_cast<int>(phi_kspace.size()) < nEleAtoms)
    {
        phi_kspace.resize(nEleAtoms);
        phi_sr.resize(nEleAtoms);
        phi_self.resize(nEleAtoms);
        phi_3dc.resize(nEleAtoms);
    }

    // Step 1: k-space contribution (reuses splines from gmx_pme_calc_phik)
    gmx_pme_calc_Aq_kspace(pme, x_electrode, q_electrode, nEleAtoms, box,
                           phi_kspace.data());

    // Step 2: short-range contribution
    gmx_pme_calc_Aq_sr(pme, x_electrode, q_electrode, nEleAtoms, box,
                       beta, rcoulomb, pme->epsilon_r, phi_sr.data(), nthreads, b3dc);

    // Step 3: self-term - use per-atom eta
    gmx_pme_calc_Aq_self(q_electrode, pme->eta_per_atom, nEleAtoms, beta, pme->epsilon_r, phi_self.data());

    // Step 4: 3DC correction (if enabled)
    if (b3dc)
    {
        gmx_pme_calc_Aq_3dc(x_electrode, q_electrode, nEleAtoms, box, pme->epsilon_r, phi_3dc.data());
    }

    // Sum all contributions - final result should be in V (volts)
    // to match b_target = setPot - (psr_coul + plr_coul) * potUnit which is in V
    //
    // phi_kspace from PME is in kJ/mol/e (with charges in e)
    // phi_sr, phi_self, phi_3dc are in 1/nm (matching cpm_matrix.cpp)
    //
    // Unit conversion constants derived from units.h
    // potUnit converts kJ/mol/e to V (eV): 1 eV = ELECTRONVOLT kJ/mol
    // evscale converts V*nm to e: evscale = 1/(ONE_4PI_EPS0 * potUnit)
    constexpr real potUnit = 1.0 / ELECTRONVOLT;
    constexpr real evscale = ELECTRONVOLT / ONE_4PI_EPS0;
    // printf("potUnit=%g, evscale=%g\n", potUnit, evscale);

    // phi_kspace (kJ/mol/e) * potUnit gives V (since potUnit converts kJ/mol/e to eV = V)
    // But we need 1/nm to match other components, so multiply by evscale * potUnit

    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; i++)
    {
        // Convert phi_kspace from kJ/mol/e to 1/nm by multiplying by evscale * potUnit
        real phi_ks_nm = phi_kspace[i] * evscale * potUnit;

        // Sum all in 1/nm units - this is A*q where A is in 1/nm
        // The result represents potential in V (since q is in V*nm during CG)
        Aq[i] = phi_ks_nm + phi_sr[i] + phi_self[i];
        if (b3dc)
        {
            Aq[i] += phi_3dc[i];
        }
        // bisheng 2026 - Thomas-Fermi diagonal (same term cpm_matrix.cpp adds to A_ii)
        if (pme->bThomasFermi)
        {
            Aq[i] += pme->tf_diag[i] * q_electrode[i];
        }
        // No further conversion needed - result is already in correct units
        // to match b_target which is in V
    }

    // Debug: print component statistics on first call
    static bool bDebugAq = false;
    if (bDebugAq)
    {
        double sum_ks = 0, sum_sr = 0, sum_self = 0, sum_3dc = 0, sum_Aq = 0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            sum_ks += phi_kspace[i];
            sum_sr += phi_sr[i];
            sum_self += phi_self[i];
            if (b3dc) sum_3dc += phi_3dc[i];
            sum_Aq += Aq[i];
        }
        printf("DEBUG Aq components (GROMACS units, before potUnit): kspace=%g, sr=%g, self=%g, 3dc=%g\n",
               sum_ks, sum_sr, sum_self, sum_3dc);
        printf("DEBUG Aq total (eV): sum=%g, mean=%g\n", sum_Aq, sum_Aq/nEleAtoms);
        printf("DEBUG first atom: kspace[0]=%g, sr[0]=%g, self[0]=%g, Aq[0]=%g eV\n",
               phi_kspace[0], phi_sr[0], phi_self[0], Aq[0]);
        bDebugAq = false;
    }
}

// bisheng 2025 - Matrix-free CG solver for electrode charges (LAMMPS-style projected CG)
// Solves (P*A*P)*q = P*b where P projects onto charge-neutral subspace
// This implicitly solves: A*q = b + k*[1,1,...,1] with Σq = 0
// Fixed to follow LAMMPS electrode package approach for robust convergence
int gmx_pme_solve_electrode_charges_cg(gmx_pme_t*                     pme,
                                       gmx::ArrayRef<const gmx::RVec> x_electrode,
                                       const real*                    b,
                                       int                            nEleAtoms,
                                       const matrix                   box,
                                       real                           beta,
                                       real                           eta,
                                       real                           rcoulomb,
                                       gmx_bool                       b3dc,
                                       real*                          q_out,
                                       real                           tol,
                                       int                            maxiter,
                                       int                            nthreads,
                                       gmx_bool                       bColdStart)
{
    // Helper lambda: project vector onto charge-neutral subspace (v -= mean(v))
    auto project_neutral = [nEleAtoms](real* v) {
        // bisheng 2026: accumulate in double and project twice.
        // In float, summing thousands of terms leaves ~1e-6 of the constraint
        // direction in v; A^-1 amplifies that leak along the softest mode and the
        // CG recurrence then diverges. Double + a second pass makes the leak
        // negligible at float storage precision.
        for (int pass = 0; pass < 2; pass++)
        {
            double sum = 0.0;
            for (int i = 0; i < nEleAtoms; i++) sum += static_cast<double>(v[i]);
            const double mean = sum / nEleAtoms;
            for (int i = 0; i < nEleAtoms; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) - mean);
        }
    };

    // Static storage for warm start
    // bisheng 2026 - warm start lives in pme so it can be checkpointed
    std::vector<real>& q_prev = pme->cpm_qPrev;
    const bool bFirstCall = (static_cast<int>(q_prev.size()) != nEleAtoms);

    // Allocate working arrays
    std::vector<real> q(nEleAtoms);            // Current solution (always neutral)
    std::vector<real> r(nEleAtoms);            // Residual (projected)
    std::vector<real> d(nEleAtoms);            // Search direction (projected)
    std::vector<real> Ad(nEleAtoms);           // A*d product

    // Initialize q: warm start or zeros
    if (bColdStart || bFirstCall || static_cast<int>(q_prev.size()) != nEleAtoms)
    {
        q_prev.resize(nEleAtoms, 0.0);
        std::fill(q.begin(), q.end(), 0.0);
        // (bFirstCall is recomputed from cpm_qPrev's size each call)
    }
    else
    {
        q = q_prev;
    }

    if (cpm_gpu_cg_prepare(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads))
    {
        /* bisheng 2026 - the whole CG runs on the device; q carries the warm start in */
        double    U     = 0.0;
        const int itGpu = cpm_gpu_cg_solve(pme, b, q.data(), nEleAtoms, 0, 0.0, 0.0, tol, maxiter, &U);
        for (int i = 0; i < nEleAtoms; i++)
        {
            q_out[i] = q[i];
            if (!bColdStart)
            {
                q_prev[i] = q[i];
            }
        }
        return itGpu;
    }

    // Ensure initial q is neutral
    project_neutral(q.data());

    // Compute initial residual: r = b - A*q
    gmx::ArrayRef<const real> q_ref(q);
    gmx_pme_calc_Aq(pme, x_electrode, q_ref, nEleAtoms, box, beta, rcoulomb,
                    b3dc, r.data(), nthreads);

    for (int i = 0; i < nEleAtoms; i++)
    {
        r[i] = b[i] - r[i];
    }
    project_neutral(r.data());

    // LAMMPS-style: Initial search direction d = projection(r)
    // dot_old = dot(r, d) = ||projected_r||^2
    for (int i = 0; i < nEleAtoms; i++)
    {
        d[i] = r[i];
    }
    double dot_old = 0.0;
    for (int i = 0; i < nEleAtoms; i++)
    {
        dot_old += r[i] * d[i];
    }
    double delta = dot_old;  // Convergence criterion: delta = dot(r, d)

    // CG iteration with projection (LAMMPS-style)
    int iter;
    double dot_new = 0.0, beta_cg = 0.0, dAd = 0.0, alpha = 0.0;  //  Declare outside loop for debug
    for (iter = 0; iter < maxiter && delta > tol; iter++)
    {
        // Compute A*d
        gmx::ArrayRef<const real> d_ref(d);
        gmx_pme_calc_Aq(pme, x_electrode, d_ref, nEleAtoms, box, beta, rcoulomb,
                        b3dc, Ad.data(), nthreads);

        // alpha = dot_old / (d·A*d) - LAMMPS style
        dAd = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            dAd += d[i] * Ad[i];
        }

        if (std::abs(dAd) < 1e-30)
        {
            if (iter > 100)
            {
                printf("CG DEBUG [potential] iter=%d: BREAK due to |dAd|<1e-30, dAd=%g\n", iter, dAd);
            }
            break;
        }

        alpha = dot_old / dAd;

        // Update q and r
        for (int i = 0; i < nEleAtoms; i++)
        {
            q[i] += alpha * d[i];
            r[i] -= alpha * Ad[i];
        }

        // Periodic refresh for numerical stability (every 20 iterations, like LAMMPS)
        bool bRestart = false;
        if ((iter + 1) % 20 == 0)
        {
            // Recompute r from scratch to avoid accumulating errors
            project_neutral(q.data());
            gmx::ArrayRef<const real> q_ref_refresh(q);
            gmx_pme_calc_Aq(pme, x_electrode, q_ref_refresh, nEleAtoms, box, beta, rcoulomb,
                            b3dc, r.data(), nthreads);
            for (int i = 0; i < nEleAtoms; i++)
            {
                r[i] = b[i] - r[i];
            }
            // After refresh, force restart (beta=0) to ensure d is consistent with fresh r
            bRestart = true;
        }

        // Project residual onto constraint subspace
        project_neutral(r.data());

        // LAMMPS-style: dot_new = dot(r, projection(r)) = ||projected_r||^2
        dot_new = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            dot_new += r[i] * r[i];
        }

        // Update search direction: d = r + (dot_new/dot_old) * d
        // After refresh, use beta=0 to restart CG with consistent state
        beta_cg = bRestart ? 0.0 : dot_new / dot_old;
        for (int i = 0; i < nEleAtoms; i++)
        {
            d[i] = r[i] + beta_cg * d[i];
        }
        project_neutral(d.data());

        // Update for next iteration
        delta = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            delta += r[i] * d[i];
        }

        // Debug logging when iterations exceed 100
        if (iter > 100 && (iter % 10 == 0 || iter > maxiter - 10))
        {
            // Compute residual statistics
            real r_min = r[0], r_max = r[0], r_sum = 0.0;
            real q_min = q[0], q_max = q[0], q_sum = 0.0;
            for (int i = 0; i < nEleAtoms; i++)
            {
                r_sum += r[i];
                if (r[i] < r_min) r_min = r[i];
                if (r[i] > r_max) r_max = r[i];
                q_sum += q[i];
                if (q[i] < q_min) q_min = q[i];
                if (q[i] > q_max) q_max = q[i];
            }
            printf("CG DEBUG [potential] iter=%d: delta=%g, tol=%g, dot_old=%g, dot_new=%g\n",
                   iter, delta, tol, dot_old, dot_new);
            printf("  dAd=%g, alpha=%g, beta_cg=%g\n", dAd, alpha, beta_cg);
            printf("  r: sum=%g, min=%g, max=%g, ||r||^2=%g\n", r_sum, r_min, r_max, dot_new);
            printf("  q: sum=%g, min=%g, max=%g\n", q_sum, q_min, q_max);
        }

        dot_old = dot_new;
    }

    // Final neutrality enforcement
    project_neutral(q.data());

    // Copy result and save for warm start
    for (int i = 0; i < nEleAtoms; i++)
    {
        q_out[i] = q[i];
        if (!bColdStart)
        {
            q_prev[i] = q[i];
        }
    }

    if (iter >= maxiter && delta >= tol)
    {
        return -iter;
    }
    return iter;
}

// bisheng 2025 - Matrix-free CG solver with CHARGE constraint (instead of potential constraint)
// For two electrodes: enforces sum(q_electrode1) = Q_set and sum(q_electrode2) = -Q_set
// The electrode potential "floats" to satisfy the charge constraint
// Fixed to follow LAMMPS electrode package approach for robust convergence
int gmx_pme_solve_electrode_charges_cg_charge_constrained(
                                       gmx_pme_t*                     pme,
                                       gmx::ArrayRef<const gmx::RVec> x_electrode,
                                       const real*                    b,
                                       int                            nEleAtoms,
                                       const matrix                   box,
                                       real                           beta,
                                       real                           eta,
                                       real                           rcoulomb,
                                       gmx_bool                       b3dc,
                                       real*                          q_out,
                                       real                           Q_set,
                                       real                           Q2_set,
                                       real                           tol,
                                       int                            maxiter,
                                       int                            nthreads,
                                       gmx_bool                       bColdStart)
{
    // Check that nEleAtoms is even (required for two-electrode setup)
    if (nEleAtoms % 2 != 0)
    {
        gmx_fatal(FARGS, "Matrix-free CG with charge constraint requires even number of electrode atoms.\n"
                         "nEleAtoms = %d is odd. Please check your electrode setup.", nEleAtoms);
    }

    int halfN = nEleAtoms / 2;

    // bisheng 2026: Q_set arrives in elementary charges, but this
    // solver's charge vector stays in internal V*nm units until the final
    // evscale conversion; convert the constraint target to internal units so the
    // realised total charge equals Q_set in e (was low by evscale = 0.6945).
    Q_set  = Q_set * (ONE_4PI_EPS0 / ELECTRONVOLT);
    Q2_set = Q2_set * (ONE_4PI_EPS0 / ELECTRONVOLT);   // bisheng 2026 - electrode 2's own target
    // Helper lambda: project vector onto charge-constrained subspace
    // Enforces: sum(v[0..halfN-1]) = Q_set, sum(v[halfN..N-1]) = Q2_set (default -Q_set)
    auto project_charge_constrained = [nEleAtoms, halfN, Q_set, Q2_set](real* v) {
        // bisheng 2026: double accumulation + second pass, as above.
        for (int pass = 0; pass < 2; pass++)
        {
            double sum1 = 0.0, sum2 = 0.0;
            for (int i = 0; i < halfN; i++) sum1 += static_cast<double>(v[i]);
            for (int i = halfN; i < nEleAtoms; i++) sum2 += static_cast<double>(v[i]);

            const double shift1 = (static_cast<double>(Q_set) - sum1) / halfN;
            const double shift2 = (static_cast<double>(Q2_set) - sum2) / (nEleAtoms - halfN);

            for (int i = 0; i < halfN; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) + shift1);
            for (int i = halfN; i < nEleAtoms; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) + shift2);
        }
    };

    // Helper lambda: project vector onto zero-charge subspace for search directions
    // For CG, the search direction must preserve the constraint:
    // sum(d_electrode1) = 0 and sum(d_electrode2) = 0
    auto project_zero_charge = [nEleAtoms, halfN](real* v) {
        // bisheng 2026: see note on the global projector -- double
        // accumulation and a second pass. This one matters most: one of the two
        // block-uniform directions it removes is the soft electrode-to-electrode
        // charge-transfer mode, where A^-1 amplifies any leak.
        for (int pass = 0; pass < 2; pass++)
        {
            double sum1 = 0.0, sum2 = 0.0;
            for (int i = 0; i < halfN; i++) sum1 += static_cast<double>(v[i]);
            for (int i = halfN; i < nEleAtoms; i++) sum2 += static_cast<double>(v[i]);

            const double mean1 = sum1 / halfN;
            const double mean2 = sum2 / (nEleAtoms - halfN);

            for (int i = 0; i < halfN; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) - mean1);
            for (int i = halfN; i < nEleAtoms; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) - mean2);
        }
    };

    // Static storage for warm start
    // bisheng 2026 - warm start lives in pme so it can be checkpointed
    std::vector<real>& q_prev_cc = pme->cpm_qPrev;
    const bool bFirstCall_cc = (static_cast<int>(q_prev_cc.size()) != nEleAtoms);

    // Allocate working arrays
    std::vector<real> q(nEleAtoms);
    std::vector<real> r(nEleAtoms);
    std::vector<real> d(nEleAtoms);
    std::vector<real> Ad(nEleAtoms);

    // Initialize q: warm start or initial guess with correct charge
    if (bColdStart || bFirstCall_cc || static_cast<int>(q_prev_cc.size()) != nEleAtoms)
    {
        q_prev_cc.resize(nEleAtoms, 0.0);
        // Initialize with uniform charge distribution satisfying constraint
        real q1 = Q_set / halfN;
        real q2 = Q2_set / (nEleAtoms - halfN);
        for (int i = 0; i < halfN; i++) q[i] = q1;
        for (int i = halfN; i < nEleAtoms; i++) q[i] = q2;
        // (bFirstCall_cc is recomputed from cpm_qPrev's size each call)
    }
    else
    {
        q = q_prev_cc;
    }

    if (cpm_gpu_cg_prepare(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads))
    {
        /* bisheng 2026 - the whole CG runs on the device; q carries the warm start in */
        double    U     = 0.0;
        const int itGpu = cpm_gpu_cg_solve(pme, b, q.data(), nEleAtoms, 1, Q_set, Q2_set, tol, maxiter, &U);
        pme->cpm_cell_voltage = U;
        for (int i = 0; i < nEleAtoms; i++)
        {
            q_out[i] = q[i];
            if (!bColdStart)
            {
                q_prev_cc[i] = q[i];
            }
        }
        return itGpu;
    }

    // Ensure initial q satisfies charge constraint
    project_charge_constrained(q.data());

    // Compute initial residual: r = b - A*q
    gmx::ArrayRef<const real> q_ref(q);
    gmx_pme_calc_Aq(pme, x_electrode, q_ref, nEleAtoms, box, beta, rcoulomb,
                    b3dc, r.data(), nthreads);

    for (int i = 0; i < nEleAtoms; i++)
    {
        r[i] = b[i] - r[i];
    }
    project_zero_charge(r.data());

    // LAMMPS-style: Initial search direction d = projection(r)
    // dot_old = dot(r, d) = ||projected_r||^2
    for (int i = 0; i < nEleAtoms; i++)
    {
        d[i] = r[i];
    }
    double dot_old = 0.0;
    for (int i = 0; i < nEleAtoms; i++)
    {
        dot_old += r[i] * d[i];
    }
    double delta = dot_old;  // Convergence criterion: delta = dot(r, d)

    // bisheng 2026: A*v is evaluated by PME in single precision, so
    // this is an INEXACT-operator CG. On the two-block charge-constrained problem
    // the residual descends for a few hundred iterations and then destabilises
    // (measured: delta 2.1e-5 at 1e3 iterations, 1.3e-2 at 5e3, 8.7e+1 at 4e4).
    // Returning the final iterate therefore returns a diverged one. Keep the best
    // iterate seen instead, and let the caller know it did not converge.
    std::vector<real> q_best(q.begin(), q.end());
    double            delta_best = delta;

    // CG iteration with projection (LAMMPS-style)
    int iter;
    for (iter = 0; iter < maxiter && delta > tol; iter++)
    {
        // Compute A*d
        gmx::ArrayRef<const real> d_ref(d);
        gmx_pme_calc_Aq(pme, x_electrode, d_ref, nEleAtoms, box, beta, rcoulomb,
                        b3dc, Ad.data(), nthreads);

        // alpha = dot_old / (d·A*d) - LAMMPS style
        // bisheng 2026: these accumulators must be double. In single
        // precision a dot product over thousands of electrode atoms loses enough
        // significance to corrupt the CG recurrence, and the iteration then
        // converges to a solution that satisfies the constraints but not the
        // stationarity condition. The potential-constrained solver above already
        // uses double here and agrees with matrix inversion at the 1e-6 level.
        double dAd = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            dAd += d[i] * Ad[i];
        }

        static const bool bCgTrace = (getenv("GMX_CPM_CG_TRACE") != nullptr);
        if (bCgTrace && iter < 12)
        {
            printf("CGTRACE iter=%d delta=%.6e dot_old=%.6e dAd=%.6e tol=%.3e\n",
                   iter, delta, dot_old, dAd, tol);
        }
        if (std::abs(dAd) < 1e-30)
        {
            break;
        }

        double alpha = dot_old / dAd;

        // Update q and r
        for (int i = 0; i < nEleAtoms; i++)
        {
            q[i] += alpha * d[i];
            r[i] -= alpha * Ad[i];
        }

        // Periodic refresh for numerical stability (every 20 iterations, like LAMMPS)
        bool bRestart = false;
        if ((iter + 1) % 20 == 0)
        {
            // Recompute r from scratch to avoid accumulating errors
            project_charge_constrained(q.data());
            gmx::ArrayRef<const real> q_ref_refresh(q);
            gmx_pme_calc_Aq(pme, x_electrode, q_ref_refresh, nEleAtoms, box, beta, rcoulomb,
                            b3dc, r.data(), nthreads);
            for (int i = 0; i < nEleAtoms; i++)
            {
                r[i] = b[i] - r[i];
            }
            // After refresh, force restart (beta=0) to ensure d is consistent with fresh r
            bRestart = true;
        }

        // Project residual onto constraint subspace
        project_zero_charge(r.data());

        // LAMMPS-style: dot_new = dot(r, projection(r)) = ||projected_r||^2
        double dot_new = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            dot_new += r[i] * r[i];
        }

        // Update search direction: d = r + (dot_new/dot_old) * d
        // After refresh, use beta=0 to restart CG with consistent state
        double beta_cg = bRestart ? 0.0 : dot_new / dot_old;
        for (int i = 0; i < nEleAtoms; i++)
        {
            d[i] = r[i] + beta_cg * d[i];
        }
        project_zero_charge(d.data());

        // Update for next iteration
        delta = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            delta += r[i] * d[i];
        }
        dot_old = dot_new;

        if (delta < delta_best)
        {
            delta_best = delta;
            std::copy(q.begin(), q.end(), q_best.begin());
        }
    }

    // Fall back on the best iterate whenever the run ended worse than it started.
    if (delta > delta_best)
    {
        std::copy(q_best.begin(), q_best.end(), q.begin());
        delta = delta_best;
    }

    if (getenv("GMX_CPM_CG_TRACE") != nullptr)
    {
        printf("CGEND iters=%d delta=%.6e tol=%.3e maxiter=%d %s\n", iter, delta, tol, maxiter,
               (iter >= maxiter) ? "HIT-MAXITER" : "converged-or-broke");
    }
    // Final charge constraint enforcement
    project_charge_constrained(q.data());

    // bisheng 2026 - realised electrode potentials. b is setPot + b_field in volts
    // with setPot = 0 in charge mode, and A*q (q in internal units) is the
    // potential the charges produce, so at convergence r = b - A*q is uniform on
    // each electrode and equals minus that electrode's potential. One extra A*q
    // on the final iterate keeps this consistent with the returned charges.
    {
        gmx::ArrayRef<const real> qf(q);
        gmx_pme_calc_Aq(pme, x_electrode, qf, nEleAtoms, box, beta, rcoulomb, b3dc, r.data(), nthreads);
        double s1 = 0.0, s2 = 0.0;
        for (int i = 0; i < halfN; i++) s1 += static_cast<double>(b[i]) - static_cast<double>(r[i]);
        for (int i = halfN; i < nEleAtoms; i++) s2 += static_cast<double>(b[i]) - static_cast<double>(r[i]);
        const double psi1 = -s1 / halfN, psi2 = -s2 / (nEleAtoms - halfN);
        pme->cpm_cell_voltage = psi1 - psi2;
    }

    // Copy result and save for warm start
    for (int i = 0; i < nEleAtoms; i++)
    {
        q_out[i] = q[i];
        if (!bColdStart)
        {
            q_prev_cc[i] = q[i];
        }
    }

    if (iter >= maxiter && delta >= tol)
    {
        return -iter;
    }
    return iter;
}

// bisheng 2025 - Matrix-free CG solver for SINGLE electrode with charge constraint
// Constraint: sum(q) = Q_set (total charge on the single electrode)
// The electrode potential floats to satisfy the charge constraint
// Fixed to follow LAMMPS electrode package approach for robust convergence
int gmx_pme_solve_electrode_charges_cg_single_electrode(
                                       gmx_pme_t*                     pme,
                                       gmx::ArrayRef<const gmx::RVec> x_electrode,
                                       const real*                    b,
                                       int                            nEleAtoms,
                                       const matrix                   box,
                                       real                           beta,
                                       real                           eta,
                                       real                           rcoulomb,
                                       gmx_bool                       b3dc,
                                       real*                          q_out,
                                       real                           Q_set,
                                       real                           tol,
                                       int                            maxiter,
                                       int                            nthreads,
                                       gmx_bool                       bColdStart)
{
    // bisheng 2026: Q_set arrives in elementary charges, but this
    // solver's charge vector stays in internal V*nm units until the final
    // evscale conversion; convert the constraint target to internal units so the
    // realised total charge equals Q_set in e (was low by evscale = 0.6945).
    Q_set = Q_set * (ONE_4PI_EPS0 / ELECTRONVOLT);
    // Helper lambda: project vector to enforce sum(q) = Q_set
    auto project_charge = [nEleAtoms, Q_set](real* v) {
        real sum = 0.0;
        for (int i = 0; i < nEleAtoms; i++) sum += v[i];
        real shift = (Q_set - sum) / nEleAtoms;
        for (int i = 0; i < nEleAtoms; i++) v[i] += shift;
    };

    // Helper lambda: project vector to zero-sum subspace (for search directions)
    auto project_zero = [nEleAtoms](real* v) {
        // bisheng 2026: accumulate in double and project twice.
        // In float, summing thousands of terms leaves ~1e-6 of the constraint
        // direction in v; A^-1 amplifies that leak along the softest mode and the
        // CG recurrence then diverges. Double + a second pass makes the leak
        // negligible at float storage precision.
        for (int pass = 0; pass < 2; pass++)
        {
            double sum = 0.0;
            for (int i = 0; i < nEleAtoms; i++) sum += static_cast<double>(v[i]);
            const double mean = sum / nEleAtoms;
            for (int i = 0; i < nEleAtoms; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) - mean);
        }
    };

    // bisheng 2026 - warm start lives in pme so it can be checkpointed
    std::vector<real>& q_prev_single = pme->cpm_qPrev;
    const bool bFirstCall_single = (static_cast<int>(q_prev_single.size()) != nEleAtoms);

    // Allocate working arrays
    std::vector<real> q(nEleAtoms);
    std::vector<real> r(nEleAtoms);
    std::vector<real> d(nEleAtoms);
    std::vector<real> Ad(nEleAtoms);

    // Initialize q: warm start or initial guess with correct charge
    if (bColdStart || bFirstCall_single || static_cast<int>(q_prev_single.size()) != nEleAtoms)
    {
        q_prev_single.resize(nEleAtoms, 0.0);
        // Initialize with uniform charge distribution satisfying constraint
        real q_uniform = Q_set / nEleAtoms;
        for (int i = 0; i < nEleAtoms; i++) q[i] = q_uniform;
        if (false)
        {
        }
    }
    else
    {
        q = q_prev_single;
    }

    if (cpm_gpu_cg_prepare(pme, x_electrode, nEleAtoms, box, beta, rcoulomb, b3dc, nthreads))
    {
        /* bisheng 2026 - the whole CG runs on the device; q carries the warm start in */
        double    U     = 0.0;
        const int itGpu = cpm_gpu_cg_solve(pme, b, q.data(), nEleAtoms, 2, Q_set, 0.0, tol, maxiter, &U);
        pme->cpm_cell_voltage = U;
        for (int i = 0; i < nEleAtoms; i++)
        {
            q_out[i] = q[i];
            if (!bColdStart)
            {
                q_prev_single[i] = q[i];
            }
        }
        return itGpu;
    }

    // Ensure initial q satisfies charge constraint
    project_charge(q.data());

    // Compute initial residual: r = b - A*q
    gmx::ArrayRef<const real> q_ref(q);
    gmx_pme_calc_Aq(pme, x_electrode, q_ref, nEleAtoms, box, beta, rcoulomb,
                    b3dc, r.data(), nthreads);

    for (int i = 0; i < nEleAtoms; i++)
    {
        r[i] = b[i] - r[i];
    }
    project_zero(r.data());

    // LAMMPS-style: Initial search direction d = projection(r)
    // dot_old = dot(r, d) = ||projected_r||^2
    for (int i = 0; i < nEleAtoms; i++)
    {
        d[i] = r[i];
    }
    double dot_old = 0.0;
    for (int i = 0; i < nEleAtoms; i++)
    {
        dot_old += r[i] * d[i];
    }
    double delta = dot_old;  // Convergence criterion: delta = dot(r, d)

    // CG iteration with projection (LAMMPS-style)
    int iter;
    for (iter = 0; iter < maxiter && delta > tol; iter++)
    {
        // Compute A*d
        gmx::ArrayRef<const real> d_ref(d);
        gmx_pme_calc_Aq(pme, x_electrode, d_ref, nEleAtoms, box, beta, rcoulomb,
                        b3dc, Ad.data(), nthreads);

        // alpha = dot_old / (d·A*d) - LAMMPS style
        double dAd = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            dAd += d[i] * Ad[i];
        }

        if (std::abs(dAd) < 1e-30)
        {
            break;
        }

        double alpha = dot_old / dAd;

        // Update q and r
        for (int i = 0; i < nEleAtoms; i++)
        {
            q[i] += alpha * d[i];
            r[i] -= alpha * Ad[i];
        }

        // Periodic refresh for numerical stability (every 20 iterations, like LAMMPS)
        bool bRestart = false;
        if ((iter + 1) % 20 == 0)
        {
            // Recompute r from scratch to avoid accumulating errors
            project_charge(q.data());
            gmx::ArrayRef<const real> q_ref_refresh(q);
            gmx_pme_calc_Aq(pme, x_electrode, q_ref_refresh, nEleAtoms, box, beta, rcoulomb,
                            b3dc, r.data(), nthreads);
            for (int i = 0; i < nEleAtoms; i++)
            {
                r[i] = b[i] - r[i];
            }
            // After refresh, force restart (beta=0) to ensure d is consistent with fresh r
            bRestart = true;
        }

        // Project residual onto constraint subspace
        project_zero(r.data());

        // LAMMPS-style: dot_new = dot(r, projection(r)) = ||projected_r||^2
        double dot_new = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            dot_new += r[i] * r[i];
        }

        // Update search direction: d = r + (dot_new/dot_old) * d
        // After refresh, use beta=0 to restart CG with consistent state
        double beta_cg = bRestart ? 0.0 : dot_new / dot_old;
        for (int i = 0; i < nEleAtoms; i++)
        {
            d[i] = r[i] + beta_cg * d[i];
        }
        project_zero(d.data());

        // Update for next iteration
        delta = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            delta += r[i] * d[i];
        }
        dot_old = dot_new;
    }

    // Final charge constraint enforcement
    project_charge(q.data());

    // bisheng 2026 - realised electrode potential (relative to the Ewald zero of
    // the periodic cell): r = b - A*q is uniform on the electrode at convergence
    // and equals minus its potential; see the two-electrode solver.
    {
        gmx::ArrayRef<const real> qf(q);
        gmx_pme_calc_Aq(pme, x_electrode, qf, nEleAtoms, box, beta, rcoulomb, b3dc, r.data(), nthreads);
        double s = 0.0;
        for (int i = 0; i < nEleAtoms; i++) s += static_cast<double>(b[i]) - static_cast<double>(r[i]);
        pme->cpm_cell_voltage = -s / nEleAtoms;
    }

    // Copy result and save for warm start
    for (int i = 0; i < nEleAtoms; i++)
    {
        q_out[i] = q[i];
        if (!bColdStart)
        {
            q_prev_single[i] = q[i];
        }
    }

    if (iter >= maxiter && delta >= tol)
    {
        return -iter;
    }
    return iter;
}

// bisheng 2025 - Extended Lagrangian method for electrode charges
// Instead of solving A*q = b exactly, we propagate charge dynamics:
//   F_q = b - A*q  (restoring force toward CPM solution)
//   qdot += F_q * dt / μ
//   q += qdot * dt
// This conserves the extended Hamiltonian: H_ext = H_MD + μ/2 * qdot² + 1/2 * q·A·q - q·b
//
// Supported modes:
// - Two electrodes, potential constraint (pot_or_charge=0): sum(q) = 0 (neutrality)
// - Two electrodes, charge constraint (pot_or_charge=1): sum(q1) = Q_set, sum(q2) = -Q_set
// - Single electrode, charge constraint: sum(q) = Q_set
void gmx_pme_extended_lagrangian_step(gmx_pme_t*                     pme,
                                       gmx::ArrayRef<const gmx::RVec> x_electrode,
                                       const real*                    b,
                                       int                            nEleAtoms,
                                       const matrix                   box,
                                       real                           beta,
                                       real                           eta,
                                       real                           rcoulomb,
                                       gmx_bool                       b3dc,
                                       real*                          q_out,
                                       int                            nthreads,
                                       int                            oneEle_or_not,
                                       int                            pot_or_charge,
                                       real                           Q_set,
                                       real                           Q2_set)
{
    // bisheng 2026: Q_set arrives in elementary charges, but this
    // solver's charge vector stays in internal V*nm units until the final
    // evscale conversion; convert the constraint target to internal units so the
    // realised total charge equals Q_set in e (was low by evscale = 0.6945).
    const real Q_set_input  = Q_set;  // as supplied by the user, in e (for logging)
    const real Q2_set_input = Q2_set; // bisheng 2026 - electrode 2's own target, in e
    Q_set                   = Q_set * (ONE_4PI_EPS0 / ELECTRONVOLT);
    Q2_set                  = Q2_set * (ONE_4PI_EPS0 / ELECTRONVOLT);
    real mass = pme->ext_lag_mass;
    real dt   = pme->ext_lag_dt;

    // For two electrodes, we need halfN
    int halfN = nEleAtoms / 2;

    // Initialize on first call
    if (pme->ext_lag_q.empty() || static_cast<int>(pme->ext_lag_q.size()) != nEleAtoms)
    {
        pme->ext_lag_q.resize(nEleAtoms, 0.0);
        pme->ext_lag_qdot.resize(nEleAtoms, 0.0);

        // bisheng 2026: seed from a converged solve.
        //
        // These charges are propagated by undamped dynamics, so whatever
        // displacement they start with never decays -- it oscillates forever at
        // amplitude |q(0) - q*| and angular frequency sqrt(A/mu). Starting from
        // zero (or from a uniform distribution with the right total but the
        // wrong shape) therefore left the charges ringing about the correct
        // answer with a phase-averaged relative error of 1/sqrt(2) ~ 71%,
        // independent of mu -- measured at 74% over the slab cell, with the
        // period tracking sqrt(mu) exactly as that picture predicts.
        //
        // One converged solve costs a single extra CG at step 0 and puts the
        // charges on the Born-Oppenheimer surface, where the dynamics are meant
        // to start. Note this uses the MATRIX-FREE solver: extended Lagrangian
        // requires userint2=2, for which no elastance matrix is loaded at all.
        // Q_set_input (elementary charges) is passed rather than the converted
        // Q_set, because these solvers do their own unit conversion.
        {
            static const real seedTol = 1.0e-6;
            int               seedIter = 0;
            if (oneEle_or_not == 1)
            {
                seedIter = gmx_pme_solve_electrode_charges_cg_single_electrode(
                        pme, x_electrode, b, nEleAtoms, box, beta, eta, rcoulomb, b3dc,
                        pme->ext_lag_q.data(), Q_set_input, seedTol, 1000, nthreads, TRUE);
            }
            else if (pot_or_charge == 1)
            {
                seedIter = gmx_pme_solve_electrode_charges_cg_charge_constrained(
                        pme, x_electrode, b, nEleAtoms, box, beta, eta, rcoulomb, b3dc,
                        pme->ext_lag_q.data(), Q_set_input, Q2_set_input, seedTol, 1000, nthreads, TRUE);
            }
            else
            {
                seedIter = gmx_pme_solve_electrode_charges_cg(
                        pme, x_electrode, b, nEleAtoms, box, beta, eta, rcoulomb, b3dc,
                        pme->ext_lag_q.data(), seedTol, 1000, nthreads, TRUE);
            }
            printf("Extended Lagrangian: charges seeded from a converged solve "
                   "(%d iterations); velocities start at zero.\n", std::abs(seedIter));
        }

        // Initialize charges based on electrode type and constraint
        if (oneEle_or_not == 1)
        {
            printf("Extended Lagrangian initialized (single electrode, charge constraint):\n");
            printf("  mass=%.4f, dt=%.6f, nEleAtoms=%d, Q_set=%.4f e\n", mass, dt, nEleAtoms, Q_set_input);
        }
        else if (pot_or_charge == 1)
        {
            printf("Extended Lagrangian initialized (two electrodes, charge constraint):\n");
            printf("  mass=%.4f, dt=%.6f, nEleAtoms=%d, Q_set=%.4f e\n", mass, dt, nEleAtoms, Q_set_input);
        }
        else
        {
            // Two electrodes with potential constraint: sum(q) = 0
            printf("Extended Lagrangian initialized (two electrodes, potential constraint):\n");
            printf("  mass=%.4f, dt=%.6f, nEleAtoms=%d\n", mass, dt, nEleAtoms);
        }
    }

    // Define projection lambdas based on mode
    // For force/velocity: project to zero-sum subspace
    // For charge: project to satisfy constraint

    // project_zero_force: projects force/velocity to appropriate zero-sum subspace
    auto project_zero_force = [nEleAtoms, halfN, oneEle_or_not, pot_or_charge](real* v) {
        if (oneEle_or_not == 1 || pot_or_charge == 0)
        {
            // Single electrode or two electrodes with potential: global zero-sum
            real sum = 0.0;
            for (int i = 0; i < nEleAtoms; i++) sum += v[i];
            real mean = sum / nEleAtoms;
            for (int i = 0; i < nEleAtoms; i++) v[i] -= mean;
        }
        else
        {
            // Two electrodes with charge constraint: zero-sum on each electrode separately
            real sum1 = 0.0, sum2 = 0.0;
            for (int i = 0; i < halfN; i++) sum1 += v[i];
            for (int i = halfN; i < nEleAtoms; i++) sum2 += v[i];
            real mean1 = sum1 / halfN;
            real mean2 = sum2 / (nEleAtoms - halfN);
            for (int i = 0; i < halfN; i++) v[i] -= mean1;
            for (int i = halfN; i < nEleAtoms; i++) v[i] -= mean2;
        }
    };

    // project_charge: projects charge to satisfy constraint
    auto project_charge = [nEleAtoms, halfN, oneEle_or_not, pot_or_charge, Q_set, Q2_set](real* v) {
        if (oneEle_or_not == 1)
        {
            // Single electrode: sum(q) = Q_set
            // bisheng 2026: double accumulation + second pass.
            for (int pass = 0; pass < 2; pass++)
            {
                double sum = 0.0;
                for (int i = 0; i < nEleAtoms; i++) sum += static_cast<double>(v[i]);
                const double shift = (static_cast<double>(Q_set) - sum) / nEleAtoms;
                for (int i = 0; i < nEleAtoms; i++) v[i] = static_cast<real>(static_cast<double>(v[i]) + shift);
            }
        }
        else if (pot_or_charge == 1)
        {
            // Two electrodes with charge constraint: sum(q1) = Q_set, sum(q2) = Q2_set (default -Q_set)
            real sum1 = 0.0, sum2 = 0.0;
            for (int i = 0; i < halfN; i++) sum1 += v[i];
            for (int i = halfN; i < nEleAtoms; i++) sum2 += v[i];
            real shift1 = (Q_set - sum1) / halfN;
            real shift2 = (Q2_set - sum2) / (nEleAtoms - halfN);
            for (int i = 0; i < halfN; i++) v[i] += shift1;
            for (int i = halfN; i < nEleAtoms; i++) v[i] += shift2;
        }
        else
        {
            // Two electrodes with potential constraint: sum(q) = 0 (neutrality)
            real sum = 0.0;
            for (int i = 0; i < nEleAtoms; i++) sum += v[i];
            real mean = sum / nEleAtoms;
            for (int i = 0; i < nEleAtoms; i++) v[i] -= mean;
        }
    };

    // Compute A*q (potential from current charges)
    std::vector<real> Aq(nEleAtoms);
    gmx::ArrayRef<const real> q_ref(pme->ext_lag_q);
    gmx_pme_calc_Aq(pme, x_electrode, q_ref, nEleAtoms, box, beta, rcoulomb,
                    b3dc, Aq.data(), nthreads);

    // Compute force: F_q = b - A*q
    // Project both b and F onto appropriate zero-sum subspace
    std::vector<real> F_q(nEleAtoms);
    std::vector<real> b_proj(b, b + nEleAtoms);
    project_zero_force(b_proj.data());

    for (int i = 0; i < nEleAtoms; i++)
    {
        F_q[i] = b_proj[i] - Aq[i];
    }
    project_zero_force(F_q.data());

    // Velocity Verlet integration (or simple Euler for now)
    // First half-step for velocity: qdot += 0.5 * F_q * dt / mass
    // Full step for position: q += qdot * dt
    // Recompute F_q at new position
    // Second half-step for velocity: qdot += 0.5 * F_q * dt / mass

    // For simplicity, using leapfrog/velocity Verlet:
    // qdot(t+dt/2) = qdot(t-dt/2) + F_q(t) * dt / mass
    // q(t+dt) = q(t) + qdot(t+dt/2) * dt


    // bisheng 2026 - extended-Lagrangian energy bookkeeping (DIAGNOSTIC).
    //
    // The conserved quantity of these dynamics is NOT the physical energy but
    //     H_ext = KE_atoms + U_elyte + U_q + T_fict,   U_q = 1/2 q.A.q - q.b
    // and only T_fict was being logged, so a rise in the .edr total energy could
    // not be told apart from a rise in H_ext.  This block logs the remaining
    // pieces, plus the magnitude of the two constraint projections (which are a
    // position projection without a matching RATTLE-style velocity correction and
    // could therefore inject energy).
    //
    // UNITS.  Inside this solver q is in V*nm (q_e = q * evscale with
    // evscale = ELECTRONVOLT/ONE_4PI_EPS0, see the conversion at the call site),
    // A is in 1/nm and b, setPot are in V.  So q.A.q, q.b, q.V and mu*qdot^2 are
    // all in V^2*nm and ONE factor converts all of them to kJ/mol:
    //     C = ELECTRONVOLT * evscale = ELECTRONVOLT^2 / ONE_4PI_EPS0
    //       = 96.4853^2 / 138.9354 = 67.0053 kJ/mol per V^2*nm.
    // C is printed into the file header, computed from the code's own constants.
    //
    // Enabled by GMX_CPM_EXTLAG_UQ=1.  Off by default: the binary is shared.
    static const bool bUqLog = (getenv("GMX_CPM_EXTLAG_UQ") != nullptr);

    double uq_qAq = 0.0, uq_qb = 0.0, uq_sq2 = 0.0, uq_Q1 = 0.0, uq_Q2 = 0.0;
    double uq_F2 = 0.0, uq_Told = 0.0, uq_gself = 0.0;
    std::vector<real> uq_qdotOld;
    std::vector<real> uq_qPre, uq_vPre;
    if (bUqLog)
    {
        const real inv_sqrt_pi_uq = 1.0 / std::sqrt(M_PI);
        for (int i = 0; i < nEleAtoms; i++)
        {
            const double qi = static_cast<double>(pme->ext_lag_q[i]);
            uq_qAq += qi * static_cast<double>(Aq[i]);
            uq_qb += qi * static_cast<double>(b[i]);
            uq_sq2 += qi * qi;
            uq_F2 += static_cast<double>(F_q[i]) * static_cast<double>(F_q[i]);
            uq_Told += static_cast<double>(pme->ext_lag_qdot[i])
                       * static_cast<double>(pme->ext_lag_qdot[i]);
            // diagonal of 1/2 q.(A_gauss - A_point).q : the Gaussian self term
            // sqrt(2)*eta/sqrt(pi), which is the part of the electrode-electrode
            // energy that the .edr (point charges) does not contain.
            uq_gself += 0.5 * std::sqrt(2.0) * static_cast<double>(pme->eta_per_atom[i])
                        * inv_sqrt_pi_uq * qi * qi;
            if (i < halfN)
            {
                uq_Q1 += qi;
            }
            else
            {
                uq_Q2 += qi;
            }
        }
        uq_Told *= 0.5 * static_cast<double>(mass);
        uq_qdotOld.assign(pme->ext_lag_qdot.begin(), pme->ext_lag_qdot.end());

        // One-shot check that A is symmetric.  If it is not, F_q = b - A q is not
        // the gradient of anything and no U_q exists, which would settle the
        // question before any energy series is looked at.
        static bool bUqSymDone = false;
        if (!bUqSymDone)
        {
            bUqSymDone = true;
            std::mt19937                           rngSym(4242);
            std::uniform_real_distribution<double> dSym(-1.0, 1.0);
            std::vector<real>                      uS(nEleAtoms), vS(nEleAtoms);
            std::vector<real>                      AuS(nEleAtoms), AvS(nEleAtoms);
            double                                 su = 0.0, sv = 0.0;
            for (int i = 0; i < nEleAtoms; i++)
            {
                uS[i] = static_cast<real>(dSym(rngSym));
                vS[i] = static_cast<real>(dSym(rngSym));
                su += uS[i];
                sv += vS[i];
            }
            for (int i = 0; i < nEleAtoms; i++)
            {
                uS[i] -= static_cast<real>(su / nEleAtoms);
                vS[i] -= static_cast<real>(sv / nEleAtoms);
            }
            gmx::ArrayRef<const real> uRef(uS);
            gmx::ArrayRef<const real> vRef(vS);
            gmx_pme_calc_Aq(pme, x_electrode, uRef, nEleAtoms, box, beta, rcoulomb, b3dc,
                            AuS.data(), nthreads);
            gmx_pme_calc_Aq(pme, x_electrode, vRef, nEleAtoms, box, beta, rcoulomb, b3dc,
                            AvS.data(), nthreads);
            double uAv = 0.0, vAu = 0.0;
            for (int i = 0; i < nEleAtoms; i++)
            {
                uAv += static_cast<double>(uS[i]) * static_cast<double>(AvS[i]);
                vAu += static_cast<double>(vS[i]) * static_cast<double>(AuS[i]);
            }
            printf("ExtLag A-symmetry probe: u.Av = %.10e, v.Au = %.10e, rel.asym = %.3e\n",
                   uAv, vAu, std::fabs(uAv - vAu) / std::max(1.0e-30, std::fabs(uAv)));
            fflush(stdout);
        }
    }

    real dt_over_mass = dt / mass;

    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; i++)
    {
        // Update velocity
        pme->ext_lag_qdot[i] += F_q[i] * dt_over_mass;
        // Update position
        pme->ext_lag_q[i] += pme->ext_lag_qdot[i] * dt;
    }

    // Project q to maintain charge constraint, qdot to zero-sum
    if (bUqLog)
    {
        uq_qPre.assign(pme->ext_lag_q.begin(), pme->ext_lag_q.end());
        uq_vPre.assign(pme->ext_lag_qdot.begin(), pme->ext_lag_qdot.end());
    }
    project_charge(pme->ext_lag_q.data());
    project_zero_force(pme->ext_lag_qdot.data());

    // bisheng 2026 - periodic quench of the charge velocities (DIAGNOSTIC / FIX).
    //
    // The charge degrees of freedom carry no thermostat and nothing returns them
    // to the Born-Oppenheimer surface, so T_fict climbs monotonically toward
    // equipartition with the ions and the charges settle a large, mu-independent
    // distance off the constant-potential solution.  Setting
    // GMX_CPM_EXTLAG_QUENCH=N zeroes qdot every N steps, which caps T_fict at
    // whatever it accumulates in N steps.  Unset or 0 = off (the default), so a
    // shared binary is unaffected.
    {
        static const int quenchEvery = getenv("GMX_CPM_EXTLAG_QUENCH")
                                               ? atoi(getenv("GMX_CPM_EXTLAG_QUENCH"))
                                               : 0;
        static int64_t   quenchCall  = 0;
        if (quenchEvery > 0)
        {
            if (quenchCall == 0)
            {
                printf("Extended Lagrangian: charge velocities quenched every %d steps "
                       "(GMX_CPM_EXTLAG_QUENCH)\n", quenchEvery);
                fflush(stdout);
            }
            if (quenchCall > 0 && (quenchCall % quenchEvery) == 0)
            {
                std::fill(pme->ext_lag_qdot.begin(), pme->ext_lag_qdot.end(), 0.0_real);
            }
            quenchCall++;
        }
    }

    // bisheng 2026 - write the bookkeeping row (see the UNITS note above).
    if (bUqLog)
    {
        static FILE*   uqFp   = nullptr;
        static int64_t uqCall = 0;
        if (uqFp == nullptr)
        {
            uqFp = fopen("extlag_uq.dat", "a");
            if (uqFp != nullptr && ftell(uqFp) == 0)
            {
                const double Cconv = static_cast<double>(ELECTRONVOLT)
                                     * static_cast<double>(ELECTRONVOLT)
                                     / static_cast<double>(ONE_4PI_EPS0);
                fprintf(uqFp,
                        "# extended-Lagrangian energy bookkeeping (GMX_CPM_EXTLAG_UQ)\n"
                        "# internal units: q [V nm], A [1/nm], b and setPot [V], mu [ps^2/nm]\n"
                        "# every energy-like column below is in V^2 nm; multiply by\n"
                        "# C = ELECTRONVOLT^2/ONE_4PI_EPS0 = %.6f kJ/mol per V^2 nm\n"
                        "# mu = %.6g, dt = %.6g ps, nEleAtoms = %d, oneEle = %d, potOrQ = %d\n"
                        "# U_q = 0.5*qAq - qb.  gauss_self is the diagonal of\n"
                        "# 0.5*q.(A_gauss-A_point).q, i.e. the electrode-electrode energy the\n"
                        "# .edr (point charges) is missing.  qV is the potentiostat term.\n"
                        "# 1:call 2:qAq 3:qb 4:U_q 5:qV 6:sum_q2 7:Q1 8:Q2 9:T_old 10:T_new "
                        "11:T_cross 12:normdq 13:maxdq 14:normdqdot 15:maxdqdot 16:W_proj "
                        "17:normF 18:gauss_self\n",
                        Cconv, static_cast<double>(mass), static_cast<double>(dt), nEleAtoms,
                        oneEle_or_not, pot_or_charge);
            }
        }
        if (uqFp != nullptr)
        {
            double dq2 = 0.0, dqmax = 0.0, dv2 = 0.0, dvmax = 0.0, wproj = 0.0;
            double Tnew = 0.0, Tcross = 0.0;
            for (int i = 0; i < nEleAtoms; i++)
            {
                const double dq = static_cast<double>(pme->ext_lag_q[i])
                                  - static_cast<double>(uq_qPre[i]);
                const double dv = static_cast<double>(pme->ext_lag_qdot[i])
                                  - static_cast<double>(uq_vPre[i]);
                dq2 += dq * dq;
                dv2 += dv * dv;
                dqmax = std::max(dqmax, std::fabs(dq));
                dvmax = std::max(dvmax, std::fabs(dv));
                wproj += static_cast<double>(F_q[i]) * dq;
                Tnew += static_cast<double>(pme->ext_lag_qdot[i])
                        * static_cast<double>(pme->ext_lag_qdot[i]);
                Tcross += static_cast<double>(pme->ext_lag_qdot[i])
                          * static_cast<double>(uq_qdotOld[i]);
            }
            Tnew *= 0.5 * static_cast<double>(mass);
            Tcross *= 0.5 * static_cast<double>(mass);
            // q.setPot for the two-electrode potential constraint, where
            // setPot = +V/2 on the first half and -V/2 on the second (pme.cpp
            // "setPot[i] = pme->applied_potential/2").  Not defined otherwise.
            double qV = 0.0;
            if (oneEle_or_not == 0 && pot_or_charge == 0)
            {
                qV = 0.5 * static_cast<double>(pme->applied_potential) * (uq_Q1 - uq_Q2);
            }
            fprintf(uqFp,
                    "%8lld %16.8e %16.8e %16.8e %16.8e %16.8e %14.6e %14.6e "
                    "%16.8e %16.8e %16.8e %12.4e %12.4e %12.4e %12.4e %14.6e %14.6e %16.8e\n",
                    static_cast<long long>(uqCall), uq_qAq, uq_qb, 0.5 * uq_qAq - uq_qb, qV,
                    uq_sq2, uq_Q1, uq_Q2, uq_Told, Tnew, Tcross, std::sqrt(dq2), dqmax,
                    std::sqrt(dv2), dvmax, wproj, std::sqrt(uq_F2), uq_gself);
            fflush(uqFp);
        }
        uqCall++;
    }

    // Copy result to output
    for (int i = 0; i < nEleAtoms; i++)
    {
        q_out[i] = pme->ext_lag_q[i];
    }

    // bisheng 2026: the fictitious kinetic energy of the charge
    // degrees of freedom, T_fict = (mu/2) sum_i qdot_i^2. The physical energy
    // is NOT the conserved quantity of these dynamics -- the extended
    // Hamiltonian is H_ext = H_MD + T_fict - sum_i U_i q_i -- so characterising
    // conservation on H_MD alone cannot distinguish genuine integrator drift
    // from energy traded with the charge subsystem. T_fict is the piece that
    // was missing; its DRIFT is the adiabaticity diagnostic (a Car-Parrinello
    // style check: if it grows secularly the charges are heating and the
    // dynamics have left the Born-Oppenheimer surface).
    //
    // Values are in the solver's internal units (q in V nm, mu in ps^2/nm), so
    // read the trend rather than the absolute number.
    {
        static FILE* elFp  = nullptr;
        static int64_t elCall = 0;
        if (elFp == nullptr)
        {
            elFp = fopen("extlag_energy.dat", "a");
            if (elFp != nullptr && ftell(elFp) == 0)
            {
                fprintf(elFp, "# call  T_fict(internal)  rms_qdot(internal)  sum_q\n");
            }
        }
        if (elFp != nullptr)
        {
            double t2 = 0.0, sq = 0.0;
            for (int i = 0; i < nEleAtoms; i++)
            {
                t2 += static_cast<double>(pme->ext_lag_qdot[i]) * pme->ext_lag_qdot[i];
                sq += static_cast<double>(pme->ext_lag_q[i]);
            }
            fprintf(elFp, "%8lld  %16.8e  %18.8e  %12.4e\n",
                    static_cast<long long>(elCall), 0.5 * mass * t2,
                    std::sqrt(t2 / nEleAtoms), sq);
            fflush(elFp);
        }
        elCall++;
    }


    // Debug output (first few steps and every nstenergy steps)
    static int ext_lag_step = 0;
    if (ext_lag_step < 5 || (pme->nstenergy > 0 && ext_lag_step % pme->nstenergy == 0))
    {
        real sum_q = 0, sum_qdot = 0, sum_F = 0;
        real max_qdot = 0, max_F = 0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            sum_q += pme->ext_lag_q[i];
            sum_qdot += pme->ext_lag_qdot[i];
            sum_F += F_q[i];
            if (std::abs(pme->ext_lag_qdot[i]) > max_qdot) max_qdot = std::abs(pme->ext_lag_qdot[i]);
            if (std::abs(F_q[i]) > max_F) max_F = std::abs(F_q[i]);
        }
        printf("ExtLag step %d: sum_q=%.6e, sum_qdot=%.6e, max_qdot=%.6e, max_F=%.6e\n",
               ext_lag_step, sum_q, sum_qdot, max_qdot, max_F);
    }
    ext_lag_step++;
}

// bisheng 2025
// bisheng 2026 - the O(N_el^2) part of the matrix-inversion solve: y = A^{-1} x.
// On the device when the GPU solve is enabled (cpm_gpu_solve.cu, keyed on the
// matrix pointer so the upload happens once), otherwise the original host loop,
// which accumulates j ascending per row exactly as the code always did and is
// therefore bit-identical to the pre-port results.
#include "cpm_gpu_solve.h"
static bool g_cpmUseGpuSolve    = false;
static bool g_cpmGpuSolveSingle = false;
static void cpm_inv_gemv(const double* invA, const double* x, double* y, const int n, const int nthreads)
{
    if (g_cpmUseGpuSolve && cpm_gpu_solve_init(invA, n, g_cpmGpuSolveSingle) && cpm_gpu_solve_gemv(x, y, n))
    {
        return;
    }
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < n; ++i)
    {
        double acc = 0.0;
        for (int j = 0; j < n; ++j)
        {
            acc += invA[static_cast<size_t>(i) * n + j] * x[j];
        }
        y[i] = acc;
    }
}

void compute_and_neutralize_electrode_charge(
    double *eleCharge,                  // Output: array, size nEleAtoms
    const double *setPot,               // Input: electrode potential (eV), size nEleAtoms
    const real *psr_coul,             // Input: array, size nEleAtoms
    const real *plr_coul,             // Input: array, size nEleAtoms
    const double *inv_MatrixA,          // Input: flattened matrix, size nEleAtoms*nEleAtoms
    int nEleAtoms,                      // Number of electrode atoms
    int nthreads,                       // Number of OpenMP threads
    double potUnit = 1.0 / ELECTRONVOLT,       // Potential unit conversion factor
    double evscale = ELECTRONVOLT / ONE_4PI_EPS0           // 1/ke, in unit of V*nm/e, which convert V*nm to e as q = 1/ke * V * r
) {
    // bisheng 2026: cache A^-1.1 across steps, but key it on the matrix
    // pointer and the electrode count so a resized electrode or a second pme
    // instance rebuilds it instead of silently reusing a stale vector. The
    // std::vector also replaces a calloc that was never freed.
    static std::vector<double> dEleCharge1V_storage;
    static double              sumCharge1V             = 0.0;
    static bool                bSumCharge1VInitialized = true;
    static const double*       cachedMatrix            = nullptr;
    if (static_cast<int>(dEleCharge1V_storage.size()) != nEleAtoms || cachedMatrix != inv_MatrixA)
    {
        dEleCharge1V_storage.assign(nEleAtoms, 0.0);
        cachedMatrix            = inv_MatrixA;
        bSumCharge1VInitialized = true;
    }
    double* dEleCharge1V = dEleCharge1V_storage.data();

    // 1. Compute eleCharge (reset to 0)
    // Debug: print elePot stats for comparison with CG
    static bool bDebugMatrix = false;
    if (bDebugMatrix)
    {
        double pot_sum = 0, pot_min = 1e30, pot_max = -1e30;
        for (int ii = 0; ii < nEleAtoms; ++ii) {
            double elePot = setPot[ii] - static_cast<double>(psr_coul[ii] + plr_coul[ii]) * potUnit;
            pot_sum += elePot;
            if (elePot < pot_min) pot_min = elePot;
            if (elePot > pot_max) pot_max = elePot;
        }
        printf("Matrix elePot (eV): sum=%g, min=%g, max=%g, mean=%g\n",
               pot_sum, pot_min, pot_max, pot_sum/nEleAtoms);
        printf("Matrix elePot[0]=%g, elePot[1144]=%g\n",
               setPot[0] - (psr_coul[0] + plr_coul[0]) * potUnit,
               setPot[nEleAtoms/2] - (psr_coul[nEleAtoms/2] + plr_coul[nEleAtoms/2]) * potUnit);
        printf("Matrix: setPot[0]=%g, psr_coul[0]=%g, plr_coul[0]=%g\n",
               setPot[0], psr_coul[0], plr_coul[0]);
        bDebugMatrix = false;
    }

    {
        static std::vector<double> elePotV;
        elePotV.resize(nEleAtoms);
        #pragma omp parallel for num_threads(nthreads)
        for (int ii = 0; ii < nEleAtoms; ++ii)
            elePotV[ii] = setPot[ii] - static_cast<double>(psr_coul[ii] + plr_coul[ii]) * potUnit;
        cpm_inv_gemv(inv_MatrixA, elePotV.data(), eleCharge, nEleAtoms, nthreads);
        #pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < nEleAtoms; ++i)
            eleCharge[i] *= evscale;
    }

    // 2. Compute dEleCharge1V and sumCharge1V only once
    if (bSumCharge1VInitialized) {
        std::vector<double> ones(nEleAtoms, 1.0);
        cpm_inv_gemv(inv_MatrixA, ones.data(), dEleCharge1V, nEleAtoms, nthreads);
        sumCharge1V = 0.0;
        #pragma omp parallel for num_threads(nthreads) reduction(+:sumCharge1V)
        for (int i = 0; i < nEleAtoms; ++i)
            sumCharge1V += dEleCharge1V[i];
        bSumCharge1VInitialized = false;
        printf("sumCharge1V = %f\n", sumCharge1V);
    }

    // 3. Sum all charges (neutralization)
    double sumchargetemp = 0.0;
    #pragma omp parallel for num_threads(nthreads) reduction(+:sumchargetemp)
    for (int i = 0; i < nEleAtoms; ++i)
        sumchargetemp += eleCharge[i];

    double coeffdQ = -sumchargetemp / sumCharge1V;

    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i)
        eleCharge[i] += coeffdQ * dEleCharge1V[i];

    // for debug
    if(0)
    {
        double sumchargetemp = 0.0;
        for(int i=0;i<nEleAtoms;i++)
        {
            sumchargetemp += eleCharge[i];
            printf("i=%d, eleCharge=%f\n",i,eleCharge[i]);
        }
        printf("sum of charge on electrode = %f\n",sumchargetemp);			
    }
}

// bisheng CG algorithm 

// Helper: Matrix-vector product (row-major)
void matvec_cg(const double *A, const double *x, double *y, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += A[i * n + j] * x[j];
        y[i] = sum;
    }
}

// Helper: Dot product
double dot_cg(const double *x, const double *y, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
}

// Main function
void compute_and_neutralize_electrode_charge_cg(
    double *eleCharge,                  // Output: array, size nEleAtoms
    const double *setPot,               // Input: electrode potential (eV), size nEleAtoms
    const real *psr_coul,               // Input: array, size nEleAtoms
    const real *plr_coul,               // Input: array, size nEleAtoms
    const double *MatrixA,              // Input: flattened matrix, size nEleAtoms*nEleAtoms (row-major)
    int nEleAtoms,                      // Number of electrode atoms
    int nthreads,                       // Number of OpenMP threads
    double potUnit = 1.0 / ELECTRONVOLT,       // Potential unit conversion factor
    double evscale = ELECTRONVOLT / ONE_4PI_EPS0,          // 1/ke, in unit of V*nm/e, which convert V*nm to e as q = 1/ke * V * r
    int cg_maxiter = 1000,
    double cg_tol = 1e-8
) {
    // Compute right-hand side: b = setPot[ii] - (psr_coul[ii] + plr_coul[ii])*potUnit, scaled by evscale
    std::vector<double> b(nEleAtoms);
    #pragma omp parallel for num_threads(nthreads)
    for (int ii = 0; ii < nEleAtoms; ++ii)
        b[ii] = (setPot[ii] - static_cast<double>(psr_coul[ii] + plr_coul[ii]) * potUnit) * evscale;

    // ---- Conjugate Gradient Solve for MatrixA * eleCharge = b ----
    std::vector<double> r(nEleAtoms), d(nEleAtoms), q(nEleAtoms), y(nEleAtoms);
    // initial guess: eleCharge = 0
    // std::fill(eleCharge, eleCharge + nEleAtoms, 0.0);

    // r = b - A*Q
    matvec_cg(MatrixA, eleCharge, r.data(), nEleAtoms);
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i) r[i] = b[i] - r[i];

    // d = r
    std::copy(r.begin(), r.end(), d.begin());

    double delta_new = dot_cg(r.data(), r.data(), nEleAtoms);
    double delta_0 = delta_new;
    double delta_old = 0.0;
    int iter = 0;
    // while (iter < cg_maxiter && delta_new > cg_tol * cg_tol * delta_0) {
    while (iter < cg_maxiter && delta_new > cg_tol ) {
        matvec_cg(MatrixA, d.data(), y.data(), nEleAtoms);
        double denom = dot_cg(d.data(), y.data(), nEleAtoms);
        double alpha = (denom == 0.0) ? 0.0 : delta_new / denom;

        #pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < nEleAtoms; ++i) eleCharge[i] += alpha * d[i];

        if ((iter % 50 == 0) && (iter != 0)) {
            matvec_cg(MatrixA, eleCharge, r.data(), nEleAtoms);
            #pragma omp parallel for num_threads(nthreads)
            for (int i = 0; i < nEleAtoms; ++i) r[i] = b[i] - r[i];
        } else {
            #pragma omp parallel for num_threads(nthreads)
            for (int i = 0; i < nEleAtoms; ++i) r[i] -= alpha * y[i];
        }

        delta_old = delta_new;
        delta_new = dot_cg(r.data(), r.data(), nEleAtoms);
        double beta = (delta_old == 0.0) ? 0.0 : delta_new / delta_old;

        #pragma omp parallel for num_threads(nthreads)
        for (int i = 0; i < nEleAtoms; ++i) d[i] = r[i] + beta * d[i];

        ++iter;
    }
    // Give a warning if the CG did not converge 
    if (iter == cg_maxiter) {
        fprintf(stderr, "Warning: Conjugate Gradient did not converge after %d iterations.\n", iter);
    }
    // printf("iterations = %d, final delta_new = %e\n", iter, delta_new);
    // eleCharge is now the solution

    // ---- Neutralization ----
    // dEleCharge1V = A^{-1} * ones = charge response to uniform 1V potential
    // For MatCG, we solve A * dEleCharge1V = ones using CG (since we don't have inv_A)
    // bisheng 2026: cache A^-1.1 across steps, but key it on the matrix
    // pointer and the electrode count so a resized electrode or a second pme
    // instance rebuilds it instead of silently reusing a stale vector. The
    // std::vector also replaces a calloc that was never freed.
    static std::vector<double> dEleCharge1V_storage;
    static double              sumCharge1V             = 0.0;
    static bool                bSumCharge1VInitialized = true;
    static const double*       cachedMatrix            = nullptr;
    if (static_cast<int>(dEleCharge1V_storage.size()) != nEleAtoms || cachedMatrix != MatrixA)
    {
        dEleCharge1V_storage.assign(nEleAtoms, 0.0);
        cachedMatrix            = MatrixA;
        bSumCharge1VInitialized = true;
    }
    double* dEleCharge1V = dEleCharge1V_storage.data();
    if (bSumCharge1VInitialized) {
        // Solve A * dEleCharge1V = ones using CG
        std::vector<double> ones_vec(nEleAtoms, 1.0);
        std::vector<double> r_n(nEleAtoms), d_n(nEleAtoms), y_n(nEleAtoms);

        // Initial guess: dEleCharge1V = 0 (zeroed on (re)allocation above)
        // r = ones - A * dEleCharge1V = ones
        std::copy(ones_vec.begin(), ones_vec.end(), r_n.begin());
        std::copy(r_n.begin(), r_n.end(), d_n.begin());
        double dn_new = dot_cg(r_n.data(), r_n.data(), nEleAtoms);
        double dn_old;

        for (int it = 0; it < cg_maxiter && dn_new > cg_tol; ++it) {
            matvec_cg(MatrixA, d_n.data(), y_n.data(), nEleAtoms);
            double denom = dot_cg(d_n.data(), y_n.data(), nEleAtoms);
            double alpha_n = (denom == 0.0) ? 0.0 : dn_new / denom;

            #pragma omp parallel for num_threads(nthreads)
            for (int i = 0; i < nEleAtoms; ++i) dEleCharge1V[i] += alpha_n * d_n[i];

            if ((it % 50 == 0) && (it != 0)) {
                matvec_cg(MatrixA, dEleCharge1V, r_n.data(), nEleAtoms);
                #pragma omp parallel for num_threads(nthreads)
                for (int i = 0; i < nEleAtoms; ++i) r_n[i] = ones_vec[i] - r_n[i];
            } else {
                #pragma omp parallel for num_threads(nthreads)
                for (int i = 0; i < nEleAtoms; ++i) r_n[i] -= alpha_n * y_n[i];
            }

            dn_old = dn_new;
            dn_new = dot_cg(r_n.data(), r_n.data(), nEleAtoms);
            double beta_n = (dn_old == 0.0) ? 0.0 : dn_new / dn_old;
            #pragma omp parallel for num_threads(nthreads)
            for (int i = 0; i < nEleAtoms; ++i) d_n[i] = r_n[i] + beta_n * d_n[i];
        }

        sumCharge1V = 0.0;
        #pragma omp parallel for num_threads(nthreads) reduction(+:sumCharge1V)
        for (int i = 0; i < nEleAtoms; ++i)
            sumCharge1V += dEleCharge1V[i];

        bSumCharge1VInitialized = false;
        printf("sumCharge1V = %f\n", sumCharge1V);
    }
    double sumchargetemp = 0.0;
    #pragma omp parallel for num_threads(nthreads) reduction(+:sumchargetemp)
    for (int i = 0; i < nEleAtoms; ++i)
        sumchargetemp += eleCharge[i];

    double coeffdQ = -sumchargetemp / sumCharge1V;
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i)
        eleCharge[i] += coeffdQ * dEleCharge1V[i];
}

void compute_electrode_charge_double_constrain(
    double *eleCharge,                  // Output: array, size nEleAtoms
    const real *psr_coul,               // Input: array, size nEleAtoms
    const real *plr_coul,               // Input: array, size nEleAtoms
    const double *inv_MatrixA,          // Input: flattened matrix, size nEleAtoms*nEleAtoms
    int nEleAtoms,                      // Number of electrode atoms (should be even)
    int nthreads,                       // Number of OpenMP threads
    double Q1 = 0.0,                    // Desired charge on electrode 1 (first nEleAtoms/2 atoms)
    double potUnit = 1.0 / ELECTRONVOLT,       // Potential unit conversion factor
    double evscale = ELECTRONVOLT / ONE_4PI_EPS0,          // 1/ke, in unit of V*nm/e, which convert V*nm to e as q = 1/ke * V * r
    double *cellVoltageOut = nullptr,          // bisheng 2026 - optional: realised cell voltage (V)
    double Q2 = std::numeric_limits<double>::quiet_NaN()   // bisheng 2026 - charge on electrode 2; NaN -> -Q1
) {
    int n1 = nEleAtoms / 2;
    if (std::isnan(Q2))
    {
        Q2 = -Q1;
    }

    // Right-hand-side b (no setPot, just field)
    std::vector<double> b(nEleAtoms);
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i)
        b[i] = -(static_cast<double>(psr_coul[i] + plr_coul[i]) * potUnit) * evscale;

    // Ainv * b
    std::vector<double> Ainv_b(nEleAtoms, 0.0);
    cpm_inv_gemv(inv_MatrixA, b.data(), Ainv_b.data(), nEleAtoms, nthreads);

    // Constraint vectors: v1 = indicator of electrode 1, v2 = indicator of electrode 2.
    // bisheng 2026 - independent electrode sums allow arbitrary Q1 and Q2.
    // With Q2 = -Q1, these constraints also impose total electrode neutrality.
    std::vector<double> v1(nEleAtoms, 0.0), v2(nEleAtoms, 0.0);
    for (int i = 0; i < n1; ++i) v1[i] = 1.0;
    for (int i = n1; i < nEleAtoms; ++i) v2[i] = 1.0;

    // Ainv * v1, Ainv * v2: constant for a fixed matrix, so computed once and
    // cached (keyed on the matrix pointer and size) -- two dense products per
    // step fewer than before, on the host path as well.
    static std::vector<double> Ainv_v1, Ainv_v2;
    static const double*       cachedMatrixDC = nullptr;
    if (cachedMatrixDC != inv_MatrixA || static_cast<int>(Ainv_v1.size()) != nEleAtoms)
    {
        Ainv_v1.assign(nEleAtoms, 0.0); Ainv_v2.assign(nEleAtoms, 0.0);
        cpm_inv_gemv(inv_MatrixA, v1.data(), Ainv_v1.data(), nEleAtoms, nthreads);
        cpm_inv_gemv(inv_MatrixA, v2.data(), Ainv_v2.data(), nEleAtoms, nthreads);
        cachedMatrixDC = inv_MatrixA;
    }

    // Dot products for constraint system (all can be precomputed if needed)
    double v1_Ainv_b = 0, v2_Ainv_b = 0, v1_Ainv_v1 = 0, v1_Ainv_v2 = 0, v2_Ainv_v2 = 0;
    #pragma omp parallel for reduction(+:v1_Ainv_b,v2_Ainv_b,v1_Ainv_v1,v1_Ainv_v2,v2_Ainv_v2)
    for (int i = 0; i < nEleAtoms; ++i) {
        v1_Ainv_b  += v1[i] * Ainv_b[i];
        v2_Ainv_b  += v2[i] * Ainv_b[i];
        v1_Ainv_v1 += v1[i] * Ainv_v1[i];
        v1_Ainv_v2 += v1[i] * Ainv_v2[i];
        v2_Ainv_v2 += v2[i] * Ainv_v2[i];
    }

    // Build the 2x2 matrix and RHS for Lagrange multipliers [λ1, λ2]
    // | v1_Ainv_v1  v1_Ainv_v2 |   | λ1 |   = | Q1 - v1_Ainv_b |
    // | v1_Ainv_v2  v2_Ainv_v2 |   | λ2 |     | Q2 - v2_Ainv_b |

    double D = v1_Ainv_v1 * v2_Ainv_v2 - v1_Ainv_v2 * v1_Ainv_v2;
    double rhs1 = Q1 - v1_Ainv_b;
    double rhs2 = Q2 - v2_Ainv_b;

    double lambda1 = (rhs1 * v2_Ainv_v2 - rhs2 * v1_Ainv_v2) / D;
    double lambda2 = (rhs2 * v1_Ainv_v1 - rhs1 * v1_Ainv_v2) / D;

    // Q = Ainv_b + lambda1 * Ainv_v1 + lambda2 * Ainv_v2
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i)
        eleCharge[i] = Ainv_b[i] + lambda1 * Ainv_v1[i] + lambda2 * Ainv_v2[i];

    // bisheng 2026 - the multipliers ARE the electrode potentials:
    // matching q = Ainv(setPot*evscale) + Ainv*b against the line above gives
    // setPot*evscale = lambda1*v1 + lambda2*v2 with v1, v2 the electrode
    // indicators, so V1 = lambda1/evscale and V2 = lambda2/evscale and the cell
    // voltage is U = (lambda1 - lambda2)/evscale (the same quantity the previous
    // all-ones v2 gave as lambda1/evscale).
    if (cellVoltageOut) *cellVoltageOut = (lambda1 - lambda2) / evscale;
}

void compute_electrode_charge_single_constrain(
    double *eleCharge,                  // Output: array, size nEleAtoms
    const real *psr_coul,               // Input: array, size nEleAtoms
    const real *plr_coul,               // Input: array, size nEleAtoms
    const double *inv_MatrixA,          // Input: flattened matrix, size nEleAtoms*nEleAtoms
    int nEleAtoms,                      // Number of electrode atoms
    int nthreads,                       // Number of OpenMP threads
    double Qtarget = 0.0,               // Desired total charge (usually zero for neutral)
    double potUnit = 1.0 / ELECTRONVOLT,       // Potential unit conversion factor
    double evscale = ELECTRONVOLT / ONE_4PI_EPS0,          // 1/ke, in unit of V*nm/e, which convert V*nm to e as q = 1/ke * V * r
    double *electrodePotentialOut = nullptr    // bisheng 2026 - optional: realised electrode potential (V, vs the Ewald zero)
) {
    // Right-hand-side b (as before)
    std::vector<double> b(nEleAtoms);
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i)
        b[i] = -(static_cast<double>(psr_coul[i] + plr_coul[i]) * potUnit) * evscale;

    // Ainv * b
    std::vector<double> Ainv_b(nEleAtoms, 0.0);
    cpm_inv_gemv(inv_MatrixA, b.data(), Ainv_b.data(), nEleAtoms, nthreads);

    // Constraint vector: v = [1, 1, ..., 1]
    std::vector<double> v(nEleAtoms, 1.0);

    // Ainv * v: constant for a fixed matrix, cached (bisheng 2026)
    static std::vector<double> Ainv_v;
    static const double*       cachedMatrixSC = nullptr;
    if (cachedMatrixSC != inv_MatrixA || static_cast<int>(Ainv_v.size()) != nEleAtoms)
    {
        Ainv_v.assign(nEleAtoms, 0.0);
        cpm_inv_gemv(inv_MatrixA, v.data(), Ainv_v.data(), nEleAtoms, nthreads);
        cachedMatrixSC = inv_MatrixA;
    }

    // Dot products
    double v_Ainv_b = 0.0, v_Ainv_v = 0.0;
    #pragma omp parallel for reduction(+:v_Ainv_b,v_Ainv_v)
    for (int i = 0; i < nEleAtoms; ++i) {
        v_Ainv_b += v[i] * Ainv_b[i];
        v_Ainv_v += v[i] * Ainv_v[i];
    }

    // Solve for Lagrange multiplier lambda
    double lambda = (Qtarget - v_Ainv_b) / v_Ainv_v;

    // Final result: Q = Ainv_b + lambda * Ainv_v
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < nEleAtoms; ++i)
        eleCharge[i] = Ainv_b[i] + lambda * Ainv_v[i];

    // bisheng 2026 - as in the two-electrode solver, the multiplier is the electrode
    // potential in evscale units: setPot*evscale = lambda*v, so V = lambda/evscale.
    if (electrodePotentialOut) *electrodePotentialOut = lambda / evscale;
}

/*! \brief Calculate initial Lorentz-Berthelot coefficients for LJ-PME */
static void calc_initial_lb_coeffs(gmx::ArrayRef<real> coefficient, const real* local_c6, const real* local_sigma)
{
    for (gmx::index i = 0; i < coefficient.ssize(); ++i)
    {
        real sigma4    = local_sigma[i];
        sigma4         = sigma4 * sigma4;
        sigma4         = sigma4 * sigma4;
        coefficient[i] = local_c6[i] / sigma4;
    }
}

/*! \brief Calculate next Lorentz-Berthelot coefficients for LJ-PME */
static void calc_next_lb_coeffs(gmx::ArrayRef<real> coefficient, const real* local_sigma)
{
    for (gmx::index i = 0; i < coefficient.ssize(); ++i)
    {
        coefficient[i] *= local_sigma[i];
    }
}

/* bisheng 2026 - CPM restart state (cpm_state.dat): everything the solver keeps
 * between steps that the GROMACS checkpoint does not know about. Written at
 * checkpoint cadence, loaded when continuing from a checkpoint. */
static const int c_cpmStateVersion = 1;

static void cpm_write_vector(FILE* fp, const char* name, const std::vector<real>& v)
{
    fprintf(fp, "%s %zu\n", name, v.size());
    for (const real x : v)
    {
        fprintf(fp, "%.9e\n", static_cast<double>(x));
    }
}

static bool cpm_read_vector(FILE* fp, const char* name, std::vector<real>* v)
{
    char   tag[64];
    size_t n = 0;
    if (fscanf(fp, "%63s %zu", tag, &n) != 2 || std::strcmp(tag, name) != 0)
    {
        return false;
    }
    v->resize(n);
    for (size_t i = 0; i < n; i++)
    {
        double x;
        if (fscanf(fp, "%lf", &x) != 1)
        {
            return false;
        }
        (*v)[i] = static_cast<real>(x);
    }
    return true;
}

void gmx_pme_cpm_set_force_after_solve(gmx_pme_t* pme, const bool value)
{
    pme->cpmForceAfterSolve = value;
}

const std::vector<real>& gmx_pme_cpm_solved_charges(const gmx_pme_t* pme)
{
    return pme->cpm_qSolved;
}

void gmx_pme_cpm_set_time(struct gmx_pme_t* pme, real t)
{
    pme->current_time = t;
}

void gmx_pme_cpm_set_step(struct gmx_pme_t* pme, int64_t step)
{
    pme->current_step = step;
}

/* bisheng 2026 - galvanostatic cycling with voltage cut-offs ("mode = cutoff").
 * Returns the imposed charge of electrode 1 for the current step. The charge is
 * integrated in time, Q_set += dir*I*(t - t_prev), so repeated evaluations at the
 * same time (the step-0 seeding pass, energy-only steps) do not advance it, and
 * the direction flips on the realised cell voltage of the PREVIOUS solve, which
 * is what pme->cpm_cell_voltage holds when this runs (the current step has not
 * been solved yet). Every switch goes to stdout and to gcd_switches.dat. */
static double cpm_gcd_cutoff_target(gmx_pme_t* pme, const gmx::CpmPotentialControl::CutoffProtocol& c)
{
    const double t = pme->current_time;
    if (!pme->cpm_gcd_started)
    {
        pme->cpm_gcd_started = true;
        pme->cpm_gcd_dir     = c.direction;
        pme->cpm_gcd_qset    = c.q0;
        pme->cpm_gcd_tprev   = t;
        pme->cpm_gcd_nswitch = 0;
        printf("CPM GCD cutoff: start at t = %.4f ps with Q_set = %g e, direction %+d "
               "(|I| = %g e/ps, Umax = %g V, Umin = %g V)\n",
               t, c.q0, c.direction, c.current, c.uMax, c.uMin);
        return pme->cpm_gcd_qset;
    }
    const double dt = t - pme->cpm_gcd_tprev;
    if (dt <= 0.0)
    {
        return pme->cpm_gcd_qset;
    }
    const double U = pme->cpm_cell_voltage;
    // The per-step voltage fluctuates by ~0.3 V; with Utau > 0 the cut-offs are
    // compared with an exponential moving average, the response time of the cycler.
    double Ucmp = U;
    if (c.uTau > 0.0)
    {
        if (!pme->cpm_gcd_ufinit)
        {
            pme->cpm_gcd_ufilt  = U;
            pme->cpm_gcd_ufinit = true;
        }
        else
        {
            const double w = std::min(1.0, dt / c.uTau);
            pme->cpm_gcd_ufilt += w * (U - pme->cpm_gcd_ufilt);
        }
        Ucmp = pme->cpm_gcd_ufilt;
    }
    int dir = pme->cpm_gcd_dir;
    if (dir > 0 && Ucmp >= c.uMax)
    {
        dir = -1;
    }
    else if (dir < 0 && Ucmp <= c.uMin)
    {
        dir = +1;
    }
    if (dir != pme->cpm_gcd_dir)
    {
        pme->cpm_gcd_dir = dir;
        pme->cpm_gcd_nswitch++;
        printf("CPM GCD cutoff: switch %" PRId64 " at step %" PRId64 ", t = %.4f ps: Q_set = %.6f e, "
               "U = %.4f V (filtered %.4f V) -> direction %+d\n",
               pme->cpm_gcd_nswitch, pme->current_step, t, pme->cpm_gcd_qset, U, Ucmp, dir);
        FILE* fp = fopen("gcd_switches.dat", "a");
        if (fp != nullptr)
        {
            if (ftell(fp) == 0)
            {
                fprintf(fp, "# switch  step  time(ps)  Q_set(e)  U(V)  new_direction  U_filtered(V)\n");
            }
            fprintf(fp, "%8" PRId64 " %12" PRId64 " %14.6f %14.6f %12.6f %+d %12.6f\n",
                    pme->cpm_gcd_nswitch, pme->current_step, t, pme->cpm_gcd_qset, U, dir, Ucmp);
            fclose(fp);
        }
    }
    pme->cpm_gcd_qset += pme->cpm_gcd_dir * c.current * dt;
    pme->cpm_gcd_tprev = t;
    return pme->cpm_gcd_qset;
}

void gmx_pme_cpm_write_restart(struct gmx_pme_t* pme, int64_t step, const real* chargeA)
{
    const int         nEle = pme->nEleAtoms;
    const std::string tmp  = "cpm_state.dat.tmp";
    FILE*             fp   = fopen(tmp.c_str(), "w");
    if (fp == nullptr)
    {
        fprintf(stderr, "CPM: could not write %s; solver state will cold-start on restart\n",
                tmp.c_str());
        return;
    }
    fprintf(fp, "cpm_state_version %d\n", c_cpmStateVersion);
    fprintf(fp, "step %" PRId64 "\n", step);
    fprintf(fp, "nEleAtoms %d\n", nEle);
    // bisheng 2026 - Thomas-Fermi provenance: a matrix-free run restarted in a
    // directory without electrode_tf.dat would otherwise continue silently with
    // TF off (there is no cached matrix on that path to trip the MatrixA.meta guard).
    fprintf(fp, "thomasFermi %d tfLtfSum %.12g tfVolSum %.12g\n",
            pme->bThomasFermi ? 1 : 0, pme->tf_ltf_sum, pme->tf_vol_sum);
    // bisheng 2026 - cut-off cycler state, so a restart continues the same cycle
    // (record present only once the cycler has started; readers before it skip it).
    if (pme->cpm_gcd_started)
    {
        fprintf(fp, "gcdCutoff dir %d qset %.12g tprev %.12g nswitch %" PRId64 " ufilt %.12g ufinit %d\n",
                pme->cpm_gcd_dir, pme->cpm_gcd_qset, pme->cpm_gcd_tprev, pme->cpm_gcd_nswitch,
                pme->cpm_gcd_ufilt, pme->cpm_gcd_ufinit ? 1 : 0);
    }
    std::vector<real> charges(chargeA, chargeA + nEle);
    cpm_write_vector(fp, "chargeA", charges);
    cpm_write_vector(fp, "qPrev", pme->cpm_qPrev);
    cpm_write_vector(fp, "qNm1", pme->cpm_qNm1);
    cpm_write_vector(fp, "lastError", pme->cpm_lastError);
    fprintf(fp, "predStep %" PRId64 "\n", pme->cpm_predStep);
    cpm_write_vector(fp, "extLagQ", pme->ext_lag_q);
    cpm_write_vector(fp, "extLagQdot", pme->ext_lag_qdot);
    fclose(fp);
    if (std::rename(tmp.c_str(), "cpm_state.dat") != 0)
    {
        fprintf(stderr, "CPM: could not rename %s to cpm_state.dat\n", tmp.c_str());
    }
}

// bisheng 2026 - drop the CPM solver's per-run history. Called between the step-0
// charge-seeding force evaluation and the force evaluation that is actually used, so
// that the seeding pass leaves behind only the electrode charges in md->chargeA and
// neither the charge predictor nor the extended-Lagrangian charge dynamics get
// advanced by an extra step.
void gmx_pme_cpm_reset_solver_state(struct gmx_pme_t* pme)
{
    if (pme == nullptr)
    {
        return;
    }
    pme->cpm_qPrev.clear();
    pme->cpm_qNm1.clear();
    pme->cpm_qPredicted.clear();
    pme->cpm_lastError.clear();
    pme->cpm_predStep = 0;
    pme->ext_lag_q.clear();
    pme->ext_lag_qdot.clear();
}

bool gmx_pme_cpm_read_restart(struct gmx_pme_t* pme, int64_t expectedStep, int nEleAtomsExpected, real* chargeA)
{
    FILE* fp = fopen("cpm_state.dat", "r");
    if (fp == nullptr)
    {
        return false;
    }
    int     version = 0;
    int64_t step    = -1;
    int     nEle    = 0;
    bool    ok = (fscanf(fp, "cpm_state_version %d step %" SCNd64 " nEleAtoms %d", &version, &step, &nEle) == 3)
              && (version == c_cpmStateVersion) && (nEle == nEleAtomsExpected);
    // optional TF record (absent in files written before the TF model existed)
    {
        long   pos = ftell(fp);
        int    tf  = 0;
        double ls = 0.0, vs = 0.0;
        if (ok && fscanf(fp, " thomasFermi %d tfLtfSum %lf tfVolSum %lf", &tf, &ls, &vs) == 3)
        {
            pme->cpm_restart_tf = tf; pme->cpm_restart_ltf_sum = ls; pme->cpm_restart_vol_sum = vs;
        }
        else
        {
            fseek(fp, pos, SEEK_SET);
            pme->cpm_restart_tf = 0;   // pre-TF writer: TF was necessarily off
        }
    }
    // optional cut-off cycler record (bisheng 2026)
    {
        long    pos = ftell(fp);
        int     dir = 0;
        double  qs = 0.0, tp = 0.0;
        int64_t ns = 0;
        double  uf = 0.0;
        int     ui = 0;
        if (ok && fscanf(fp, " gcdCutoff dir %d qset %lf tprev %lf nswitch %" SCNd64 " ufilt %lf ufinit %d",
                         &dir, &qs, &tp, &ns, &uf, &ui) == 6)
        {
            pme->cpm_gcd_started = true;
            pme->cpm_gcd_dir     = dir;
            pme->cpm_gcd_qset    = qs;
            pme->cpm_gcd_tprev   = tp;
            pme->cpm_gcd_nswitch = ns;
            pme->cpm_gcd_ufilt   = uf;
            pme->cpm_gcd_ufinit  = (ui != 0);
            printf("CPM GCD cutoff: restored Q_set = %.6f e, direction %+d, %" PRId64 " switches so far, "
                   "last advanced at t = %.4f ps, filtered U = %.4f V\n", qs, dir, ns, tp, uf);
        }
        else
        {
            fseek(fp, pos, SEEK_SET);
        }
    }
    std::vector<real> charges;
    ok = ok && cpm_read_vector(fp, "chargeA", &charges) && static_cast<int>(charges.size()) == nEle;
    ok = ok && cpm_read_vector(fp, "qPrev", &pme->cpm_qPrev);
    ok = ok && cpm_read_vector(fp, "qNm1", &pme->cpm_qNm1);
    ok = ok && cpm_read_vector(fp, "lastError", &pme->cpm_lastError);
    char tag[64];
    ok = ok && (fscanf(fp, "%63s %" SCNd64, tag, &pme->cpm_predStep) == 2)
       && std::strcmp(tag, "predStep") == 0;
    ok = ok && cpm_read_vector(fp, "extLagQ", &pme->ext_lag_q);
    ok = ok && cpm_read_vector(fp, "extLagQdot", &pme->ext_lag_qdot);
    fclose(fp);
    if (!ok)
    {
        fprintf(stderr,
                "CPM: cpm_state.dat is unreadable or does not match this system; "
                "the solver will cold-start (results differ from an uninterrupted run).\n");
        pme->cpm_qPrev.clear();
        pme->cpm_qNm1.clear();
        pme->cpm_lastError.clear();
        pme->ext_lag_q.clear();
        pme->ext_lag_qdot.clear();
        pme->cpm_predStep = 0;
        return false;
    }
    if (step != expectedStep)
    {
        fprintf(stderr,
                "CPM: cpm_state.dat was written at step %" PRId64 " but the checkpoint restarts at "
                "step %" PRId64 "; using it anyway (small transient possible).\n",
                step, expectedStep);
    }
    std::copy(charges.begin(), charges.end(), chargeA);
    // The state file is written at checkpoint time, i.e. after the predictor
    // overwrote chargeA, so the stored charges ARE the current prediction.
    pme->cpm_qPredicted = charges;
    printf("CPM: restored solver state (charges, warm starts, predictor%s) from cpm_state.dat "
           "written at step %" PRId64 "\n",
           pme->ext_lag_q.empty() ? "" : ", extended-Lagrangian q/qdot", step);
    return true;
}

/* bisheng 2026 - CPM: solves the electrode charges from the accumulated short-range
 * potential (pme->psr_coul) and the long-range potential. If \p plrExternal is
 * non-null (GPU PME path) it is used directly; otherwise the potential is computed
 * with the CPU PME machinery via gmx_pme_calc_phik(). Writes the new charges into
 * \p chargeA[0..nEleAtoms). Extracted verbatim from the bCalcQ block of gmx_pme_do().
 */
int gmx_pme_cpm_solve_charges(struct gmx_pme_t*              pme,
                              gmx::ArrayRef<const gmx::RVec> coordinates,
                              real                           chargeA[],
                              const matrix                   box,
                              const real*                    plrExternal)
{
            // some variables are static to avoid reallocation
            static int nEleAtoms = pme->nEleAtoms;
            // For matrix-free CG (inv_or_cg == 2), electrodes are flexible so we must
            // recompute splines every step. For matrix methods, electrodes are fixed.
            static gmx_bool bFixElectrode = (pme->inv_or_cg != 2); 

            static real *plr_coul = nullptr;
            static double *setPot = nullptr;  
            static double *eleCharge = nullptr;

            if (plr_coul == nullptr)
            {
                plr_coul = (real *)calloc(pme->nEleAtoms, sizeof(real));
            }

            if (eleCharge == nullptr)
            {
                eleCharge = (double *)calloc(nEleAtoms, sizeof(double));

                // bisheng 2026 - device solve for the matrix-inversion path: on
                // whenever PME runs on the GPU (or GMX_CPM_GPU_SOLVE=1), off with
                // GMX_CPM_NO_GPU_SOLVE=1; GMX_CPM_GPU_SOLVE_SINGLE=1 uses a float copy.
                if (pme->inv_or_cg == 0)
                {
                    const bool avail  = cpm_gpu_solve_available();
                    const bool wanted = (pme_gpu_task_enabled(pme) || getenv("GMX_CPM_GPU_SOLVE") != nullptr)
                                        && getenv("GMX_CPM_NO_GPU_SOLVE") == nullptr;
                    g_cpmUseGpuSolve    = avail && wanted;
                    g_cpmGpuSolveSingle = getenv("GMX_CPM_GPU_SOLVE_SINGLE") != nullptr;
                    printf("CPM: matrix-inversion solve on the %s%s\n",
                           g_cpmUseGpuSolve ? "GPU" : "CPU",
                           g_cpmUseGpuSolve ? (g_cpmGpuSolveSingle ? " (single precision)" : " (double precision)")
                                            : (avail && !wanted ? " (device available; set GMX_CPM_GPU_SOLVE=1 to use it)" : ""));
                }

                // bisheng 2025 - Validate parameter combinations and fallback if unsupported
                // userint2 (inv_or_cg): 0=matrix inversion, 1=matrix CG, 2=matrix-free CG
                // userint3 (oneEle_or_not): 0=two electrodes, 1=single electrode
                // userint4 (pot_or_charge): 0=potential constraint, 1=charge constraint
                // userreal3 (ext_lag_mass): >0 means Extended Lagrangian

                // Validate basic parameter ranges
                if (pme->inv_or_cg < 0 || pme->inv_or_cg > 2)
                {
                    gmx_fatal(FARGS, "Invalid userint2=%d. Valid values: 0 (matrix inversion), 1 (matrix CG), 2 (matrix-free CG)", pme->inv_or_cg);
                }
                if (pme->oneEle_or_not < 0 || pme->oneEle_or_not > 1)
                {
                    gmx_fatal(FARGS, "Invalid userint3=%d. Valid values: 0 (two electrodes), 1 (single electrode)", pme->oneEle_or_not);
                }
                if (pme->pot_or_charge < 0 || pme->pot_or_charge > 1)
                {
                    gmx_fatal(FARGS, "Invalid userint4=%d. Valid values: 0 (potential constraint), 1 (charge constraint)", pme->pot_or_charge);
                }

                // Check for unsupported combinations and fallback
                bool bFallback = false;

                // Single electrode MUST use charge constraint (potential constraint is physically impossible)
                if (pme->oneEle_or_not == 1 && pme->pot_or_charge == 0)
                {
                    printf("\nWARNING: Single electrode (userint3=1) cannot use potential constraint (userint4=0).\n");
                    printf("         Reason: Setting absolute potential requires net charge on electrode,\n");
                    printf("                 which conflicts with system neutrality constraint.\n");
                    printf("         Falling back to charge constraint (userint4=1).\n\n");
                    pme->pot_or_charge = 1;
                    bFallback = true;
                }

                // Matrix CG (userint2=1) only supports two electrodes + potential constraint
                if (pme->inv_or_cg == 1 && (pme->oneEle_or_not == 1 || pme->pot_or_charge == 1))
                {
                    printf("\nWARNING: Matrix CG (userint2=1) only supports two electrodes (userint3=0) with potential constraint (userint4=0).\n");
                    printf("         Falling back to matrix inversion (userint2=0).\n\n");
                    pme->inv_or_cg = 0;
                    bFallback = true;
                }

                // Extended Lagrangian only works with matrix-free CG
                if (pme->ext_lag_mass > 0)
                {
                    if (pme->inv_or_cg != 2)
                    {
                        printf("\nWARNING: Extended Lagrangian (userreal3>0) only works with matrix-free CG (userint2=2).\n");
                        printf("         Disabling Extended Lagrangian (setting userreal3=0).\n\n");
                        pme->ext_lag_mass = 0;
                        bFallback = true;
                    }
                    // Note: Extended Lagrangian now supports all matrix-free CG configurations:
                    // - Two electrodes with potential constraint
                    // - Two electrodes with charge constraint
                    // - Single electrode with charge constraint
                }

                if (bFallback)
                {
                    printf("Supported CPM configurations:\n");
                    printf("================================================================================\n");
                    printf("  userint2  userint3  userint4  userreal3  Description\n");
                    printf("--------------------------------------------------------------------------------\n");
                    printf("     0         0         0         0       Matrix inversion, two electrodes, potential\n");
                    printf("     0         0         1         0       Matrix inversion, two electrodes, charge\n");
                    printf("     0         1         1         0       Matrix inversion, single electrode, charge\n");
                    printf("     1         0         0         0       Matrix CG, two electrodes, potential\n");
                    printf("     2         0         0         0       Matrix-free CG, two electrodes, potential\n");
                    printf("     2         0         1         0       Matrix-free CG, two electrodes, charge\n");
                    printf("     2         1         1         0       Matrix-free CG, single electrode, charge\n");
                    printf("     2         0         0        >0       Matrix-free CG + ExtLag, two electrodes, potential\n");
                    printf("     2         0         1        >0       Matrix-free CG + ExtLag, two electrodes, charge\n");
                    printf("     2         1         1        >0       Matrix-free CG + ExtLag, single electrode, charge\n");
                    printf("================================================================================\n");
                    printf("Note: Single electrode (userint3=1) only supports charge constraint (userint4=1)\n\n");
                }

                // Print final configuration being used
                printf("CPM Configuration:\n");
                printf("  Solver: ");
                if (pme->inv_or_cg == 0)
                {
                    printf("Matrix inversion (userint2=0)\n");
                }
                else if (pme->inv_or_cg == 1)
                {
                    printf("Matrix CG (userint2=1)\n");
                }
                else if (pme->inv_or_cg == 2)
                {
                    printf("Matrix-free CG (userint2=2)\n");
                    printf("    eta = %g nm^-1, rcoulomb = %g nm, ewaldcoeff_q = %g nm^-1\n",
                           pme->eta, pme->rcoulomb, pme->ewaldcoeff_q);
                    if (pme->ext_lag_mass > 0)
                    {
                        printf("    Extended Lagrangian enabled: mass = %g\n", pme->ext_lag_mass);
                    }
                }
                printf("  Electrodes: %s (userint3=%d)\n",
                       pme->oneEle_or_not == 0 ? "Two electrodes" : "Single electrode",
                       pme->oneEle_or_not);
                printf("  Constraint: %s (userint4=%d)\n",
                       pme->pot_or_charge == 0 ? "Potential" : "Charge",
                       pme->pot_or_charge);
                printf("  Electrode atoms: %d\n", nEleAtoms);
            }


            // bisheng 2025 - Step counter for time-dependent control debug output
            static int64_t timeDepStepCount = 0;

            if (setPot == nullptr)
            {
                setPot = (double *)calloc(nEleAtoms, sizeof(double));
                // check if pme->applied_potential is set and if nEleAtoms is even
                if (pme->bTimeDependentControl)
                {
                    printf("Time-dependent %s control enabled.\n",
                           pme->pot_or_charge == 0 ? "potential" : "charge");
                }
                else if (!pme->pot_or_charge)
                    printf("The applied potential is set to %f\n", pme->applied_potential);
                else
                    printf("The applied charge is set to %f\n", pme->applied_total_charge);

                if (nEleAtoms % 2 != 0 && pme->oneEle_or_not ==0)
                {
                    gmx_fatal(FARGS, "nEleAtoms is not even, cannot calculate electrode potential.");
                }
                // bisheng 2026 - second-electrode charge target and cell neutrality.
                if (pme->pot_or_charge == 1)
                {
                    pme->applied_charge2 = (pme->oneEle_or_not == 0) ? -static_cast<double>(pme->applied_total_charge) : 0.0;
                    double q2v = 0.0;
                    if (pme->oneEle_or_not == 0 && pme->bTimeDependentControl
                        && pme->timeDepControl->getParameter("Q2", &q2v))
                    {
                        pme->applied_charge2 = q2v;
                        pme->bCharge2Set     = true;
                        printf("CPM: electrode 2 charge target Q2 = %g e from potential_control.dat "
                               "(without it, Q2 = -Q1)\n", q2v);
                    }
                }
            }

            // bisheng 2025 - Update setPot each step for time-dependent control
            // For constant control, only initialize once (static check)
            static bool bSetPotInitialized = false;
            if (pme->bTimeDependentControl)
            {
                // Evaluate the time-dependent expression, or advance the cut-off cycler
                real currentValue;
                const gmx::CpmPotentialControl::CutoffProtocol& gcd = pme->timeDepControl->cutoff();
                if (gcd.enabled)
                {
                    if (pme->pot_or_charge == 0)
                    {
                        gmx_fatal(FARGS, "mode = cutoff in potential_control.dat needs the charge constraint (userint4 = 1)");
                    }
                    currentValue = static_cast<real>(cpm_gcd_cutoff_target(pme, gcd));
                }
                else
                {
                    currentValue = pme->timeDepControl->evaluate(pme->current_time);
                }

                if (pme->pot_or_charge == 0)
                {
                    // Potential constraint mode: V(t) in Volts
                    // Update applied_potential and setPot for two-electrode case
                    pme->applied_potential = currentValue;
                    if (pme->oneEle_or_not == 0)
                    {
                        int halfEleAtoms = nEleAtoms / 2;
                        for (int i = 0; i < halfEleAtoms; ++i)
                        {
                            setPot[i] = currentValue/2;
                        }
                        for (int i = halfEleAtoms; i < nEleAtoms; ++i)
                        {
                            setPot[i] = -currentValue/2;
                        }
                    }
                }
                else
                {
                    // Charge constraint mode: Q(t) in elementary charges
                    pme->applied_total_charge = currentValue;
                    if (!pme->bCharge2Set && pme->oneEle_or_not == 0)
                    {
                        pme->applied_charge2 = -static_cast<double>(currentValue);   // capacitor default follows Q1(t)
                    }
                }

                // Print debug output periodically (every nstenergy steps)
                if (pme->nstenergy > 0 && timeDepStepCount % pme->nstenergy == 0)
                {
                    if (pme->pot_or_charge == 0)
                        printf("CPM Time-dep: t = %.3f ps, V(t) = %.6f V\n",
                               pme->current_time, currentValue);
                    else if (gcd.enabled)
                        printf("CPM Time-dep: t = %.3f ps, Q_set = %.6f e, direction %+d, U(prev) = %.4f V\n",
                               pme->current_time, currentValue, pme->cpm_gcd_dir, pme->cpm_cell_voltage);
                    else
                        printf("CPM Time-dep: t = %.3f ps, Q(t) = %.6f e\n",
                               pme->current_time, currentValue);
                }
                timeDepStepCount++;
            }
            else if (!bSetPotInitialized)
            {
                // Static potential constraint initialization (original code)
                if (pme->pot_or_charge == 0 && pme->oneEle_or_not == 0)
                {
                    int halfEleAtoms = nEleAtoms / 2;
                    for (int i = 0; i < halfEleAtoms; ++i)
                    {
                        setPot[i] = pme->applied_potential/2; // convert to eV
                    }
                    for (int i = halfEleAtoms; i < nEleAtoms; ++i)
                    {
                        setPot[i] = -pme->applied_potential/2; // convert to eV
                    }
                }
                bSetPotInitialized = true;
            }
            // bisheng 2026 - cell neutrality: the electrolyte must carry -(Q1+Q2). Checked
            // once, HERE, after both targets are current -- in time-dependent mode Q1 is
            // only assigned in the block above, so checking it inside the one-time setup
            // saw Q1 = 0 and warned spuriously (found on the two-nanotube runs).
            static bool bNeutralityChecked = false;
            if (pme->pot_or_charge == 1 && !bNeutralityChecked)
            {
                bNeutralityChecked = true;
                double qElec = 0.0;
                for (int i = nEleAtoms; i < static_cast<int>(coordinates.size()); ++i)
                {
                    qElec += chargeA[i];
                }
                const double qTarget = static_cast<double>(pme->applied_total_charge)
                                       + (pme->oneEle_or_not == 0 ? pme->applied_charge2 : 0.0);
                if (std::fabs(qElec + qTarget) > 1e-3)
                {
                    fprintf(stderr,
                            "\nWARNING: CPM charge constraint: electrode targets sum to %+.4f e and the "
                            "electrolyte carries %+.4f e, a net cell charge of %+.4f e.\n"
                            "         PME neutralises it with a uniform background, which is not the "
                            "physics you want; give the electrolyte %+.4f e of excess ions.\n\n",
                            qTarget, qElec, qElec + qTarget, -qTarget);
                }
                else
                {
                    printf("CPM: cell is neutral (electrode targets %+.4f e, electrolyte %+.4f e)\n", qTarget, qElec);
                }
            }
            // bisheng 2026 - GPU PME path supplies the long-range potential directly
            if (plrExternal != nullptr)
            {
                std::memcpy(plr_coul, plrExternal, nEleAtoms * sizeof(real));
            }
            else
            {
                // dummy charge for getting potential per atom, if charge is zero, gmx won't spread to get spline coefficients
                static std::vector<real> constantCharge(nEleAtoms, 0.1);
                gmx_pme_calc_phik(pme, coordinates.subArray(0, nEleAtoms), gmx::arrayRefFromArray(constantCharge.data(), nEleAtoms), plr_coul, bFixElectrode);
            }

            // Debug: print plr_coul stats to check consistency across steps
            static bool bDebugplr_coul = false;
            if(bDebugplr_coul) 
            {
                static int step_count = 0;
                step_count++;
                real plr_sum = 0, plr_max = 0;
                for (int i = 0; i < nEleAtoms; i++) {
                    plr_sum += plr_coul[i];
                    if (std::abs(plr_coul[i]) > plr_max) plr_max = std::abs(plr_coul[i]);
                }
                printf("DEBUG Step %d: plr_coul sum=%g, max=%g, plr_coul[0]=%g, plr_coul[%d]=%g\n",
                       step_count, plr_sum, plr_max, plr_coul[0], nEleAtoms-1, plr_coul[nEleAtoms-1]);
                bDebugplr_coul = false;
            }

            // DEBUG: Print ALL electrode atoms to file
            // bisheng 2026: potential diagnostics are opt-in through the
            // GMX_CPM_DEBUG_POTENTIALS environment variable.
            static bool bDebugPotentials = (getenv("GMX_CPM_DEBUG_POTENTIALS") != nullptr);
            static int debug_step = 0;
            static FILE* debug_file = nullptr;

            if (bDebugPotentials)
            {
                if (debug_file == nullptr) {
                    // append, so a checkpoint restart does not truncate the earlier leg
                    debug_file = fopen("debug_potentials_full.dat", "a");
                    if (debug_file) {
                        fprintf(debug_file, "# step atom psr_coul(eV) plr_coul(eV) total(eV)\n");
                    }
                }

                if (debug_file && debug_step < 10)  // First 10 steps, all atoms
                {
                    debug_step++;
                    const double potUnit = 1.0 / ELECTRONVOLT;

                    for (int i = 0; i < nEleAtoms; i++) {
                        double psr = pme->psr_coul[i] * potUnit;
                        double plr = plr_coul[i] * potUnit;
                        fprintf(debug_file, "%6d %6d %16.8f %16.8f %16.8f\n",
                                debug_step, i, psr, plr, psr + plr);
                    }
                    fflush(debug_file);
                }
            }

            // bisheng 2026: the matrix-free / extended-Lagrangian solvers
            // handle BOTH geometries -- they take oneEle_or_not and pot_or_charge and
            // call gmx_pme_solve_electrode_charges_cg_single_electrode when asked.
            // Route matrix-free/EL here for either geometry; the matrix
            // solvers keep their own single-electrode branch further down.
            if(pme->oneEle_or_not == 0 || pme->inv_or_cg == 2)
            {   
                if(pme->inv_or_cg == 0)
                {
                    // Compute the electrode charge using the original method
                    if(pme->pot_or_charge == 0)
                        compute_and_neutralize_electrode_charge(eleCharge, setPot, pme->psr_coul, plr_coul, pme->inv_MatrixA, nEleAtoms, pme->nthread);
                    else if(pme->pot_or_charge == 1)
                        compute_electrode_charge_double_constrain(eleCharge, pme->psr_coul, plr_coul, pme->inv_MatrixA, nEleAtoms, pme->nthread, pme->applied_total_charge,
                                                                 1.0 / ELECTRONVOLT, ELECTRONVOLT / ONE_4PI_EPS0, &pme->cpm_cell_voltage,
                                                                 pme->applied_charge2);
                }
                else if (pme->inv_or_cg == 1)
                {
                    // Compute the electrode charge using the CG method (with matrix)
                    if(pme->pot_or_charge == 0)
                        compute_and_neutralize_electrode_charge_cg(eleCharge, setPot, pme->psr_coul, plr_coul, pme->MatrixA, nEleAtoms, pme->nthread);
                }
                else if (pme->inv_or_cg == 2)
                {
                    // bisheng 2025 - Matrix-free CG for flexible electrodes
                    // Compute target potential b = setPot - (psr_coul + plr_coul) * potUnit (in eV units)
                    // This matches the elePot calculation in matrix method
                    static const double potUnit = 1.0 / ELECTRONVOLT;  // kJ/mol/e -> V
                    static std::vector<real> b_target;
                    if (static_cast<int>(b_target.size()) < nEleAtoms)
                    {
                        b_target.resize(nEleAtoms);
                    }

                    #pragma omp parallel for num_threads(pme->nthread)
                    for (int i = 0; i < nEleAtoms; i++)
                    {
                        // Target potential in eV (same as elePot in matrix method)
                        b_target[i] = static_cast<real>(setPot[i] - (pme->psr_coul[i] + plr_coul[i]) * potUnit);
                    }

                    // bisheng 2026 - CG tolerance: default 1e-6, tunable for
                    // accuracy studies via GMX_CPM_CG_TOL
                    static const int cgMaxIter = getenv("GMX_CPM_CG_MAXITER")
                                                        ? atoi(getenv("GMX_CPM_CG_MAXITER"))
                                                        : 1000;
                    static const real cgTol = getenv("GMX_CPM_CG_TOL")
                                                      ? static_cast<real>(atof(getenv("GMX_CPM_CG_TOL")))
                                                      : 1e-6_real;
                    // Call matrix-free CG solver
                    static std::vector<real> q_solved;
                    if (static_cast<int>(q_solved.size()) < nEleAtoms)
                    {
                        q_solved.resize(nEleAtoms);
                    }

                    // Determine if 3DC correction is needed (from ewald_geometry in MDP)
                    gmx_bool b3dc = pme->bEwald3DC;

                    // bisheng 2025 - Use Extended Lagrangian if mass > 0, otherwise use CG
                    if (pme->ext_lag_mass > 0)
                    {
                        // Extended Lagrangian method: propagate charge dynamics
                        // Works for all supported configurations
                        gmx_pme_extended_lagrangian_step(
                            pme,
                            coordinates.subArray(0, nEleAtoms),
                            b_target.data(),
                            nEleAtoms,
                            box,
                            pme->ewaldcoeff_q,
                            pme->eta,
                            pme->rcoulomb,
                            b3dc,
                            q_solved.data(),
                            pme->nthread,
                            pme->oneEle_or_not,
                            pme->pot_or_charge,
                            pme->applied_total_charge,  // Q_set for charge constraint
                            static_cast<real>(pme->applied_charge2)   // Q2_set (electrode 2)
                        );
                    }
                    else
                    {
                        // Standard CG solver
                        int niter;
                        if (pme->oneEle_or_not == 1)
                        {
                            // Single electrode: always uses charge constraint
                            niter = gmx_pme_solve_electrode_charges_cg_single_electrode(
                                pme,
                                coordinates.subArray(0, nEleAtoms),
                                b_target.data(),
                                nEleAtoms,
                                box,
                                pme->ewaldcoeff_q,
                                pme->eta,
                                pme->rcoulomb,
                                b3dc,
                                q_solved.data(),
                                pme->applied_total_charge,  // Q_set
                                cgTol,  // tolerance
                                cgMaxIter,   // max iterations
                                pme->nthread,
                                false   // bColdStart
                            );
                        }
                        else if (pme->pot_or_charge == 0)
                        {
                            // Two electrodes, potential constraint: solve for charges that match target potential
                            niter = gmx_pme_solve_electrode_charges_cg(
                                pme,
                                coordinates.subArray(0, nEleAtoms),
                                b_target.data(),
                                nEleAtoms,
                                box,
                                pme->ewaldcoeff_q,
                                pme->eta,
                                pme->rcoulomb,
                                b3dc,
                                q_solved.data(),
                                cgTol,  // tolerance
                                cgMaxIter,   // max iterations
                                pme->nthread
                            );
                        }
                        else
                        {
                            // Two electrodes, charge constraint: solve for charges with fixed total charge per electrode
                            niter = gmx_pme_solve_electrode_charges_cg_charge_constrained(
                                pme,
                                coordinates.subArray(0, nEleAtoms),
                                b_target.data(),
                                nEleAtoms,
                                box,
                                pme->ewaldcoeff_q,
                                pme->eta,
                                pme->rcoulomb,
                                b3dc,
                                q_solved.data(),
                                pme->applied_total_charge,  // Q_set
                                static_cast<real>(pme->applied_charge2),  // Q2_set (electrode 2; -Q1 unless Q2 given)
                                cgTol,  // tolerance
                                cgMaxIter,   // max iterations
                                pme->nthread,
                                false   // bColdStart
                            );
                        }

                        if (niter < 0)
                        {
                            // bisheng 2026 - use the best iterate instead of aborting.
                            // The residual floor is set by the single-precision A*q
                            // evaluation (the PME cycle runs in float), so tolerances
                            // much below ~1e-6 CANNOT converge no matter how many
                            // iterations run. Retain the iterate with the smallest residual.
                            static bool warnedOnce = false;
                            if (!warnedOnce)
                            {
                                printf("\nWARNING: Matrix-free CG reached %d iterations without "
                                       "meeting tol=%g; using the best iterate. Tolerances below "
                                       "~1e-6 are unreachable in single precision "
                                       "(this message prints once).\n\n",
                                       -niter, static_cast<double>(cgTol));
                                warnedOnce = true;
                            }
                        }
                        else
                        {
                            static bool bPrintedOnce = false;
                            if (!bPrintedOnce)
                            {
                                printf("Matrix-free CG converged in %d iterations\n", niter);
                                bPrintedOnce = true;
                            }
                        }
                    }

                    // Debug: print b_target stats and compare
                    static bool bDebugPrinted = false;
                    if (bDebugPrinted)
                    {
                        double b_sum = 0, b_min = b_target[0], b_max = b_target[0];
                        for (int i = 0; i < nEleAtoms; i++)
                        {
                            b_sum += b_target[i];
                            if (b_target[i] < b_min) b_min = b_target[i];
                            if (b_target[i] > b_max) b_max = b_target[i];
                        }
                        printf("CG b_target: sum=%g, min=%g, max=%g, mean=%g\n",
                               b_sum, b_min, b_max, b_sum/nEleAtoms);
                        printf("CG b_target[0]=%g, b_target[1144]=%g (first atoms of each electrode)\n",
                               b_target[0], b_target[nEleAtoms/2]);

                        double q_sum = 0, q_min = q_solved[0], q_max = q_solved[0];
                        for (int i = 0; i < nEleAtoms; i++)
                        {
                            q_sum += q_solved[i];
                            if (q_solved[i] < q_min) q_min = q_solved[i];
                            if (q_solved[i] > q_max) q_max = q_solved[i];
                        }
                        printf("CG q_solved: sum=%g, min=%g, max=%g, mean=%g\n",
                               q_sum, q_min, q_max, q_sum/nEleAtoms);

                        bDebugPrinted = false;
                    }

                    // Copy to eleCharge with evscale conversion
                    // CG solver now uses LAMMPS-style projected CG - q is already neutral
                    static const double evscale = ELECTRONVOLT / ONE_4PI_EPS0;  // V*nm -> e
                    #pragma omp parallel for num_threads(pme->nthread)
                    for (int i = 0; i < nEleAtoms; i++)
                    {
                        eleCharge[i] = static_cast<double>(q_solved[i] * evscale);
                    }

                    // Debug: verify neutralization (should be ~0 from projected CG)
                    static bool bDebugNeutralize = false;
                    if (bDebugNeutralize)
                    {
                        double sum_q = 0.0;
                        for (int i = 0; i < nEleAtoms; i++)
                        {
                            sum_q += eleCharge[i];
                        }
                        printf("Projected CG: sum(eleCharge) = %g (should be ~0)\n", sum_q);
                    }
                }
            }
            else if(pme->oneEle_or_not == 1)
            {
                // Compute the electrode charge using the single constraint method
                compute_electrode_charge_single_constrain(eleCharge, pme->psr_coul, plr_coul, pme->inv_MatrixA, nEleAtoms, pme->nthread, pme->applied_total_charge,
                                                          1.0 / ELECTRONVOLT, ELECTRONVOLT / ONE_4PI_EPS0, &pme->cpm_cell_voltage);
            }

            // Copy the computed charges to the output arrays
            #pragma omp parallel for num_threads(pme->nthread)
            for (int i = 0; i < nEleAtoms; ++i)
            {
                chargeA[i] = static_cast<real>(eleCharge[i]);

            }

            // Per-atom charge output is handled in md.cpp at nstenergy frequency
            // (enabled via GMX_CPM_PER_ATOM_CHARGES env var)

            // For debugging, compare eleCharge_CG and eleCharge
            static bool bDebugFinalCharge = false;
            if (bDebugFinalCharge)
            {
                printf("pme->inv_or_cg == %d \n", pme->inv_or_cg);
                for (int i = 0; i < nEleAtoms; ++i)
                {
                    printf("i=%d, eleCharge=%f\n", i, eleCharge[i]);
                }
            }

            // early return if we only want to calculate the electrode charge
            return 0;
    return 0;
}

int gmx_pme_do(struct gmx_pme_t*              pme,
               gmx::ArrayRef<const gmx::RVec> coordinates,
               gmx::ArrayRef<gmx::RVec>       forces,
               real                           chargeA[],
               real                           chargeB[],
               real                           c6A[],
               real                           c6B[],
               real                           sigmaA[],
               real                           sigmaB[],
               const matrix                   box,
               const t_commrec*               cr,
               int                            maxshift_x,
               int                            maxshift_y,
               t_nrnb*                        nrnb,
               gmx_wallcycle*                 wcycle,
               matrix                         vir_q,
               matrix                         vir_lj,
               real*                          energy_q,
               real*                          energy_lj,
               real                           lambda_q,
               real                           lambda_lj,
               real*                          dvdlambda_q,
               real*                          dvdlambda_lj,
               int                            flags)
{
    GMX_ASSERT(pme->runMode == PmeRunMode::CPU,
               "gmx_pme_do should not be called on the GPU PME run.");

    int                  d, npme, grid_index, max_grid_index;
    PmeAtomComm&         atc         = pme->atc[0];
    pmegrids_t*          pmegrid     = nullptr;
    real*                grid        = nullptr;
    real*                coefficient = nullptr;
    PmeOutput            output[2]; // The second is used for the B state with FEP
    real                 scale, lambda;
    gmx_bool             bClearF;
    gmx_parallel_3dfft_t pfft_setup;
    real*                fftgrid;
    t_complex*           cfftgrid;
    int                  thread;
    gmx_bool             bFirst, bDoSplines;
    int                  fep_state;
    int                  fep_states_lj = pme->bFEP_lj ? 2 : 1;
    const gmx_bool       bCalcEnerVir  = (flags & GMX_PME_CALC_ENER_VIR) != 0;
    const gmx_bool       bBackFFT      = (flags & (GMX_PME_CALC_F | GMX_PME_CALC_POT)) != 0;
    const gmx_bool       bCalcF        = (flags & GMX_PME_CALC_F) != 0;
    // bisheng 2025
    const gmx_bool       bCalcQ     = (flags & GMX_PME_CALC_Q) != 0;

    /* We could be passing lambda!=1 while no q or LJ is actually perturbed */
    if (!pme->bFEP_q)
    {
        lambda_q = 1;
    }
    if (!pme->bFEP_lj)
    {
        lambda_lj = 1;
    }

    assert(pme->nnodes > 0);
    assert(pme->nnodes == 1 || pme->ndecompdim > 0);

    if (pme->nnodes > 1)
    {
        atc.pd.resize(coordinates.ssize());
        for (int d = pme->ndecompdim - 1; d >= 0; d--)
        {
            PmeAtomComm& atc = pme->atc[d];
            atc.maxshift     = (atc.dimind == 0 ? maxshift_x : maxshift_y);
        }
    }
    else
    {
        GMX_ASSERT(coordinates.ssize() == atc.numAtoms(), "We expect atc.numAtoms() coordinates");
        GMX_ASSERT(forces.ssize() >= atc.numAtoms(),
                   "We need a force buffer with at least atc.numAtoms() elements");

        atc.x = coordinates;
        atc.f = forces;
    }

    matrix scaledBox;
    pme->boxScaler->scaleBox(box, scaledBox);

    gmx::invertBoxMatrix(scaledBox, pme->recipbox);
    bFirst = TRUE;

    /* For simplicity, we construct the splines for all particles if
     * more than one PME calculations is needed. Some optimization
     * could be done by keeping track of which atoms have splines
     * constructed, and construct new splines on each pass for atoms
     * that don't yet have them.
     */

    bDoSplines = pme->bFEP || (pme->doCoulomb && pme->doLJ);

    /* We need a maximum of four separate PME calculations:
     * grid_index=0: Coulomb PME with charges from state A
     * grid_index=1: Coulomb PME with charges from state B
     * grid_index=2: LJ PME with C6 from state A
     * grid_index=3: LJ PME with C6 from state B
     * For Lorentz-Berthelot combination rules, a separate loop is used to
     * calculate all the terms
     */

    /* If we are doing LJ-PME with LB, we only do Q here */
    max_grid_index = (pme->ljpme_combination_rule == eljpmeLB) ? DO_Q : DO_Q_AND_LJ;

    for (grid_index = 0; grid_index < max_grid_index; ++grid_index)
    {
        /* Check if we should do calculations at this grid_index
         * If grid_index is odd we should be doing FEP
         * If grid_index < 2 we should be doing electrostatic PME
         * If grid_index >= 2 we should be doing LJ-PME
         */
        if ((grid_index < DO_Q && (!pme->doCoulomb || (grid_index == 1 && !pme->bFEP_q)))
            || (grid_index >= DO_Q && (!pme->doLJ || (grid_index == 3 && !pme->bFEP_lj))))
        {
            continue;
        }
        /* Unpack structure */
        pmegrid    = &pme->pmegrid[grid_index];
        fftgrid    = pme->fftgrid[grid_index];
        cfftgrid   = pme->cfftgrid[grid_index];
        pfft_setup = pme->pfft_setup[grid_index];
        switch (grid_index)
        {
            case 0: coefficient = chargeA; break;
            case 1: coefficient = chargeB; break;
            case 2: coefficient = c6A; break;
            case 3: coefficient = c6B; break;
        }

        grid = pmegrid->grid.grid;

        if (debug)
        {
            fprintf(debug, "PME: number of ranks = %d, rank = %d\n", cr->nnodes, cr->nodeid);
            fprintf(debug, "Grid = %p\n", static_cast<void*>(grid));
            if (grid == nullptr)
            {
                gmx_fatal(FARGS, "No grid!");
            }
        }

        if (pme->nnodes == 1)
        {
            atc.coefficient = gmx::arrayRefFromArray(coefficient, coordinates.size());
        }
        else
        {
            wallcycle_start(wcycle, ewcPME_REDISTXF);
            do_redist_pos_coeffs(pme, cr, bFirst, coordinates, coefficient);

            wallcycle_stop(wcycle, ewcPME_REDISTXF);
        }

        if (debug)
        {
            fprintf(debug, "Rank= %6d, pme local particles=%6d\n", cr->nodeid, atc.numAtoms());
        }

        if (flags & GMX_PME_SPREAD)
        {
            wallcycle_start(wcycle, ewcPME_SPREAD);

            /* Spread the coefficients on a grid */
            spread_on_grid(pme, &atc, pmegrid, bFirst, TRUE, fftgrid, bDoSplines, grid_index);

            if (bFirst)
            {
                inc_nrnb(nrnb, eNR_WEIGHTS, DIM * atc.numAtoms());
            }
            inc_nrnb(nrnb, eNR_SPREADBSP,
                     pme->pme_order * pme->pme_order * pme->pme_order * atc.numAtoms());

            if (!pme->bUseThreads)
            {
                wrap_periodic_pmegrid(pme, grid);

                /* sum contributions to local grid from other nodes */
                if (pme->nnodes > 1)
                {
                    gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_FORWARD);
                }

                copy_pmegrid_to_fftgrid(pme, grid, fftgrid, grid_index);
            }

            wallcycle_stop(wcycle, ewcPME_SPREAD);

            /* TODO If the OpenMP and single-threaded implementations
               converge, then spread_on_grid() and
               copy_pmegrid_to_fftgrid() will perhaps live in the same
               source file.
             */
        }

        /* Here we start a large thread parallel region */
#pragma omp parallel num_threads(pme->nthread) private(thread)
        {
            try
            {
                thread = gmx_omp_get_thread_num();
                if (flags & GMX_PME_SOLVE)
                {
                    int loop_count;

                    /* do 3d-fft */
                    if (thread == 0)
                    {
                        wallcycle_start(wcycle, ewcPME_FFT);
                    }
                    gmx_parallel_3dfft_execute(pfft_setup, GMX_FFT_REAL_TO_COMPLEX, thread, wcycle);
                    if (thread == 0)
                    {
                        wallcycle_stop(wcycle, ewcPME_FFT);
                    }

                    /* solve in k-space for our local cells */
                    if (thread == 0)
                    {
                        wallcycle_start(wcycle, (grid_index < DO_Q ? ewcPME_SOLVE : ewcLJPME));
                    }
                    if (grid_index < DO_Q)
                    {
                        loop_count = solve_pme_yzx(
                                pme, cfftgrid, scaledBox[XX][XX] * scaledBox[YY][YY] * scaledBox[ZZ][ZZ],
                                bCalcEnerVir, pme->nthread, thread);
                    }
                    else
                    {
                        loop_count = solve_pme_lj_yzx(
                                pme, &cfftgrid, FALSE,
                                scaledBox[XX][XX] * scaledBox[YY][YY] * scaledBox[ZZ][ZZ],
                                bCalcEnerVir, pme->nthread, thread);
                    }

                    if (thread == 0)
                    {
                        wallcycle_stop(wcycle, (grid_index < DO_Q ? ewcPME_SOLVE : ewcLJPME));
                        inc_nrnb(nrnb, eNR_SOLVEPME, loop_count);
                    }
                }

                if (bBackFFT)
                {
                    /* do 3d-invfft */
                    if (thread == 0)
                    {
                        wallcycle_start(wcycle, ewcPME_FFT);
                    }
                    gmx_parallel_3dfft_execute(pfft_setup, GMX_FFT_COMPLEX_TO_REAL, thread, wcycle);
                    if (thread == 0)
                    {
                        wallcycle_stop(wcycle, ewcPME_FFT);


                        if (pme->nodeid == 0)
                        {
                            real ntot = pme->nkx * pme->nky * pme->nkz;
                            npme      = static_cast<int>(ntot * std::log(ntot) / std::log(2.0));
                            inc_nrnb(nrnb, eNR_FFT, 2 * npme);
                        }

                        /* Note: this wallcycle region is closed below
                           outside an OpenMP region, so take care if
                           refactoring code here. */
                        wallcycle_start(wcycle, ewcPME_GATHER);
                    }

                    copy_fftgrid_to_pmegrid(pme, fftgrid, grid, grid_index, pme->nthread, thread);
                }
            }
            GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
        }
        /* End of thread parallel section.
         * With MPI we have to synchronize here before gmx_sum_qgrid_dd.
         */

        if (bBackFFT)
        {
            /* distribute local grid to all nodes */
            if (pme->nnodes > 1)
            {
                gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_BACKWARD);
            }

            unwrap_periodic_pmegrid(pme, grid);
        }

        // bisheng 2025
        if(bCalcQ)
        {
            // bisheng 2026 - close the gather counter before the charge-pass return.
            wallcycle_stop(wcycle, ewcPME_GATHER);
            // bisheng 2026 - solver extracted so the GPU PME path can reuse it
            gmx_pme_cpm_solve_charges(pme, coordinates, chargeA, box, nullptr);
            return 0;
        }

        if (bCalcF)
        {
            /* interpolate forces for our local atoms */

            // bisheng 2025 - Add electrode-electrode SR force correction for flexible electrodes
            // This corrects PME's point-charge assumption for Gaussian electrode charges
            // Only needed for matrix-free CG (inv_or_cg == 2) which supports flexible electrodes
            // bisheng 2026 - GMX_CPM_NO_SR_FORCE switches it off, as on 2026, so its size can be
            // measured (it acts on electrode atoms only, so a frozen electrode never shows it).
            static const bool cpmNoSrForce = (getenv("GMX_CPM_NO_SR_FORCE") != nullptr);
            if (pme->inv_or_cg == 2 && pme->nEleAtoms > 0 && !cpmNoSrForce)
            {
                const int nEleAtoms = pme->nEleAtoms;
                auto x_ele = coordinates.subArray(0, nEleAtoms);
                auto q_ele = gmx::constArrayRefFromArray(chargeA, nEleAtoms);
                auto f_ele = forces.subArray(0, nEleAtoms);
                real coulomb_factor = ONE_4PI_EPS0 / pme->epsilon_r;

                gmx_pme_calc_Aq_sr_force(pme, x_ele, q_ele, nEleAtoms, box,
                                         pme->rcoulomb, coulomb_factor, f_ele,
                                         pme->nthread, pme->bEwald3DC);
            }

            /* If we are running without parallelization,
             * atc->f is the actual force array, not a buffer,
             * therefore we should not clear it.
             */
            lambda  = grid_index < DO_Q ? lambda_q : lambda_lj;
            bClearF = (bFirst && PAR(cr));
#pragma omp parallel for num_threads(pme->nthread) schedule(static)
            for (thread = 0; thread < pme->nthread; thread++)
            {
                try
                {
                    gather_f_bsplines(pme, grid, bClearF, &atc, &atc.spline[thread],
                                      pme->bFEP ? (grid_index % 2 == 0 ? 1.0 - lambda : lambda) : 1.0);
                }
                GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
            }


            inc_nrnb(nrnb, eNR_GATHERFBSP,
                     pme->pme_order * pme->pme_order * pme->pme_order * atc.numAtoms());
            /* Note: this wallcycle region is opened above inside an OpenMP
               region, so take care if refactoring code here. */
            wallcycle_stop(wcycle, ewcPME_GATHER);
        }

        if (bCalcEnerVir)
        {
            /* This should only be called on the master thread
             * and after the threads have synchronized.
             */
            if (grid_index < 2)
            {
                get_pme_ener_vir_q(pme->solve_work, pme->nthread, &output[grid_index % 2]);
            }
            else
            {
                get_pme_ener_vir_lj(pme->solve_work, pme->nthread, &output[grid_index % 2]);
            }
        }
        bFirst = FALSE;
    } /* of grid_index-loop */

    /* For Lorentz-Berthelot combination rules in LJ-PME, we need to calculate
     * seven terms. */

    if (pme->doLJ && pme->ljpme_combination_rule == eljpmeLB)
    {
        /* Loop over A- and B-state if we are doing FEP */
        for (fep_state = 0; fep_state < fep_states_lj; ++fep_state)
        {
            real *local_c6 = nullptr, *local_sigma = nullptr, *RedistC6 = nullptr, *RedistSigma = nullptr;
            gmx::ArrayRef<real> coefficientBuffer;
            if (pme->nnodes == 1)
            {
                pme->lb_buf1.resize(atc.numAtoms());
                coefficientBuffer = pme->lb_buf1;
                switch (fep_state)
                {
                    case 0:
                        local_c6    = c6A;
                        local_sigma = sigmaA;
                        break;
                    case 1:
                        local_c6    = c6B;
                        local_sigma = sigmaB;
                        break;
                    default: gmx_incons("Trying to access wrong FEP-state in LJ-PME routine");
                }
            }
            else
            {
                coefficientBuffer = atc.coefficientBuffer;
                switch (fep_state)
                {
                    case 0:
                        RedistC6    = c6A;
                        RedistSigma = sigmaA;
                        break;
                    case 1:
                        RedistC6    = c6B;
                        RedistSigma = sigmaB;
                        break;
                    default: gmx_incons("Trying to access wrong FEP-state in LJ-PME routine");
                }
                wallcycle_start(wcycle, ewcPME_REDISTXF);

                do_redist_pos_coeffs(pme, cr, bFirst, coordinates, RedistC6);
                pme->lb_buf1.resize(atc.numAtoms());
                pme->lb_buf2.resize(atc.numAtoms());
                local_c6 = pme->lb_buf1.data();
                for (int i = 0; i < atc.numAtoms(); ++i)
                {
                    local_c6[i] = atc.coefficient[i];
                }

                do_redist_pos_coeffs(pme, cr, FALSE, coordinates, RedistSigma);
                local_sigma = pme->lb_buf2.data();
                for (int i = 0; i < atc.numAtoms(); ++i)
                {
                    local_sigma[i] = atc.coefficient[i];
                }

                wallcycle_stop(wcycle, ewcPME_REDISTXF);
            }
            atc.coefficient = coefficientBuffer;
            calc_initial_lb_coeffs(coefficientBuffer, local_c6, local_sigma);

            /*Seven terms in LJ-PME with LB, grid_index < 2 reserved for electrostatics*/
            for (grid_index = 2; grid_index < 9; ++grid_index)
            {
                /* Unpack structure */
                pmegrid    = &pme->pmegrid[grid_index];
                fftgrid    = pme->fftgrid[grid_index];
                pfft_setup = pme->pfft_setup[grid_index];
                calc_next_lb_coeffs(coefficientBuffer, local_sigma);
                grid = pmegrid->grid.grid;

                if (flags & GMX_PME_SPREAD)
                {
                    wallcycle_start(wcycle, ewcPME_SPREAD);
                    /* Spread the c6 on a grid */
                    spread_on_grid(pme, &atc, pmegrid, bFirst, TRUE, fftgrid, bDoSplines, grid_index);

                    if (bFirst)
                    {
                        inc_nrnb(nrnb, eNR_WEIGHTS, DIM * atc.numAtoms());
                    }

                    inc_nrnb(nrnb, eNR_SPREADBSP,
                             pme->pme_order * pme->pme_order * pme->pme_order * atc.numAtoms());
                    if (pme->nthread == 1)
                    {
                        wrap_periodic_pmegrid(pme, grid);
                        /* sum contributions to local grid from other nodes */
                        if (pme->nnodes > 1)
                        {
                            gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_FORWARD);
                        }
                        copy_pmegrid_to_fftgrid(pme, grid, fftgrid, grid_index);
                    }
                    wallcycle_stop(wcycle, ewcPME_SPREAD);
                }
                /*Here we start a large thread parallel region*/
#pragma omp parallel num_threads(pme->nthread) private(thread)
                {
                    try
                    {
                        thread = gmx_omp_get_thread_num();
                        if (flags & GMX_PME_SOLVE)
                        {
                            /* do 3d-fft */
                            if (thread == 0)
                            {
                                wallcycle_start(wcycle, ewcPME_FFT);
                            }

                            gmx_parallel_3dfft_execute(pfft_setup, GMX_FFT_REAL_TO_COMPLEX, thread, wcycle);
                            if (thread == 0)
                            {
                                wallcycle_stop(wcycle, ewcPME_FFT);
                            }
                        }
                    }
                    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
                }
                bFirst = FALSE;
            }
            if (flags & GMX_PME_SOLVE)
            {
                /* solve in k-space for our local cells */
#pragma omp parallel num_threads(pme->nthread) private(thread)
                {
                    try
                    {
                        int loop_count;
                        thread = gmx_omp_get_thread_num();
                        if (thread == 0)
                        {
                            wallcycle_start(wcycle, ewcLJPME);
                        }

                        loop_count = solve_pme_lj_yzx(
                                pme, &pme->cfftgrid[2], TRUE,
                                scaledBox[XX][XX] * scaledBox[YY][YY] * scaledBox[ZZ][ZZ],
                                bCalcEnerVir, pme->nthread, thread);
                        if (thread == 0)
                        {
                            wallcycle_stop(wcycle, ewcLJPME);
                            inc_nrnb(nrnb, eNR_SOLVEPME, loop_count);
                        }
                    }
                    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
                }
            }

            if (bCalcEnerVir)
            {
                /* This should only be called on the master thread and
                 * after the threads have synchronized.
                 */
                get_pme_ener_vir_lj(pme->solve_work, pme->nthread, &output[fep_state]);
            }

            if (bBackFFT)
            {
                bFirst = !pme->doCoulomb;
                calc_initial_lb_coeffs(coefficientBuffer, local_c6, local_sigma);
                for (grid_index = 8; grid_index >= 2; --grid_index)
                {
                    /* Unpack structure */
                    pmegrid    = &pme->pmegrid[grid_index];
                    fftgrid    = pme->fftgrid[grid_index];
                    pfft_setup = pme->pfft_setup[grid_index];
                    grid       = pmegrid->grid.grid;
                    calc_next_lb_coeffs(coefficientBuffer, local_sigma);
#pragma omp parallel num_threads(pme->nthread) private(thread)
                    {
                        try
                        {
                            thread = gmx_omp_get_thread_num();
                            /* do 3d-invfft */
                            if (thread == 0)
                            {
                                wallcycle_start(wcycle, ewcPME_FFT);
                            }

                            gmx_parallel_3dfft_execute(pfft_setup, GMX_FFT_COMPLEX_TO_REAL, thread, wcycle);
                            if (thread == 0)
                            {
                                wallcycle_stop(wcycle, ewcPME_FFT);


                                if (pme->nodeid == 0)
                                {
                                    real ntot = pme->nkx * pme->nky * pme->nkz;
                                    npme = static_cast<int>(ntot * std::log(ntot) / std::log(2.0));
                                    inc_nrnb(nrnb, eNR_FFT, 2 * npme);
                                }
                                wallcycle_start(wcycle, ewcPME_GATHER);
                            }

                            copy_fftgrid_to_pmegrid(pme, fftgrid, grid, grid_index, pme->nthread, thread);
                        }
                        GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
                    } /*#pragma omp parallel*/

                    /* distribute local grid to all nodes */
                    if (pme->nnodes > 1)
                    {
                        gmx_sum_qgrid_dd(pme, grid, GMX_SUM_GRID_BACKWARD);
                    }

                    unwrap_periodic_pmegrid(pme, grid);

                    if (bCalcF)
                    {
                        /* interpolate forces for our local atoms */
                        bClearF = (bFirst && PAR(cr));
                        scale   = pme->bFEP ? (fep_state < 1 ? 1.0 - lambda_lj : lambda_lj) : 1.0;
                        scale *= lb_scale_factor[grid_index - 2];

#pragma omp parallel for num_threads(pme->nthread) schedule(static)
                        for (thread = 0; thread < pme->nthread; thread++)
                        {
                            try
                            {
                                gather_f_bsplines(pme, grid, bClearF, &pme->atc[0],
                                                  &pme->atc[0].spline[thread], scale);
                            }
                            GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
                        }


                        inc_nrnb(nrnb, eNR_GATHERFBSP,
                                 pme->pme_order * pme->pme_order * pme->pme_order * pme->atc[0].numAtoms());
                    }
                    wallcycle_stop(wcycle, ewcPME_GATHER);

                    bFirst = FALSE;
                } /* for (grid_index = 8; grid_index >= 2; --grid_index) */
            }     /* if (bCalcF) */
        }         /* for (fep_state = 0; fep_state < fep_states_lj; ++fep_state) */
    }             /* if ((flags & GMX_PME_DO_LJ) && pme->ljpme_combination_rule == eljpmeLB) */

    if (bCalcF && pme->nnodes > 1)
    {
        wallcycle_start(wcycle, ewcPME_REDISTXF);
        for (d = 0; d < pme->ndecompdim; d++)
        {
            gmx::ArrayRef<gmx::RVec> forcesRef;
            if (d == pme->ndecompdim - 1)
            {
                const size_t numAtoms = coordinates.size();
                GMX_ASSERT(forces.size() >= numAtoms, "Need at least numAtoms forces");
                forcesRef = forces.subArray(0, numAtoms);
            }
            else
            {
                forcesRef = pme->atc[d + 1].f;
            }
            if (DOMAINDECOMP(cr))
            {
                dd_pmeredist_f(pme, &pme->atc[d], forcesRef, d == pme->ndecompdim - 1 && pme->bPPnode);
            }
        }

        wallcycle_stop(wcycle, ewcPME_REDISTXF);
    }

    if (bCalcEnerVir)
    {
        if (pme->doCoulomb)
        {
            if (!pme->bFEP_q)
            {
                *energy_q = output[0].coulombEnergy_;
                m_add(vir_q, output[0].coulombVirial_, vir_q);
            }
            else
            {
                *energy_q = (1.0 - lambda_q) * output[0].coulombEnergy_ + lambda_q * output[1].coulombEnergy_;
                *dvdlambda_q += output[1].coulombEnergy_ - output[0].coulombEnergy_;
                for (int i = 0; i < DIM; i++)
                {
                    for (int j = 0; j < DIM; j++)
                    {
                        vir_q[i][j] += (1.0 - lambda_q) * output[0].coulombVirial_[i][j]
                                       + lambda_q * output[1].coulombVirial_[i][j];
                    }
                }
            }
            if (debug)
            {
                fprintf(debug, "Electrostatic PME mesh energy: %g\n", *energy_q);
            }
        }
        else
        {
            *energy_q = 0;
        }

        if (pme->doLJ)
        {
            if (!pme->bFEP_lj)
            {
                *energy_lj = output[0].lennardJonesEnergy_;
                m_add(vir_lj, output[0].lennardJonesVirial_, vir_lj);
            }
            else
            {
                *energy_lj = (1.0 - lambda_lj) * output[0].lennardJonesEnergy_
                             + lambda_lj * output[1].lennardJonesEnergy_;
                *dvdlambda_lj += output[1].lennardJonesEnergy_ - output[0].lennardJonesEnergy_;
                for (int i = 0; i < DIM; i++)
                {
                    for (int j = 0; j < DIM; j++)
                    {
                        vir_lj[i][j] += (1.0 - lambda_lj) * output[0].lennardJonesVirial_[i][j]
                                        + lambda_lj * output[1].lennardJonesVirial_[i][j];
                    }
                }
            }
            if (debug)
            {
                fprintf(debug, "Lennard-Jones PME mesh energy: %g\n", *energy_lj);
            }
        }
        else
        {
            *energy_lj = 0;
        }
    }
    return 0;
}

void gmx_pme_destroy(gmx_pme_t* pme)
{
    if (!pme)
    {
        return;
    }

    delete pme->boxScaler;

    sfree(pme->nnx);
    sfree(pme->nny);
    sfree(pme->nnz);
    sfree(pme->fshx);
    sfree(pme->fshy);
    sfree(pme->fshz);

    for (int i = 0; i < pme->ngrids; ++i)
    {
        pmegrids_destroy(&pme->pmegrid[i]);
    }
    if (pme->pfft_setup)
    {
        for (int i = 0; i < pme->ngrids; ++i)
        {
            gmx_parallel_3dfft_destroy(pme->pfft_setup[i]);
        }
    }
    sfree(pme->fftgrid);
    sfree(pme->cfftgrid);
    sfree(pme->pfft_setup);

    for (int i = 0; i < DIM; i++)
    {
        sfree(pme->bsp_mod[i]);
    }

    sfree(pme->bufv);
    sfree(pme->bufr);

    if (pme->solve_work)
    {
        pme_free_all_work(&pme->solve_work, pme->nthread);
    }

    sfree(pme->sum_qgrid_tmp);
    sfree(pme->sum_qgrid_dd_tmp);

    destroy_pme_spline_work(pme->spline_work);

    if (pme_gpu_active(pme) && pme->gpu)
    {
        pme_gpu_destroy(pme->gpu);
    }

    delete pme;
}

void gmx_pme_reinit_atoms(gmx_pme_t* pme, const int numAtoms, const real* charges)
{
    if (pme_gpu_active(pme))
    {
        pme_gpu_reinit_atoms(pme->gpu, numAtoms, charges);
    }
    else
    {
        pme->atc[0].setNumAtoms(numAtoms);
        // TODO: set the charges here as well
    }
}

// bisheng 2026 - realised cell voltage from the CHARGE-constrained solve (volts).
// Zero unless userint4 = 1 with the matrix-inversion solver.
double gmx_pme_get_cell_voltage(const gmx_pme_t* pme)
{
    return pme ? pme->cpm_cell_voltage : 0.0;
}
