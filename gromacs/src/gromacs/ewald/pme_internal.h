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
 * \brief This file contains function declarations necessary for
 * computing energies and forces for the PME long-ranged part (Coulomb
 * and LJ).
 *
 * \author Berk Hess <hess@kth.se>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 * \ingroup module_ewald
 */

/* TODO This file is a temporary holding area for stuff local to the
 * PME code, before it acquires some more normal ewald/file.c and
 * ewald/file.h structure.  In future clean up, get rid of this file,
 * to build more normal. */

#ifndef GMX_EWALD_PME_INTERNAL_H
#define GMX_EWALD_PME_INTERNAL_H

#include "config.h"

#include "gromacs/math/gmxcomplex.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/defaultinitializationallocator.h"
#include "gromacs/utility/gmxmpi.h"
#include "gromacs/utility/smalloc.h"

#include <memory>
#include <vector>

#include "pme_gpu_types_host.h"
#include "cpm_potential_control.h"

//! A repeat of typedef from parallel_3dfft.h
typedef struct gmx_parallel_3dfft* gmx_parallel_3dfft_t;

struct t_commrec;
struct t_inputrec;
struct PmeGpu;

//@{
//! Grid indices for A state for charge and Lennard-Jones C6
#define PME_GRID_QA 0
#define PME_GRID_C6A 2
//@}

//@{
/*! \brief Flags that indicate the number of PME grids in use */
#define DO_Q 2           /* Electrostatic grids have index q<2 */
#define DO_Q_AND_LJ 4    /* non-LB LJ grids have index 2 <= q < 4 */
#define DO_Q_AND_LJ_LB 9 /* With LB rules we need a total of 2+7 grids */
//@}

/*! \brief Pascal triangle coefficients scaled with (1/2)^6 for LJ-PME with LB-rules */
static const real lb_scale_factor[] = { 1.0 / 64,  6.0 / 64, 15.0 / 64, 20.0 / 64,
                                        15.0 / 64, 6.0 / 64, 1.0 / 64 };

/*! \brief Pascal triangle coefficients used in solve_pme_lj_yzx, only need to do 4 calculations due to symmetry */
static const real lb_scale_factor_symm[] = { 2.0 / 64, 12.0 / 64, 30.0 / 64, 20.0 / 64 };

/*! \brief We only define a maximum to be able to use local arrays without allocation.
 * An order larger than 12 should never be needed, even for test cases.
 * If needed it can be changed here.
 */
#define PME_ORDER_MAX 12

/*! \brief As gmx_pme_init, but takes most settings, except the grid/Ewald coefficients, from
 * pme_src. This is only called when the PME cut-off/grid size changes.
 */
void gmx_pme_reinit(struct gmx_pme_t** pmedata,
                    const t_commrec*   cr,
                    struct gmx_pme_t*  pme_src,
                    const t_inputrec*  ir,
                    const ivec         grid_size,
                    real               ewaldcoeff_q,
                    real               ewaldcoeff_lj);


/* Temporary suppression until these structs become opaque and don't live in
 * a header that is included by other headers. Also, until then I have no
 * idea what some of the names mean. */

//! @cond Doxygen_Suppress

/*! \brief Data structure for grid communication */
struct pme_grid_comm_t
{
    int send_id; //!< Source rank id
    int send_index0;
    int send_nindex;
    int recv_id; //!< Destination rank id
    int recv_index0;
    int recv_nindex;
    int recv_size = 0; //!< Receive buffer width, used with OpenMP
};

/*! \brief Data structure for grid overlap communication in a single dimension */
struct pme_overlap_t
{
    MPI_Comm                     mpi_comm;  //!< MPI communcator
    int                          nnodes;    //!< Number of ranks
    int                          nodeid;    //!< Unique rank identifcator
    std::vector<int>             s2g0;      //!< The local interpolation grid start
    std::vector<int>             s2g1;      //!< The local interpolation grid end
    int                          send_size; //!< Send buffer width, used with OpenMP
    std::vector<pme_grid_comm_t> comm_data; //!< All the individual communication data for each rank
    std::vector<real>            sendbuf;   //!< Shared buffer for sending
    std::vector<real>            recvbuf;   //!< Shared buffer for receiving
};

template<typename T>
using AlignedVector = std::vector<T, gmx::AlignedAllocator<T>>;

template<typename T>
using FastVector = std::vector<T, gmx::DefaultInitializationAllocator<T>>;

/*! \brief Data structure for organizing particle allocation to threads */
struct AtomToThreadMap
{
    //! Cumulative counts of the number of particles per thread
    int* n = nullptr;
    //! Storage buffer for n
    std::vector<int> nBuffer;
    //! Particle indices ordered on thread index (n)
    FastVector<int> i;
};

/*! \brief Helper typedef for spline vectors */
typedef real* splinevec[DIM];

/*! \internal
 * \brief Coefficients for theta or dtheta
 */
class SplineCoefficients
{
public:
    //! Reallocate for use with up to nalloc coefficients
    void realloc(int nalloc);

    //! Pointers to the coefficient buffer for x, y, z
    splinevec coefficients = { nullptr };

private:
    //! Storage for x coefficients
    std::vector<real> bufferX_;
    //! Storage for y coefficients
    std::vector<real> bufferY_;
    //! Storage for z coefficients, aligned for SIMD load
    AlignedVector<real> bufferZ_;
};

/*! \brief Data structure for beta-spline interpolation */
struct splinedata_t
{
    int                n = 0;
    FastVector<int>    ind;
    SplineCoefficients theta;
    SplineCoefficients dtheta;
    int                nalloc = 0;
};

/*! \brief PME slab MPI communication setup */
struct SlabCommSetup
{
    //! The nodes to send x and q to with DD
    int node_dest;
    //! The nodes to receive x and q from with DD
    int node_src;
    //! Index for commnode into the buffers
    int buf_index;
    //! The number of atoms to receive
    int rcount;
};

/*! \internal
 * \brief Data structure for coordinating transfers between PME ranks along one dimension
 *
 * Also used for passing coordinates, coefficients and forces to and from PME routines.
 */
class PmeAtomComm
{
public:
    //! Constructor, \p PmeMpiCommunicator is the communicator for this dimension
    PmeAtomComm(MPI_Comm PmeMpiCommunicator, int numThreads, int pmeOrder, int dimIndex, bool doSpread);

    //! Set the atom count and when necessary resizes atom buffers
    void setNumAtoms(int numAtoms);

    //! Returns the atom count
    int numAtoms() const { return numAtoms_; }

    //! Returns the number of atoms to send to each rank
    gmx::ArrayRef<int> sendCount()
    {
        GMX_ASSERT(!count_thread.empty(), "Need at least one thread_count");
        return count_thread[0];
    }

    //! The index of the dimension, 0=x, 1=y
    int dimind = 0;
    //! The number of slabs and ranks this dimension is decomposed over
    int nslab = 1;
    //! Our MPI rank index
    int nodeid = 0;
    //! Communicator for this dimension
    MPI_Comm mpi_comm;

    //! Communication setup for each slab, only present with nslab > 1
    std::vector<SlabCommSetup> slabCommSetup;
    //! The maximum communication distance counted in MPI ranks
    int maxshift = 0;

    //! The target slab index for each particle
    FastVector<int> pd;
    //! Target particle counts for each slab, for each thread
    std::vector<std::vector<int>> count_thread;

private:
    //! The number of atoms
    int numAtoms_ = 0;

public:
    //! The coordinates
    gmx::ArrayRef<const gmx::RVec> x;
    //! The coefficient, charges or LJ C6
    gmx::ArrayRef<const real> coefficient;
    //! The forces
    gmx::ArrayRef<gmx::RVec> f;
    //! Coordinate buffer, used only with nslab > 1
    FastVector<gmx::RVec> xBuffer;
    //! Coefficient buffer, used only with nslab > 1
    FastVector<real> coefficientBuffer;
    //! Force buffer, used only with nslab > 1
    FastVector<gmx::RVec> fBuffer;
    //! Tells whether these coordinates are used for spreading
    bool bSpread;
    //! The PME order
    int pme_order;
    //! The grid index per atom
    FastVector<gmx::IVec> idx;
    //! Fractional atom coordinates relative to the lower cell boundary
    FastVector<gmx::RVec> fractx;

    //! The number of threads to use in PME
    int nthread;
    //! Thread index for each atom
    FastVector<int>              thread_idx;
    std::vector<AtomToThreadMap> threadMap;
    std::vector<splinedata_t>    spline;
};

/*! \brief Data structure for a single PME grid */
struct pmegrid_t
{
    ivec  ci;     /* The spatial location of this grid         */
    ivec  n;      /* The used size of *grid, including order-1 */
    ivec  offset; /* The grid offset from the full node grid   */
    int   order;  /* PME spreading order                       */
    ivec  s;      /* The allocated size of *grid, s >= n       */
    real* grid;   /* The grid local thread, size n             */
};

/*! \brief Data structures for PME grids */
struct pmegrids_t
{
    pmegrid_t  grid;         /* The full node grid (non thread-local)            */
    int        nthread;      /* The number of threads operating on this grid     */
    ivec       nc;           /* The local spatial decomposition over the threads */
    pmegrid_t* grid_th;      /* Array of grids for each thread                   */
    real*      grid_all;     /* Allocated array for the grids in *grid_th        */
    int*       g2t[DIM];     /* The grid to thread index                         */
    ivec       nthread_comm; /* The number of threads to communicate with        */
};

/*! \brief Data structure for spline-interpolation working buffers */
struct pme_spline_work;

/*! \brief Data structure for working buffers */
struct pme_solve_work_t;

/*! \brief Master PME data structure */
struct gmx_pme_t
{                   //NOLINT(clang-analyzer-optin.performance.Padding)
    int ndecompdim; /* The number of decomposition dimensions */
    int nodeid;     /* Our nodeid in mpi->mpi_comm */
    int nodeid_major;
    int nodeid_minor;
    int nnodes; /* The number of nodes doing PME */
    int nnodes_major;
    int nnodes_minor;

    MPI_Comm mpi_comm;
    MPI_Comm mpi_comm_d[2]; /* Indexed on dimension, 0=x, 1=y */
#if GMX_MPI
    MPI_Datatype rvec_mpi; /* the pme vector's MPI type */
#endif

    gmx_bool bUseThreads; /* Does any of the PME ranks have nthread>1 ?  */
    int      nthread;     /* The number of threads doing PME on our rank */

    gmx_bool bPPnode;   /* Node also does particle-particle forces */
    bool     doCoulomb; /* Apply PME to electrostatics */
    bool     doLJ;      /* Apply PME to Lennard-Jones r^-6 interactions */
    gmx_bool bFEP;      /* Compute Free energy contribution */
    gmx_bool bFEP_q;
    gmx_bool bFEP_lj;
    int      nkx, nky, nkz; /* Grid dimensions */
    gmx_bool bP3M;          /* Do P3M: optimize the influence function */
    int      pme_order;
    real     ewaldcoeff_q;  /* Ewald splitting coefficient for Coulomb */
    real     ewaldcoeff_lj; /* Ewald splitting coefficient for r^-6 */
    real     epsilon_r;

    /* bisheng 2025 - Parameters for matrix-free CG */
    real     eta;           /* Electrode Gaussian width (nm^-1) for CPM (global, used if userreal1 > 0) */
    real     rcoulomb;      /* Coulomb cutoff (nm) for CPM short-range */
    std::vector<real> eta_per_atom; /* Per-atom Gaussian width (size = nEleAtoms, used if userreal1 < 0) */
    bool     bPerAtomEta;   /* True if using per-atom eta from electrode_eta.dat */


    enum PmeRunMode runMode; /* Which codepath is the PME runner taking - CPU, GPU, mixed;
                              * TODO: this is the information that should be owned by the task
                              * scheduler, and ideally not be duplicated here.
                              */

    PmeGpu* gpu; /* A pointer to the GPU data.
                  * TODO: this should be unique or a shared pointer.
                  * Currently in practice there is a single gmx_pme_t instance while a code
                  * is partially set up for many of them. The PME tuning calls gmx_pme_reinit()
                  * which fully reinitializes the one and only PME structure anew while maybe
                  * keeping the old grid buffers if they were already large enough.
                  * This small choice should be made clear in the later refactoring -
                  * do we store many PME objects for different grid sizes,
                  * or a single PME object that handles different grid sizes gracefully.
                  */


    class EwaldBoxZScaler* boxScaler; /**< The scaling data Ewald uses with walls (set at pme_init constant for the entire run) */

    gmx_bool bEwald3DC; /**< True if ewald_geometry = 3dc (slab correction needed) */

    int ljpme_combination_rule; /* Type of combination rule in LJ-PME */

    int ngrids; /* number of grids we maintain for pmegrid, (c)fftgrid and pfft_setups*/

    pmegrids_t pmegrid[DO_Q_AND_LJ_LB]; /* Grids on which we do spreading/interpolation,
                                         * includes overlap Grid indices are ordered as
                                         * follows:
                                         * 0: Coloumb PME, state A
                                         * 1: Coloumb PME, state B
                                         * 2-8: LJ-PME
                                         * This can probably be done in a better way
                                         * but this simple hack works for now
                                         */

    /* The PME coefficient spreading grid sizes/strides, includes pme_order-1 */
    int pmegrid_nx, pmegrid_ny, pmegrid_nz;
    /* pmegrid_nz might be larger than strictly necessary to ensure
     * memory alignment, pmegrid_nz_base gives the real base size.
     */
    int pmegrid_nz_base;
    /* The local PME grid starting indices */
    int pmegrid_start_ix, pmegrid_start_iy, pmegrid_start_iz;

    /* Work data for spreading and gathering */
    pme_spline_work* spline_work;

    real** fftgrid; /* Grids for FFT. With 1D FFT decomposition this can be a pointer */
    /* inside the interpolation grid, but separate for 2D PME decomp. */
    int fftgrid_nx, fftgrid_ny, fftgrid_nz;

    t_complex** cfftgrid; /* Grids for complex FFT data */

    int cfftgrid_nx, cfftgrid_ny, cfftgrid_nz;

    gmx_parallel_3dfft_t* pfft_setup;

    int * nnx, *nny, *nnz;
    real *fshx, *fshy, *fshz;

    std::vector<PmeAtomComm> atc; /* Indexed on decomposition index */
    matrix                   recipbox;
    real                     boxVolume;
    splinevec                bsp_mod;
    /* Buffers to store data for local atoms for L-B combination rule
     * calculations in LJ-PME. lb_buf1 stores either the coefficients
     * for spreading/gathering (in serial), or the C6 coefficient for
     * local atoms (in parallel).  lb_buf2 is only used in parallel,
     * and stores the sigma values for local atoms. */
    FastVector<real> lb_buf1;
    FastVector<real> lb_buf2;

    pme_overlap_t overlap[2]; /* Indexed on dimension, 0=x, 1=y */

    /* Atom step for energy only calculation in gmx_pme_calc_energy() */
    std::unique_ptr<PmeAtomComm> atc_energy;

    /* bisheng 2025 Atom step for energy only plr_coul in gmx_pme_calc_phik()
     * Also reused by gmx_pme_calc_Aq_kspace for spline sharing */
    std::unique_ptr<PmeAtomComm> atc_phik;
    /* bisheng 2026: whether atc_phik currently holds cached spline
     * coefficients for frozen electrodes. Must live with the pme instance: as a
     * function-scope static it survived gmx_pme_destroy(), so a second pme would
     * skip spread_on_grid() and then gather from splines that were never built. */
    bool cpm_splineFrozen = false;

    /* Communication buffers */
    rvec* bufv;       /* Communication buffer */
    real* bufr;       /* Communication buffer */
    int   buf_nalloc; /* The communication buffer size */

    /* thread local work data for solve_pme */
    struct pme_solve_work_t* solve_work;

    /* Work data for sum_qgrid */
    real* sum_qgrid_tmp;
    real* sum_qgrid_dd_tmp;

    // bisheng 2025 psr_coul and Matrix
    real* psr_coul;
    double* MatrixA;
    double* inv_MatrixA;
    real applied_potential; // the voltage drop between two electrodes
    real applied_total_charge; // the total charge constrain
    // bisheng 2026 - charge target of the SECOND electrode in two-electrode
    // charge mode. Default -applied_total_charge (a capacitor); a "Q2 = ..."
    // parameter in potential_control.dat sets it independently (e.g. two
    // like-charged bodies, with the electrolyte carrying -(Q1+Q2)).
    double applied_charge2 = 0.0;
    bool   bCharge2Set     = false;
    // bisheng 2026 - cell voltage realised in CHARGE-constrained mode.
    // The charge-constrained solve determines the electrode potentials as
    // Lagrange multipliers; U = lambda1/evscale is otherwise discarded.
    // Written to electrode_potential.dat so galvanostatic runs record the
    // voltage response, which is the measured quantity in a GCD experiment.
    double cpm_cell_voltage = 0.0;

    // bisheng 2026 - galvanostatic cycling with voltage cut-offs ("mode = cutoff"
    // in potential_control.dat, charge-constraint mode). The imposed charge is
    // integrated here, Q_set += dir*I*dt, and dir flips when the realised cell
    // voltage of the previous step reaches Umax (charging) or Umin (discharging).
    // All of it is saved in cpm_state.dat so a restart continues the same cycle.
    bool    cpm_gcd_started = false;  // Q_set initialised (from Q0 or the state file)
    int     cpm_gcd_dir     = +1;     // +1 charging, -1 discharging
    double  cpm_gcd_qset    = 0.0;    // imposed charge of electrode 1, e
    double  cpm_gcd_tprev   = 0.0;    // time at which Q_set was last advanced, ps
    int64_t cpm_gcd_nswitch = 0;      // number of direction switches so far
    double  cpm_gcd_ufilt   = 0.0;    // exponential moving average of U (Utau > 0), V
    bool    cpm_gcd_ufinit  = false;  // the average has been seeded
    int64_t current_step    = 0;      // MD step, for the switch log

    // bisheng 2026 - Thomas-Fermi electrode model (Scalfi, Dufils, Reeves, Rotenberg,
    // Salanne, J. Chem. Phys. 153, 174704 (2020); MetalWalls src/thomas_fermi.f90).
    // The electron gas of a non-ideal metal costs (k_e/2)*(4*pi*l_TF^2/V_i)*q_i^2 per
    // atom to hold charge q_i, so the elastance matrix gains the DIAGONAL term
    //     A_ii += 4*pi*l_TF,i^2 / V_i      [1/nm; l_TF in nm, V_i (Voronoi volume) in nm^3]
    // Finite l_TF lets charge penetrate into inner electrode layers and lowers the
    // capacitance; l_TF = 0 is the perfect metal, i.e. the standard CPM. Enabled by
    // electrode_tf.dat in the working directory (see force.cpp); zero otherwise.
    bool              bThomasFermi = false;
    std::vector<real> tf_diag;              // 4*pi*l_TF,i^2/V_i per electrode atom, 1/nm
    double            tf_ltf_sum = 0.0;     // sum_i l_TF,i (nm)    -- provenance, MatrixA.meta
    double            tf_vol_sum = 0.0;     // sum_i V_i   (nm^3)   -- provenance, MatrixA.meta
    // TF record read back from cpm_state.dat on restart, compared against the
    // electrode_tf.dat actually loaded (force.cpp). -1 = no record (pre-TF file).
    int               cpm_restart_tf      = -1;
    double            cpm_restart_ltf_sum = 0.0;
    double            cpm_restart_vol_sum = 0.0;
    int nEleAtoms; // the number of atoms in the electrode
    int inv_or_cg; // 0 for matrix_inversion and 1 for conjugate_gradient
    int oneEle_or_not; // 1 for single electrode, 0 for two electrodes
    int pot_or_charge; // 0 for potential constrain, 1 for total charge constrain.

    // bisheng 2025 - Time-dependent potential/charge control
    // When userreal2 < 0, read expression from potential_control.dat
    std::unique_ptr<gmx::CpmPotentialControl> timeDepControl;  // Expression evaluator for V(t) or Q(t)
    gmx_bool bTimeDependentControl;  // True when userreal2 < 0
    real current_time;               // Current simulation time in ps (step * dt)
    int64_t nstenergy;               // Energy output interval (for debug print frequency)

    // bisheng 2025 - Extended Lagrangian parameters for matrix-free CG
    // When ext_lag_mass > 0, use Extended Lagrangian instead of CG solver
    real ext_lag_mass;    // Fictitious mass μ for charge dynamics (from userreal3)
    real ext_lag_dt;      // MD timestep for Extended Lagrangian propagation
    std::vector<real> ext_lag_q;      // Charge positions (in V*nm units)
    std::vector<real> ext_lag_qdot;   // Charge velocities

    /* bisheng 2026 - CPM solver state promoted from function statics so that it
     * can be saved/restored across checkpoint restarts (cpm_state.dat). */
    std::vector<real> cpm_qPrev;      // CG warm-start charges (whichever CG variant runs)
    std::vector<real> cpm_qNm1;       // charge predictor: q(N-1)
    std::vector<real> cpm_qPredicted; // charge predictor: prediction made for the current step
    std::vector<real> cpm_lastError;  // charge predictor: last prediction error
    /* bisheng 2026 - the electrode charges solved at the current step, kept
     * separately because the predictor overwrites md->chargeA before do_force
     * regains control (GMX_CPM_FORCE_AFTER_SOLVE pushes these to the GPU
     * nonbonded kernel). */
    std::vector<real> cpm_qSolved;
    /* bisheng 2026 - set by the force schedule when the GPU nonbonded force
     * kernel runs after the charge solve with the solved charges; the reciprocal
     * force pass then uses the solved charges too, so both halves of the
     * electrostatic force are self-consistent. */
    bool cpmForceAfterSolve = false;
    int64_t           cpm_predStep = 0; // charge predictor: step counter

    // bisheng 2025 - Neighbor list for electrode short-range interactions
    // CSR format: row_ptr[i] to row_ptr[i+1] gives neighbors of atom i
    std::vector<int>  sr_nlist_row_ptr;   // Row pointers [nEleAtoms+1]
    std::vector<int>  sr_nlist_col_idx;   // Column indices (neighbor atom j)
    std::vector<real> sr_nlist_coeff;     // Precomputed: erfc(beta*r)/r - erfc(eta_scale*r)/r
    std::vector<gmx::RVec> sr_nlist_x_ref; // Reference positions for rebuild check
    real sr_nlist_buffer;                 // Buffer distance for neighbor list (nm)
    gmx_bool sr_nlist_valid;              // Is neighbor list valid?
};

//! @endcond

/*! \brief
 * Finds out if PME is currently running on GPU.
 * TODO: should this be removed eventually?
 *
 * \param[in] pme  The PME structure.
 * \returns        True if PME runs on GPU currently, false otherwise.
 */
inline bool pme_gpu_active(const gmx_pme_t* pme)
{
    return (pme != nullptr) && (pme->runMode != PmeRunMode::CPU);
}

/*! \brief Tell our PME-only node to switch to a new grid size */
void gmx_pme_send_switchgrid(const t_commrec* cr, ivec grid_size, real ewaldcoeff_q, real ewaldcoeff_lj);

#endif
