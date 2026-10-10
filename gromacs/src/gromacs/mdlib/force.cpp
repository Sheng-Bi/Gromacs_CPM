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
#include "gmxpre.h"

#include "force.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gromacs/domdec/dlbtiming.h"
#include "gromacs/domdec/domdec.h"
#include "gromacs/domdec/domdec_struct.h"
#include "gromacs/ewald/cpm_potential_control.h"
#include "gromacs/ewald/ewald.h"
#include "gromacs/ewald/long_range_correction.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/gmxlib/network.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vecdump.h"
#include "gromacs/mdlib/forcerec_threading.h"
#include "gromacs/mdlib/qmmm.h"
#include "gromacs/mdlib/rf_util.h"
#include "gromacs/mdlib/wall.h"
#include "gromacs/mdtypes/commrec.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdatom.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/mshift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/smalloc.h"
// bisheng 2025 add header
#include "gromacs/nbnxm/nbnxm.h"
#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/math/units.h"
#include "gromacs/ewald/pme_internal.h"
#include "gromacs/ewald/cpm_matrix.h" // bisheng 2025 - internal matrix calculation

static void clearEwaldThreadOutput(ewald_corr_thread_t* ewc_t)
{
    ewc_t->Vcorr_q        = 0;
    ewc_t->Vcorr_lj       = 0;
    ewc_t->dvdl[efptCOUL] = 0;
    ewc_t->dvdl[efptVDW]  = 0;
    clear_mat(ewc_t->vir_q);
    clear_mat(ewc_t->vir_lj);
}

static void reduceEwaldThreadOuput(int nthreads, ewald_corr_thread_t* ewc_t)
{
    ewald_corr_thread_t& dest = ewc_t[0];

    for (int t = 1; t < nthreads; t++)
    {
        dest.Vcorr_q += ewc_t[t].Vcorr_q;
        dest.Vcorr_lj += ewc_t[t].Vcorr_lj;
        dest.dvdl[efptCOUL] += ewc_t[t].dvdl[efptCOUL];
        dest.dvdl[efptVDW] += ewc_t[t].dvdl[efptVDW];
        m_add(dest.vir_q, ewc_t[t].vir_q, dest.vir_q);
        m_add(dest.vir_lj, ewc_t[t].vir_lj, dest.vir_lj);
    }
}

// bisheng 2026 - CPM charge consistency.
//
// Within one MD step the short-range and the reciprocal halves of the
// electrostatic force must be evaluated with the SAME electrode charges,
// otherwise the total force is not the gradient of any energy and does net work
// whenever the charges move.  The short-range kernel runs first, with whatever
// md->chargeA holds on entry to do_force; the CPM solver then determines new
// charges and writes them into md->chargeA. The reciprocal force pass must use
// the same charge snapshot as the short-range force pass.
//
// The predictor schedule applies newly determined charges at the next step.
// GMX_CPM_LEGACY_Q_ORDER=1 enables an inconsistent ordering for diagnostics only.
static bool cpmLegacyQOrder()
{
    static const bool legacy    = (getenv("GMX_CPM_LEGACY_Q_ORDER") != nullptr);
    static bool       announced = false;
    if (!announced)
    {
        announced = true;
        printf("CPM: reciprocal force pass uses the %s electrode charges "
               "(GMX_CPM_LEGACY_Q_ORDER %s)\n",
               legacy ? "NEWLY SOLVED (legacy, inconsistent with short-range)"
                      : "same charges as the short-range kernel (consistent)",
               legacy ? "set" : "unset");
        fflush(stdout);
    }
    return legacy;
}

// bisheng 2026 - the reciprocal force pass takes the freshly solved charges
// either in the legacy A/B order or when the GPU nonbonded force kernel itself
// runs after the solve with those charges (GMX_CPM_FORCE_AFTER_SOLVE): then the
// two halves of the electrostatic force are self-consistent at q*(N).
static bool cpmForcePassUsesSolved(const gmx_pme_t* pme)
{
    return cpmLegacyQOrder() || pme->cpmForceAfterSolve;
}

// bisheng 2026 - checksum of an electrode charge vector at one point in the
// step; enabled by GMX_CPM_Q_TRACE=1.  Used to verify the invariant above: the
// "sr" and "recip" lines of a step must agree exactly.
static void cpmQTrace(const char* what, const real* q, int n)
{
    static const bool on = (getenv("GMX_CPM_Q_TRACE") != nullptr);
    if (!on)
    {
        return;
    }
    double s = 0, a = 0;
    for (int i = 0; i < n; i++)
    {
        s += static_cast<double>(q[i]);
        a += std::fabs(static_cast<double>(q[i]));
    }
    fprintf(stderr, "CPM-QTRACE %-6s sum=%.12f sumabs=%.12f\n", what, s, a);
}

// bisheng 2025
void accumulateAndResetPsrCoul(nonbonded_verlet_t* nbv, const t_mdatoms* md, real* psr_coul_global, int nEleAtoms, int nthreads)
{
    gmx::ArrayRef<const int> atomOrder = nbv->getLocalAtomOrder();
    // int nthreads = nbv->nbat->out.size();

    // Allocate if needed
    if (psr_coul_global == nullptr)
    {
        gmx_fatal(FARGS, "psr_coul_global must be preallocated before calling accumulateAndResetPsrCoul");
    }

    // bisheng 2026 - the number of nbnxm output buffers is the number of CPU
    // nonbonded threads, or 1 when the nonbonded kernel runs on a GPU (in that
    // case gpu_cpm_collect_psr() has downloaded the GPU result into out[0]).
    const int nout = static_cast<int>(nbv->nbat->out.size());

    // Accumulate thread-local psr_coul into global array
    #pragma omp parallel for num_threads(nthreads)
    for (int i = 0; i < atomOrder.size(); ++i)
    {
        int global_index = atomOrder[i];
        if (global_index == -1 || global_index >= nEleAtoms) continue;

        real temp = 0.0F;
        for (int t = 0; t < nout; ++t)
        {
            temp += nbv->nbat->out[t].psr_coul[i];
        }
        psr_coul_global[global_index] = temp;
    }

    // Reset per-thread psr_coul to zero
    #pragma omp parallel for num_threads(nthreads)
    for (int t = 0; t < nout; ++t)
    {
        std::fill(nbv->nbat->out[t].psr_coul.begin(),
                  nbv->nbat->out[t].psr_coul.end(), 0.0F);
    }
    if (0) // bisheng 2025 debug, will upgrade to CI test in the future
    {
        for (int i = 0; i < md->nr; ++i)
        {
            printf("psr_coul[gro %d] = %g\n", i + 1, psr_coul_global[i]* 0.010364272);
        }
    }
}

void do_force_lowlevel(t_forcerec*                         fr,
                       const t_inputrec*                   ir,
                       const t_idef*                       idef,
                       const t_commrec*                    cr,
                       const gmx_multisim_t*               ms,
                       t_nrnb*                             nrnb,
                       gmx_wallcycle_t                     wcycle,
                       const t_mdatoms*                    md,
                       gmx::ArrayRefWithPadding<gmx::RVec> coordinates,
                       history_t*                          hist,
                       gmx::ForceOutputs*                  forceOutputs,
                       gmx_enerdata_t*                     enerd,
                       t_fcdata*                           fcd,
                       const matrix                        box,
                       const real*                         lambda,
                       const t_graph*                      graph,
                       const rvec*                         mu_tot,
                       const gmx::StepWorkload&            stepWork,
                       const DDBalanceRegionHandler&       ddBalanceRegionHandler)
{
    // TODO: Replace all uses of x by const coordinates
    rvec* x = as_rvec_array(coordinates.paddedArrayRef().data());

    auto& forceWithVirial = forceOutputs->forceWithVirial();

    // bisheng 2025
    // Step 1: Read inv_MatrixA.dat and MatrixA.dat
    static int nEleAtoms = fr->userint1;
    static double *inv_MatrixA = nullptr, *MatrixA = nullptr;

    // Safety check: CPM requires all electrode atoms on single rank
    // Domain decomposition would split atoms across ranks, breaking the calculation
    // bisheng 2026: this guard is CPM-specific -- without the userint1
    // gate every multi-rank run of this build, CPM or not, aborted here.
    static bool bCheckedDD = false;
    if (fr->userint1 > 0 && !bCheckedDD)
    {
        if (DOMAINDECOMP(cr))
        {
            gmx_fatal(FARGS,
                "CPM (Constant Potential Method) does not support domain decomposition.\n"
                "Please run with -ntmpi 1 to disable MPI parallelism.\n"
                "OpenMP parallelism (-ntomp N) is supported and recommended.");
        }
        bCheckedDD = true;
    }

    // bisheng 2026 - the dense electrode matrix is only consumed by the
    // matrix-inversion (userint2=0) and matrix-CG (userint2=1) solvers; the
    // matrix-free CG / extended-Lagrangian path (userint2=2) rebuilds A*q on
    // the fly and must neither allocate nor load/build the N x N matrix.
    const bool bNeedMatrix = (fr->userint2 != 2);
    if (bNeedMatrix && nEleAtoms > 0 && inv_MatrixA == nullptr)
    {
        inv_MatrixA = (double*)calloc(nEleAtoms*nEleAtoms, sizeof(double));
    }
    if (bNeedMatrix && nEleAtoms > 0 && MatrixA == nullptr)
    {
        MatrixA = (double*)calloc(nEleAtoms*nEleAtoms, sizeof(double));
    }

    // bisheng 2025 - Initialize pme->eta_per_atom from nbat->eta_source_data
    // nbat->eta_source_data is initialized in atomdata.cpp (in set_charges)
    // We copy it to pme for matrix-free CG calculations (A*q operations)
    static bool bInitializedEta = false;
    // bisheng 2026: gate on CPM being active. With nEleAtoms == 0 the
    // average below evaluates 0.0/0 and writes NaN into pmedata->eta / ic->eta,
    // and fr->pmedata is legitimately null for non-PME electrostatics.
    if (fr->userint1 > 0 && !bInitializedEta)
    {
        nonbonded_verlet_t* nbv_eta = fr->nbv.get();
        fr->pmedata->eta_per_atom.resize(nEleAtoms);

        // Copy eta values from nbat to pme
        double eta_sum = 0.0;
        for (int i = 0; i < nEleAtoms; i++)
        {
            fr->pmedata->eta_per_atom[i] = nbv_eta->nbat->eta_source_data[i];
            eta_sum += nbv_eta->nbat->eta_source_data[i];
        }

        // Set average eta for code that uses global eta
        fr->pmedata->eta = static_cast<real>(eta_sum / nEleAtoms);
        fr->pmedata->bPerAtomEta = (fr->userreal1 <= 0);

        // Update ic->eta with the (averaged) value
        fr->ic->eta = fr->pmedata->eta;
        bInitializedEta = true;

        // bisheng 2026 - Thomas-Fermi electrode model, enabled by electrode_tf.dat in
        // the working directory (Scalfi et al., J. Chem. Phys. 153, 174704 (2020)).
        // Format, mirroring electrode_eta.dat:
        //   # comments
        //   l_TF  V_at                 one global line: applies to every electrode atom
        //   atom_id  l_TF  V_at        optional per-atom overrides (0-based, like electrode_eta.dat)
        // l_TF in nm, V_at (Voronoi / atomic volume) in nm^3. Each atom's elastance
        // diagonal gains 4*pi*l_TF^2/V_at (1/nm). Absent file -> perfect metal (unchanged).
        {
            gmx_pme_t* pme = fr->pmedata;
            pme->tf_diag.assign(nEleAtoms, static_cast<real>(0.0));
            pme->bThomasFermi = false;
            pme->tf_ltf_sum = pme->tf_vol_sum = 0.0;
            FILE* ftf = fopen("electrode_tf.dat", "r");
            if (ftf)
            {
                std::vector<double> ltf(nEleAtoms, -1.0), vol(nEleAtoms, -1.0);
                double gl = -1.0, gv = -1.0;
                char   line[512];
                int    lineNum = 0, nGlobal = 0, nAtom = 0;
                while (fgets(line, sizeof line, ftf))
                {
                    lineNum++;
                    char* ptr = line;
                    while (*ptr == ' ' || *ptr == '\t') ptr++;
                    if (*ptr == '#' || *ptr == '\n' || *ptr == '\0') continue;
                    int    id;
                    double a, b, c;
                    int    n = sscanf(ptr, "%lf %lf %lf", &a, &b, &c);
                    if (n == 2)
                    {
                        gl = a; gv = b; nGlobal++;
                    }
                    else if (n == 3 && sscanf(ptr, "%d", &id) == 1)
                    {
                        if (id < 0 || id >= nEleAtoms)
                            gmx_fatal(FARGS, "CPM Thomas-Fermi: electrode_tf.dat line %d: atom_id %d out of range 0-%d", lineNum, id, nEleAtoms - 1);
                        ltf[id] = b; vol[id] = c; nAtom++;
                    }
                    else
                    {
                        gmx_fatal(FARGS, "CPM Thomas-Fermi: cannot parse electrode_tf.dat line %d:\n  %s", lineNum, line);
                    }
                }
                fclose(ftf);
                if (nGlobal > 1)
                    gmx_fatal(FARGS, "CPM Thomas-Fermi: electrode_tf.dat has %d global 'l_TF V_at' lines; at most one is allowed", nGlobal);
                double lmin = 1e30, lmax = -1e30, vmin = 1e30, vmax = -1e30;
                for (int i = 0; i < nEleAtoms; ++i)
                {
                    double l = ltf[i] >= 0 ? ltf[i] : gl;
                    double v = vol[i] >= 0 ? vol[i] : gv;
                    if (l < 0 || v <= 0)
                        gmx_fatal(FARGS, "CPM Thomas-Fermi: no l_TF/V_at for electrode atom %d (need a global line or a per-atom line; V_at must be > 0)", i);
                    pme->tf_diag[i]  = static_cast<real>(4.0 * M_PI * l * l / v);
                    pme->tf_ltf_sum += l;
                    pme->tf_vol_sum += v;
                    lmin = std::min(lmin, l); lmax = std::max(lmax, l);
                    vmin = std::min(vmin, v); vmax = std::max(vmax, v);
                }
                pme->bThomasFermi = true;
                // bisheng 2026 - unit sanity. MetalWalls takes l_TF and V_at in bohr and
                // bohr^3; ours are nm and nm^3. A value copied across unconverted is
                // off by 19x (l) or 6.7e3 (V) and produces no error, only wrong charges.
                if (lmax > 1.0 || vmin < 1e-3 || vmax > 1.0)
                {
                    printf("\nWARNING: CPM Thomas-Fermi: electrode_tf.dat values look out of range for nm units\n"
                           "  (l_TF = %g..%g nm, V_at = %g..%g nm^3). Typical: l_TF 0.05-0.5 nm, V_at ~1e-2 nm^3\n"
                           "  (graphite 0.00878). MetalWalls inputs are in bohr / bohr^3 -- convert them.\n",
                           lmin, lmax, vmin, vmax);
                }
                printf("\nCPM: Thomas-Fermi electrode model ON (electrode_tf.dat: %d global, %d per-atom lines)\n"
                       "     l_TF = %g .. %g nm, V_at = %g .. %g nm^3, diagonal 4*pi*l_TF^2/V = %g .. %g 1/nm\n",
                       nGlobal, nAtom, lmin, lmax, vmin, vmax,
                       4.0 * M_PI * lmin * lmin / vmax, 4.0 * M_PI * lmax * lmax / vmin);
                if (fr->userint2 == 1)
                    gmx_fatal(FARGS, "CPM Thomas-Fermi: the dense-matrix CG solver (userint2=1) is not supported with TF; use userint2=0 or 2");
            }
            // bisheng 2026 - restart consistency: the TF record written into cpm_state.dat by
            // the previous leg must match what this leg loaded (or did not load).
            if (pme->cpm_restart_tf >= 0)
            {
                const bool prevTF = (pme->cpm_restart_tf == 1);
                auto rel = [](double a, double b) { double m = std::max(std::fabs(a), std::fabs(b)); return m > 0 ? std::fabs(a - b) / m : 0.0; };
                if (prevTF != pme->bThomasFermi
                    || (prevTF && (rel(pme->cpm_restart_ltf_sum, pme->tf_ltf_sum) > 1e-6
                                   || rel(pme->cpm_restart_vol_sum, pme->tf_vol_sum) > 1e-6)))
                {
                    gmx_fatal(FARGS, "CPM Thomas-Fermi: this run continues from cpm_state.dat written with the TF model %s\n"
                              "(sum l_TF = %g nm, sum V_at = %g nm^3), but electrode_tf.dat in this directory gives TF %s\n"
                              "(sum l_TF = %g, sum V_at = %g). Restore the same electrode_tf.dat (or remove it) before continuing.",
                              prevTF ? "ON" : "OFF", pme->cpm_restart_ltf_sum, pme->cpm_restart_vol_sum,
                              pme->bThomasFermi ? "ON" : "OFF", pme->tf_ltf_sum, pme->tf_vol_sum);
                }
            }
        }
    }

    static gmx_bool bReadMatrix = TRUE;
    // bisheng 2026 - nEleAtoms guard: non-CPM runs must not attempt to load or
    // internally compute a zero-sized electrode matrix.
    if (bReadMatrix && nEleAtoms > 0 && !bNeedMatrix)
    {
        printf("\nNOTE: Matrix-free CG (userint2=2): electrode matrix files are neither loaded nor computed.\n");
        bReadMatrix = FALSE;
    }
    if (bReadMatrix && nEleAtoms > 0)
    {
        // bisheng 2025 - Try to read matrix files first, compute internally if not found
        FILE* f_inv = fopen("inv_MatrixA.dat", "r");
        FILE* f_mat = fopen("MatrixA.dat", "r");

        if (f_inv && f_mat)
        {
            // Read inv_MatrixA.dat
            {
                fseek(f_inv, 0, SEEK_END);
                long filesize = ftell(f_inv);
                rewind(f_inv);
                char* buffer = (char*)malloc(filesize + 1);
                if (!buffer) { printf("\nNOTE: Memory allocation failed for inv_MatrixA buffer.\n"); exit(0); }
                size_t nread = fread(buffer, 1, filesize, f_inv);
                buffer[filesize] = '\0';
                fclose(f_inv);

                char* ptr = buffer;
                for (int idx = 0; idx < nEleAtoms; ++idx) {
                    for (int index = 0; index < nEleAtoms; ++index) {
                        double val = strtod(ptr, &ptr);
                        inv_MatrixA[idx*nEleAtoms+index] = val;
                    }
                }
                free(buffer);
            }
            // Read MatrixA.dat
            {
                fseek(f_mat, 0, SEEK_END);
                long filesize = ftell(f_mat);
                rewind(f_mat);
                char* buffer = (char*)malloc(filesize + 1);
                if (!buffer) { printf("\nNOTE: Memory allocation failed for MatrixA buffer.\n"); exit(0); }
                size_t nread = fread(buffer, 1, filesize, f_mat);
                buffer[filesize] = '\0';
                fclose(f_mat);

                char* ptr = buffer;
                for (int idx = 0; idx < nEleAtoms; ++idx) {
                    for (int index = 0; index < nEleAtoms; ++index) {
                        double val = strtod(ptr, &ptr);
                        MatrixA[idx*nEleAtoms+index] = val;
                    }
                }
                free(buffer);
            }
            printf("\nNOTE: inv_MatrixA and MatrixA files loaded successfully.\n");

            // bisheng 2026 - Provenance check against MatrixA.meta, written by
            // compute_cpm_matrix() alongside the matrices.  A cached matrix is
            // otherwise accepted verbatim, so one inherited from another run
            // (copied or symlinked working directory, changed cutoff/eta/box)
            // would be used silently.
            //
            // Severity follows the physics.  For a neutral two-electrode cell a
            // beta mismatch is a near-uniform offset that the sum(q)=0 constraint
            // annihilates, so it is a warning.  For a single electrode or a
            // fixed total charge, sum(q) = Q != 0 and nothing cancels, so any
            // mismatch is fatal.  A wrong atom count or geometry is always
            // fatal: it is a different system, not a different accuracy.
            {
                const bool chargeConstrained = (fr->userint3 == 1 || fr->userint4 == 1);
                FILE* fm = fopen("MatrixA.meta", "r");
                if (!fm)
                {
                    printf("\nWARNING: MatrixA.meta not found next to MatrixA.dat, so the cached\n"
                           "  electrode matrix cannot be checked against this run's settings.\n"
                           "  It is used as-is.  Delete MatrixA.dat and inv_MatrixA.dat to rebuild\n"
                           "  from the current MDP parameters (this also writes the .meta file).\n");
                    if (chargeConstrained)
                    {
                        printf("  NOTE: this run is charge-constrained (userint3=1 or userint4=1), where\n"
                               "  a matrix built at different settings is NOT harmless.  Rebuilding is\n"
                               "  strongly recommended.\n");
                    }
                }
                else
                {
                    int    m_n = -1, m_3dc = -1, m_tf = 0;
                    double m_eta = 0, m_beta = 0, m_cut = 0, m_rtol = 0, m_geom = 0, m_tfl = 0, m_tfv = 0;
                    double m_box[3] = { 0, 0, 0 };
                    char   key[64];
                    while (fscanf(fm, "%63s", key) == 1)
                    {
                        if      (!strcmp(key, "nEleAtoms"))     { if (fscanf(fm, "%d", &m_n) != 1) break; }
                        else if (!strcmp(key, "eta_sum"))       { if (fscanf(fm, "%lf", &m_eta) != 1) break; }
                        else if (!strcmp(key, "ewaldcoeff_q"))  { if (fscanf(fm, "%lf", &m_beta) != 1) break; }
                        else if (!strcmp(key, "cutoff"))        { if (fscanf(fm, "%lf", &m_cut) != 1) break; }
                        else if (!strcmp(key, "ewald_rtol"))    { if (fscanf(fm, "%lf", &m_rtol) != 1) break; }
                        else if (!strcmp(key, "box"))           { if (fscanf(fm, "%lf %lf %lf", &m_box[0], &m_box[1], &m_box[2]) != 3) break; }
                        else if (!strcmp(key, "use3dc"))        { if (fscanf(fm, "%d", &m_3dc) != 1) break; }
                        else if (!strcmp(key, "geom_checksum")) { if (fscanf(fm, "%lf", &m_geom) != 1) break; }
                        else if (!strcmp(key, "thomas_fermi"))  { if (fscanf(fm, "%d", &m_tf) != 1) break; }
                        else if (!strcmp(key, "tf_ltf_sum"))    { if (fscanf(fm, "%lf", &m_tfl) != 1) break; }
                        else if (!strcmp(key, "tf_vol_sum"))    { if (fscanf(fm, "%lf", &m_tfv) != 1) break; }
                        else { char rest[512]; if (!fgets(rest, sizeof rest, fm)) break; }
                    }
                    fclose(fm);

                    // recompute this run's values exactly as the writer did
                    double eta_sum = 0.0, geom = 0.0;
                    for (int i = 0; i < nEleAtoms; ++i)
                    {
                        eta_sum += static_cast<double>(fr->pmedata->eta_per_atom[i]);
                        geom += (i + 1.0) * (x[i][XX] + 2.0 * x[i][YY] + 3.0 * x[i][ZZ]);
                    }
                    auto rel = [](double a, double b) {
                        double d = std::fabs(a - b), m = std::max(std::fabs(a), std::fabs(b));
                        return m > 0 ? d / m : 0.0;
                    };
                    const double tolParam = 1e-6, tolGeom = 1e-8;
                    const bool use3dc = (ir->ewald_geometry == eewg3DC);

                    // always fatal: a different system
                    if (m_n != nEleAtoms)
                    {
                        gmx_fatal(FARGS, "CPM: cached MatrixA.dat was built for %d electrode atoms but this run has %d.\n"
                                  "Delete MatrixA.dat and inv_MatrixA.dat to rebuild.", m_n, nEleAtoms);
                    }
                    if (rel(m_geom, geom) > tolGeom)
                    {
                        gmx_fatal(FARGS, "CPM: cached MatrixA.dat was built for a different electrode geometry or\n"
                                  "atom ordering (geometry checksum %.10g vs %.10g). Delete MatrixA.dat and\n"
                                  "inv_MatrixA.dat to rebuild.", m_geom, geom);
                    }
                    // Thomas-Fermi: a DIAGONAL term, so unlike a beta change it is not
                    // absorbed by the sum(q)=0 constraint -- a mismatch is a different
                    // model in every mode. Older .meta files have no TF keys (m_tf=0).
                    {
                        const bool run_tf = fr->pmedata->bThomasFermi;
                        const bool bad = (m_tf != (run_tf ? 1 : 0))
                            || (run_tf && (rel(m_tfl, fr->pmedata->tf_ltf_sum) > tolParam
                                           || rel(m_tfv, fr->pmedata->tf_vol_sum) > tolParam));
                        if (bad)
                        {
                            gmx_fatal(FARGS, "CPM: cached MatrixA.dat was built %s Thomas-Fermi (sum l_TF %.8g nm, sum V %.8g nm^3)\n"
                                      "but this run is %s Thomas-Fermi (sum l_TF %.8g nm, sum V %.8g nm^3).\n"
                                      "The TF term is diagonal and changes the charges in every mode. Delete\n"
                                      "MatrixA.dat and inv_MatrixA.dat to rebuild for the current electrode_tf.dat.",
                                      m_tf ? "WITH" : "WITHOUT", m_tfl, m_tfv,
                                      run_tf ? "WITH" : "WITHOUT", fr->pmedata->tf_ltf_sum, fr->pmedata->tf_vol_sum);
                        }
                    }
                    if (m_3dc != (use3dc ? 1 : 0))
                    {
                        gmx_fatal(FARGS, "CPM: cached MatrixA.dat was built with ewald_geometry %s but this run uses %s.\n"
                                  "Delete MatrixA.dat and inv_MatrixA.dat to rebuild.",
                                  m_3dc ? "3dc" : "3d", use3dc ? "3dc" : "3d");
                    }

                    // accuracy-class mismatches: fatal when sum(q) != 0, warning otherwise
                    char why[1024] = "";
                    if (rel(m_beta, fr->ic->ewaldcoeff_q) > tolParam)
                        snprintf(why + strlen(why), sizeof why - strlen(why), "  ewaldcoeff_q (from rcoulomb/ewald_rtol): cached %.8g, run %.8g\n", m_beta, static_cast<double>(fr->ic->ewaldcoeff_q));
                    if (rel(m_cut, fr->ic->rcoulomb) > tolParam)
                        snprintf(why + strlen(why), sizeof why - strlen(why), "  rcoulomb: cached %.8g, run %.8g\n", m_cut, static_cast<double>(fr->ic->rcoulomb));
                    if (rel(m_rtol, ir->ewald_rtol) > tolParam)
                        snprintf(why + strlen(why), sizeof why - strlen(why), "  ewald_rtol: cached %.8g, run %.8g\n", m_rtol, static_cast<double>(ir->ewald_rtol));
                    if (rel(m_eta, eta_sum) > tolParam)
                        snprintf(why + strlen(why), sizeof why - strlen(why), "  eta (sum over electrode): cached %.10g, run %.10g\n", m_eta, eta_sum);
                    if (rel(m_box[0], box[XX][XX]) > tolParam || rel(m_box[1], box[YY][YY]) > tolParam || rel(m_box[2], box[ZZ][ZZ]) > tolParam)
                        snprintf(why + strlen(why), sizeof why - strlen(why), "  box: cached %.8g %.8g %.8g, run %.8g %.8g %.8g\n",
                                 m_box[0], m_box[1], m_box[2], static_cast<double>(box[XX][XX]), static_cast<double>(box[YY][YY]), static_cast<double>(box[ZZ][ZZ]));

                    if (why[0] != '\0')
                    {
                        if (chargeConstrained)
                        {
                            gmx_fatal(FARGS, "CPM: cached MatrixA.dat was built at different settings:\n%s"
                                      "This run is charge-constrained (userint3=1 or userint4=1), where sum(q) != 0\n"
                                      "and the resulting matrix error does NOT cancel. Delete MatrixA.dat and\n"
                                      "inv_MatrixA.dat to rebuild from the current MDP parameters.", why);
                        }
                        printf("\nWARNING: cached MatrixA.dat was built at different settings:\n%s"
                               "  For a neutral two-electrode cell the effect is a near-uniform offset that\n"
                               "  the sum(q)=0 constraint cancels, so the run continues.  Delete MatrixA.dat\n"
                               "  and inv_MatrixA.dat to rebuild if you want an exactly consistent matrix.\n", why);
                    }
                    else
                    {
                        printf("NOTE: MatrixA.meta matches this run's electrode, eta, Ewald and box settings.\n");
                    }
                }
            }
        }
        else
        {
            // Files not found - compute internally
            if (f_inv) fclose(f_inv);
            if (f_mat) fclose(f_mat);

            printf("\nNOTE: Matrix files not found, computing internally...\n");

            // bisheng 2025 - Compute the CPM matrix internally using optimized LAMMPS-style k-space
            // Note: PME grid size N corresponds to k-vectors from -N/2 to +N/2, so kmax = N/2
            compute_cpm_matrix(
                nEleAtoms,
                x,                              // electrode coords (first nEleAtoms)
                box,
                fr->pmedata->eta_per_atom.data(), // per-atom eta from userreal1 or electrode_eta.dat
                fr->ic->ewaldcoeff_q,           // beta from ewald_rtol (nm^-1)
                fr->ic->rcoulomb,               // cutoff (nm)
                fr->pmedata->nkx / 2,           // k-vector limit = PME grid / 2
                fr->pmedata->nky / 2,
                fr->pmedata->nkz / 2,
                ir->ewald_rtol,                 // accuracy for k-space (from MDP)
                ir->ewald_geometry == eewg3DC,  // 3DC correction flag
                MatrixA,
                inv_MatrixA,
                true,                           // save to files for future runs
                fr->pmedata->bThomasFermi ? fr->pmedata->tf_diag.data() : nullptr,
                fr->pmedata->tf_ltf_sum,
                fr->pmedata->tf_vol_sum
            );
        }

        // pass to PME data structure
        fr->pmedata->inv_MatrixA = inv_MatrixA;
        fr->pmedata->MatrixA = MatrixA;

        bReadMatrix = FALSE;
    }

    // Step 2: Get psr_coul and add 3dc correction if needed
    wallcycle_start(wcycle, ewcPMEMESH);
    nonbonded_verlet_t*          nbv      = fr->nbv.get();
    static real *psr_coul = nullptr;
    static int nthreads = fr->pmedata->nthread;

    if (psr_coul == nullptr)
    {
        psr_coul = (real *)calloc(nEleAtoms, sizeof(real));
    }
    accumulateAndResetPsrCoul(nbv, md, psr_coul, nEleAtoms, nthreads);

    //printf("ir->epsilon_r = %f\n",ir->epsilon_r);
    if(ir->ewald_geometry == eewg3DC)
    {
        static double* pcorr_coul = nullptr;
        if (pcorr_coul == nullptr)
        {
            pcorr_coul = (double*)calloc(nEleAtoms, sizeof(double));
        }

        static double ONE_4pi_eps = ONE_4PI_EPS0/ir->epsilon_r; //fr->epsilon_r?
        // bisheng 2026 - recomputed every step (was captured on the first step
        // only, which silently broke any box change; matrix-free CG now supports
        // a changing box, matrix modes are rejected at init when pcoupl is on)
        const double boxZ = static_cast<double>(box[ZZ][ZZ]);
        const double boxVolume = static_cast<double>(box[XX][XX] * box[YY][YY] * box[ZZ][ZZ]);
        double ElectricField;
        double ThreeDCZtemp1 = 0.0;
        #pragma omp parallel for reduction(+:ThreeDCZtemp1) num_threads(nthreads)
        for (int index = nEleAtoms; index < md->nr; index++)
        {
            ThreeDCZtemp1 += static_cast<double>(x[index][ZZ]) * static_cast<double>(md->chargeA[index]);
            // printf("index=%d, z=%f, charge=%f, ThreeDCZtemp1=%f\n", index, x[index][ZZ], md->chargeA[index], ThreeDCZtemp1);
        }
        ElectricField = 4*M_PI*ONE_4pi_eps/boxVolume*ThreeDCZtemp1*ENM2DEBYE*DEBYE2ENM;
        // printf("4*M_PI*ONE_4pi_eps/box= %f, ONE_4pi_eps=%f, box=%f, ElectricField = %f\n", 4*M_PI*ONE_4pi_eps/(box[XX][XX]*box[YY][YY]*box[ZZ][ZZ]), ONE_4pi_eps, box[XX][XX]*box[YY][YY]*box[ZZ][ZZ], ElectricField);	/* get phi_3dc if necessary */
        
		#pragma omp parallel for num_threads(nthreads)
		for(int i=0;i<nEleAtoms;i++)
		{
			pcorr_coul[i] = (static_cast<double>(x[i][ZZ])-boxZ/2)*ElectricField;
			// printf("i=%d, x=%f, y=%f, z=%f, Pot3dc=%f\n",i,x[i][XX],x[i][YY],x[i][ZZ],pcorr_coul[i]* 0.010364272);
            psr_coul[i] += static_cast<real>(pcorr_coul[i]); // accumulate the potential correction
            // printf("psr_coul[%d] = %f\n", i, psr_coul[i] * 0.010364272); // debug output
		}
    }

    // // Step 3: Compute charges on the electrodes
    // // We need to pass the psr_coul to the PME data structure
    // fr->pmedata->psr_coul = psr_coul; // pass psr_coul to PME data structure
    // fr->pmedata->applied_potential = fr->userreal2; // pass applied_potential to PME data structure
    // fr->pmedata->nEleAtoms = nEleAtoms; // pass nEleAtoms to PME data structure
    // // before computing the charge, we reset the charge of electrode atoms to zero
    // #pragma omp parallel for num_threads(nthreads) 
    // for(int i = 0; i< nEleAtoms; i++)
    // {
    //     md->chargeA[i] = 0;
    // }
    // ewald_corr_thread_t ewaldOutput;
    // real Vlr_q = 0, Vlr_lj = 0;
    // // Here we only compute the charges on the electrodes, we do not compute the forces
    // static int pme_flags_charge = GMX_PME_SPREAD | GMX_PME_SOLVE | GMX_PME_CALC_POT | GMX_PME_CALC_Q;
    // // static int pme_flags_charge = GMX_PME_SOLVE | GMX_PME_CALC_POT | GMX_PME_CALC_Q;
    // int status = gmx_pme_do(
    //         fr->pmedata,
    //         gmx::constArrayRefFromArray(coordinates.unpaddedConstArrayRef().data(),
    //                                     md->homenr - fr->n_tpi),
    //         forceWithVirial.force_, md->chargeA, md->chargeB, md->sqrt_c6A,
    //         md->sqrt_c6B, md->sigmaA, md->sigmaB, box, cr,
    //         DOMAINDECOMP(cr) ? dd_pme_maxshift_x(cr->dd) : 0,
    //         DOMAINDECOMP(cr) ? dd_pme_maxshift_y(cr->dd) : 0, nrnb, wcycle,
    //         ewaldOutput.vir_q, ewaldOutput.vir_lj, &Vlr_q, &Vlr_lj,
    //         lambda[efptCOUL], lambda[efptVDW], &ewaldOutput.dvdl[efptCOUL],
    //         &ewaldOutput.dvdl[efptVDW],  pme_flags_charge);
    
    // // debug
    // if(0)
    // {
    //     printf("After computing Charge\n");
    //     for(int i = 0; i< nEleAtoms; i++)
    //     {
    //         printf("chargeA[%d]=%f\n",i, md->chargeA[i]);
    //     }
    // }
    // wallcycle_stop(wcycle, ewcPMEMESH);

    /* do QMMM first if requested */
    if (fr->bQMMM)
    {
        enerd->term[F_EQM] = calculate_QMMM(cr, &forceOutputs->forceWithShiftForces(), fr);
    }

    /* Call the short range functions all in one go. */

    if (ir->nwall)
    {
        /* foreign lambda component for walls */
        real dvdl_walls = do_walls(*ir, *fr, box, *md, x, &forceWithVirial, lambda[efptVDW],
                                   enerd->grpp.ener[egLJSR].data(), nrnb);
        enerd->dvdl_lin[efptVDW] += dvdl_walls;
    }

    /* Shift the coordinates. Must be done before listed forces and PPPM,
     * but is also necessary for SHAKE and update, therefore it can NOT
     * go when no listed forces have to be evaluated.
     *
     * The shifting and PBC code is deliberately not timed, since with
     * the Verlet scheme it only takes non-zero time with triclinic
     * boxes, and even then the time is around a factor of 100 less
     * than the next smallest counter.
     */


    /* Here sometimes we would not need to shift with NBFonly,
     * but we do so anyhow for consistency of the returned coordinates.
     */
    if (graph)
    {
        shift_self(graph, box, x);
        if (TRICLINIC(box))
        {
            inc_nrnb(nrnb, eNR_SHIFTX, 2 * graph->nnodes);
        }
        else
        {
            inc_nrnb(nrnb, eNR_SHIFTX, graph->nnodes);
        }
    }

    {
        t_pbc pbc;

        /* Check whether we need to take into account PBC in listed interactions. */
        const auto needPbcForListedForces =
                fr->bMolPBC && stepWork.computeListedForces && haveCpuListedForces(*fr, *idef, *fcd);
        if (needPbcForListedForces)
        {
            /* Since all atoms are in the rectangular or triclinic unit-cell,
             * only single box vector shifts (2 in x) are required.
             */
            set_pbc_dd(&pbc, fr->ePBC, DOMAINDECOMP(cr) ? cr->dd->nc : nullptr, TRUE, box);
        }

        do_force_listed(wcycle, box, ir->fepvals, cr, ms, idef, x, hist, forceOutputs, fr, &pbc,
                        graph, enerd, nrnb, lambda, md, fcd,
                        DOMAINDECOMP(cr) ? cr->dd->globalAtomIndices.data() : nullptr, stepWork);
    }

    const bool computePmeOnCpu = (EEL_PME(fr->ic->eeltype) || EVDW_PME(fr->ic->vdwtype))
                                 && thisRankHasDuty(cr, DUTY_PME)
                                 && (pme_run_mode(fr->pmedata) == PmeRunMode::CPU);

    // bisheng 2026 - CPM with GPU PME: the reciprocal-space work runs on the GPU
    // (driven from do_force in sim_util.cpp), but the electrode-charge solve
    // still happens here, between the charge pass and the force pass.
    const bool useGpuPmeCpm = (fr->userint1 > 0) && EEL_PME(fr->ic->eeltype)
                              && thisRankHasDuty(cr, DUTY_PME)
                              && (pme_run_mode(fr->pmedata) == PmeRunMode::GPU);
    if (fr->userint1 > 0 && fr->pmedata != nullptr && pme_run_mode(fr->pmedata) == PmeRunMode::Mixed)
    {
        gmx_fatal(FARGS,
                  "CPM (constant potential) does not support mixed-mode PME (-pme gpu with CPU "
                  "FFT). Use -pme gpu (full GPU) or -pme cpu.");
    }

    const bool haveEwaldSurfaceTerm = haveEwaldSurfaceContribution(*ir);

    /* Do long-range electrostatics and/or LJ-PME
     * and compute PME surface terms when necessary.
     */
    if (computePmeOnCpu || fr->ic->eeltype == eelEWALD || haveEwaldSurfaceTerm || useGpuPmeCpm)
    {
        int  status = 0;
        real Vlr_q = 0, Vlr_lj = 0;

        /* We reduce all virial, dV/dlambda and energy contributions, except
         * for the reciprocal energies (Vlr_q, Vlr_lj) into the same struct.
         */
        ewald_corr_thread_t& ewaldOutput = fr->ewc_t[0];
        clearEwaldThreadOutput(&ewaldOutput);

        if (EEL_PME_EWALD(fr->ic->eeltype) || EVDW_PME(fr->ic->vdwtype))
        {
            /* Calculate the Ewald surface force and energy contributions, when necessary */
            if (haveEwaldSurfaceTerm)
            {
                wallcycle_sub_start(wcycle, ewcsEWALD_CORRECTION);

                if (fr->n_tpi > 0)
                {
                    gmx_fatal(FARGS,
                              "TPI with PME currently only works in a 3D geometry with tin-foil "
                              "boundary conditions");
                }

                // bisheng 2025, we need to update mu_tot, because we have computed charges on the electrode
                // Note that we only update mu_tot[0][ZZ], because 1. we don't have (or use) md->chargeB in our case
                // 2. we only need it for 3dc correction.
                // if(ir->ewald_geometry == eewg3DC)
                // {
                //     double temp = 0.0;
                //     for(int i = 0; i < nEleAtoms; i++)
                //     {
                //          temp += static_cast<double>(x[i][ZZ]*md->chargeA[i]);
                //         //  printf("x[%d][ZZ]=%f, chargeA[%d]=%f, temp=%f\n", i, x[i][ZZ], i, md->chargeA[i], temp);
                //     }
                //     const_cast<rvec*>(mu_tot)[0][ZZ] += static_cast<real>(temp*ENM2DEBYE);
                // }
                // check mu_tot[0][ZZ]
                // printf("mu_tot[0][ZZ] = %f\n", mu_tot[0][ZZ]);

                int nthreads = fr->nthread_ewc;
#pragma omp parallel for num_threads(nthreads) schedule(static)
                for (int t = 0; t < nthreads; t++)
                {
                    try
                    {
                        ewald_corr_thread_t& ewc_t = fr->ewc_t[t];
                        if (t > 0)
                        {
                            clearEwaldThreadOutput(&ewc_t);
                        }

                        /* Threading is only supported with the Verlet cut-off
                         * scheme and then only single particle forces (no
                         * exclusion forces) are calculated, so we can store
                         * the forces in the normal, single forceWithVirial->force_ array.
                         */
                        
                        ewald_LRcorrection(md->homenr, cr, nthreads, t, *fr, *ir, md->chargeA,
                                           md->chargeB, (md->nChargePerturbed != 0), x, box, mu_tot,
                                           as_rvec_array(forceWithVirial.force_.data()),
                                           &ewc_t.Vcorr_q, lambda[efptCOUL], &ewc_t.dvdl[efptCOUL]);
                    }
                    GMX_CATCH_ALL_AND_EXIT_WITH_FATAL_ERROR
                }
                if (nthreads > 1)
                {
                    reduceEwaldThreadOuput(nthreads, fr->ewc_t);
                }
                wallcycle_sub_stop(wcycle, ewcsEWALD_CORRECTION);
            }

            if (EEL_PME_EWALD(fr->ic->eeltype) && fr->n_tpi == 0)
            {
                /* This is not in a subcounter because it takes a
                   negligible and constant-sized amount of time */
                ewaldOutput.Vcorr_q += ewald_charge_correction(
                        cr, fr, lambda[efptCOUL], box, &ewaldOutput.dvdl[efptCOUL], ewaldOutput.vir_q);
            }

            if (computePmeOnCpu || useGpuPmeCpm)
            {
                /* Do reciprocal PME for Coulomb and/or LJ. */
                assert(fr->n_tpi >= 0);
                if (fr->n_tpi == 0 || stepWork.stateChanged)
                {
                    int pme_flags = GMX_PME_SPREAD | GMX_PME_SOLVE;

                    if (stepWork.computeForces)
                    {
                        pme_flags |= GMX_PME_CALC_F;
                    }
                    if (stepWork.computeVirial)
                    {
                        pme_flags |= GMX_PME_CALC_ENER_VIR;
                    }
                    if (fr->n_tpi > 0)
                    {
                        /* We don't calculate f, but we do want the potential */
                        pme_flags |= GMX_PME_CALC_POT;
                    }

                    /* With domain decomposition we close the CPU side load
                     * balancing region here, because PME does global
                     * communication that acts as a global barrier.
                     */
                    ddBalanceRegionHandler.closeAfterForceComputationCpu();


                    // bisheng 2026 - non-CPM runs (userint1 <= 0) get the standard
                    // single PME call, without a CPM charge pass.
                    if (fr->userint1 <= 0)
                    {
                        // (the enclosing ewcPMEMESH counter is already running)
                        status = gmx_pme_do(
                                fr->pmedata,
                                gmx::constArrayRefFromArray(coordinates.unpaddedConstArrayRef().data(),
                                                            md->homenr - fr->n_tpi),
                                forceWithVirial.force_, md->chargeA, md->chargeB, md->sqrt_c6A,
                                md->sqrt_c6B, md->sigmaA, md->sigmaB, box, cr,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_x(cr->dd) : 0,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_y(cr->dd) : 0, nrnb, wcycle,
                                ewaldOutput.vir_q, ewaldOutput.vir_lj, &Vlr_q, &Vlr_lj,
                                lambda[efptCOUL], lambda[efptVDW], &ewaldOutput.dvdl[efptCOUL],
                                &ewaldOutput.dvdl[efptVDW], pme_flags);
                    }
                    else
                    {
                    // Now we update the electrode charges
                    // We need to pass the psr_coul to the PME data structure
                    fr->pmedata->psr_coul = psr_coul; // pass psr_coul to PME data structure
                    fr->pmedata->nEleAtoms = nEleAtoms; // pass nEleAtoms to PME data structure
                    // fr->pmedata->inv_or_cg = fr->userint2; // pass inv_or_cg to PME data structure, 0 for matrix_inv and 1 for cg
                    // fr->pmedata->oneEle_or_not = fr->userint3; // pass oneEle to PME data structure, 0 for two electrodes, 1 for single electrode
                    // fr->pmedata->pot_or_charge = fr->userint4; // pass constrain to PME data structure, 0 for potential constrain, 1 for charge constrain
                    // fr->pmedata->applied_potential = fr->userreal2; // pass applied_potential to PME data structure
                    // fr->pmedata->applied_total_charge = fr->userreal2; // pass applied_total_charge to PME data structure

                    static bool bInitializedElectrodeSettings = false;
                    if (!bInitializedElectrodeSettings)
                    {
                        // bisheng 2026 - pressure coupling changes the box: the
                        // precomputed capacitance matrix would go stale, so only the
                        // matrix-free CG solver (which rebuilds A*q from the current
                        // geometry every step) may run under NPT.
                        if (ir->epc != epcNO)
                        {
                            if (fr->userint2 != 2)
                            {
                                gmx_fatal(FARGS,
                                          "CPM with pressure coupling requires the matrix-free CG "
                                          "solver (userint2=2); the precomputed electrode matrix is "
                                          "only valid for the grompp-time box.");
                            }
                            printf("\nWARNING: CPM with pressure coupling (matrix-free CG) is "
                                   "experimental; the 3DC correction now tracks the box each "
                                   "step, but validate against fixed-box references.\n\n");
                        }
                        fr->pmedata->inv_or_cg = fr->userint2;           // 0 for matrix_inv, 1 for matrix-cg, 2 for matrix-free cg
                        fr->pmedata->oneEle_or_not = fr->userint3;       // 0 for two electrodes, 1 for single electrode
                        fr->pmedata->pot_or_charge = fr->userint4;       // 0 for potential constrain, 1 for charge constrain

                        // bisheng 2025 - Time-dependent potential/charge control
                        // If userreal2 < 0, enable time-dependent mode and load expression from file
                        if (fr->userreal2 < 0)
                        {
                            fr->pmedata->bTimeDependentControl = TRUE;
                            fr->pmedata->timeDepControl = std::make_unique<gmx::CpmPotentialControl>();
                            if (!fr->pmedata->timeDepControl->loadFromFile("potential_control.dat"))
                            {
                                gmx_fatal(FARGS, "Failed to load time-dependent control from potential_control.dat");
                            }
                            // Initialize applied values to 0, will be updated each step
                            fr->pmedata->applied_potential = 0;
                            fr->pmedata->applied_total_charge = 0;
                        }
                        else
                        {
                            fr->pmedata->bTimeDependentControl = FALSE;
                            fr->pmedata->applied_potential = fr->userreal2;  // applied potential
                            fr->pmedata->applied_total_charge = fr->userreal2; // total applied charge
                        }

                        // bisheng 2025 - Parameters for matrix-free CG
                        fr->pmedata->rcoulomb = fr->ic->rcoulomb;        // Coulomb cutoff (nm)

                        // bisheng 2025 - Extended Lagrangian parameters
                        // userreal3 > 0: use Extended Lagrangian with fictitious mass = userreal3
                        // userreal3 <= 0: use standard CG solver
                        fr->pmedata->ext_lag_mass = fr->userreal3;       // fictitious mass μ
                        fr->pmedata->ext_lag_dt = ir->delta_t;           // MD timestep

                        if (fr->pmedata->ext_lag_mass > 0 && fr->pmedata->inv_or_cg == 2)
                        {
                            printf("Extended Lagrangian enabled for matrix-free CG:\n");
                            printf("  Fictitious mass (userreal3) = %.4f\n", fr->pmedata->ext_lag_mass);
                            printf("  Timestep = %.6f ps\n", fr->pmedata->ext_lag_dt);
                        }

                        // If only one electrode, check charge neutrality
                        if (fr->pmedata->oneEle_or_not == 1)
                        {
                            double totalCharge = 0.0;
                            for (int i = nEleAtoms; i < md->nr; ++i)
                            {
                                totalCharge += md->chargeA[i];
                            }

                            totalCharge += fr->pmedata->applied_total_charge;

                            if (std::abs(totalCharge) > 1e-4) // tolerance for floating-point sum
                            {
                                printf("WARNING: Total charge of single electrode system is not neutral. "
                                    "Total charge of the system is %g now\n", totalCharge);

                                // Adjust applied_total_charge to ensure neutrality
                                double corrected_charge = fr->pmedata->applied_total_charge - totalCharge;
                                printf("Adjusting applied_total_charge from %g to %g to ensure neutrality.\n",
                                    fr->pmedata->applied_total_charge, corrected_charge);

                                fr->pmedata->applied_total_charge = corrected_charge;
                            }
                        }

                        bInitializedElectrodeSettings = true;
                    }

                    // bisheng 2026 - current_time is set from the true simulation
                    // time in do_force (sim_util.cpp), which survives checkpoint
                    // restarts and honors tinit for V(t)/Q(t) protocols.

                    // bisheng 2026 - the force-pass gmx_pme_do below must write its
                    // reciprocal energy/virial/dvdl into the OUTER ewaldOutput and
                    // Vlr_q of this block so those contributions reach the energy output.
                    static int pme_flags_charge = GMX_PME_SPREAD | GMX_PME_SOLVE | GMX_PME_CALC_POT | GMX_PME_CALC_Q;
                    // bisheng 2026 - uses the OUTER status (stays 0 on the GPU PME path)

                    // bisheng 2026 - GPU PME path: the charge pass (spread with
                    // electrode charges zeroed, FFT, solve, back-FFT) has already
                    // run on the GPU, launched from do_force. Gather the per-atom
                    // potential from the grid and solve for the charges; the
                    // force pass with the new charges is launched from do_force
                    // right after this function returns.
                    if (useGpuPmeCpm)
                    {
                        const bool extLagGpu = (fr->userreal3 > 0 && fr->userint2 == 2);
                        auto       cpmCoords = gmx::constArrayRefFromArray(
                                coordinates.unpaddedConstArrayRef().data(), md->homenr - fr->n_tpi);

                        // bisheng 2026 - the stock GPU gather computes point-charge PME
                        // forces only; for flexible electrodes (userint2=2) the CPU path
                        // additionally applies the electrode-electrode Gaussian
                        // short-range force correction inside gmx_pme_do. Mirror it
                        // here with the charges the force pass actually uses: q(N) for
                        // the ExtLag ordering (force pass already ran), the freshly
                        // solved q(N) below for the standard ordering.
                        static const bool cpmNoSrForce = (getenv("GMX_CPM_NO_SR_FORCE") != nullptr);
                        const auto applySrForceCorrection = [&](const real* qForcePass) {
                            if (fr->userint2 == 2 && nEleAtoms > 0 && !cpmNoSrForce)
                            {
                                auto x_ele = cpmCoords.subArray(0, nEleAtoms);
                                auto q_ele = gmx::constArrayRefFromArray(qForcePass, nEleAtoms);
                                auto f_ele = forceWithVirial.force_.subArray(0, nEleAtoms);
                                const real coulomb_factor = ONE_4PI_EPS0 / fr->pmedata->epsilon_r;
                                gmx_pme_calc_Aq_sr_force(fr->pmedata, x_ele, q_ele, nEleAtoms, box,
                                                         fr->pmedata->rcoulomb, coulomb_factor, f_ele,
                                                         fr->pmedata->nthread, fr->pmedata->bEwald3DC);
                            }
                        };

                        // bisheng 2026 - the electrode charges the short-range kernel
                        // used for THIS step's forces. On the GPU PME path md->chargeA
                        // is still untouched here (the charge pass runs from a separate
                        // electrode-zeroed buffer), so this is the value on entry to
                        // do_force. gmx_pme_cpm_solve_charges below overwrites it.
                        static std::vector<real> qUsedBySr;
                        qUsedBySr.assign(md->chargeA, md->chargeA + nEleAtoms);
                        cpmQTrace("sr", qUsedBySr.data(), nEleAtoms);

                        if (extLagGpu)
                        {
                            // Extended Lagrangian ordering: the FORCE pass with the
                            // current charges q(N) already ran as the normal GPU PME
                            // (sim_util launches it unmodified in this mode). Apply the
                            // SR force correction with q(N), then run the charge pass
                            // so the update below produces q(N+1).
                            // bisheng 2026 - this ordering ALREADY satisfies the charge
                            // consistency invariant: the short-range kernel and the
                            // reciprocal force pass both saw q(N). Nothing to switch.
                            cpmQTrace("recip", md->chargeA, nEleAtoms);
                            applySrForceCorrection(md->chargeA);
                            static std::vector<real> zeroedCharges;
                            zeroedCharges.assign(md->chargeA, md->chargeA + md->homenr);
                            std::fill(zeroedCharges.begin(), zeroedCharges.begin() + nEleAtoms, 0._real);
                            pme_gpu_cpm_run_kspace_cycle(fr->pmedata, zeroedCharges.data());
                        }
                        static std::vector<real> plrGpu;
                        if (static_cast<int>(plrGpu.size()) < nEleAtoms)
                        {
                            plrGpu.resize(nEleAtoms);
                        }
                        pme_gpu_cpm_gather_potential(fr->pmedata, plrGpu.data(), nEleAtoms);
                        gmx_pme_cpm_solve_charges(fr->pmedata, cpmCoords, md->chargeA, box, plrGpu.data());
                        fr->pmedata->cpm_qSolved.assign(md->chargeA, md->chargeA + nEleAtoms);

                        if (!extLagGpu)
                        {
                            cpmQTrace("solved", md->chargeA, nEleAtoms);
                            // bisheng 2026 - stage the pass-2 (force) PME coefficients
                            // NOW: the charge-prediction block further down overwrites
                            // md->chargeA with the next step's predicted charges, which
                            // must not enter this step's forces.
                            // Stage into a PAGEABLE buffer first: md->chargeA is pinned
                            // when a PME GPU task exists, so an async copy straight from
                            // it would still be DMA-reading when the predictor writes
                            // (pageable async transfers are staged on return, so the
                            // snapshot below is race-free by CUDA semantics).
                            //
                            // Charge consistency: the force pass gets the electrode
                            // charges the short-range kernel already used (qUsedBySr),
                            // NOT the ones just solved. The solved ones stay in
                            // md->chargeA for the predictor and the logging, and reach
                            // the forces at the next step.
                            static std::vector<real> forcePassCharges;
                            forcePassCharges.assign(md->chargeA, md->chargeA + md->homenr);
                            if (!cpmForcePassUsesSolved(fr->pmedata))
                            {
                                std::copy(qUsedBySr.begin(), qUsedBySr.end(),
                                          forcePassCharges.begin());
                            }
                            cpmQTrace("recip", forcePassCharges.data(), nEleAtoms);
                            pme_gpu_cpm_set_charges(fr->pmedata, forcePassCharges.data());
                            applySrForceCorrection(forcePassCharges.data());
                        }
                    }
                    // bisheng 2025 - Extended Lagrangian: compute forces FIRST with q(N),
                    // then update charges to q(N+1) at the END. This ensures SR and LR use same q(N).
                    else if (fr->userreal3 > 0 && fr->userint2 == 2)
                    {
                        // Step 1: Compute LR forces using current charges q(N) in chargeA
                        // bisheng 2026 - this ordering ALREADY satisfies the charge
                        // consistency invariant: the short-range kernel and the
                        // reciprocal force pass both use q(N). Nothing to switch here;
                        // the mode flag only affects the solver paths.
                        cpmQTrace("sr", md->chargeA, nEleAtoms);
                        cpmQTrace("recip", md->chargeA, nEleAtoms);
                        wallcycle_start(wcycle, ewcPMEMESH);
                        status = gmx_pme_do(
                                fr->pmedata,
                                gmx::constArrayRefFromArray(coordinates.unpaddedConstArrayRef().data(),
                                                            md->homenr - fr->n_tpi),
                                forceWithVirial.force_, md->chargeA, md->chargeB, md->sqrt_c6A,
                                md->sqrt_c6B, md->sigmaA, md->sigmaB, box, cr,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_x(cr->dd) : 0,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_y(cr->dd) : 0, nrnb, wcycle,
                                ewaldOutput.vir_q, ewaldOutput.vir_lj, &Vlr_q, &Vlr_lj,
                                lambda[efptCOUL], lambda[efptVDW], &ewaldOutput.dvdl[efptCOUL],
                                &ewaldOutput.dvdl[efptVDW], pme_flags);
                        wallcycle_stop(wcycle, ewcPMEMESH);
                        // Step 2: Reset electrode charges to 0, then Extended Lagrangian computes q(N+1)
                        #pragma omp parallel for num_threads(nthreads)
                        for(int i = 0; i < nEleAtoms; i++)
                        {
                            md->chargeA[i] = 0;
                        }
                        status = gmx_pme_do(
                                fr->pmedata,
                                gmx::constArrayRefFromArray(coordinates.unpaddedConstArrayRef().data(),
                                                            md->homenr - fr->n_tpi),
                                forceWithVirial.force_, md->chargeA, md->chargeB, md->sqrt_c6A,
                                md->sqrt_c6B, md->sigmaA, md->sigmaB, box, cr,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_x(cr->dd) : 0,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_y(cr->dd) : 0, nrnb, wcycle,
                                ewaldOutput.vir_q, ewaldOutput.vir_lj, &Vlr_q, &Vlr_lj,
                                lambda[efptCOUL], lambda[efptVDW], &ewaldOutput.dvdl[efptCOUL],
                                &ewaldOutput.dvdl[efptVDW], pme_flags_charge);
                    }
                    else
                    {
                        // bisheng 2026 - the electrode charges the short-range kernel
                        // used for THIS step's forces (what the previous step
                        // published, post-predictor). The charge pass below zeroes the
                        // electrode entries and the solver overwrites them, so snapshot
                        // them first.
                        static std::vector<real> qUsedBySr;
                        qUsedBySr.assign(md->chargeA, md->chargeA + nEleAtoms);
                        cpmQTrace("sr", qUsedBySr.data(), nEleAtoms);

                        // Original order: compute charges first, then forces
                        #pragma omp parallel for num_threads(nthreads)
                        for(int i = 0; i < nEleAtoms; i++)
                        {
                            md->chargeA[i] = 0;
                        }
                        status = gmx_pme_do(
                                fr->pmedata,
                                gmx::constArrayRefFromArray(coordinates.unpaddedConstArrayRef().data(),
                                                            md->homenr - fr->n_tpi),
                                forceWithVirial.force_, md->chargeA, md->chargeB, md->sqrt_c6A,
                                md->sqrt_c6B, md->sigmaA, md->sigmaB, box, cr,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_x(cr->dd) : 0,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_y(cr->dd) : 0, nrnb, wcycle,
                                ewaldOutput.vir_q, ewaldOutput.vir_lj, &Vlr_q, &Vlr_lj,
                                lambda[efptCOUL], lambda[efptVDW], &ewaldOutput.dvdl[efptCOUL],
                                &ewaldOutput.dvdl[efptVDW], pme_flags_charge);

                        // bisheng 2026 - charge consistency: md->chargeA now holds the
                        // freshly solved q*(N). Keep it for the predictor and the
                        // logging, but run the reciprocal FORCE pass with the charges
                        // the short-range kernel used, so both halves of the
                        // electrostatic force come from one charge vector. q*(N) reaches
                        // the forces at the next step (through the predictor).
                        static std::vector<real> qSolved;
                        cpmQTrace("solved", md->chargeA, nEleAtoms);
                        fr->pmedata->cpm_qSolved.assign(md->chargeA, md->chargeA + nEleAtoms);
                        if (!cpmForcePassUsesSolved(fr->pmedata))
                        {
                            qSolved.assign(md->chargeA, md->chargeA + nEleAtoms);
                            std::copy(qUsedBySr.begin(), qUsedBySr.end(), md->chargeA);
                        }
                        cpmQTrace("recip", md->chargeA, nEleAtoms);

                        wallcycle_start(wcycle, ewcPMEMESH);
                        status = gmx_pme_do(
                                fr->pmedata,
                                gmx::constArrayRefFromArray(coordinates.unpaddedConstArrayRef().data(),
                                                            md->homenr - fr->n_tpi),
                                forceWithVirial.force_, md->chargeA, md->chargeB, md->sqrt_c6A,
                                md->sqrt_c6B, md->sigmaA, md->sigmaB, box, cr,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_x(cr->dd) : 0,
                                DOMAINDECOMP(cr) ? dd_pme_maxshift_y(cr->dd) : 0, nrnb, wcycle,
                                ewaldOutput.vir_q, ewaldOutput.vir_lj, &Vlr_q, &Vlr_lj,
                                lambda[efptCOUL], lambda[efptVDW], &ewaldOutput.dvdl[efptCOUL],
                                &ewaldOutput.dvdl[efptVDW], pme_flags);
                        wallcycle_stop(wcycle, ewcPMEMESH);

                        // Restore the solved charges: they are what the predictor
                        // extrapolates from and what gets reported/logged.
                        if (!cpmForcePassUsesSolved(fr->pmedata))
                        {
                            std::copy(qSolved.begin(), qSolved.end(), md->chargeA);
                        }
                    }


                    // bisheng 2025 - Charge prediction for energy conservation
                    // Use linear extrapolation with error feedback: q_pred(N+1) = 2*q(N) - q(N-1) + γ*error
                    // This ensures forces at step N+1 use predicted q(N+1) instead of lagged q(N)
                    // NOTE: Skip prediction when Extended Lagrangian is used (userreal3 > 0 with matrix-free CG)
                    //       Extended Lagrangian handles energy conservation through its own charge dynamics
                    if (!(fr->userreal3 > 0 && fr->userint2 == 2))
                    {
                        // bisheng 2026 - predictor state lives in pme so it can be
                        // saved/restored across checkpoint restarts (cpm_state.dat)
                        std::vector<real>& q_Nm1       = fr->pmedata->cpm_qNm1;
                        std::vector<real>& q_predicted = fr->pmedata->cpm_qPredicted;
                        std::vector<real>& last_error  = fr->pmedata->cpm_lastError;
                        int64_t&           pred_step   = fr->pmedata->cpm_predStep;
                        static const real gamma = 0.5;            // error feedback coefficient

                        // Initialize on first call
                        if (q_Nm1.empty()) {
                            q_Nm1.resize(nEleAtoms, 0.0);
                            q_predicted.resize(nEleAtoms, 0.0);
                            last_error.resize(nEleAtoms, 0.0);
                        }

                        // Compute prediction error: actual q(N) vs predicted q(N)
                        // Only available from step 2 onwards (first prediction was made at step 1)
                        static real total_charge_actual = 0.0;
                        static real total_charge_predicted = 0.0;
                        static real rmse = 0.0;
                        static real max_error = 0.0;

                        if (pred_step >= 2) {
                            real error_sum = 0.0;
                            max_error = 0.0;
                            total_charge_actual = 0.0;
                            total_charge_predicted = 0.0;

                            // Determine number of atoms to sum for total charge
                            // userint3=0: two electrodes, report first electrode (nEleAtoms/2)
                            // userint3=1: single electrode, report all (nEleAtoms)
                            int nAtomsForTotal = (fr->userint3 == 0) ? nEleAtoms / 2 : nEleAtoms;

                            for (int i = 0; i < nAtomsForTotal; i++) {
                                total_charge_actual += md->chargeA[i];
                                total_charge_predicted += q_predicted[i];
                            }

                            // Compute RMSE and max error over ALL electrode atoms
                            for (int i = 0; i < nEleAtoms; i++) {
                                last_error[i] = md->chargeA[i] - q_predicted[i];
                                error_sum += last_error[i] * last_error[i];
                                if (std::abs(last_error[i]) > max_error) {
                                    max_error = std::abs(last_error[i]);
                                }
                            }
                            rmse = std::sqrt(error_sum / nEleAtoms);
                        }

                        // Log prediction error to file (every nstlog steps)
                        // Format: step, total_actual, total_predicted, rmse, max_error
                        static FILE* pred_log = nullptr;
                        static int nstlog = ir->nstlog > 0 ? ir->nstlog : 100;

                        if (pred_log == nullptr) {
                            pred_log = fopen("charge_prediction.dat", "a"); // bisheng 2026 - append across restarts
                            if (pred_log) {
                                fprintf(pred_log, "# step  total_actual  total_predicted  rmse  max_error\n");
                            }
                        }

                        if (pred_log && pred_step >= 2 && pred_step % nstlog == 0) {
                            fprintf(pred_log, "%8lld  %12.6f  %12.6f  %12.6e  %12.6e\n",
                                    static_cast<long long>(pred_step), total_charge_actual, total_charge_predicted, rmse, max_error);
                            fflush(pred_log);
                        }

                        // Predict q(N+1) for next step's force calculation
                        for (int i = 0; i < nEleAtoms; i++) {
                            if (pred_step >= 1) {
                                // Linear extrapolation: q(N+1) = 2*q(N) - q(N-1)
                                // Plus error correction: + γ * (q(N) - q_predicted(N))
                                real q_linear = 2.0 * md->chargeA[i] - q_Nm1[i];
                                q_predicted[i] = q_linear + gamma * last_error[i];
                            } else {
                                // Step 0: no history yet, just use current charge
                                q_predicted[i] = md->chargeA[i];
                            }

                            // Shift history: current q(N) becomes q(N-1) for next step
                            q_Nm1[i] = md->chargeA[i];
                        }

                        pred_step++;

                        // Set predicted charges for NEXT step's force calculation
                        for (int i = 0; i < nEleAtoms; i++) {
                            md->chargeA[i] = q_predicted[i];
                        }
                    } // end of charge prediction (skip when Extended Lagrangian is used)
                    } // bisheng 2026 - end of CPM branch (userint1 > 0)

                    wallcycle_stop(wcycle, ewcPMEMESH);

                    
                    if (status != 0)
                    {
                        gmx_fatal(FARGS, "Error %d in reciprocal PME routine", status);
                    }

                    /* We should try to do as little computation after
                     * this as possible, because parallel PME synchronizes
                     * the nodes, so we want all load imbalance of the
                     * rest of the force calculation to be before the PME
                     * call.  DD load balancing is done on the whole time
                     * of the force call (without PME).
                     */
                }

                if (fr->n_tpi > 0)
                {
                    if (EVDW_PME(ir->vdwtype))
                    {

                        gmx_fatal(FARGS, "Test particle insertion not implemented with LJ-PME");
                    }
                    /* Determine the PME grid energy of the test molecule
                     * with the PME grid potential of the other charges.
                     */
                    gmx_pme_calc_energy(
                            fr->pmedata,
                            coordinates.unpaddedConstArrayRef().subArray(md->homenr - fr->n_tpi, fr->n_tpi),
                            gmx::arrayRefFromArray(md->chargeA + md->homenr - fr->n_tpi, fr->n_tpi),
                            &Vlr_q);
                }
            }
        }

        if (fr->ic->eeltype == eelEWALD)
        {
            Vlr_q = do_ewald(ir, x, as_rvec_array(forceWithVirial.force_.data()), md->chargeA,
                             md->chargeB, box, cr, md->homenr, ewaldOutput.vir_q, fr->ic->ewaldcoeff_q,
                             lambda[efptCOUL], &ewaldOutput.dvdl[efptCOUL], fr->ewald_table);
        }

        /* Note that with separate PME nodes we get the real energies later */
        // TODO it would be simpler if we just accumulated a single
        // long-range virial contribution.
        forceWithVirial.addVirialContribution(ewaldOutput.vir_q);
        forceWithVirial.addVirialContribution(ewaldOutput.vir_lj);
        enerd->dvdl_lin[efptCOUL] += ewaldOutput.dvdl[efptCOUL];
        enerd->dvdl_lin[efptVDW] += ewaldOutput.dvdl[efptVDW];
        enerd->term[F_COUL_RECIP] = Vlr_q + ewaldOutput.Vcorr_q;
        enerd->term[F_LJ_RECIP]   = Vlr_lj + ewaldOutput.Vcorr_lj;

        if (debug)
        {
            fprintf(debug, "Vlr_q = %g, Vcorr_q = %g, Vlr_corr_q = %g\n", Vlr_q,
                    ewaldOutput.Vcorr_q, enerd->term[F_COUL_RECIP]);
            pr_rvecs(debug, 0, "vir_el_recip after corr", ewaldOutput.vir_q, DIM);
            rvec* fshift = as_rvec_array(forceOutputs->forceWithShiftForces().shiftForces().data());
            pr_rvecs(debug, 0, "fshift after LR Corrections", fshift, SHIFTS);
            fprintf(debug, "Vlr_lj: %g, Vcorr_lj = %g, Vlr_corr_lj = %g\n", Vlr_lj,
                    ewaldOutput.Vcorr_lj, enerd->term[F_LJ_RECIP]);
            pr_rvecs(debug, 0, "vir_lj_recip after corr", ewaldOutput.vir_lj, DIM);
        }
    }

    if (debug)
    {
        print_nrnb(debug, nrnb);
    }

    if (debug)
    {
        rvec* fshift = as_rvec_array(forceOutputs->forceWithShiftForces().shiftForces().data());
        pr_rvecs(debug, 0, "fshift after bondeds", fshift, SHIFTS);
    }
}
