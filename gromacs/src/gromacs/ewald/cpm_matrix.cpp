/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2025, by the GROMACS development team, led by
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

/*! \file
 * \brief
 * Implements internal CPM electrode matrix computation.
 *
 * \author Bisheng (2025)
 * \ingroup module_ewald
 */

// bisheng 2025 - Internal CPM matrix calculation implementation

#include "cpm_matrix.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <atomic>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "gromacs/math/vec.h"
#include "gromacs/utility/fatalerror.h"

// bisheng 2025 - Progress tracking for matrix computation
static std::atomic<int> g_progress_count{0};
static int g_progress_total = 0;
static std::chrono::steady_clock::time_point g_start_time;

static void reset_progress(int total)
{
    g_progress_count = 0;
    g_progress_total = total;
    g_start_time = std::chrono::steady_clock::now();
}

static void update_progress(int increment = 1)
{
    g_progress_count += increment;
}

static void print_progress(const char* stage)
{
    if (g_progress_total <= 0) return;

    int count = g_progress_count.load();
    double percent = 100.0 * count / g_progress_total;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - g_start_time).count();

    double eta = 0.0;
    if (count > 0 && percent < 100.0)
    {
        eta = elapsed * (g_progress_total - count) / count;
    }

    // Build progress bar (40 characters wide)
    const int barWidth = 40;
    int filled = static_cast<int>(barWidth * percent / 100.0);
    if (filled > barWidth) filled = barWidth;

    // Use ANSI escape code to clear line and return to start
    // \033[2K clears entire line, \r returns to start
    fprintf(stderr, "\033[2K\r  %s: [", stage);
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < filled)
            fprintf(stderr, "=");
        else if (i == filled)
            fprintf(stderr, ">");
        else
            fprintf(stderr, " ");
    }
    fprintf(stderr, "] %5.2f%%", percent);

    // Show ETA
    if (percent >= 100.0)
    {
        fprintf(stderr, " Done!      ");
    }
    else if (eta > 0)
    {
        if (eta < 60)
        {
            fprintf(stderr, " ETA: %4.0fs", eta);
        }
        else if (eta < 3600)
        {
            fprintf(stderr, " ETA: %4.1fm", eta / 60.0);
        }
        else
        {
            fprintf(stderr, " ETA: %4.1fh", eta / 3600.0);
        }
    }
    else
    {
        fprintf(stderr, " ETA: calculating...");
    }

    fflush(stderr);
}

// Constants
static const double c_pi = M_PI;
static const double c_sqrt_pi = std::sqrt(M_PI);
static const double c_sqrt2 = std::sqrt(2.0);

/*! \brief Compute erfc using polynomial approximation.
 *
 * This is the same approximation used in calcMatrix for consistency.
 */
static double my_erfc(double x)
{
    constexpr double EWALD_P = 0.3275911;
    constexpr double A1      = 0.254829592;
    constexpr double A2      = -0.284496736;
    constexpr double A3      = 1.421413741;
    constexpr double A4      = -1.453152027;
    constexpr double A5      = 1.061405429;

    double expm2 = std::exp(-x * x);
    double t     = 1.0 / (1.0 + EWALD_P * x);
    return t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
}

/*! \brief Compute real-space contribution to the matrix.
 *
 * Computes: (erfc(beta*r) - erfc(eta_ij*r/sqrt(2))) / r
 * with periodic images in xy-plane.
 * bisheng 2025 - Modified to support per-atom eta
 * eta_ij = eta_i * eta_j / sqrt(eta_i^2 + eta_j^2)
 * bisheng 2026 - wrapZ: with ewald_geometry = 3d the cell is periodic in z as
 * well, so dz takes its minimum image (cutoff < box_z/2 is enforced by grompp,
 * so one z-image suffices). A slab (3DC) run keeps dz unwrapped, as before.
 * Found on a nanotube bonded across the z boundary: unwrapped dz gave per-atom
 * charges 1e-3 e off the (correct, z-wrapping) matrix-free solver.
 */
static void calc_matrix_real(int           nEleAtoms,
                             const rvec*   x,
                             const double* boxSize,
                             const real*   eta_per_atom,
                             double        beta,
                             double        cutoff,
                             bool          wrapZ,
                             double*       rReal)
{
    const double cutoff2 = cutoff * cutoff;

    // Periodic image ratios for xy-plane (9 images)
    const int nImages = 9;
    const double ratio[9][2] = {
        {0, 0}, {0, 1}, {0, -1}, {1, 0}, {1, 1}, {1, -1}, {-1, 0}, {-1, 1}, {-1, -1}
    };

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nEleAtoms; ++i)
    {
        const double eta_i = eta_per_atom[i];
        const double eta_i2 = eta_i * eta_i;

        for (int img = 0; img < nImages; ++img)
        {
            for (int j = 0; j < nEleAtoms; ++j)
            {
                double dx = x[i][XX] - (x[j][XX] + boxSize[XX] * ratio[img][0]);
                double dy = x[i][YY] - (x[j][YY] + boxSize[YY] * ratio[img][1]);
                double dz = x[i][ZZ] - x[j][ZZ];
                if (wrapZ)
                {
                    dz -= boxSize[ZZ] * std::round(dz / boxSize[ZZ]);
                }
                double r2 = dx * dx + dy * dy + dz * dz;

                if (r2 < cutoff2 && r2 > 1e-14)
                {
                    double r = std::sqrt(r2);

                    // Compute combined eta_ij = eta_i * eta_j / sqrt(eta_i^2 + eta_j^2)
                    // For same eta: eta_ij = eta/sqrt(2), matching MetalWalls
                    const double eta_j = eta_per_atom[j];
                    const double eta_j2 = eta_j * eta_j;
                    const double eta_ij = (eta_i * eta_j) / std::sqrt(eta_i2 + eta_j2);

                    double val = (my_erfc(beta * r) - my_erfc(eta_ij * r)) / r;
                    #pragma omp atomic
                    rReal[i * nEleAtoms + j] += val;
                }
            }
        }
    }
}

/*! \brief Compute self-interaction contribution (diagonal).
 *
 * Computes: (sqrt(2)*eta[i] - 2*beta) / sqrt(pi) for each atom
 * bisheng 2025 - Modified to support per-atom eta
 */
static void calc_matrix_self(int nEleAtoms, const real* eta_per_atom, double beta, double* rSelf)
{
    for (int i = 0; i < nEleAtoms; ++i)
    {
        double selfTerm = (c_sqrt2 * eta_per_atom[i] - 2.0 * beta) / c_sqrt_pi;
        rSelf[i * nEleAtoms + i] = selfTerm;
    }
}

/*! \brief Compute 3DC slab correction contribution.
 *
 * Computes: 4*pi*z[i]*z[j] / volume
 * Only applied if use3dc is true.
 */
static void calc_matrix_slab(int           nEleAtoms,
                             const rvec*   x,
                             double        volume,
                             double        boxZ,
                             double*       rSlab)
{
    // Shift z-coordinates so center of box is at z=0
    // (matching calcMatrix convention)
    const double zShift = boxZ / 2.0;
    const double factor = 4.0 * c_pi / volume;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nEleAtoms; ++i)
    {
        for (int j = 0; j < nEleAtoms; ++j)
        {
            double zi = x[i][ZZ] - zShift;
            double zj = x[j][ZZ] - zShift;
            rSlab[i * nEleAtoms + j] = factor * zi * zj;
        }
    }
}

/*! \brief Compute RMS error estimate for k-space Ewald (LAMMPS formula).
 *
 * From LAMMPS ewald.cpp: estimates the RMS error in forces from truncating
 * the k-space sum at kmax in a given dimension.
 *
 * \param[in] km    k-vector limit in this dimension
 * \param[in] prd   box length in this dimension (nm)
 * \param[in] beta  Ewald splitting parameter (nm^-1)
 * \param[in] natoms number of atoms
 * \return RMS error estimate
 */
static double kspace_rms(int km, double prd, double beta, int natoms)
{
    if (natoms == 0) natoms = 1;
    // LAMMPS formula: 2 * q2 * beta / prd * sqrt(1/(π*km*N)) * exp(-π²*km²/(β²*prd²))
    // We assume q2 ≈ natoms (unit charges), which gives a conservative estimate
    double q2 = static_cast<double>(natoms);
    double value = 2.0 * q2 * beta / prd *
                   std::sqrt(1.0 / (c_pi * km * natoms)) *
                   std::exp(-c_pi * c_pi * km * km / (beta * beta * prd * prd));
    return value;
}

/*! \brief Auto-determine k-vector limits based on accuracy.
 *
 * Following LAMMPS approach: increment kmax until RMS error < target accuracy.
 *
 * \param[in]  boxSize   Box dimensions (nm)
 * \param[in]  beta      Ewald splitting parameter (nm^-1)
 * \param[in]  natoms    Number of atoms
 * \param[in]  accuracy  Target accuracy (default 1e-5)
 * \param[out] kxmax     K-vector limit in x
 * \param[out] kymax     K-vector limit in y
 * \param[out] kzmax     K-vector limit in z
 */
static void auto_kmax(const double* boxSize, double beta, int natoms, double accuracy,
                      int& kxmax, int& kymax, int& kzmax)
{
    // Start with kmax = 1 and increment until error < accuracy
    kxmax = 1;
    while (kspace_rms(kxmax, boxSize[XX], beta, natoms) > accuracy)
    {
        kxmax++;
        if (kxmax > 500) break;  // Safety limit
    }

    kymax = 1;
    while (kspace_rms(kymax, boxSize[YY], beta, natoms) > accuracy)
    {
        kymax++;
        if (kymax > 500) break;
    }

    kzmax = 1;
    while (kspace_rms(kzmax, boxSize[ZZ], beta, natoms) > accuracy)
    {
        kzmax++;
        if (kzmax > 500) break;
    }
}

/*! \brief Compute k-space contribution using LAMMPS-style optimized Ewald summation.
 *
 * bisheng 2025 - Optimized k-space calculation following LAMMPS ELECTRODE approach
 * (see lammps/src/ELECTRODE/ewald_electrode.cpp)
 *
 * Key optimization: Precompute cos/sin values for each atom and k-vector component,
 * then combine using angle addition formulas. This reduces cos/sin calls from
 * O(N² * k³) to O(N * (kx + ky + kz)).
 *
 * Formula: A[i][j] = Σ_k 2 * ug[k] * cos(k·(r_i - r_j))
 * where ug[k] = (4π/V) * exp(-k²/(4β²)) / k²
 *
 * Using: cos(k·(ri-rj)) = cos(k·ri)*cos(k·rj) + sin(k·ri)*sin(k·rj)
 */
static void calc_matrix_kspace_optimized(int           nEleAtoms,
                                         const rvec*   x,
                                         const double* boxSize,
                                         double        volume,
                                         double        beta,
                                         int           kxmax_in,
                                         int           kymax_in,
                                         int           kzmax_in,
                                         double        accuracy,
                                         double*       rKspace)
{
    // Auto-determine k-vector limits based on accuracy (LAMMPS approach)
    // accuracy comes from ewald_rtol in MDP file (default 1e-5)
    int kxmax, kymax, kzmax;

    // As current matrix calculation is fast, we don't really need to run this,
    // We will just use the values from mdp.
    // auto_kmax(boxSize, beta, nEleAtoms, accuracy, kxmax, kymax, kzmax);
    // printf("    Auto k-limits (accuracy=%.0e): kx=%d, ky=%d, kz=%d\n",
    //        accuracy, kxmax, kymax, kzmax);

    kxmax = kxmax_in;
    kymax = kymax_in;
    kzmax = kzmax_in;

    printf("    Input k-limits (from PME/2):   kx=%d, ky=%d, kz=%d\n",
           kxmax_in, kymax_in, kzmax_in);

    // Use auto-calculated limits (ignore input)
    // If you want to use input limits, comment out the auto_kmax call above

    // Unit k-vectors (2π/L for each dimension)
    const double unitk[3] = {
        2.0 * c_pi / boxSize[XX],
        2.0 * c_pi / boxSize[YY],
        2.0 * c_pi / boxSize[ZZ]
    };

    // Cutoff in k-space (following LAMMPS: use MAX)
    const double gsqxmx = unitk[XX] * unitk[XX] * kxmax * kxmax;
    const double gsqymx = unitk[YY] * unitk[YY] * kymax * kymax;
    const double gsqzmx = unitk[ZZ] * unitk[ZZ] * kzmax * kzmax;
    const double gsqmx = std::max({gsqxmx, gsqymx, gsqzmx}) * 1.00001;

    const double beta_sq_inv = 1.0 / (beta * beta);
    const double preu = 4.0 * c_pi / volume;  // LAMMPS uses 4π/V

    printf("    Precomputing cos/sin tables (direct method)...\n");

    // Precompute cos/sin for each atom and k-value along each dimension
    // cs[dim][k][i] = cos(k * unitk[dim] * x[i][dim])
    // sn[dim][k][i] = sin(k * unitk[dim] * x[i][dim])
    // Using direct std::cos/std::sin for maximum accuracy (no recurrence drift)
    std::vector<std::vector<std::vector<double>>> cs(3), sn(3);
    int kmax[3] = {kxmax, kymax, kzmax};

    for (int dim = 0; dim < 3; ++dim)
    {
        cs[dim].resize(kmax[dim] + 1, std::vector<double>(nEleAtoms));
        sn[dim].resize(kmax[dim] + 1, std::vector<double>(nEleAtoms));

        #pragma omp parallel for
        for (int i = 0; i < nEleAtoms; ++i)
        {
            // k=0: cos=1, sin=0
            cs[dim][0][i] = 1.0;
            sn[dim][0][i] = 0.0;

            // Use direct std::cos/std::sin for all k values (more accurate than recurrence)
            double base_arg = unitk[dim] * x[i][dim];
            for (int k = 1; k <= kmax[dim]; ++k)
            {
                double arg = k * base_arg;
                cs[dim][k][i] = std::cos(arg);
                sn[dim][k][i] = std::sin(arg);
            }
        }
    }

    printf("    Computing k-space matrix elements...\n");

    // Build list of k-vectors with their weights (following LAMMPS structure)
    struct KVector {
        int kx, ky, kz;
        double ug;  // weight = preu * exp(-k²/(4β²)) / k²
    };
    std::vector<KVector> kvecs;

    // Generate all k-vectors (excluding k=0)
    // (k,0,0), (0,l,0), (0,0,m)
    for (int m = 1; m <= kxmax; ++m)
    {
        double sqk = (m * unitk[XX]) * (m * unitk[XX]);
        if (sqk <= gsqmx)
        {
            kvecs.push_back({m, 0, 0, preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk});
        }
    }
    for (int m = 1; m <= kymax; ++m)
    {
        double sqk = (m * unitk[YY]) * (m * unitk[YY]);
        if (sqk <= gsqmx)
        {
            kvecs.push_back({0, m, 0, preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk});
        }
    }
    for (int m = 1; m <= kzmax; ++m)
    {
        double sqk = (m * unitk[ZZ]) * (m * unitk[ZZ]);
        if (sqk <= gsqmx)
        {
            kvecs.push_back({0, 0, m, preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk});
        }
    }

    // (k,l,0)
    for (int k = 1; k <= kxmax; ++k)
    {
        for (int l = 1; l <= kymax; ++l)
        {
            double sqk = (k * unitk[XX]) * (k * unitk[XX]) + (l * unitk[YY]) * (l * unitk[YY]);
            if (sqk <= gsqmx)
            {
                double ug = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                kvecs.push_back({k, l, 0, ug});
                kvecs.push_back({k, -l, 0, ug});
            }
        }
    }

    // (0,l,m)
    for (int l = 1; l <= kymax; ++l)
    {
        for (int m = 1; m <= kzmax; ++m)
        {
            double sqk = (l * unitk[YY]) * (l * unitk[YY]) + (m * unitk[ZZ]) * (m * unitk[ZZ]);
            if (sqk <= gsqmx)
            {
                double ug = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                kvecs.push_back({0, l, m, ug});
                kvecs.push_back({0, l, -m, ug});
            }
        }
    }

    // (k,0,m)
    for (int k = 1; k <= kxmax; ++k)
    {
        for (int m = 1; m <= kzmax; ++m)
        {
            double sqk = (k * unitk[XX]) * (k * unitk[XX]) + (m * unitk[ZZ]) * (m * unitk[ZZ]);
            if (sqk <= gsqmx)
            {
                double ug = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                kvecs.push_back({k, 0, m, ug});
                kvecs.push_back({k, 0, -m, ug});
            }
        }
    }

    // (k,l,m) - all combinations
    for (int k = 1; k <= kxmax; ++k)
    {
        for (int l = 1; l <= kymax; ++l)
        {
            for (int m = 1; m <= kzmax; ++m)
            {
                double sqk = (k * unitk[XX]) * (k * unitk[XX])
                           + (l * unitk[YY]) * (l * unitk[YY])
                           + (m * unitk[ZZ]) * (m * unitk[ZZ]);
                if (sqk <= gsqmx)
                {
                    double ug = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                    kvecs.push_back({k, l, m, ug});
                    kvecs.push_back({k, -l, m, ug});
                    kvecs.push_back({k, l, -m, ug});
                    kvecs.push_back({k, -l, -m, ug});
                }
            }
        }
    }

    printf("    Number of k-vectors: %zu\n", kvecs.size());

    // Reset progress tracking
    reset_progress(static_cast<int>(kvecs.size()));
    // Print every 2% to reduce terminal buffer issues
    int printInterval = std::max(1, static_cast<int>(kvecs.size()) / 50);

    // Get number of threads for thread-local storage
    int nthreads = 1;
    #ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        nthreads = omp_get_num_threads();
    }
    #endif
    printf("    Using %d OpenMP threads\n", nthreads);

    // Print initial progress bar immediately
    print_progress("K-space (optimized)");

    // Allocate thread-local matrices to avoid atomic operations
    std::vector<std::vector<double>> rKspace_local(nthreads,
        std::vector<double>(nEleAtoms * nEleAtoms, 0.0));

    // Compute matrix elements: A[i][j] = Σ_k 2 * ug[k] * cos(k·(r_i - r_j))
    // Using: cos(k·(ri-rj)) = cos(k·ri)*cos(k·rj) + sin(k·ri)*sin(k·rj)
    // Parallelize over k-vectors (each thread accumulates into its own buffer)
    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        double* local_buf = rKspace_local[tid].data();

        #pragma omp for schedule(dynamic, 16)
        for (size_t kidx = 0; kidx < kvecs.size(); ++kidx)
        {
            const KVector& kv = kvecs[kidx];
            const int kx = std::abs(kv.kx);
            const int ky = std::abs(kv.ky);
            const int kz = std::abs(kv.kz);
            const int sign_ky = (kv.ky >= 0) ? 1 : -1;
            const int sign_kz = (kv.kz >= 0) ? 1 : -1;

            // Precompute cos(k·ri) and sin(k·ri) for all atoms (thread-local)
            std::vector<double> cos_kr(nEleAtoms), sin_kr(nEleAtoms);

            for (int i = 0; i < nEleAtoms; ++i)
            {
                // cos(kx*xi + ky*yi + kz*zi) using angle addition
                double cos_kxky = cs[0][kx][i] * cs[1][ky][i] - sn[0][kx][i] * sn[1][ky][i] * sign_ky;
                double sin_kxky = sn[0][kx][i] * cs[1][ky][i] + cs[0][kx][i] * sn[1][ky][i] * sign_ky;
                cos_kr[i] = cos_kxky * cs[2][kz][i] - sin_kxky * sn[2][kz][i] * sign_kz;
                sin_kr[i] = sin_kxky * cs[2][kz][i] + cos_kxky * sn[2][kz][i] * sign_kz;
            }

            // Accumulate matrix elements into thread-local buffer
            const double weight = 2.0 * kv.ug;

            // Outer product with vectorizable inner loop
            for (int i = 0; i < nEleAtoms; ++i)
            {
                const double ci = cos_kr[i];
                const double si = sin_kr[i];
                double* __restrict__ row = &local_buf[i * nEleAtoms];
                const double* __restrict__ cj = cos_kr.data();
                const double* __restrict__ sj = sin_kr.data();

                // Inner loop - compiler can vectorize this (SIMD)
                #pragma omp simd
                for (int j = 0; j < nEleAtoms; ++j)
                {
                    row[j] += weight * (ci * cj[j] + si * sj[j]);
                }
            }

            // Update progress (any thread can print, protected by critical section)
            int count = g_progress_count.fetch_add(1) + 1;
            if (count % printInterval == 0)
            {
                #pragma omp critical
                {
                    print_progress("K-space (optimized)");
                }
            }
        }
    }

    // Final progress print (before reduction message)
    g_progress_count = static_cast<int>(kvecs.size());
    print_progress("K-space (optimized)");
    fprintf(stderr, "\n");

    // Reduce thread-local buffers into final result
    printf("    Reducing thread-local buffers...\n");
    #pragma omp parallel for
    for (int idx = 0; idx < nEleAtoms * nEleAtoms; ++idx)
    {
        double sum = 0.0;
        for (int t = 0; t < nthreads; ++t)
        {
            sum += rKspace_local[t][idx];
        }
        rKspace[idx] = sum;
    }
}

/*! \brief Compute reciprocal-space (k-space) contribution with progress tracking (SLOW - FALLBACK).
 *
 * Computes Ewald sum over k-vectors:
 * A[i][j] += fc * cos(k · (r_i - r_j))
 * where fc = (8*pi/volume) * exp(-k²/(4*beta²)) / k²
 *
 * bisheng 2025 - This is the slow direct method using the same algorithm in calMatrix. 
 */
static void calc_matrix_kspace_direct(int           nEleAtoms,
                               const rvec*   x,
                               const double* boxSize,
                               double        volume,
                               double        beta,
                               int           kxmax,
                               int           kymax,
                               int           kzmax,
                               double*       rKspace)
{
    const double unitk[3] = {
        2.0 * c_pi / boxSize[XX],
        2.0 * c_pi / boxSize[YY],
        2.0 * c_pi / boxSize[ZZ]
    };

    const int kmax = std::max({kxmax, kymax, kzmax});
    const double gsqxmx = unitk[XX] * unitk[XX] * kxmax * kxmax;
    const double gsqymx = unitk[YY] * unitk[YY] * kymax * kymax;
    const double gsqzmx = unitk[ZZ] * unitk[ZZ] * kzmax * kzmax;
    double gsqmx = std::max({gsqxmx, gsqymx, gsqzmx}) * 1.00001;

    const double beta_sq_inv = 1.0 / (beta * beta);
    const double preu = 8.0 * c_pi / volume;

    // Calculate total number of k-vector operations for progress tracking
    int totalOps = 3 * kmax                           // (k,0,0), (0,l,0), (0,0,m)
                 + kxmax * kymax                      // (k,l,0)
                 + kymax * kzmax                      // (0,l,m)
                 + kxmax * kzmax                      // (k,0,m)
                 + kxmax * kymax * kzmax;             // (k,l,m)
    reset_progress(totalOps);

    int printInterval = std::max(1, totalOps / 10000); // Print every 0.01%
    int lastPrint = 0;

    // (k,0,0), (0,l,0), (0,0,m) contributions
    for (int m = 1; m <= kmax; ++m)
    {
        // (m,0,0)
        double sqk = (m * unitk[XX]) * (m * unitk[XX]);
        if (sqk <= gsqmx)
        {
            double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nEleAtoms; ++i)
            {
                for (int j = 0; j < nEleAtoms; ++j)
                {
                    double contrib = fc * std::cos(m * unitk[XX] * (x[i][XX] - x[j][XX]));
                    #pragma omp atomic
                    rKspace[i * nEleAtoms + j] += contrib;
                }
            }
        }
        update_progress();

        // (0,m,0)
        sqk = (m * unitk[YY]) * (m * unitk[YY]);
        if (sqk <= gsqmx)
        {
            double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nEleAtoms; ++i)
            {
                for (int j = 0; j < nEleAtoms; ++j)
                {
                    double contrib = fc * std::cos(m * unitk[YY] * (x[i][YY] - x[j][YY]));
                    #pragma omp atomic
                    rKspace[i * nEleAtoms + j] += contrib;
                }
            }
        }
        update_progress();

        // (0,0,m)
        sqk = (m * unitk[ZZ]) * (m * unitk[ZZ]);
        if (sqk <= gsqmx)
        {
            double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < nEleAtoms; ++i)
            {
                for (int j = 0; j < nEleAtoms; ++j)
                {
                    double contrib = fc * std::cos(m * unitk[ZZ] * (x[i][ZZ] - x[j][ZZ]));
                    #pragma omp atomic
                    rKspace[i * nEleAtoms + j] += contrib;
                }
            }
        }
        update_progress();

        // Print progress periodically
        int currentCount = g_progress_count.load();
        if (currentCount - lastPrint >= printInterval)
        {
            print_progress("K-space");
            lastPrint = currentCount;
        }
    }

    // (k,l,0) contributions
    for (int k = 1; k <= kxmax; ++k)
    {
        for (int l = 1; l <= kymax; ++l)
        {
            double sqk = (unitk[XX] * k) * (unitk[XX] * k) + (unitk[YY] * l) * (unitk[YY] * l);
            if (sqk <= gsqmx)
            {
                double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nEleAtoms; ++i)
                {
                    for (int j = 0; j < nEleAtoms; ++j)
                    {
                        double contrib = 2.0 * fc
                            * std::cos(k * unitk[XX] * (x[i][XX] - x[j][XX]))
                            * std::cos(l * unitk[YY] * (x[i][YY] - x[j][YY]));
                        #pragma omp atomic
                        rKspace[i * nEleAtoms + j] += contrib;
                    }
                }
            }
            update_progress();
        }

        // Print progress periodically
        int currentCount = g_progress_count.load();
        if (currentCount - lastPrint >= printInterval)
        {
            print_progress("K-space");
            lastPrint = currentCount;
        }
    }

    // (0,l,m) contributions
    for (int l = 1; l <= kymax; ++l)
    {
        for (int m = 1; m <= kzmax; ++m)
        {
            double sqk = (unitk[YY] * l) * (unitk[YY] * l) + (unitk[ZZ] * m) * (unitk[ZZ] * m);
            if (sqk <= gsqmx)
            {
                double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nEleAtoms; ++i)
                {
                    for (int j = 0; j < nEleAtoms; ++j)
                    {
                        double contrib = 2.0 * fc
                            * std::cos(l * unitk[YY] * (x[i][YY] - x[j][YY]))
                            * std::cos(m * unitk[ZZ] * (x[i][ZZ] - x[j][ZZ]));
                        #pragma omp atomic
                        rKspace[i * nEleAtoms + j] += contrib;
                    }
                }
            }
            update_progress();
        }

        // Print progress periodically
        int currentCount = g_progress_count.load();
        if (currentCount - lastPrint >= printInterval)
        {
            print_progress("K-space");
            lastPrint = currentCount;
        }
    }

    // (k,0,m) contributions
    for (int k = 1; k <= kxmax; ++k)
    {
        for (int m = 1; m <= kzmax; ++m)
        {
            double sqk = (unitk[XX] * k) * (unitk[XX] * k) + (unitk[ZZ] * m) * (unitk[ZZ] * m);
            if (sqk <= gsqmx)
            {
                double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                #pragma omp parallel for collapse(2)
                for (int i = 0; i < nEleAtoms; ++i)
                {
                    for (int j = 0; j < nEleAtoms; ++j)
                    {
                        double contrib = 2.0 * fc
                            * std::cos(k * unitk[XX] * (x[i][XX] - x[j][XX]))
                            * std::cos(m * unitk[ZZ] * (x[i][ZZ] - x[j][ZZ]));
                        #pragma omp atomic
                        rKspace[i * nEleAtoms + j] += contrib;
                    }
                }
            }
            update_progress();
        }

        // Print progress periodically
        int currentCount = g_progress_count.load();
        if (currentCount - lastPrint >= printInterval)
        {
            print_progress("K-space");
            lastPrint = currentCount;
        }
    }

    // (k,l,m) contributions - the largest part
    for (int k = 1; k <= kxmax; ++k)
    {
        for (int l = 1; l <= kymax; ++l)
        {
            for (int m = 1; m <= kzmax; ++m)
            {
                double sqk = (unitk[XX] * k) * (unitk[XX] * k)
                           + (unitk[YY] * l) * (unitk[YY] * l)
                           + (unitk[ZZ] * m) * (unitk[ZZ] * m);
                if (sqk <= gsqmx)
                {
                    double fc = preu * std::exp(-0.25 * sqk * beta_sq_inv) / sqk;
                    #pragma omp parallel for collapse(2)
                    for (int i = 0; i < nEleAtoms; ++i)
                    {
                        for (int j = 0; j < nEleAtoms; ++j)
                        {
                            double contrib = 4.0 * fc
                                * std::cos(k * unitk[XX] * (x[i][XX] - x[j][XX]))
                                * std::cos(l * unitk[YY] * (x[i][YY] - x[j][YY]))
                                * std::cos(m * unitk[ZZ] * (x[i][ZZ] - x[j][ZZ]));
                            #pragma omp atomic
                            rKspace[i * nEleAtoms + j] += contrib;
                        }
                    }
                }
                update_progress();
            }
        }

        // Print progress periodically
        int currentCount = g_progress_count.load();
        if (currentCount - lastPrint >= printInterval)
        {
            print_progress("K-space");
            lastPrint = currentCount;
        }
    }

    // Final progress print
    print_progress("K-space");
    fprintf(stderr, "\n");
}

/*! \brief Invert a matrix using Gauss-Jordan elimination.
 *
 * This is a fallback when LAPACK is not available.
 * Works in-place on the input matrix.
 */
static bool invert_matrix_gauss_jordan(int n, double* A, double* Ainv)
{
    // Initialize Ainv as identity
    std::memset(Ainv, 0, n * n * sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        Ainv[i * n + i] = 1.0;
    }

    // Make a working copy of A
    std::vector<double> work(A, A + n * n);

    for (int col = 0; col < n; ++col)
    {
        // Find pivot
        int pivot = col;
        double maxVal = std::abs(work[col * n + col]);
        for (int row = col + 1; row < n; ++row)
        {
            if (std::abs(work[row * n + col]) > maxVal)
            {
                maxVal = std::abs(work[row * n + col]);
                pivot = row;
            }
        }

        if (maxVal < 1e-15)
        {
            return false; // Singular matrix
        }

        // Swap rows
        if (pivot != col)
        {
            for (int k = 0; k < n; ++k)
            {
                std::swap(work[col * n + k], work[pivot * n + k]);
                std::swap(Ainv[col * n + k], Ainv[pivot * n + k]);
            }
        }

        // Scale pivot row
        double scale = 1.0 / work[col * n + col];
        for (int k = 0; k < n; ++k)
        {
            work[col * n + k] *= scale;
            Ainv[col * n + k] *= scale;
        }

        // Eliminate column
        for (int row = 0; row < n; ++row)
        {
            if (row != col)
            {
                double factor = work[row * n + col];
                for (int k = 0; k < n; ++k)
                {
                    work[row * n + k] -= factor * work[col * n + k];
                    Ainv[row * n + k] -= factor * Ainv[col * n + k];
                }
            }
        }
    }

    return true;
}

/*! \brief Save matrix to file in full precision format. */
// bisheng 2026 - Provenance sidecar for the cached electrode matrix.
//
// MatrixA.dat / inv_MatrixA.dat are loaded verbatim if present, in preference to
// being rebuilt (force.cpp).  That is the right default -- the build costs
// minutes at N ~ 2000 -- but it means a matrix inherited from another run, or a
// copied/symlinked working directory, is accepted without complaint.  The matrix
// depends on the electrode geometry, eta, the Ewald splitting parameter (itself
// set by rcoulomb and ewald_rtol), the box and the 3DC flag; none of that is
// recoverable from the bare numbers.
//
// For a neutral two-electrode cell a beta mismatch is benign: Ewald theory makes
// the total independent of the splitting, so the difference is a near-uniform
// offset, and a uniform offset contributes c*sum(q) to every row, which the
// sum(q) = 0 constraint annihilates.  That cancellation does NOT occur for a
// single electrode or a fixed total charge, where sum(q) = Q != 0.  Those are
// exactly the cases this file protects.
static void save_matrix_meta(int          nEleAtoms,
                             const rvec*  x,
                             const matrix box,
                             const real*  eta_per_atom,
                             real         ewaldcoeff_q,
                             real         cutoff,
                             real         ewald_rtol,
                             bool         use3dc,
                             bool         tf_on,
                             double       tf_ltf_sum,
                             double       tf_vol_sum)
{
    FILE* f = fopen("MatrixA.meta", "w");
    if (!f)
    {
        fprintf(stderr, "Warning: Could not open MatrixA.meta for writing.\n");
        return;
    }
    double eta_sum = 0.0, geom = 0.0;
    for (int i = 0; i < nEleAtoms; ++i)
    {
        eta_sum += static_cast<double>(eta_per_atom[i]);
        // order-sensitive reduction: catches a different electrode geometry or ordering
        geom += (i + 1.0) * (x[i][XX] + 2.0 * x[i][YY] + 3.0 * x[i][ZZ]);
    }
    fprintf(f, "# CPM electrode matrix provenance -- checked on load (force.cpp)\n");
    fprintf(f, "nEleAtoms %d\n", nEleAtoms);
    fprintf(f, "eta_sum %.12g\n", eta_sum);
    fprintf(f, "ewaldcoeff_q %.12g\n", static_cast<double>(ewaldcoeff_q));
    fprintf(f, "cutoff %.12g\n", static_cast<double>(cutoff));
    fprintf(f, "ewald_rtol %.12g\n", static_cast<double>(ewald_rtol));
    fprintf(f, "box %.12g %.12g %.12g\n",
            static_cast<double>(box[XX][XX]), static_cast<double>(box[YY][YY]),
            static_cast<double>(box[ZZ][ZZ]));
    fprintf(f, "use3dc %d\n", use3dc ? 1 : 0);
    fprintf(f, "geom_checksum %.12g\n", geom);
    // Thomas-Fermi: a diagonal term, NOT annihilated by sum(q)=0, so any mismatch is fatal on load
    fprintf(f, "thomas_fermi %d\n", tf_on ? 1 : 0);
    fprintf(f, "tf_ltf_sum %.12g\n", tf_ltf_sum);
    fprintf(f, "tf_vol_sum %.12g\n", tf_vol_sum);
    fclose(f);
}

static void save_matrix_to_file(const char* filename, int n, const double* matrix)
{
    FILE* f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Warning: Could not open %s for writing.\n", filename);
        return;
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (j > 0) fprintf(f, " ");
            fprintf(f, "%.17e", matrix[i * n + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

// bisheng 2025 - Optimized k-space calculation using LAMMPS approach
// bisheng 2025 - Modified to support per-atom eta
void compute_cpm_matrix(int          nEleAtoms,
                        const rvec*  x,
                        const matrix box,
                        const real*  eta_per_atom,
                        real         ewaldcoeff_q,
                        real         cutoff,
                        int          nkx,
                        int          nky,
                        int          nkz,
                        real         ewald_rtol,
                        bool         use3dc,
                        double*      MatrixA,
                        double*      inv_MatrixA,
                        bool         saveToFile,
                        const real*  tf_diag,
                        double       tf_ltf_sum,
                        double       tf_vol_sum)
{
    printf("\n=== Computing CPM electrode matrix internally ===\n");
    printf("  Number of electrode atoms: %d\n", nEleAtoms);

    // Compute eta statistics for printing
    real eta_min = eta_per_atom[0];
    real eta_max = eta_per_atom[0];
    real eta_sum = 0.0;
    for (int i = 0; i < nEleAtoms; i++)
    {
        eta_sum += eta_per_atom[i];
        if (eta_per_atom[i] < eta_min) eta_min = eta_per_atom[i];
        if (eta_per_atom[i] > eta_max) eta_max = eta_per_atom[i];
    }
    real eta_avg = eta_sum / nEleAtoms;

    if (eta_min == eta_max)
    {
        printf("  eta (nm^-1): %.6f (uniform)\n", eta_min);
    }
    else
    {
        printf("  eta (nm^-1): %.6f - %.6f (avg: %.6f, per-atom)\n", eta_min, eta_max, eta_avg);
    }

    printf("  ewaldcoeff_q (nm^-1): %.6f\n", ewaldcoeff_q);
    printf("  cutoff (nm): %.3f\n", cutoff);
    printf("  k-space grid: %d x %d x %d\n", nkx, nky, nkz);
    printf("  ewald_rtol: %.2e\n", ewald_rtol);
    printf("  3DC correction: %s\n", use3dc ? "yes" : "no");

    const int matrixSize = nEleAtoms * nEleAtoms;

    // Get box dimensions (in nm, GROMACS native units)
    double boxSize[3] = { box[XX][XX], box[YY][YY], box[ZZ][ZZ] };
    double volume = boxSize[XX] * boxSize[YY] * boxSize[ZZ];
    printf("  Box: %.3f x %.3f x %.3f nm, Volume: %.3f nm^3\n",
           boxSize[XX], boxSize[YY], boxSize[ZZ], volume);

    // Allocate temporary arrays
    std::vector<double> rReal(matrixSize, 0.0);
    std::vector<double> rSelf(matrixSize, 0.0);
    std::vector<double> rSlab(matrixSize, 0.0);
    std::vector<double> rKspace(matrixSize, 0.0);

    // Compute each contribution (all in nm units)
    printf("  Computing real-space contribution...\n");
    calc_matrix_real(nEleAtoms, x, boxSize, eta_per_atom, ewaldcoeff_q, cutoff, !use3dc, rReal.data());

    printf("  Computing self-interaction contribution...\n");
    calc_matrix_self(nEleAtoms, eta_per_atom, ewaldcoeff_q, rSelf.data());

    if (use3dc)
    {
        printf("  Computing 3DC slab correction...\n");
        calc_matrix_slab(nEleAtoms, x, volume, boxSize[ZZ], rSlab.data());
    }

    // bisheng 2025 - K-space method selection:
    // Default: Optimized Ewald (LAMMPS-style, fast and reliable)
    // GMX_CPM_DIRECT_KSPACE=1: Use slow direct Ewald (for debugging/validation)
    const char* envDirect = std::getenv("GMX_CPM_DIRECT_KSPACE");
    bool useDirect = (envDirect != nullptr && std::string(envDirect) == "1");

    if (useDirect)
    {
        printf("  Computing k-space contribution (direct method)...\n");
        calc_matrix_kspace_direct(nEleAtoms, x, boxSize, volume, ewaldcoeff_q, nkx, nky, nkz, rKspace.data());
    }
    else
    {
        printf("  Computing k-space contribution (optimized Ewald)...\n");
        calc_matrix_kspace_optimized(nEleAtoms, x, boxSize, volume, ewaldcoeff_q, nkx, nky, nkz,
                                     static_cast<double>(ewald_rtol), rKspace.data());
    }

    // Sum all contributions
    #pragma omp parallel for
    for (int i = 0; i < matrixSize; ++i)
    {
        MatrixA[i] = (rReal[i] + rSelf[i] + rSlab[i] + rKspace[i]); // in unit of 1/nm
    }

    // bisheng 2026 - Thomas-Fermi: diagonal 4*pi*l_TF^2/V_i (1/nm), the electron-gas
    // kinetic-energy penalty on each atom's charge (MetalWalls thomas_fermi.f90:109)
    if (tf_diag != nullptr)
    {
        double tf_min = tf_diag[0], tf_max = tf_diag[0];
        for (int i = 0; i < nEleAtoms; ++i)
        {
            MatrixA[i * nEleAtoms + i] += static_cast<double>(tf_diag[i]);
            tf_min = std::min(tf_min, static_cast<double>(tf_diag[i]));
            tf_max = std::max(tf_max, static_cast<double>(tf_diag[i]));
        }
        printf("  Thomas-Fermi diagonal added: 4*pi*l_TF^2/V = %.6g .. %.6g 1/nm\n", tf_min, tf_max);
    }

    // Invert the matrix
    printf("  Inverting matrix...\n");
    if (!invert_matrix_gauss_jordan(nEleAtoms, MatrixA, inv_MatrixA))
    {
        gmx_fatal(FARGS, "Failed to invert CPM electrode matrix - matrix may be singular");
    }

    // Save to files if requested
    if (saveToFile)
    {
        printf("  Saving MatrixA.dat...\n");
        save_matrix_to_file("MatrixA.dat", nEleAtoms, MatrixA);
        printf("  Saving inv_MatrixA.dat...\n");
        save_matrix_to_file("inv_MatrixA.dat", nEleAtoms, inv_MatrixA);
        printf("  Saving MatrixA.meta (provenance for the load-time check)...\n");
        save_matrix_meta(nEleAtoms, x, box, eta_per_atom, ewaldcoeff_q, cutoff,
                         ewald_rtol, use3dc, tf_diag != nullptr, tf_ltf_sum, tf_vol_sum);

        // Also save individual components for debugging
        printf("  Saving individual component matrices for debugging...\n");
        save_matrix_to_file("MatrixA_sr.dat", nEleAtoms, rReal.data());
        save_matrix_to_file("MatrixA_self.dat", nEleAtoms, rSelf.data());
        save_matrix_to_file("MatrixA_3dc.dat", nEleAtoms, rSlab.data());
        save_matrix_to_file("MatrixA_kspace.dat", nEleAtoms, rKspace.data());
    }

    printf("=== CPM matrix computation complete ===\n\n");
}
