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
 * Declares functions for computing the CPM electrode matrix internally.
 *
 * \author Bisheng (2025)
 * \ingroup module_ewald
 */

// bisheng 2025 - Internal CPM matrix calculation header

#ifndef GMX_EWALD_CPM_MATRIX_H
#define GMX_EWALD_CPM_MATRIX_H

#include "gromacs/math/vectypes.h"
#include "gromacs/utility/real.h"

/*! \brief Compute the CPM electrode matrix internally.
 *
 * Computes the capacitance matrix A for the Constant Potential Method.
 * The matrix is computed as: A = rReal + rSelf + rSlab + rKspace
 *
 * All parameters are in GROMACS native units (nm).
 *
 * K-space is computed using an optimized LAMMPS-style Ewald summation with
 * precomputed cos/sin tables. Set GMX_CPM_DIRECT_KSPACE=1 to use the original
 * direct method for comparison.
 *
 * bisheng 2025 - Modified to support per-atom eta
 *
 * \param[in]  nEleAtoms     Number of electrode atoms (from userint1)
 * \param[in]  x             System coordinates, electrode atoms are first nEleAtoms
 * \param[in]  box           Simulation box (nm)
 * \param[in]  eta_per_atom  Per-atom Gaussian width parameter array (nm^-1), size nEleAtoms
 * \param[in]  ewaldcoeff_q  Ewald splitting parameter beta (nm^-1)
 * \param[in]  cutoff        Real-space cutoff (nm)
 * \param[in]  nkx           K-space grid dimension in x
 * \param[in]  nky           K-space grid dimension in y
 * \param[in]  nkz           K-space grid dimension in z
 * \param[in]  ewald_rtol    Ewald tolerance for k-space accuracy (from MDP)
 * \param[in]  use3dc        True if using 3DC slab correction
 * \param[out] MatrixA       Output matrix (nEleAtoms x nEleAtoms), must be pre-allocated
 * \param[out] inv_MatrixA   Output inverse matrix, must be pre-allocated
 * \param[in]  saveToFile    If true, save MatrixA.dat and inv_MatrixA.dat
 */
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
                        const real*  tf_diag = nullptr,   // Thomas-Fermi 4*pi*l_TF^2/V_i (1/nm), or nullptr
                        double       tf_ltf_sum = 0.0,    // provenance for MatrixA.meta
                        double       tf_vol_sum = 0.0);

#endif /* GMX_EWALD_CPM_MATRIX_H */
