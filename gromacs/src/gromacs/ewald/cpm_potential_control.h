/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2024, by the GROMACS development team, led by
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
 * \brief This file declares the CpmPotentialControl class for time-dependent
 * electrode potential/charge control in the Constant Potential Method (CPM).
 *
 * \author Bisheng
 * \ingroup module_ewald
 */

#ifndef GMX_EWALD_CPM_POTENTIAL_CONTROL_H
#define GMX_EWALD_CPM_POTENTIAL_CONTROL_H

#include <memory>
#include <string>

#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/real.h"

namespace gmx
{

/*! \internal
 * \brief Class for evaluating time-dependent electrode potential or charge.
 *
 * This class uses the exprtk library to parse and evaluate mathematical
 * expressions that define how the applied potential (or charge) varies
 * with simulation time.
 *
 * File format (potential_control.dat):
 * \verbatim
 * # Comments start with #
 * # Parameter definitions
 * V0 = 4.0
 * w = 0.001
 * period = 100
 *
 * # Expression (last non-comment line)
 * # Available variable: t (simulation time in ps)
 * V0 * sin(w * t)
 * \endverbatim
 */
class CpmPotentialControl
{
public:
    /*! \brief Default constructor */
    CpmPotentialControl();

    /*! \brief Destructor */
    ~CpmPotentialControl();

    /*! \brief Move constructor */
    CpmPotentialControl(CpmPotentialControl&& other) noexcept;

    /*! \brief Move assignment operator */
    CpmPotentialControl& operator=(CpmPotentialControl&& other) noexcept;

    // Disable copy
    CpmPotentialControl(const CpmPotentialControl&)            = delete;
    CpmPotentialControl& operator=(const CpmPotentialControl&) = delete;

    /*! \brief Load control expression from file
     *
     * \param[in] filename  Path to the control file (e.g., "potential_control.dat")
     * \returns true if file was loaded and expression compiled successfully
     *
     * The file format supports:
     * - Comment lines starting with #
     * - Parameter definitions: name = value
     * - The last non-comment line is the expression
     * - Available variable: t (time in ps)
     */
    bool loadFromFile(const std::string& filename);

    /*! \brief Evaluate the expression at given time
     *
     * \param[in] t  Simulation time in picoseconds
     * \returns The evaluated potential (V) or charge (e) at time t
     */
    real evaluate(real t);

    /*! \brief Check if time-dependent control is enabled
     *
     * \returns true if an expression has been loaded successfully
     */
    bool isEnabled() const;

    /*! \brief Get the expression string for logging
     *
     * \returns The mathematical expression string
     */
    const std::string& getExpression() const;
    /*! \brief Look up a user-defined parameter of the control file by name.
     * bisheng 2026 - lets the solver read named constants such as Q2 (the
     * second electrode's charge target) from the same file as the expression.
     * \returns true and the value if the parameter was defined. */
    bool getParameter(const std::string& name, double* value) const;

    /*! \internal
     * \brief Galvanostatic cycling with voltage cut-offs (charge-constraint mode).
     *
     * bisheng 2026 - selected by a "mode = cutoff" line in potential_control.dat
     * instead of an expression. The controller then integrates the imposed
     * charge itself, Q_set(t+dt) = Q_set(t) + s*I*dt, and flips the sign s when
     * the realised cell voltage of the previous step reaches Umax (charging)
     * or Umin (discharging), like an experimental charge-discharge cycler.
     * \verbatim
     * mode = cutoff
     * I = 0.096        # |current|, e/ps
     * Umax = 2.0       # V, switch to discharge when U >= Umax
     * Umin = 0.0       # V, switch to charge when U <= Umin
     * Q0 = 0.0         # optional start charge, e
     * direction = +1   # optional initial direction (+1 charge, -1 discharge)
     * Utau = 0.5       # optional, ps: compare the cut-offs with an exponential
     *                  # moving average of U (time constant Utau) instead of the
     *                  # raw per-step voltage, which fluctuates by ~0.3 V
     * \endverbatim
     */
    struct CutoffProtocol
    {
        bool   enabled   = false;
        double current   = 0.0;  //!< |I| in e/ps
        double uMax      = 0.0;  //!< upper cut-off, V
        double uMin      = 0.0;  //!< lower cut-off, V
        double q0        = 0.0;  //!< start charge, e
        int    direction = +1;   //!< initial direction
        double uTau      = 0.0;  //!< filter time constant, ps (0 = raw voltage)
    };

    /*! \brief The cut-off protocol (enabled == false in expression mode). */
    const CutoffProtocol& cutoff() const;

private:
    //! Forward declaration of implementation class (PIMPL pattern)
    class Impl;
    //! Pointer to implementation
    std::unique_ptr<Impl> impl_;
};

} // namespace gmx

#endif // GMX_EWALD_CPM_POTENTIAL_CONTROL_H
