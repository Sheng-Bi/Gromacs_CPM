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
 * \brief Implementation of CpmPotentialControl for time-dependent
 * electrode potential/charge control using exprtk expression parser.
 *
 * \author Bisheng
 * \ingroup module_ewald
 */

#include "gmxpre.h"

#include "cpm_potential_control.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Disable some warnings for the exprtk header
#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-parameter"
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    pragma GCC diagnostic ignored "-Wsign-compare"
#    pragma GCC diagnostic ignored "-Wfloat-conversion"
#    pragma GCC diagnostic ignored "-Wconversion"
#endif

#include "external/exprtk/exprtk.hpp"

#if defined(__GNUC__) || defined(__clang__)
#    pragma GCC diagnostic pop
#endif

#include "gromacs/utility/fatalerror.h"

namespace gmx
{

/*! \internal
 * \brief Implementation class for CpmPotentialControl (PIMPL pattern)
 *
 * Using PIMPL to hide the exprtk types from the header file, which would
 * significantly increase compilation time for any file that includes the header.
 */
class CpmPotentialControl::Impl
{
public:
    //! Time variable for expression evaluation
    double timeVariable_ = 0.0;

    //! User-defined parameters
    std::map<std::string, double> parameters_;

    //! Storage for parameter values (needed for exprtk symbol table)
    std::vector<double> parameterValues_;

    //! The original expression string
    std::string expressionString_;

    //! Flag indicating if expression is ready
    bool isEnabled_ = false;

    //! bisheng 2026 - voltage-cut-off galvanostatic protocol ("mode = cutoff")
    CpmPotentialControl::CutoffProtocol cutoff_;

    //! exprtk symbol table
    exprtk::symbol_table<double> symbolTable_;

    //! exprtk expression object
    exprtk::expression<double> expression_;

    //! exprtk parser
    exprtk::parser<double> parser_;
};

CpmPotentialControl::CpmPotentialControl() : impl_(std::make_unique<Impl>()) {}

CpmPotentialControl::~CpmPotentialControl() = default;

CpmPotentialControl::CpmPotentialControl(CpmPotentialControl&& other) noexcept = default;

CpmPotentialControl& CpmPotentialControl::operator=(CpmPotentialControl&& other) noexcept = default;

/*! \brief Helper function to trim whitespace from string */
static std::string trim(const std::string& str)
{
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Note: startsWith is already provided by gromacs/utility/stringutil.h (included via fatalerror.h)

bool CpmPotentialControl::loadFromFile(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        gmx_fatal(FARGS, "Cannot open potential control file: %s", filename.c_str());
        return false;
    }

    std::vector<std::string>       nonCommentLines;
    std::map<std::string, double>& params = impl_->parameters_;

    std::string line;
    while (std::getline(file, line))
    {
        // Trim the line
        line = trim(line);

        // Skip empty lines and comments
        if (line.empty() || startsWith(line, "#"))
        {
            continue;
        }

        // Check if this is a parameter definition (contains '=')
        size_t eqPos = line.find('=');
        if (eqPos != std::string::npos)
        {
            std::string name  = trim(line.substr(0, eqPos));
            std::string value = trim(line.substr(eqPos + 1));

            // Validate parameter name (must start with letter, contain only alphanumeric and _)
            if (name.empty() || !std::isalpha(static_cast<unsigned char>(name[0])))
            {
                gmx_fatal(FARGS, "Invalid parameter name in %s: %s", filename.c_str(), name.c_str());
                return false;
            }

            // bisheng 2026 - "mode = cutoff" selects the voltage-cut-off cycler;
            // it is a keyword, not a numeric parameter.
            if (name == "mode")
            {
                if (value == "cutoff")
                {
                    impl_->cutoff_.enabled = true;
                }
                else if (value != "expression")
                {
                    gmx_fatal(FARGS, "Unknown mode in %s: %s (use cutoff or expression)",
                              filename.c_str(), value.c_str());
                    return false;
                }
                continue;
            }

            // Parse the value
            try
            {
                double val  = std::stod(value);
                params[name] = val;
            }
            catch (const std::exception&)
            {
                gmx_fatal(FARGS, "Cannot parse parameter value in %s: %s = %s", filename.c_str(),
                          name.c_str(), value.c_str());
                return false;
            }
        }
        else
        {
            // This is a non-parameter line (likely the expression)
            nonCommentLines.push_back(line);
        }
    }
    file.close();

    // bisheng 2026 - cut-off cycler: no expression; the controller integrates
    // Q_set(t) itself from I, Umax, Umin (see pme.cpp).
    if (impl_->cutoff_.enabled)
    {
        CutoffProtocol& c = impl_->cutoff_;
        auto need = [&](const char* key, double* out) {
            const auto it = params.find(key);
            if (it == params.end())
            {
                gmx_fatal(FARGS, "mode = cutoff in %s needs '%s = value'", filename.c_str(), key);
            }
            *out = it->second;
        };
        need("I", &c.current);
        need("Umax", &c.uMax);
        need("Umin", &c.uMin);
        c.current = std::fabs(c.current);
        if (params.count("Q0"))
        {
            c.q0 = params["Q0"];
        }
        if (params.count("direction"))
        {
            c.direction = (params["direction"] < 0) ? -1 : +1;
        }
        if (params.count("Utau"))
        {
            c.uTau = std::fabs(params["Utau"]);
        }
        if (c.current <= 0.0 || !(c.uMax > c.uMin))
        {
            gmx_fatal(FARGS, "mode = cutoff in %s: need I > 0 and Umax > Umin (I = %g, Umax = %g, Umin = %g)",
                      filename.c_str(), c.current, c.uMax, c.uMin);
        }
        if (!nonCommentLines.empty())
        {
            fprintf(stderr, "CPM: mode = cutoff ignores the expression line '%s'\n",
                    nonCommentLines.back().c_str());
        }
        impl_->expressionString_ = "cutoff cycler";
        impl_->isEnabled_        = true;
        fprintf(stderr,
                "\nCPM Time-dependent control loaded from %s\n"
                "  Mode: galvanostatic cycling with voltage cut-offs\n"
                "  |I| = %g e/ps, Umax = %g V, Umin = %g V, Q0 = %g e, initial direction %+d, "
                "voltage filter Utau = %g ps%s\n\n",
                filename.c_str(), c.current, c.uMax, c.uMin, c.q0, c.direction, c.uTau,
                c.uTau > 0 ? "" : " (raw per-step voltage)");
        return true;
    }

    // The last non-comment line is the expression
    if (nonCommentLines.empty())
    {
        gmx_fatal(FARGS, "No expression found in potential control file: %s", filename.c_str());
        return false;
    }

    impl_->expressionString_ = nonCommentLines.back();

    // Set up the symbol table
    impl_->symbolTable_.add_variable("t", impl_->timeVariable_);

    // Add mathematical constants
    impl_->symbolTable_.add_constants();

    // Add user-defined parameters
    impl_->parameterValues_.clear();
    impl_->parameterValues_.reserve(params.size());
    for (auto& kv : params)
    {
        impl_->parameterValues_.push_back(kv.second);
    }

    // Now add parameters to symbol table
    size_t idx = 0;
    for (auto& kv : params)
    {
        impl_->symbolTable_.add_variable(kv.first, impl_->parameterValues_[idx]);
        ++idx;
    }

    // Register the symbol table with the expression
    impl_->expression_.register_symbol_table(impl_->symbolTable_);

    // Compile the expression
    if (!impl_->parser_.compile(impl_->expressionString_, impl_->expression_))
    {
        gmx_fatal(FARGS, "Failed to compile expression in %s: %s\nError: %s", filename.c_str(),
                  impl_->expressionString_.c_str(), impl_->parser_.error().c_str());
        return false;
    }

    impl_->isEnabled_ = true;

    // Print info about loaded expression
    fprintf(stderr, "\nCPM Time-dependent control loaded from %s\n", filename.c_str());
    fprintf(stderr, "  Parameters:\n");
    for (const auto& kv : params)
    {
        fprintf(stderr, "    %s = %g\n", kv.first.c_str(), kv.second);
    }
    fprintf(stderr, "  Expression: %s\n", impl_->expressionString_.c_str());

    // Test evaluate at t=0
    real testVal = evaluate(0.0);
    fprintf(stderr, "  Value at t=0: %g\n\n", testVal);

    return true;
}

real CpmPotentialControl::evaluate(real t)
{
    if (!impl_->isEnabled_)
    {
        return 0.0;
    }

    impl_->timeVariable_ = static_cast<double>(t);
    return static_cast<real>(impl_->expression_.value());
}

bool CpmPotentialControl::getParameter(const std::string& name, double* value) const
{
    const auto it = impl_->parameters_.find(name);
    if (it == impl_->parameters_.end())
    {
        return false;
    }
    *value = it->second;
    return true;
}

bool CpmPotentialControl::isEnabled() const
{
    return impl_->isEnabled_;
}

const CpmPotentialControl::CutoffProtocol& CpmPotentialControl::cutoff() const
{
    return impl_->cutoff_;
}

const std::string& CpmPotentialControl::getExpression() const
{
    return impl_->expressionString_;
}

} // namespace gmx
