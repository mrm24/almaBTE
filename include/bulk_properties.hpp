// Copyright 2015-2018 The ALMA Project Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

#pragma once

/// @file
///
/// Code related to bulk properties such as the specific heat.

#include <constants.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>

namespace alma {
/// Compute the specific heat at constant volume.
///
/// @param[in] poscar - a description of the unit cell
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] T - temperature in K
/// @return  the volumetric specific heat
inline double calc_cv(const alma::Crystal_structure& poscar,
                      const alma::Gamma_grid& grid,
                      double T) {
    double nruter = 0.;
    double nqpoints = grid.nqpoints;
    double nmodes = grid.get_spectrum_at_q(0).omega.size();

    for (std::size_t iq = 0; iq < nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < nmodes; ++im)
            nruter += alma::bose_einstein_kernel(spectrum.omega(im), T);
    }
    nruter *= alma::constants::kB / nqpoints / poscar.V;
    return nruter;
}


/// Obtain the thermal conductivity in the relaxation
/// time approximation.
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[in] w - scattering rates for all modes in each of
/// the irreducible classes of q points in the grid.
/// @param[in] T - temperature in K
/// @return the thermal conductivity tensor in SI units
Eigen::MatrixXd calc_kappa(const alma::Crystal_structure& poscar,
                           const alma::Gamma_grid& grid,
			   const alma::Symmetry_operations& syms,
                           const Eigen::Ref<const Eigen::ArrayXXd>& w,
                           double T);

/// Obtain the coherence term to the thermal conductivity
/// See 10.1103/PhysRevX.12.041011 
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[in] w - scattering rates for all modes in each of
/// the irreducible classes of q points in the grid.
/// @param[in] T - temperature in K
/// @return the thermal conductivity tensor in SI units
Eigen::MatrixXd calc_kappa_coherence(const alma::Crystal_structure& poscar,
                                     const alma::Gamma_grid& grid,
                                     const alma::Symmetry_operations& syms,
                                     const Eigen::Ref<const Eigen::ArrayXXd>& w,
                                     double T);

/// Obtain the small-grain thermal conductivity tensor.
///
/// The small-grain thermal conductivity is defined as the value of the
/// thermal conductivity tensor over the mean free path when the mean
// free
/// path is uniform across all modes.
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[in] T - temperature in K
/// @return the small-grain thermal conductivity tensor [W / (m K nm)]
Eigen::MatrixXd calc_kappa_sg(const alma::Crystal_structure& poscar,
                              const alma::Gamma_grid& grid,
                              const alma::Symmetry_operations& syms,
                              double T);

/// Obtain the thermal conductivity along a particular direction in the
/// relaxation time approximation.
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] w0 - scattering rates for all modes in each of
/// the irreducible classes of q points in the grid.
/// @param[in] T - temperature in K
/// @param[in] direction - 1D thermal transport direction in
/// Cartesian coordinates.
/// @return the thermal conductivity tensor in SI units
double calc_kappa_1d(const alma::Crystal_structure& poscar,
                     const alma::Gamma_grid& grid,
                     const Eigen::Ref<const Eigen::ArrayXXd>& w,
                     double T,
                     const Eigen::Ref<const Eigen::Vector3d>& direction);

/// Obtain the phase space and its weighted version
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] T - temperature in K
/// @param[in] processes  - three-phonon procesess
/// @param[out] P3plus    - contains the mode-resolve absorption phase space in ps/rad
/// @param[out] P3minus   - contains the mode-resolve emission phase space in ps/rad
/// @param[out] WP3plus   - contains the mode-resolve weighted absorption phase space in ps^{4}/rad^{4}
/// @param[out] WP3minus  - contains the mode-resolve weighted emission phase space in ps^{4}/rad^{4} 
/// @param[in]  world     - the mpi communicator
/// @return pair containing the total phase space (ps/rad) and the total weighted phase space (ps^{4}/rad^{4}) respectively
std::pair<double,double> calc_phase_space(const alma::Crystal_structure& poscar,
                                          const alma::Gamma_grid& grid,
                                          const double T,
                                          std::vector<alma::Threeph_process>& processes,
                                          Eigen::MatrixXd& P3plus,
                                          Eigen::MatrixXd& P3minus,
                                          Eigen::MatrixXd& WP3plus,
                                          Eigen::MatrixXd& WP3minus,
                                          const boost::mpi::communicator& world);

/// Obtain the averaged three-phonon matrix element
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] T - temperature in K
/// @param[in] processes  - three-phonon procesess
/// @param[out] P3plus    - contains the mode-resolve absorption phase space in ps/rad
/// @param[out] P3minus   - contains the mode-resolve emission phase space in ps/rad
/// @param[out] WP3plus   - contains the mode-resolve weighted absorption phase space in ps^{4}/rad^{4}
/// @param[out] WP3minus  - contains the mode-resolve weighted emission phase space in ps^{4}/rad^{4}
/// @param[in]  world     - the mpi communicator
/// @return pair containing the total phase space (ps/rad) and the total weighted phase space (ps^{4}/rad^{4}) respectively
std::pair<double,double> calc_phase_space(const alma::Crystal_structure& poscar,
                                          const alma::Gamma_grid& grid,
                                          const double T,
                                          std::vector<alma::Threeph_process>& processes,
                                          Eigen::MatrixXd& P3plus,
                                          Eigen::MatrixXd& P3minus,
                                          Eigen::MatrixXd& WP3plus,
                                          Eigen::MatrixXd& WP3minus,
                                          const boost::mpi::communicator& world);

/// Returns the mean of the energy allowed three-phonon processes
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] processes  - three-phonon procesess
/// @param[out] vp2plus   - contains the mode-resolve mean of the allowed procesess
/// @param[in]  world     - the mpi communicator
/// @return the mean of the three-phonon matrix elements
double calc_anharmonicity(const alma::Crystal_structure& poscar,
                          const alma::Gamma_grid& grid,
                          std::vector<alma::Threeph_process>& processes,
                          Eigen::MatrixXd& vp2,
                          const boost::mpi::communicator& world);

} // namespace alma
