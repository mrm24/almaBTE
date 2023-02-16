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

/// @file
/// Definitions corresponding to superlattices.hpp.

#include <boost/math/special_functions/pow.hpp>
#include <Eigen/Dense>
#include <exceptions.hpp>
#include <periodic_table.hpp>
#include <isotopic_scattering.hpp>
#include <superlattices.hpp>
#include <dynamical_matrix.hpp>
#include <qpoint_grid.hpp>
#include <green1d.hpp>

namespace alma {
void Superlattice_structure::fill_factors(const Crystal_structure& struct1,
                                          const Crystal_structure& struct2) {
    std::size_t natoms = struct1.get_natoms();
    double x1{1. - this->average};
    double x2{this->average};

    this->gfactors.resize(natoms, this->nlayers);
    this->vfactors.resize(natoms * nlayers);

    for (std::size_t i = 0; i < natoms; ++i) {
        double m1{struct1.get_mass(i)};
        double m2{struct2.get_mass(i)};
        double mbar{x1 * m1 + x2 * m2};

        for (std::size_t j = 0; j < nlayers; ++j) {
            double y1{1. - this->profile(j)};
            double y2{this->profile(j)};
            this->vfactors(natoms * j + i) =
                ((y1 - x1) * m1 + (y2 - x2) * m2) / mbar;
            // Note that disorder is considered only in the
            // directions normal to the superlattice; hence,
            // the reference mass for the gfactor is the
            // local average.
            double mlayer{y1 * m1 + y2 * m2};
            this->gfactors(i, j) = (y1 * boost::math::pow<2>(m1 - mlayer) +
                                    y2 * boost::math::pow<2>(m2 - mlayer)) /
                                   boost::math::pow<2>(mlayer);
        }
    }
}


Eigen::ArrayXXd Superlattice_structure::calc_w0_medium(
    const Gamma_grid& grid,
    const boost::mpi::communicator& comm,
    double scalebroad) const {
    std::size_t nqpoints = grid.nqpoints;
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    // Find all allowed two-phonon processes.
    auto twoph_processes = find_allowed_twoph(grid, comm, scalebroad);
    Eigen::ArrayXXd nruter{Eigen::ArrayXXd::Zero(nmodes, nqpoints)};

    // And average the scattering rates for each layer.
    for (std::size_t ilayer = 0; ilayer < nlayers; ++ilayer)
        nruter += calc_w0_twoph(*(this->vc_struct),
                                this->gfactors.col(ilayer),
                                grid,
                                twoph_processes,
                                comm);
    nruter /= static_cast<double>(this->nlayers);
    return nruter;
}


Eigen::ArrayXXd Superlattice_structure::calc_w0_barriers(
    const Gamma_grid& grid,
    const Dynamical_matrix_builder& factory,
    randutils::mt19937_rng& rng,
    const boost::mpi::communicator& comm,
    std::size_t nqline) const {
    std::size_t nqpoints = grid.nqpoints;
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    auto nprocs = comm.size();
    auto my_id = comm.rank();
    // Vectors used to convert between wave numbers in the
    // [0, 2 * pi) interval and distances.
    Eigen::VectorXi rnormal(alma::reduce_integers(this->normal));
    Eigen::Vector3d cnormal{grid.rlattvec * rnormal.cast<double>()};
    Eigen::Vector3d u{cnormal / cnormal.norm()};
    // For each q point.
    auto limits = my_jobs(nqpoints, nprocs, my_id);
    Eigen::ArrayXXd my_nruter{Eigen::ArrayXXd::Zero(nmodes, nqpoints)};

    unsigned int previous_percentage = 0;

    for (std::size_t iq = limits[0]; iq < limits[1]; ++iq) {
        unsigned int current_percentage = static_cast<unsigned int>(
            100. * (iq - limits[0] + 1) / (limits[1] - limits[0]));

        if ((my_id == 0) && (current_percentage > previous_percentage)) {
            unsigned int nchars =
                static_cast<unsigned int>(0.72 * current_percentage);

            std::cout << "[";

            for (auto i = 0u; i < nchars; ++i) {
                std::cout << "-";
            }

            std::cout << ">";

            for (auto i = nchars; i < 72; ++i) {
                std::cout << " ";
            }
            std::cout << "] " << current_percentage << "%\r";
            std::cout.flush();
            previous_percentage = current_percentage;
        }

        // Recast the problem as 1-dimensional.
        Eigen::Vector3d q0{grid.get_q(iq)};
        // Generate a random offset to avoid systematic noise
        // caused by the choice of origin.
        double deltaq = rng.uniform(0., 1.);
        Eigen::Vector3d origin = q0 - deltaq * cnormal;
        // Factory for Green's functions.
        Green1d_factory gff(
            *(this->vc_struct), origin, rnormal, nqline, factory);
        // Obtain the spectrum.
        auto spectrum = grid.get_spectrum_at_q(iq);

        // And use it to get the scattering rate for each mode.
        for (std::size_t im = 0; im < nmodes; ++im)
            if (!almost_equal(0., spectrum.omega(im)))
                my_nruter(im, iq) = gff.calc_w0_dmass(
                    2. * constants::pi * u.dot(q0) / cnormal.norm(),
                    spectrum.omega(im),
                    spectrum.wfs.col(im),
                    vfactors);
    }
    // Sum over processes so that the whole array is available
    // globally.
    Eigen::ArrayXXd nruter{Eigen::ArrayXXd::Zero(nmodes, nqpoints)};
    boost::mpi::all_reduce(comm,
                           my_nruter.data(),
                           my_nruter.size(),
                           nruter.data(),
                           std::plus<double>());
    return nruter;
}
} // namespace alma
