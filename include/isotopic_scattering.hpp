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
/// Code implementing isotopic scattering according to Tamura's
/// formula:
/// S.-I. Tamura, Isotope scattering of dispersive phonons in Ge,
///    Phys. Rev. B 27 (1983) 858–866
/// A. Kundu, N. Mingo, D.A. Broido, D.A. Stewart, Role of light and
///    heavy embedded nanoparticles on the thermal conductivity of SiGe
///    alloys, Phys. Rev. B 84 (2011) 125426.
/// The same method is used for alloys in the virtual crystal
/// approximation.
/// The classes and functions in this file are modelled after those
/// declared in processes.hpp. There are, however, some differences.
/// The most important aspects are:
/// - Conservation of momentum is not enforced for two-phonon
/// processes.
/// - The Gaussian factor of each process is computed when the object
/// is built.
/// - The matrix element of each process is not stored in the object.

#include <cstddef>
#include <cmath>
#include <array>
#include <vector>
#include <boost/mpi.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/serialization/array.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>
#include <boost/math/distributions/normal.hpp>

// Forward declarations of elements documented later on.
// Note that serialization of std::vectors of objects without
// a default constructor is broken in Boost version 1.58.0. See:
// http://stackoverflow.com/a/30437359/85371
// https://svn.boost.org/trac/boost/ticket/11342
// This will prevent ALMA from compiling. We check for that
// specific version in our cmake configuration.
namespace alma {
class Twoph_process;
}
namespace boost {
namespace serialization {
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Twoph_process* t,
                                const unsigned int file_version);
}
} // namespace boost

namespace alma {
/// Representation of a elastic two-phonon process.
class Twoph_process {
private:
    friend class boost::serialization::access;
    template <class Archive>
    friend void boost::serialization::save_construct_data(
        Archive& ar,
        const Twoph_process* t,
        const unsigned int file_version);

    friend void save_bulk_hdf5(const char* filename,
                               const std::string& description,
                               const Crystal_structure& cell,
                               const Symmetry_operations& symmetries,
                               const Gamma_grid& grid,
                               const std::vector<Threeph_process>& processes,
                               const boost::mpi::communicator& comm);

    friend std::tuple<std::string,
                      std::unique_ptr<Crystal_structure>,
                      std::unique_ptr<Symmetry_operations>,
                      std::unique_ptr<Gamma_grid>,
                      std::unique_ptr<std::vector<Threeph_process>>>
    load_bulk_hdf5(const char* filename, const boost::mpi::communicator& comm);

    /// Deviation from the conservation of energy.
    const double domega;
    /// Standard deviation of the Gaussian.
    const double sigma;
    /// Gaussian factor of the process, coming from the
    /// regularized Dirac delta.
    const double gaussian;

public:
    /// Equivalence class of the first phonon.
    const std::size_t c;
    /// q point indices of each of the two phonons involved.
    const std::array<std::size_t, 2> q;
    /// Mode indices of the two phonons involved.
    const std::array<std::size_t, 2> alpha;
    /// Basic constructor.
    Twoph_process(std::size_t _c,
                  const std::array<std::size_t, 2>& _q,
                  const std::array<std::size_t, 2>& _alpha,
                  double _domega,
                  double _sigma)
        : domega(_domega), sigma(_sigma),
          gaussian(boost::math::pdf(boost::math::normal(0., sigma), domega)),
          c(_c), q(std::move(_q)), alpha(std::move(_alpha)) {
    }


    /// Copy constructor.
    Twoph_process(const Twoph_process& original)
        : domega(original.domega), sigma(original.sigma),
          gaussian(original.gaussian), c(original.c), q(original.q),
          alpha(original.alpha) {
    }


    /// Explicit constructor used when deserializing objects of
    /// this class. Save time by avoiding the calculatio of
    /// this->gaussian.
    Twoph_process(std::size_t _c,
                  const std::array<std::size_t, 2>& _q,
                  const std::array<std::size_t, 2>& _alpha,
                  double _domega,
                  double _sigma,
                  double _gaussian)
        : domega(_domega), sigma(_sigma), gaussian(_gaussian), c(_c),
          q(std::move(_q)), alpha(std::move(_alpha)) {
    }


    /// Get the "partial scattering rate" Gamma for this process.
    ///
    /// @param[in] cell - description of the unit cell
    /// @param[in] grid - regular grid with phonon spectrum
    /// @return Gamma, the partial scattering rate
    double compute_gamma(const Crystal_structure& cell,
                         const Gamma_grid& grid) const;

    /// Get the "partial scattering rate" Gamma for this process.
    /// using a custom set of g factors.
    ///
    /// @param[in] cell - description of the unit cell
    /// @param[in] gfactors - Pearson deviation coefficient of the mass
    /// at each site.
    /// @param[in] grid - regular grid with phonon spectrum
    /// @return Gamma, the partial scattering rate
    double compute_gamma(const Crystal_structure& cell,
                         const Eigen::Ref<const Eigen::VectorXd>& gfactors,
                         const Gamma_grid& grid) const;
};
/// Look for allowed two-phonon processes in a regular grid.
///
/// Iterate over part of the irreducible q points in the grid
/// (trying to evenly split the equivalence classes over processes)
/// and look for allowed two-phonon processes involving one phonon
/// from that part and another phonon from anywhere in the grid.
/// @param[in] grid - a regular grid containing Gamma
/// @param[in] communicator - MPI communicator to use
/// @param[in] scalebroad - factor modulating all the broadenings
/// @return a vector of Twoph_process objects
std::vector<Twoph_process> find_allowed_twoph(
    const Gamma_grid& grid,
    const boost::mpi::communicator& communicator,
    double scalebroad = 1.0);

/// Compute and the two-phonon contribution to the RTA
/// scattering rates for all vibrational modes on a grid.
///
/// @param[in] cell - a description of the crystal structure
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] processes - a vector of allowed two-phonon processes
/// @param[in] comm - an mpi communicator
/// @return a set of scattering rates
Eigen::ArrayXXd calc_w0_twoph(const Crystal_structure& cell,
                              const alma::Gamma_grid& grid,
                              const std::vector<alma::Twoph_process>& processes,
                              const boost::mpi::communicator& comm);

/// Compute and the two-phonon contribution to the RTA
/// scattering rates for all vibrational modes on a grid, with a custom
/// set of g factors.
///
/// @param[in] cell - a description of the crystal structure
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] gfactors - Pearson deviation coefficient of the mass
/// at each site.
/// @param[in] processes - a vector of allowed two-phonon processes
/// @param[in] comm - an mpi communicator
/// @return a set of scattering rates
Eigen::ArrayXXd calc_w0_twoph(const Crystal_structure& cell,
                              const Eigen::Ref<const Eigen::VectorXd>& gfactors,
                              const alma::Gamma_grid& grid,
                              const std::vector<alma::Twoph_process>& processes,
                              const boost::mpi::communicator& comm);
} // namespace alma

namespace boost {
namespace serialization {
/// Overload required to serialize the const members of
/// alma::Twoph_process.
///
/// See the boost::serialization documentation for details.
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Twoph_process* t,
                                const unsigned int file_version) {
    ar << t->c;
    ar << make_array(t->q.data(), t->q.size());
    ar << make_array(t->alpha.data(), t->alpha.size());
    ar << t->domega;
    ar << t->sigma;
    ar << t->gaussian;
}


/// Overload required to deserialize the const members of
/// alma::Twoph_process by calling the non-default
/// constructor in place.
///
/// See the boost::serialization documentation for details.
template <class Archive>
inline void load_construct_data(Archive& ar,
                                alma::Twoph_process* t,
                                const unsigned int file_version) {
    std::size_t c;

    ar >> c;
    std::array<std::size_t, 2> q;
    ar >> make_array(q.data(), q.size());
    std::array<std::size_t, 2> alpha;
    ar >> make_array(alpha.data(), alpha.size());
    double domega;
    ar >> domega;
    double sigma;
    ar >> sigma;
    double gaussian;
    ar >> gaussian;
    ::new (t) alma::Twoph_process(c, q, alpha, domega, sigma, gaussian);
}
} // namespace serialization
} // namespace boost
