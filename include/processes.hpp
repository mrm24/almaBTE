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
/// Detection and representation of allowed three-phonon processes.

#include <cstddef>
#include <cmath>
#include <array>
#include <vector>
#include <functional>
#include <boost/mpi.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>

// Forward declarations of elements documented later on.
// Note that serialization of std::vectors of objects without
// a default constructor is broken in Boost version 1.58.0. See:
// http://stackoverflow.com/a/30437359/85371
// https://svn.boost.org/trac/boost/ticket/11342
// This will prevent ALMA from compiling. We check for that
// specific version in our cmake configuration.
namespace alma {
class Threeph_process;
}
namespace boost {
namespace serialization {
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Threeph_process* t,
                                const unsigned int file_version);
}
} // namespace boost

namespace alma {
/// Process type (emission or absorption).
enum class threeph_type { emission = -1, absorption = 1 };
/// Representation of a three-phonon process.
class Threeph_process {
private:
    friend class boost::serialization::access;
    template <class Archive>
    friend void boost::serialization::save_construct_data(
        Archive& ar,
        const Threeph_process* t,
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
    /// Has the phase space of the process been computed yet?
    bool gaussian_computed;
    /// Gaussian factor of the process, coming from the
    /// regularized Dirac delta.
    double gaussian;
    /// Has the matrix element of the process been computed yet?
    bool vp2_computed;
    /// Modulus squared of the matrix element of the process.
    double vp2;
    /// Serialize non-const members of the class.
    ///
    /// See the boost::serialization documentation for details.
    template <class Archive>
    void serialize(Archive& ar, const unsigned int file_version) {
        ar & this->gaussian_computed;

        if (this->gaussian_computed)
            ar & this->gaussian;
        ar & this->vp2_computed;

        if (this->vp2_computed)
            ar & this->vp2;
    }


public:
    /// Equivalence class of the first phonon.
    const std::size_t c;
    /// q point indices of each of the three phonons involved.
    const std::array<std::size_t, 3> q;
    /// Mode indices of the three phonons involved.
    const std::array<std::size_t, 3> alpha;
    /// Type of process.
    const threeph_type type;
    /// Basic constructor.
    Threeph_process(std::size_t _c,
                    const std::array<std::size_t, 3>& _q,
                    const std::array<std::size_t, 3>& _alpha,
                    threeph_type _type,
                    double _domega,
                    double _sigma)
        : domega(_domega), sigma(_sigma), gaussian_computed(false),
          vp2_computed(false), c(_c), q(std::move(_q)),
          alpha(std::move(_alpha)), type(_type) {
    }


    /// Copy constructor.
    Threeph_process(const Threeph_process& original)
        : domega(original.domega), sigma(original.sigma),
          gaussian_computed(original.gaussian_computed),
          gaussian(original.gaussian), vp2_computed(original.vp2_computed),
          vp2(original.vp2), c(original.c), q(original.q),
          alpha(original.alpha), type(original.type) {
    }


    /// Compute and return the Gaussian factor of the process,
    /// i.e., the amplitude of the regularized delta.
    ///
    /// The result is cached. It can be used to obtain the phase
    /// space volume of three-phonon processes.
    /// @return the Gaussian factor of the process
    double compute_gaussian() {
        if (!this->gaussian_computed) {
            auto distr = boost::math::normal(0., this->sigma);
            this->gaussian = boost::math::pdf(distr, this->domega);
            this->gaussian_computed = true;
        }
        return this->gaussian;
    }


    /// Compute and return a weighted version of the Gaussian factor
    /// of the process, containing essentially the same ingredients as
    /// the scattering rate except for the matrix element and some
    /// constants.
    ///
    /// The result can be used to obtain the weighted phase
    /// space volume of three-phonon processes. The Gaussian factor
    /// is cached.
    /// @param[in] grid - regular grid with phonon spectrum
    /// @param[in] T - temperature in K
    /// @return the Bose-Einstein weighted Gaussian factor of the process at the
    /// given temperature.
    double compute_weighted_gaussian(const Gamma_grid& grid, double T);


    /// Compute, return and store the modulus squared of the matrix
    /// element of the process.
    ///
    /// @param[in] cell - description of the unit cell
    /// @param[in] grid - regular grid with phonon spectrum
    /// @param[in] thirdorder - third-order ifcs
    /// @return the modulus squared of the matrix element
    /// of the process
    double compute_vp2(const Crystal_structure& cell,
                       const Gamma_grid& grid,
                       const std::vector<Thirdorder_ifcs>& thirdorder);

    /// Return the modulus squared of the matrix element of the
    /// process or throw an exception if it has not been calculated.
    ///
    /// @return the modulus squared of the matrix element
    /// of the process
    inline double get_vp2() const {
        if (!this->vp2_computed)
            throw exception("vp2 is not available");
        return this->vp2;
    }


    /// @return the value of vp2_computed.
    inline bool is_vp2_computed() const {
        return this->vp2_computed;
    }


    /// Get the "partial scattering rate" Gamma for this process.
    ///
    /// vp2 must have been precomputed.
    /// @param[in] grid - regular grid with phonon spectrum
    /// @param[in] T - temperature in K
    /// @return Gamma, the partial scattering rate
    double compute_gamma(const Gamma_grid& grid, double T);

    /// Get the reduced "partial scattering rate", meaning Gamma
    /// without the scaling factor involving the BE occupations.
    ///
    /// vp2 must have been precomputed.
    /// @param[in] grid - regular grid with phonon spectrum
    /// @param[in] T - temperature in K
    /// @return Gamma, the partial scattering rate
    double compute_gamma_reduced(const Gamma_grid& grid, double T);

    /// Get the coefficients of the three contributions to the
    /// linearized scattering operator from this process.
    ///
    /// vp2 must have been precomputed.
    /// @param[in] grid - regular grid with phonon spectrum
    /// @param[in] n0 - precomputed Bose-Einstein distribution
    /// @return an array with the three coefficients
    Eigen::ArrayXd compute_collision(const Gamma_grid& grid,
                                     const Eigen::ArrayXXd& n0);
};
/// Look for allowed three-phonon processes in a regular grid.
///
/// Iterate over part of the irreducible q points in the grid
/// (trying to evenly split the equivalence classes over processes)
/// and look for allowed three-phonon processes involving one
/// phonon from that part and two other phonons from anywhere
/// in the grid.
/// @param[in] grid - a regular grid containing Gamma
/// @param[in] communicator - MPI communicator to use
/// @param[in] scalebroad - factor modulating all the broadenings
/// @return a vector of Threeph_process objects
std::vector<Threeph_process> find_allowed_threeph(
    const Gamma_grid& grid,
    const boost::mpi::communicator& communicator,
    double scalebroad = 1.0);

/// Compute and store the three-phonon contribution to the RTA
/// scattering rates for all vibrational modes on a grid.
///
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] processes - a vector of allowed
/// three-phonon processes
/// @param[in] T - the temperature in K
/// @param[in] comm - an mpi communicator
Eigen::ArrayXXd calc_w0_threeph(const alma::Gamma_grid& grid,
                                std::vector<alma::Threeph_process>& processes,
                                double T,
                                const boost::mpi::communicator& comm);

/// Compute and store the three-phonon contribution to the RTA
/// scattering rates for all vibrational modes on a grid. Take into
/// account only those processes fullfilling a criterion.
///
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] processes - a vector of allowed
/// three-phonon processes
/// @param[in] T - the temperature in K
/// @param[in] filter - boolean function returning true if a process must
///                     be taken into account
/// @param[in] comm - an mpi communicator
Eigen::ArrayXXd calc_w0_threeph(
    const alma::Gamma_grid& grid,
    std::vector<alma::Threeph_process>& processes,
    double T,
    std::function<bool(const Threeph_process&)> filter,
    const boost::mpi::communicator& comm);
} // namespace alma

namespace boost {
namespace serialization {
/// Overload required to serialize the const members of
/// alma::Threeph_process.
///
/// See the boost::serialization documentation for details.
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Threeph_process* t,
                                const unsigned int file_version) {
    ar << t->c;
    ar << make_array(t->q.data(), t->q.size());
    ar << make_array(t->alpha.data(), t->alpha.size());
    ar << t->type;
    ar << t->domega;
    ar << t->sigma;
}


/// Overload required to deserialize the const members of
/// alma::Threeph_process by calling the non-default
/// constructor in place.
///
/// See the boost::serialization documentation for details.
template <class Archive>
inline void load_construct_data(Archive& ar,
                                alma::Threeph_process* t,
                                const unsigned int file_version) {
    std::size_t c;

    ar >> c;
    std::array<std::size_t, 3> q;
    ar >> make_array(q.data(), q.size());
    std::array<std::size_t, 3> alpha;
    ar >> make_array(alpha.data(), alpha.size());
    alma::threeph_type type;
    ar >> type;
    double domega;
    ar >> domega;
    double sigma;
    ar >> sigma;
    ::new (t) alma::Threeph_process(c, q, alpha, type, domega, sigma);
}
} // namespace serialization
} // namespace boost
