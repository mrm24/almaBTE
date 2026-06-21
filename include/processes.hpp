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
#include <msgpack.hpp>

namespace alma {

/// Three phonon process type (emission or absorption).
enum class threeph_type { emission = -1, absorption = 1 };

/// Four phonon process type
enum class fourph_type {
    recombination   = 0,
    redistribution  = 1,
    splitting       = 2
};

constexpr std::array<int,2> fourph_type_signs(fourph_type t) {
    switch (t) {
        case fourph_type::recombination:   return {+1,+1};
        case fourph_type::redistribution:  return {+1,-1};
        case fourph_type::splitting:  return {-1,-1};
    }

    throw std::runtime_error("Error(fourph_type_signs): should not be here");

    return {0,0}; 
}

/// Representation of a N-phonon process.
template<std::size_t N, class process_type>
class ph_process {
private:
    friend class boost::serialization::access;
    template <class Archive>
    friend void boost::serialization::save_construct_data(
        Archive& ar,
        const ph_process* t,
        const unsigned int file_version);

    friend struct msgpack::adaptor::pack<ph_process<4, fourph_type>>;
    friend struct msgpack::adaptor::convert<ph_process<4, fourph_type>>;

    friend void save_bulk_hdf5(const char* filename,
                               const std::string& description,
                               const Crystal_structure& cell,
                               const Symmetry_operations& symmetries,
                               const Gamma_grid& grid,
                               const std::vector<ph_process<3, threeph_type>>& processes_3ph,
                               const std::vector<ph_process<4, fourph_type>>& processes_4ph,
                               const boost::mpi::communicator& comm);
                               
    friend std::tuple<std::string,
                      std::unique_ptr<Crystal_structure>,
                      std::unique_ptr<Symmetry_operations>,
                      std::unique_ptr<Gamma_grid>,
                      std::unique_ptr<std::vector<ph_process<3, threeph_type>>>,
                      std::unique_ptr<std::vector<ph_process<4, fourph_type >>> >
                      load_bulk_hdf5(const char* filename, const boost::mpi::communicator& comm);

    /// Deviation from the conservation of energy.
    const double domega;
    /// Standard deviation of the Gaussian.
    const double sigma;
    /// Has the phase space of the process been computed yet?
    bool gaussian_computed;
    /// Gaussian factor of the process, coming from alphathe
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
    const std::array<std::size_t, N> q;
    /// Mode indices of the three phonons involved.
    const std::array<std::size_t, N> alpha;
    /// Type of process.
    const process_type type;

    /// Basic constructor.
    ph_process(std::size_t _c,
                    const std::array<std::size_t, N>& _q,
                    const std::array<std::size_t, N>& _alpha,
                    process_type _type,
                    double _domega,
                    double _sigma)
        : domega(_domega), sigma(_sigma), gaussian_computed(false),
          vp2_computed(false), c(_c), q(std::move(_q)),
          alpha(std::move(_alpha)), type(_type) {
    }


    /// Copy constructor.
    ph_process(const ph_process& original)
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
    /// the scattering rate except fthreeph_typeor the matrix element and some
    /// constants.
    ///
    /// The result can be used to obtain the weighted phase
    /// space volume of three-phonon processes. The Gaussian factor
    /// is cached.xorder
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
    /// @param[in] xorder - x-order ifcs
    /// @return the modulus squared of the matrix element
    /// of the process
    template<class xifc_t>
    double compute_vp2(const Crystal_structure& cell,
                       const Gamma_grid& grid,
                       const std::vector<xifc_t>& xorder);

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

    /// Sets vp2
    inline void set_vp2(const double vp2_) {
        this->vp2_computed = true;
        this->vp2 = vp2_;
    }

    /// @return the value of vp2_computed.
    inline bool is_vp2_computed() const {
        return this->vp2_computed;
    }


    /// Get the "partial rate" Gamma for this process.
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
                                     const Eigen::ArrayXXd& n0){};


    /// Equal operator
    bool operator==(const ph_process& other) const {
        return (this->c == other.c) &&
               (this->q == other.q) &&
               (this->alpha == other.alpha) &&
               (this->type == other.type) &&
               (this->domega == other.domega) &&
               (this->sigma == other.sigma) &&
               (this->gaussian_computed == other.gaussian_computed) &&
               (this->gaussian == other.gaussian) &&
               (this->vp2_computed == other.vp2_computed) &&
               (this->vp2 == other.vp2);
    }


    /// Not-equal operator, defined in terms of operator==.
    bool operator!=(const ph_process& other) const {
        return !(*this == other);
    }
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

/// Look for allowed four-phonon processes in a regular grid.
///
/// Iterate over part of the irreducible q points in the grid
/// (trying to evenly split the equivalence classes over processes)
/// and look for allowed four-phonon processes involving one
/// phonon from that part and three other phonons from anywhere
/// in the grid.
/// @param[in] grid - a regular grid containing Gamma
/// @param[in] communicator - MPI communicator to use
/// @param[in] scalebroad - factor modulating all the broadenings
/// @return a vector of Fourph_process objects
std::vector<Fourph_process> find_allowed_fourph(
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

/// Compute and store the four-phonon contribution to the RTA
/// scattering rates for all vibrational modes on a grid.
///
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] processes - a vector of allowed
/// four-phonon processes
/// @param[in] T - the temperature in K
/// @param[in] comm - an mpi communicator
Eigen::ArrayXXd calc_w0_fourph(const alma::Gamma_grid& grid,
                                std::vector<alma::Fourph_process>& processes,
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
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Threeph_process* t,
                                const unsigned int file_version);
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Fourph_process* t,
                                const unsigned int file_version);
/// Overload required to serialize
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
/// Overload required to serialize the const members of
/// alma::Threeph_process.
///
/// See the boost::serialization documentation for details.
template <class Archive>
inline void save_construct_data(Archive& ar,
                                const alma::Fourph_process* t,
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
                                alma::Fourph_process* t,
                                const unsigned int file_version) {
    std::size_t c;
    ar >> c;
    std::array<std::size_t, 4> q;
    ar >> make_array(q.data(), q.size());
    std::array<std::size_t, 4> alpha;
    ar >> make_array(alpha.data(), alpha.size());
    alma::fourph_type type;
    ar >> type;
    double domega;
    ar >> domega;
    double sigma;
    ar >> sigma;
    ::new (t) alma::Fourph_process(c, q, alpha, type, domega, sigma);
}
} // namespace serialization
} // namespace boost

namespace msgpack {
using fourph_process = alma::ph_process<4,alma::fourph_type>;
MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS) {
namespace adaptor {
template <>
struct pack<fourph_process> {
    /// @brief  Packs an alma::ph_process<4, alma::fourph_type> into a MessagePack stream.
    /// @param o - MessagePack packer object used to write serialized data. E.g. a std::stream
    /// @param v - The fourph_process instance to serialize.
    /// @return  Reference to the MessagePack packer after writing.
    template <typename Stream>
    msgpack::packer<Stream>& operator()(msgpack::packer<Stream>& o, fourph_process const& v) const {
        o.pack_array(10);
        o.pack(static_cast<std::uint64_t>(v.c));
        o.pack_array(4);
        for (auto& x : v.q) o.pack_uint64(x);
        o.pack_array(4);
        for (auto& x : v.alpha) o.pack_uint64(x);
        o.pack(static_cast<int>(v.type));
        o.pack(v.domega);
        o.pack(v.sigma);
        
        o.pack(v.gaussian_computed);
        o.pack(v.vp2_computed);
        o.pack(v.gaussian);
        o.pack(v.vp2);

        return o;
    }
};

template <>
struct convert<fourph_process> {
    /// @brief Deserializes a MessagePack object into a fourph_process instance.
    /// @param o - MessagePack object containing the serialized data.
    /// @param v - Destination object to be reconstructed from the serialized data.
    /// @return A reference to the input MessagePack object.
    msgpack::object const& operator()(msgpack::object const& o, fourph_process& v) const
    {
        if (o.type != msgpack::type::ARRAY || o.via.array.size != 10)
            throw msgpack::type_error();

        auto const& a = o.via.array.ptr;
        
        std::size_t c;
        std::array<std::size_t, 4> q;
        std::array<std::size_t, 4> alpha;
        alma::fourph_type type;
        double domega;
        double sigma;
        
        c = static_cast<std::size_t>(a[0].as<std::uint64_t>());
        if (a[1].type != msgpack::type::ARRAY) throw msgpack::type_error();
        for (size_t j = 0; j < 4; ++j) q[j] = a[1].via.array.ptr[j].as<std::uint64_t>();
        if (a[2].type != msgpack::type::ARRAY) throw msgpack::type_error();
        for (size_t j = 0; j < 4; ++j) alpha[j] = a[2].via.array.ptr[j].as<std::uint64_t>();

        type = static_cast<alma::fourph_type>(a[3].as<int>());
        domega = a[4].as<double>();
        sigma  = a[5].as<double>();

        v.~fourph_process();
        new (&v) fourph_process(c, q, alpha, type, domega, sigma);

        v.gaussian_computed = a[6].as<bool>();
        v.vp2_computed      = a[7].as<bool>();
        v.gaussian = a[8].as<double>();
        v.vp2      = a[9].as<double>();

        return o;
    }
};

} // namespace adaptor
} // MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
} // namespace msgpack
