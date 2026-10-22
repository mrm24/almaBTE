// Copyright 2015-2026 The ALMA Project Developers
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
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <boost/mpi.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>

// Forward declarations used by Boost construction hooks.
namespace alma {
class Threeph_process;
class Threeph_processes;
class Const_threeph_process_view;
class Threeph_process_view;
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
    friend class Const_threeph_process_view;
    friend class Threeph_processes;
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
                               const Threeph_processes& processes,
                               const boost::mpi::communicator& comm);

    friend std::tuple<std::string,
                      std::unique_ptr<Crystal_structure>,
                      std::unique_ptr<Symmetry_operations>,
                      std::unique_ptr<Gamma_grid>,
                      std::unique_ptr<Threeph_processes>>
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
          gaussian(0.), vp2_computed(false), vp2(0.), c(_c), q(std::move(_q)),
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

/// Packed mode indices, with a checked 64-bit fallback for very large grids.
/// The representation is uniform within each column, never per element.
class Threeph_indices {
    friend class boost::serialization::access;
    bool wide_ = false;
    std::vector<std::uint32_t> narrow_;
    std::vector<std::uint64_t> wide_values_;
    template <class Archive> void serialize(Archive& ar, unsigned int) {
        ar & wide_ & narrow_ & wide_values_;
    }

public:
    explicit Threeph_indices(bool wide = false) : wide_(wide) {
    }
    bool wide() const {
        return wide_;
    }
    std::size_t size() const {
        return wide_ ? wide_values_.size() : narrow_.size();
    }
    std::uint64_t operator[](std::size_t i) const {
        return wide_ ? wide_values_[i] : narrow_[i];
    }
    void push_back(std::uint64_t value);
    void reserve(std::size_t n);
    void resize(std::size_t n);
    void set(std::size_t i, std::uint64_t value);
    const std::vector<std::uint32_t>& narrow() const {
        return narrow_;
    }
    const std::vector<std::uint64_t>& wide_values() const {
        return wide_values_;
    }
    void validate() const;
};

/// One sign of the processes, grouped by incoming mode. All offsets are local,
/// zero-based, and include the terminal offset. Flags preserve partially
/// prepared processes during construction and file/MPI round trips.
struct Threeph_table {
    enum : std::uint8_t { gaussian_ready = 1, vp2_ready = 2 };
    std::vector<std::uint64_t> offsets;
    Threeph_indices second, third;
    std::vector<double> domega, sigma, gaussian, vp2;
    std::vector<std::uint8_t> flags;
    explicit Threeph_table(bool wide = false) : second(wide), third(wide) {
    }
    std::size_t size() const {
        return second.size();
    }

private:
    friend class boost::serialization::access;
    template <class Archive> void serialize(Archive& ar, unsigned int) {
        ar & offsets & second & third & domega & sigma & gaussian & vp2 & flags;
    }
};

/// A read-only process view. Metadata is decoded into the short-lived view;
/// numeric data remains in the owning columns. Views expire on structural
/// modification of their owner, just like references into a std::vector.
class Const_threeph_process_view {
protected:
    const Threeph_table* table_;
    std::size_t index_;

public:
    const std::size_t c;
    const std::array<std::size_t, 3> q, alpha;
    const threeph_type type;
    Const_threeph_process_view(const Threeph_table& table,
                               std::size_t i,
                               std::size_t row,
                               std::size_t first_row,
                               std::size_t nbranches,
                               std::size_t incoming_q,
                               threeph_type sign);
    double compute_gaussian() const;
    double compute_weighted_gaussian(const Gamma_grid& grid, double T) const;
    double get_vp2() const;
    bool is_vp2_computed() const;
    double compute_gamma(const Gamma_grid& grid, double T) const;
    double compute_gamma_reduced(const Gamma_grid& grid, double T) const;
    Eigen::ArrayXd compute_collision(const Gamma_grid& grid,
                                     const Eigen::ArrayXXd& n0) const;
    /// Explicit owning snapshot, useful at API boundaries and as a reference.
    Threeph_process to_value() const;
    operator Threeph_process() const {
        return to_value();
    }
};

/// A mutable view writes computed values back to the owning columns. Copying
/// the view does not copy a process or detach its caches from the owner.
class Threeph_process_view : public Const_threeph_process_view {
    Threeph_table* mutable_table_;

public:
    Threeph_process_view(Threeph_table& table,
                         std::size_t i,
                         std::size_t row,
                         std::size_t first_row,
                         std::size_t nbranches,
                         std::size_t incoming_q,
                         threeph_type sign);
    double compute_gaussian();
    double compute_vp2(const Crystal_structure& cell,
                       const Gamma_grid& grid,
                       const std::vector<Thirdorder_ifcs>& thirdorder);
};

/// Owning SoA storage. Rows are global irreducible-class/branch indices, with
/// a contiguous local row range on each MPI rank. No array of process objects
/// is retained. Flat iteration visits emission then absorption processes.
class Threeph_processes {
    friend class boost::serialization::access;
    std::size_t nbranches_ = 0, nqpoints_ = 0, first_row_ = 0, nrows_ = 0;
    std::vector<std::size_t> representatives_;
    std::array<Threeph_table, 2> tables_;
    bool finalized_ = true;
    std::array<std::size_t, 2> last_row_{{0, 0}};
    template <class Archive> void serialize(Archive& ar, unsigned int) {
        ar & nbranches_ & nqpoints_ & first_row_ & nrows_ & representatives_;
        ar & tables_ & finalized_ & last_row_;
        if (Archive::is_loading::value)
            validate();
    }
    std::size_t locate_row(const Threeph_table& table, std::size_t i) const;

public:
    Threeph_processes() = default;
    Threeph_processes(const Gamma_grid& grid,
                      std::size_t first_row,
                      std::size_t nrows);
    std::size_t size() const {
        return tables_[0].size() + tables_[1].size();
    }
    bool empty() const {
        return size() == 0;
    }
    std::size_t nbranches() const {
        return nbranches_;
    }
    std::size_t nqpoints() const {
        return nqpoints_;
    }
    std::size_t first_row() const {
        return first_row_;
    }
    std::size_t nrows() const {
        return nrows_;
    }
    std::size_t row_class(std::size_t row) const {
        return (first_row_ + row) / nbranches_;
    }
    std::size_t row_branch(std::size_t row) const {
        return (first_row_ + row) % nbranches_;
    }
    std::size_t row_q(std::size_t row) const {
        return representatives_[row_class(row)];
    }
    const Threeph_table& table(threeph_type sign) const {
        return tables_[sign == threeph_type::absorption ? 1 : 0];
    }
    /// Append in nondecreasing row order within each sign, then finalize once.
    void push_back(const Threeph_process& process);
    void finalize();
    void clear();
    void validate() const;
    /// Install already grouped columns, validating all lengths and indices.
    void set_table(threeph_type sign, Threeph_table table);
    Threeph_process_view view(threeph_type sign,
                              std::size_t row,
                              std::size_t i);
    Const_threeph_process_view view(threeph_type sign,
                                    std::size_t row,
                                    std::size_t i) const;
    Threeph_process_view operator[](std::size_t i);
    Const_threeph_process_view operator[](std::size_t i) const;
    Threeph_process_view at(std::size_t i);
    Const_threeph_process_view at(std::size_t i) const;

    template <bool Const> class Iterator {
        using owner_type = typename std::conditional<Const,
                                                     const Threeph_processes,
                                                     Threeph_processes>::type;
        owner_type* owner_;
        std::size_t i_;
        std::size_t row_ = 0;
        int which_ = 0;

    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Threeph_process;
        using difference_type = std::ptrdiff_t;
        using reference = typename std::conditional<Const,
                                                    Const_threeph_process_view,
                                                    Threeph_process_view>::type;
        using pointer = void;
        Iterator(owner_type* owner = nullptr, std::size_t i = 0)
            : owner_(owner), i_(i) {
            if (owner_ && i_ < owner_->size()) {
                which_ = i_ < owner_->tables_[0].size() ? 0 : 1;
                row_ = owner_->locate_row(
                    owner_->tables_[which_],
                    i_ - (which_ ? owner_->tables_[0].size() : 0));
            }
        }
        reference operator*() const {
            return owner_->view(which_ ? threeph_type::absorption
                                       : threeph_type::emission,
                                row_,
                                i_ - (which_ ? owner_->tables_[0].size() : 0));
        }
        Iterator& operator++() {
            ++i_;
            if (i_ == owner_->size())
                return *this;
            if (!which_ && i_ == owner_->tables_[0].size()) {
                which_ = 1;
                row_ = 0;
            }
            const auto local = i_ - (which_ ? owner_->tables_[0].size() : 0);
            const auto& offsets = owner_->tables_[which_].offsets;
            while (row_ + 1 < offsets.size() && offsets[row_ + 1] <= local)
                ++row_;
            return *this;
        }
        Iterator operator++(int) {
            auto copy = *this;
            ++*this;
            return copy;
        }
        bool operator==(const Iterator& other) const {
            return owner_ == other.owner_ && i_ == other.i_;
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };
    Iterator<false> begin() {
        return {this, 0};
    }
    Iterator<false> end() {
        return {this, size()};
    }
    Iterator<true> begin() const {
        return {this, 0};
    }
    Iterator<true> end() const {
        return {this, size()};
    }
    Iterator<true> cbegin() const {
        return begin();
    }
    Iterator<true> cend() const {
        return end();
    }
};

/// Prepared temperature-independent scattering coefficients. Construct after
/// matrix elements/grid frequencies are finalized; rebuild after changing them.
/// Temperature sweeps can reuse this object; iterations only need rates(T).
class Threeph_scattering {
    const Threeph_processes& processes_;
    std::vector<double> omega_;
    std::array<std::vector<double>, 2> reduced_;

public:
    Threeph_scattering(
        const Threeph_processes& processes,
        const Gamma_grid& grid,
        const std::function<bool(const Const_threeph_process_view&)>& filter =
            {});
    std::array<std::vector<double>, 2> rates(double T) const;
};

/// Matrix-free local three-phonon collision operator, prepared at one
/// temperature. The grid and processes must outlive it and remain unchanged.
/// MPI reduction is the caller's responsibility; apply() is also useful for
/// testing the operator independently of an iterative solver.
class Threeph_collision {
    const Threeph_processes& processes_;
    const Gamma_grid& grid_;
    std::array<std::vector<double>, 2> gamma_;
    std::vector<std::vector<std::vector<std::size_t>>> operations_;

public:
    Threeph_collision(const Threeph_processes& processes,
                      const Gamma_grid& grid,
                      double T);
    Eigen::MatrixXd apply(const Eigen::Ref<const Eigen::MatrixXd>& F) const;
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
Threeph_processes find_allowed_threeph(
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
                                alma::Threeph_processes& processes,
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
    alma::Threeph_processes& processes,
    double T,
    std::function<bool(const Const_threeph_process_view&)> filter,
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
