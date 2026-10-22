// Copyright 2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <processes.hpp>
#include <utilities.hpp>
#include <algorithm>
#include <stdexcept>

namespace alma {
namespace {
std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw value_error("three-phonon mode count overflows size_t");
    return a * b;
}
int sign_index(threeph_type sign) {
    if (sign != threeph_type::absorption && sign != threeph_type::emission)
        throw value_error("invalid three-phonon process sign");
    return sign == threeph_type::absorption ? 1 : 0;
}
} // namespace

void Threeph_indices::push_back(std::uint64_t value) {
    if (wide_)
        wide_values_.push_back(value);
    else {
        if (value > std::numeric_limits<std::uint32_t>::max())
            throw value_error("three-phonon mode index overflows uint32");
        narrow_.push_back(static_cast<std::uint32_t>(value));
    }
}
void Threeph_indices::reserve(std::size_t n) {
    if (wide_)
        wide_values_.reserve(n);
    else
        narrow_.reserve(n);
}
void Threeph_indices::resize(std::size_t n) {
    if (wide_)
        wide_values_.resize(n);
    else
        narrow_.resize(n);
}
void Threeph_indices::set(std::size_t i, std::uint64_t value) {
    if (!wide_ && value > std::numeric_limits<std::uint32_t>::max())
        throw value_error("three-phonon mode index overflows uint32");
    if (wide_)
        wide_values_.at(i) = value;
    else
        narrow_.at(i) = static_cast<std::uint32_t>(value);
}
void Threeph_indices::validate() const {
    if ((wide_ && !narrow_.empty()) || (!wide_ && !wide_values_.empty()))
        throw value_error("inconsistent packed three-phonon index storage");
}

Const_threeph_process_view::Const_threeph_process_view(
    const Threeph_table& table,
    std::size_t i,
    std::size_t row,
    std::size_t first_row,
    std::size_t nbranches,
    std::size_t incoming_q,
    threeph_type sign)
    : table_(&table), index_(i), c((first_row + row) / nbranches),
      q{{incoming_q,
         static_cast<std::size_t>(table.second[i] / nbranches),
         static_cast<std::size_t>(table.third[i] / nbranches)}},
      alpha{{(first_row + row) % nbranches,
             static_cast<std::size_t>(table.second[i] % nbranches),
             static_cast<std::size_t>(table.third[i] % nbranches)}},
      type(sign) {
}

Threeph_process Const_threeph_process_view::to_value() const {
    Threeph_process p(
        c, q, alpha, type, table_->domega[index_], table_->sigma[index_]);
    p.gaussian_computed = table_->flags[index_] & Threeph_table::gaussian_ready;
    p.vp2_computed = table_->flags[index_] & Threeph_table::vp2_ready;
    p.gaussian = table_->gaussian[index_];
    p.vp2 = table_->vp2[index_];
    return p;
}
double Const_threeph_process_view::compute_gaussian() const {
    if (table_->flags[index_] & Threeph_table::gaussian_ready)
        return table_->gaussian[index_];
    return boost::math::pdf(boost::math::normal(0., table_->sigma[index_]),
                            table_->domega[index_]);
}
bool Const_threeph_process_view::is_vp2_computed() const {
    return table_->flags[index_] & Threeph_table::vp2_ready;
}
double Const_threeph_process_view::get_vp2() const {
    if (!is_vp2_computed())
        throw exception("vp2 is not available");
    return table_->vp2[index_];
}
double Const_threeph_process_view::compute_weighted_gaussian(
    const Gamma_grid& grid,
    double T) const {
    return to_value().compute_weighted_gaussian(grid, T);
}
double Const_threeph_process_view::compute_gamma(const Gamma_grid& grid,
                                                 double T) const {
    return to_value().compute_gamma(grid, T);
}
double Const_threeph_process_view::compute_gamma_reduced(const Gamma_grid& grid,
                                                         double T) const {
    return to_value().compute_gamma_reduced(grid, T);
}
Eigen::ArrayXd Const_threeph_process_view::compute_collision(
    const Gamma_grid& grid,
    const Eigen::ArrayXXd& n0) const {
    return to_value().compute_collision(grid, n0);
}
Threeph_process_view::Threeph_process_view(Threeph_table& table,
                                           std::size_t i,
                                           std::size_t row,
                                           std::size_t first_row,
                                           std::size_t nbranches,
                                           std::size_t incoming_q,
                                           threeph_type sign)
    : Const_threeph_process_view(table,
                                 i,
                                 row,
                                 first_row,
                                 nbranches,
                                 incoming_q,
                                 sign),
      mutable_table_(&table) {
}
double Threeph_process_view::compute_gaussian() {
    const auto result = Const_threeph_process_view::compute_gaussian();
    mutable_table_->gaussian[index_] = result;
    mutable_table_->flags[index_] |= Threeph_table::gaussian_ready;
    return result;
}
double Threeph_process_view::compute_vp2(
    const Crystal_structure& cell,
    const Gamma_grid& grid,
    const std::vector<Thirdorder_ifcs>& thirdorder) {
    auto p = to_value();
    const auto result = p.compute_vp2(cell, grid, thirdorder);
    mutable_table_->vp2[index_] = result;
    mutable_table_->flags[index_] |= Threeph_table::vp2_ready;
    return result;
}

Threeph_processes::Threeph_processes(const Gamma_grid& grid,
                                     std::size_t first_row,
                                     std::size_t nrows)
    : nbranches_(grid.get_spectrum_at_q(0).omega.size()),
      nqpoints_(grid.nqpoints), first_row_(first_row), nrows_(nrows) {
    const auto total = checked_product(grid.get_nequivalences(), nbranches_);
    if (first_row > total || nrows > total - first_row)
        throw value_error("three-phonon row range is outside the grid");
    representatives_.reserve(grid.get_nequivalences());
    for (std::size_t c = 0; c < grid.get_nequivalences(); ++c)
        representatives_.push_back(grid.get_representative(c));
    const auto modes = checked_product(nqpoints_, nbranches_);
    const bool wide =
        modes && modes - 1 > std::numeric_limits<std::uint32_t>::max();
    for (auto& t : tables_) {
        t = Threeph_table(wide);
        t.offsets.assign(nrows_ + 1, 0);
    }
}

void Threeph_processes::push_back(const Threeph_process& p) {
    if (!nbranches_ || p.c >= representatives_.size() ||
        p.alpha[0] >= nbranches_)
        throw value_error("invalid incoming three-phonon mode");
    const auto global = p.c * nbranches_ + p.alpha[0];
    if (global < first_row_ || global - first_row_ >= nrows_)
        throw value_error(
            "incoming three-phonon mode is not owned by this rank");
    const auto row = global - first_row_;
    const auto which = sign_index(p.type);
    auto& t = tables_[which];
    if ((finalized_ && !empty()) ||
        (!t.second.size() ? false : row < last_row_[which]))
        throw value_error("append processes in row order before finalizing");
    for (std::size_t i = 0; i < 3; ++i)
        if (p.q[i] >= nqpoints_ || p.alpha[i] >= nbranches_)
            throw value_error("three-phonon partner is outside the grid");
    if (p.q[0] != row_q(row))
        throw value_error("incoming q point is not its class representative");
    finalized_ = false;
    last_row_[which] = row;
    ++t.offsets[row + 1];
    t.second.push_back(p.q[1] * nbranches_ + p.alpha[1]);
    t.third.push_back(p.q[2] * nbranches_ + p.alpha[2]);
    t.domega.push_back(p.domega);
    t.sigma.push_back(p.sigma);
    t.gaussian.push_back(p.gaussian_computed ? p.gaussian : 0.);
    t.vp2.push_back(p.vp2_computed ? p.vp2 : 0.);
    t.flags.push_back(
        (p.gaussian_computed ? Threeph_table::gaussian_ready : 0) |
        (p.vp2_computed ? Threeph_table::vp2_ready : 0));
}
void Threeph_processes::finalize() {
    if (!finalized_) {
        for (auto& t : tables_)
            for (std::size_t r = 0; r < nrows_; ++r)
                t.offsets[r + 1] += t.offsets[r];
        finalized_ = true;
    }
    validate();
}
void Threeph_processes::clear() {
    for (auto& t : tables_) {
        t = Threeph_table(t.second.wide());
        if (nbranches_)
            t.offsets.assign(nrows_ + 1, 0);
    }
    last_row_ = {{0, 0}};
    finalized_ = true;
}
void Threeph_processes::validate() const {
    if (!nbranches_) {
        if (nqpoints_ || nrows_ || first_row_ || !representatives_.empty() ||
            !empty())
            throw value_error("invalid empty three-phonon container");
        return;
    }
    const auto total_rows =
        checked_product(representatives_.size(), nbranches_);
    const auto modes = checked_product(nqpoints_, nbranches_);
    if (!finalized_ || first_row_ > total_rows ||
        nrows_ > total_rows - first_row_)
        throw value_error("invalid three-phonon row range or unfinished table");
    for (auto q : representatives_)
        if (q >= nqpoints_)
            throw value_error("invalid three-phonon representative");
    for (const auto& t : tables_) {
        t.second.validate();
        t.third.validate();
        if (t.second.wide() != t.third.wide())
            throw value_error("three-phonon partner index widths must match");
        const auto n = t.size();
        if (t.offsets.size() != nrows_ + 1 || t.offsets.front() != 0 ||
            t.offsets.back() != n ||
            !std::is_sorted(t.offsets.begin(), t.offsets.end()))
            throw value_error("invalid three-phonon row offsets");
        if (t.third.size() != n || t.domega.size() != n ||
            t.sigma.size() != n || t.gaussian.size() != n ||
            t.vp2.size() != n || t.flags.size() != n)
            throw value_error("inconsistent three-phonon column lengths");
        for (std::size_t i = 0; i < n; ++i) {
            if (t.second[i] >= modes || t.third[i] >= modes ||
                (t.flags[i] & ~3u))
                throw value_error(
                    "invalid three-phonon partner index or flags");
        }
    }
}
void Threeph_processes::set_table(threeph_type sign, Threeph_table table) {
    auto& target = tables_[sign_index(sign)];
    std::swap(target, table);
    try {
        validate();
    }
    catch (...) {
        std::swap(target, table);
        throw;
    }
}
std::size_t Threeph_processes::locate_row(const Threeph_table& t,
                                          std::size_t i) const {
    if (!finalized_)
        throw value_error("finalize three-phonon processes before access");
    return std::upper_bound(t.offsets.begin(), t.offsets.end(), i) -
           t.offsets.begin() - 1;
}
Threeph_process_view Threeph_processes::view(threeph_type sign,
                                             std::size_t row,
                                             std::size_t i) {
    return {tables_[sign_index(sign)],
            i,
            row,
            first_row_,
            nbranches_,
            row_q(row),
            sign};
}
Const_threeph_process_view Threeph_processes::view(threeph_type sign,
                                                   std::size_t row,
                                                   std::size_t i) const {
    return {tables_[sign_index(sign)],
            i,
            row,
            first_row_,
            nbranches_,
            row_q(row),
            sign};
}
Threeph_process_view Threeph_processes::operator[](std::size_t i) {
    const auto sign = i < tables_[0].size() ? threeph_type::emission
                                            : threeph_type::absorption;
    if (sign == threeph_type::absorption)
        i -= tables_[0].size();
    return view(sign, locate_row(table(sign), i), i);
}
Const_threeph_process_view Threeph_processes::operator[](std::size_t i) const {
    const auto sign = i < tables_[0].size() ? threeph_type::emission
                                            : threeph_type::absorption;
    if (sign == threeph_type::absorption)
        i -= tables_[0].size();
    return view(sign, locate_row(table(sign), i), i);
}
Threeph_process_view Threeph_processes::at(std::size_t i) {
    if (i >= size())
        throw std::out_of_range("three-phonon process index");
    return (*this)[i];
}
Const_threeph_process_view Threeph_processes::at(std::size_t i) const {
    if (i >= size())
        throw std::out_of_range("three-phonon process index");
    return (*this)[i];
}

Threeph_scattering::Threeph_scattering(
    const Threeph_processes& processes,
    const Gamma_grid& grid,
    const std::function<bool(const Const_threeph_process_view&)>& filter)
    : processes_(processes) {
    processes.validate();
    if (!processes.nbranches() && processes.empty())
        return;
    if (processes.nbranches() !=
            static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size()) ||
        processes.nqpoints() != grid.nqpoints)
        throw value_error("three-phonon processes and grid do not match");
    omega_.resize(checked_product(processes.nbranches(), grid.nqpoints));
    for (std::size_t q = 0; q < grid.nqpoints; ++q)
        for (std::size_t b = 0; b < processes.nbranches(); ++b)
            omega_[q * processes.nbranches() + b] =
                grid.get_spectrum_at_q(q).omega[b];
    for (auto sign : {threeph_type::emission, threeph_type::absorption}) {
        const auto which = sign_index(sign);
        const auto& table = processes.table(sign);
        auto& reduced = reduced_[which];
        reduced.resize(table.size());
        for (std::size_t row = 0; row < processes.nrows(); ++row) {
            const auto first = processes.row_q(row) * processes.nbranches() +
                               processes.row_branch(row);
            for (std::size_t i = table.offsets[row]; i < table.offsets[row + 1];
                 ++i) {
                if (filter && !filter(processes.view(sign, row, i))) {
                    reduced[i] = 0.;
                    continue;
                }
                if (!(table.flags[i] & Threeph_table::vp2_ready))
                    throw exception("vp2 is not available");
                const auto gaussian =
                    table.flags[i] & Threeph_table::gaussian_ready
                        ? table.gaussian[i]
                        : boost::math::pdf(
                              boost::math::normal(0., table.sigma[i]),
                              table.domega[i]);
                constexpr double factor =
                    1e6 * constants::hbar * constants::pi / 4.;
                reduced[i] = factor * gaussian * table.vp2[i] / grid.nqpoints /
                             omega_[first] / omega_[table.second[i]] /
                             omega_[table.third[i]];
            }
        }
    }
}
std::array<std::vector<double>, 2> Threeph_scattering::rates(double T) const {
    if (!std::isfinite(T) || T <= 0.)
        throw value_error("temperature must be finite and positive");
    std::vector<double> occupation(omega_.size());
    for (std::size_t i = 0; i < omega_.size(); ++i)
        occupation[i] = omega_[i] == 0. ? 0. : bose_einstein(omega_[i], T);
    std::array<std::vector<double>, 2> result;
    for (auto sign : {threeph_type::emission, threeph_type::absorption}) {
        const auto which = sign_index(sign);
        const auto& table = processes_.table(sign);
        result[which].resize(table.size());
        for (std::size_t i = 0; i < table.size(); ++i) {
            if (reduced_[which][i] == 0.) {
                result[which][i] = 0.;
                continue;
            }
            const auto n2 = occupation[table.second[i]],
                       n3 = occupation[table.third[i]];
            result[which][i] =
                reduced_[which][i] *
                (sign == threeph_type::absorption ? n2 - n3 : n2 + n3 + 1.);
        }
    }
    return result;
}
Threeph_collision::Threeph_collision(const Threeph_processes& processes,
                                     const Gamma_grid& grid,
                                     double T)
    : processes_(processes), grid_(grid),
      gamma_(Threeph_scattering(processes, grid).rates(T)),
      operations_(grid.get_nequivalences()) {
    // Group all candidate rotations for each distinct incoming image. Keep
    // alternate rotations: non-cubic grids may invalidate a partner image.
    for (std::size_t c = 0; c < operations_.size(); ++c) {
        const auto& images =
            grid.equivalent_qpoints(grid.get_representative(c));
        auto& groups = operations_[c];
        for (std::size_t op = 0; op < images.size(); ++op) {
            if (images[op] == grid.nqpoints)
                continue;
            auto group =
                std::find_if(groups.begin(),
                             groups.end(),
                             [&](const std::vector<std::size_t>& g) {
                                 return images[g.front()] == images[op];
                             });
            if (group == groups.end())
                groups.push_back({op});
            else
                group->push_back(op);
        }
    }
}
Eigen::MatrixXd Threeph_collision::apply(
    const Eigen::Ref<const Eigen::MatrixXd>& F) const {
    const auto bands = grid_.get_spectrum_at_q(0).omega.size();
    if (F.rows() != 3 * bands ||
        F.cols() != static_cast<Eigen::Index>(grid_.nqpoints))
        throw value_error(
            "three-phonon collision input has the wrong dimensions");
    Eigen::MatrixXd delta = Eigen::MatrixXd::Zero(F.rows(), F.cols());
    for (auto sign : {threeph_type::emission, threeph_type::absorption}) {
        const auto which = sign_index(sign);
        const auto factor = sign == threeph_type::absorption ? 1. : 0.5;
        const auto s = static_cast<int>(sign);
        const auto& table = processes_.table(sign);
        if (gamma_[which].size() != table.size())
            throw value_error(
                "processes changed after preparing the collision operator");
        // Rows own disjoint output modes, including their symmetry images.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
        for (std::size_t row = 0; row < processes_.nrows(); ++row) {
            const auto c = processes_.row_class(row),
                       branch = processes_.row_branch(row);
            const auto& incoming_images =
                grid_.equivalent_qpoints(processes_.row_q(row));
            for (const auto& group : operations_[c]) {
                Eigen::Vector3d sum = Eigen::Vector3d::Zero();
                for (std::size_t i = table.offsets[row];
                     i < table.offsets[row + 1];
                     ++i) {
                    const auto second = table.second[i], third = table.third[i];
                    const auto& images2 =
                        grid_.equivalent_qpoints(second / bands);
                    const auto& images3 =
                        grid_.equivalent_qpoints(third / bands);
                    for (auto op : group) {
                        const auto q2 = images2[op], q3 = images3[op];
                        if (q2 == grid_.nqpoints || q3 == grid_.nqpoints)
                            continue;
                        sum += factor * gamma_[which][i] *
                               (F.block<3, 1>(3 * (third % bands), q3) -
                                s * F.block<3, 1>(3 * (second % bands), q2));
                        break;
                    }
                }
                delta.block<3, 1>(3 * branch, incoming_images[group.front()]) +=
                    sum;
            }
        }
    }
    return delta;
}
} // namespace alma
