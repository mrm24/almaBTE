// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
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

/// @file
/// Definitions corresponding to processes.hpp.

#include <cmath>
#include <complex>
#include <constants.hpp>
#include <periodic_table.hpp>
#include <utilities.hpp>
#include <processes.hpp>

namespace alma {
Threeph_processes find_allowed_threeph(
    const Gamma_grid& grid,
    const boost::mpi::communicator& communicator,
    double scalebroad) {
    const auto limits = my_jobs(
        grid.get_nequivalences(), communicator.size(), communicator.rank());
    const std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    Threeph_processes result(
        grid, limits[0] * nmodes, (limits[1] - limits[0]) * nmodes);
    // Store each incoming mode's processes together, independently by sign.
    // The conservation tests and multiplicities are unchanged.
    for (auto ic = limits[0]; ic < limits[1]; ++ic) {
        const auto iq1 = grid.get_representative(ic);
        const auto coords1 = grid.one_to_three(iq1);
        const auto& spectrum1 = grid.get_spectrum_at_q(iq1);
        for (std::size_t im1 = 0; im1 < nmodes; ++im1) {
            if (spectrum1.omega[im1] == 0.)
                continue;
            for (std::size_t iq2 = 0; iq2 < grid.nqpoints; ++iq2) {
                const auto coords2 = grid.one_to_three(iq2);
                const auto& spectrum2 = grid.get_spectrum_at_q(iq2);
                for (int sign : {-1, 1}) {
                    auto coords3 = coords1;
                    for (int j = 0; j < 3; ++j)
                        coords3[j] += sign * coords2[j];
                    const auto iq3 = grid.three_to_one(coords3);
                    const auto& spectrum3 = grid.get_spectrum_at_q(iq3);
                    for (std::size_t im2 = 0; im2 < nmodes; ++im2) {
                        if (spectrum2.omega[im2] == 0.)
                            continue;
                        for (std::size_t im3 = 0; im3 < nmodes; ++im3) {
                            if (spectrum3.omega[im3] == 0.)
                                continue;
                            const auto v =
                                spectrum2.vg.col(im2) - spectrum3.vg.col(im3);
                            const auto sigma = scalebroad * grid.base_sigma(v);
                            const auto delta =
                                std::fabs(spectrum1.omega[im1] +
                                          sign * spectrum2.omega[im2] -
                                          spectrum3.omega[im3]);
                            if (delta <= constants::nsigma * sigma)
                                result.push_back(Threeph_process(
                                    ic,
                                    {{iq1, iq2, iq3}},
                                    {{im1, im2, im3}},
                                    static_cast<threeph_type>(sign),
                                    delta,
                                    sigma));
                        }
                    }
                }
            }
        }
    }
    result.finalize();
    return result;
}

double Threeph_process::compute_weighted_gaussian(const Gamma_grid& grid,
                                                  double T) {
    auto g = this->compute_gaussian();
    auto s = static_cast<int>(this->type);
    const auto& sp1 = grid.get_spectrum_at_q(this->q[0]);
    const auto& sp2 = grid.get_spectrum_at_q(this->q[1]);
    const auto& sp3 = grid.get_spectrum_at_q(this->q[2]);
    auto fBE2 = bose_einstein(sp2.omega[this->alpha[1]], T);
    auto fBE3 = bose_einstein(sp3.omega[this->alpha[2]], T);

    if (!this->gaussian_computed) {
        auto distr = boost::math::normal(0., this->sigma);
        this->gaussian = boost::math::pdf(distr, this->domega);
        this->gaussian_computed = true;
    }
    return (fBE2 - s * fBE3 + (1 - s) / 2) * g / sp1.omega[this->alpha[0]] /
           sp2.omega[this->alpha[1]] / sp3.omega[this->alpha[2]] /
           static_cast<double>(1 + (1 - s) / 2);
    ;
}


double Threeph_process::compute_vp2(
    const Crystal_structure& cell,
    const Gamma_grid& grid,
    const std::vector<Thirdorder_ifcs>& thirdorder) {
    // Prefactor used to convert the result to THz^4 / (kg nm^2)
    constexpr double unitfactor = 1e-6 * constants::e * constants::e /
                                  constants::amu / constants::amu /
                                  constants::amu;
    auto s = static_cast<int>(this->type);
    // Set up those variables that depend only on the q point.
    Eigen::VectorXd q2 = grid.get_q(this->q[1]);
    Eigen::VectorXd q3 = grid.get_q(this->q[2]);
    Eigen::VectorXcd wf1 =
        grid.get_spectrum_at_q(this->q[0]).wfs.col(this->alpha[0]);
    Eigen::VectorXcd wf2 =
        grid.get_spectrum_at_q(this->q[1]).wfs.col(this->alpha[1]);

    // This conditional conjugation takes care of the fact
    // that the second phonon appears on different sides of
    // the equations for emission and absorption processes.
    if (this->type == threeph_type::emission)
        wf2 = wf2.conjugate();
    Eigen::VectorXcd wf3 =
        grid.get_spectrum_at_q(this->q[2]).wfs.col(this->alpha[2]).conjugate();
    std::complex<double> vp = 0.;

// Iterate over triplets to build the matrix element.
#ifdef _OPENMP
#pragma omp declare reduction(+ : std::complex<double> : omp_out += omp_in)    \
    initializer(omp_priv = omp_orig)
#pragma omp parallel for reduction(+ : vp) if (thirdorder.size() >= 4096)
#endif
    for (auto tri_it = thirdorder.begin(); tri_it < thirdorder.end();
         ++tri_it) {
        auto& tri = *tri_it;
        double massfactor = std::sqrt(
            cell.get_mass(tri.i) * cell.get_mass(tri.j) * cell.get_mass(tri.k));
        double arg2 = s * q2.dot(tri.rj);
        double arg3 = q3.dot(tri.rk);
        std::complex<double> prefactor =
            (std::cos(arg2) + constants::imud * std::sin(arg2)) *
            (std::cos(arg3) - constants::imud * std::sin(arg3)) / massfactor;
        std::complex<double> contr = 0.;

        for (auto rr = 0; rr < 3; ++rr)
            for (auto ss = 0; ss < 3; ++ss)
                for (auto tt = 0; tt < 3; ++tt)
                    contr += wf1(tt + 3 * tri.i) * wf2(ss + 3 * tri.j) *
                             wf3(rr + 3 * tri.k) * tri.ifc(tt, ss, rr);
        vp += prefactor * contr;
    }
    // And store only its modulus squared.
    this->vp2 = unitfactor * std::norm(vp);
    this->vp2_computed = true;
    return this->vp2;
}


double Threeph_process::compute_gamma(const Gamma_grid& grid, double T) {
    auto s = static_cast<int>(this->type);
    auto gaussian = this->compute_gaussian();
    auto vp2 = this->get_vp2();
    const auto& sp1 = grid.get_spectrum_at_q(this->q[0]);
    const auto& sp2 = grid.get_spectrum_at_q(this->q[1]);
    const auto& sp3 = grid.get_spectrum_at_q(this->q[2]);
    auto fBE2 = bose_einstein(sp2.omega[this->alpha[1]], T);
    auto fBE3 = bose_einstein(sp3.omega[this->alpha[2]], T);
    const double prefactor = 1e6 * constants::hbar * constants::pi / 4.0;
    auto nruter = prefactor * (fBE2 - s * fBE3 + (1 - s) / 2) * gaussian * vp2 /
                  grid.nqpoints / sp1.omega[this->alpha[0]] /
                  sp2.omega[this->alpha[1]] / sp3.omega[this->alpha[2]];

    return nruter;
}

double Threeph_process::compute_gamma_reduced(const Gamma_grid& grid,
                                              double T) {
    auto gaussian = this->compute_gaussian();
    auto vp2 = this->get_vp2();
    const auto& sp1 = grid.get_spectrum_at_q(this->q[0]);
    const auto& sp2 = grid.get_spectrum_at_q(this->q[1]);
    const auto& sp3 = grid.get_spectrum_at_q(this->q[2]);
    const double prefactor = 1e6 * constants::hbar * constants::pi / 4.0;
    auto nruter = prefactor * gaussian * vp2 / grid.nqpoints /
                  sp1.omega[this->alpha[0]] / sp2.omega[this->alpha[1]] /
                  sp3.omega[this->alpha[2]];

    return nruter;
}

Eigen::ArrayXd Threeph_process::compute_collision(const Gamma_grid& grid,
                                                  const Eigen::ArrayXXd& n0) {
    auto s = static_cast<int>(this->type);
    auto gaussian = this->compute_gaussian();
    auto vp2 = this->get_vp2();
    const auto& sp1 = grid.get_spectrum_at_q(this->q[0]);
    const auto& sp2 = grid.get_spectrum_at_q(this->q[1]);
    const auto& sp3 = grid.get_spectrum_at_q(this->q[2]);
    double fBE1 = n0(this->alpha[0], this->q[0]);
    double fBE2 = n0(this->alpha[1], this->q[1]);
    double fBE3 = n0(this->alpha[2], this->q[2]);
    const double prefactor = 1e6 * constants::hbar * constants::pi / 4.0;
    Eigen::ArrayXd nruter(3);

    nruter.fill(prefactor * gaussian * vp2 * 2. / (3. - s) / grid.nqpoints /
                sp1.omega[this->alpha[0]] / sp2.omega[this->alpha[1]] /
                sp3.omega[this->alpha[2]]);
    // The expressions involved in this calculation are pretty
    // similar to the one in compute_gamma, but the BE distributions
    // involved are different in each case.
    nruter(0) *= s * (fBE3 - s * fBE2 + (1 + s) / 2);
    nruter(1) *= fBE3 - fBE1;
    nruter(2) *= fBE2 + s * fBE1 + (1 - s) / 2;
    return nruter;
}


namespace {
Eigen::ArrayXXd calc_w0_impl(
    const Gamma_grid& grid,
    Threeph_processes& processes,
    double T,
    const std::function<bool(const Const_threeph_process_view&)>& filter,
    const boost::mpi::communicator& comm) {
    const auto nmodes = grid.get_spectrum_at_q(0).omega.size();
    Eigen::ArrayXXd local = Eigen::ArrayXXd::Zero(nmodes, grid.nqpoints);
    std::string error;
    try {
        if (!std::isfinite(T) || T <= 0.)
            throw value_error("temperature must be finite and positive");
        if (!processes.empty()) {
            // Preserve the old lazy Gaussian cache when loading legacy files
            // whose builders computed matrix elements but not Gaussians.
            // Subsequent temperatures only scan the compact flag columns.
            if (!filter) {
                bool pending = false;
                for (auto sign :
                     {threeph_type::emission, threeph_type::absorption}) {
                    const auto& flags = processes.table(sign).flags;
                    pending =
                        pending ||
                        std::any_of(
                            flags.begin(), flags.end(), [](std::uint8_t flag) {
                                return !(flag & Threeph_table::gaussian_ready);
                            });
                }
                if (pending)
                    for (auto p : processes)
                        p.compute_gaussian();
            }
            const auto gamma =
                Threeph_scattering(processes, grid, filter).rates(T);
            for (auto sign :
                 {threeph_type::emission, threeph_type::absorption}) {
                const auto which = sign == threeph_type::absorption ? 1 : 0;
                const auto factor = sign == threeph_type::absorption ? 1. : 0.5;
                const auto& table = processes.table(sign);
                for (std::size_t row = 0; row < processes.nrows(); ++row) {
                    double rate = 0.;
                    for (std::size_t i = table.offsets[row];
                         i < table.offsets[row + 1];
                         ++i) {
                        rate += factor * gamma[which][i];
                    }
                    local(processes.row_branch(row), processes.row_q(row)) +=
                        rate;
                }
            }
        }
    }
    catch (const std::exception& e) {
        error = e.what();
    }
    int failed_rank;
    boost::mpi::all_reduce(comm,
                           error.empty() ? comm.size() : comm.rank(),
                           failed_rank,
                           boost::mpi::minimum<int>());
    if (failed_rank != comm.size()) {
        boost::mpi::broadcast(comm, error, failed_rank);
        throw exception(error);
    }
    // Each rank may own different branches; symmetry expansion precedes the
    // reduction so that partial rows and zero-work ranks remain valid.
    for (std::size_t c = 0; c < grid.get_nequivalences(); ++c) {
        const auto eq = grid.get_equivalence(c);
        for (std::size_t i = 1; i < eq.size(); ++i)
            local.col(eq[i]) = local.col(eq[0]);
    }
    Eigen::ArrayXXd result(nmodes, grid.nqpoints);
    boost::mpi::all_reduce(
        comm, local.data(), local.size(), result.data(), std::plus<double>());
    return result;
}
} // namespace
Eigen::ArrayXXd calc_w0_threeph(const Gamma_grid& grid,
                                Threeph_processes& processes,
                                double T,
                                const boost::mpi::communicator& comm) {
    return calc_w0_impl(grid, processes, T, {}, comm);
}
Eigen::ArrayXXd calc_w0_threeph(
    const Gamma_grid& grid,
    Threeph_processes& processes,
    double T,
    std::function<bool(const Const_threeph_process_view&)> filter,
    const boost::mpi::communicator& comm) {
    return calc_w0_impl(grid, processes, T, filter, comm);
}
} // namespace alma
