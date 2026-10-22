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
/// Definitions corresponding to bulk_properties.hpp

#include <bulk_properties.hpp>
#include <analytic1d.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <processes.hpp>
#include <iostream>

namespace alma {
Eigen::MatrixXd calc_kappa(const alma::Crystal_structure& poscar,
                           const alma::Gamma_grid& grid,
                           const alma::Symmetry_operations& syms,
                           const Eigen::Ref<const Eigen::ArrayXXd>& w,
                           double T) {
    auto nequiv = grid.get_nequivalences();
    auto nmodes =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());

    if ((static_cast<std::size_t>(w.rows()) != nmodes) ||
        (static_cast<std::size_t>(w.cols()) != grid.nqpoints))
        throw alma::value_error("inconsistent dimensions");
    Eigen::MatrixXd nruter(3, 3);
    nruter.fill(0.);

    // The zero frequency and/or velocity states are ignored.
    for (decltype(nequiv) iequiv = 0; iequiv < nequiv; ++iequiv) {
        auto iq0 = grid.get_representative(iequiv);
        const auto& sp0 = grid.get_spectrum_at_q(iq0);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double tau = (w(im, iq0) == 0.) ? 0. : (1. / w(im, iq0));
            Eigen::MatrixXd outer(3, 3);
            outer.fill(0.);

            if (alma::almost_equal(sp0.omega(im), 0.) or
                alma::almost_equal(sp0.vg.col(im).matrix().norm(), 0.))
                continue;

            for (auto iq : grid.get_equivalence(iequiv)) {
                const auto& sp = grid.get_spectrum_at_q(iq);
                Eigen::VectorXd vg = sp.vg.col(im);
                outer += vg * vg.transpose();
            }
            nruter +=
                alma::bose_einstein_kernel(sp0.omega[im], T) * tau * outer;
        }
    }

    nruter = (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;

    /// Symmetrise kappa tensor

    Eigen::Matrix3d nruter_accumulated;
    nruter_accumulated.fill(0.0);

    for (std::size_t nsymm = 0; nsymm < syms.get_nsym(); nsymm++) {
        nruter_accumulated += syms.rotate_m<double>(nruter, nsymm, true);
    }

    nruter = nruter_accumulated / static_cast<double>(syms.get_nsym());

    return nruter;
}

Eigen::MatrixXd calc_kappa_coherence(
    const Crystal_structure& cell,
    const Gamma_grid& grid,
    const Symmetry_operations& syms,
    const Eigen::Ref<const Eigen::ArrayXXd>& rates,
    double temperature) {
    const auto bands = grid.get_spectrum_at_q(0).omega.size();
    if (rates.rows() != bands ||
        rates.cols() != static_cast<Eigen::Index>(grid.nqpoints))
        throw value_error("inconsistent scattering-rate dimensions");
    if (!std::isfinite(temperature) || temperature <= 0. ||
        !rates.isFinite().all() || (rates < 0.).any())
        throw value_error("coherence requires positive temperature and finite "
                          "nonnegative rates");
    Eigen::Matrix3cd tensor = Eigen::Matrix3cd::Zero();
    // Use full-grid rates, keeping the operator and its band labels together.
    for (std::size_t q = 0; q < grid.nqpoints; ++q) {
        const auto& sp = grid.get_spectrum_at_q(q);
        if (sp.wigner_v.rows() != 3 || sp.wigner_v.cols() != bands * bands)
            throw value_error("coherence requires Wigner velocities; "
                              "regenerate the material file from IFCs");
        for (Eigen::Index i = 0; i < bands; ++i) {
            if (sp.omega(i) <= 0.)
                continue;
            for (Eigen::Index j = 0; j < bands; ++j) {
                // Keep degenerate blocks in the existing population treatment;
                // coherence here couples distinct frequency subspaces.
                if (sp.omega(j) <= 0. || almost_equal(sp.omega(i), sp.omega(j)))
                    continue;
                const double width = rates(i, q) + rates(j, q);
                if (width == 0.)
                    continue;
                const double difference = sp.omega(j) - sp.omega(i);
                const double lorentzian =
                    0.5 * width /
                    (difference * difference + 0.25 * width * width);
                const double capacity =
                    bose_einstein_kernel(sp.omega(i), temperature) /
                        sp.omega(i) +
                    bose_einstein_kernel(sp.omega(j), temperature) /
                        sp.omega(j);
                const Eigen::Vector3cd left = sp.wigner_v.col(i + bands * j);
                const Eigen::Vector3cd right = sp.wigner_v.col(j + bands * i);
                tensor += (sp.omega(i) + sp.omega(j)) * capacity * lorentzian *
                          left * right.transpose();
            }
        }
    }
    tensor *= 0.25e21 * constants::kB / cell.V / grid.nqpoints;
    Eigen::Matrix3cd symmetric = Eigen::Matrix3cd::Zero();
    for (std::size_t op = 0; op < syms.get_nsym(); ++op)
        symmetric += syms.rotate_m<std::complex<double>>(tensor, op, true);
    symmetric /= static_cast<double>(syms.get_nsym());
    if (symmetric.imag().cwiseAbs().maxCoeff() >
        1e-10 * std::max(1., symmetric.real().cwiseAbs().maxCoeff()))
        throw value_error(
            "coherence conductivity has a nonzero imaginary part");
    return symmetric.real();
}


Eigen::MatrixXd calc_kappa_sg(const alma::Crystal_structure& poscar,
                              const alma::Gamma_grid& grid,
                              const alma::Symmetry_operations& syms,
                              double T) {
    auto nequiv = grid.get_nequivalences();
    auto nmodes =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());
    Eigen::MatrixXd nruter(3, 3);

    nruter.fill(0.);

    // The Gamma point is ignored.
    for (decltype(nequiv) iequiv = 1; iequiv < nequiv; ++iequiv) {
        auto iq0 = grid.get_representative(iequiv);
        const auto& sp0 = grid.get_spectrum_at_q(iq0);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double modv = sp0.vg.col(im).matrix().norm();

            if (!almost_equal(0., modv)) {
                Eigen::MatrixXd outer(3, 3);
                outer.fill(0.);

                for (auto iq : grid.get_equivalence(iequiv)) {
                    const auto& sp = grid.get_spectrum_at_q(iq);
                    Eigen::VectorXd vg = sp.vg.col(im);
                    outer += vg * vg.transpose();
                }
                nruter +=
                    alma::bose_einstein_kernel(sp0.omega[im], T) * outer / modv;
            }
        }
    }

    nruter = (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;

    /// Symmetrise kappa tensor

    Eigen::Matrix3d nruter_accumulated;
    nruter_accumulated.fill(0.0);

    for (std::size_t nsymm = 0; nsymm < syms.get_nsym(); nsymm++) {
        nruter_accumulated += syms.rotate_m<double>(nruter, nsymm, true);
    }

    nruter = nruter_accumulated / static_cast<double>(syms.get_nsym());

    return nruter;
}


double calc_kappa_1d(const alma::Crystal_structure& poscar,
                     const alma::Gamma_grid& grid,
                     const Eigen::Ref<const Eigen::ArrayXXd>& w,
                     double T,
                     const Eigen::Ref<const Eigen::Vector3d>& direction) {
    if (!direction.allFinite() || direction.norm() == 0.)
        throw value_error("invalid direction");
    const Eigen::Vector3d unit = direction.normalized();
    const Eigen::Matrix3d tensor =
        calc_kappa(poscar, grid, Symmetry_operations(poscar), w, T);
    return unit.dot(tensor * unit);
}

std::pair<double, double> calc_phase_space(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const double T,
    alma::Threeph_processes& processes,
    Eigen::MatrixXd& P3plus,
    Eigen::MatrixXd& P3minus,
    Eigen::MatrixXd& WP3plus,
    Eigen::MatrixXd& WP3minus,
    const boost::mpi::communicator& world) {
    const auto nqpoints = grid.nqpoints;
    const auto nmodes = grid.get_spectrum_at_q(0).omega.size();

    if (!std::isfinite(T) || T <= 0.)
        throw value_error("phase space requires a finite positive temperature");
    // Convert before multiplying: valid large grids can overflow integer
    // powers.
    const double points = static_cast<double>(nqpoints);
    const double branches = static_cast<double>(nmodes);
    const double P3_factor =
        1. / (points * points * branches * branches * branches);

    constexpr double plus_factor = 2.0 / 3.0;
    constexpr double minus_factor = 1.0 / 3.0;

    std::pair<double, double> nruter = std::make_pair(0.0, 0.0);

    Eigen::MatrixXd my_P3plus(nmodes, nqpoints);
    my_P3plus.setZero();
    Eigen::MatrixXd my_P3minus(nmodes, nqpoints);
    my_P3minus.setZero();
    Eigen::MatrixXd my_WP3plus(nmodes, nqpoints);
    my_WP3plus.setZero();
    Eigen::MatrixXd my_WP3minus(nmodes, nqpoints);
    my_WP3minus.setZero();

    // Compute the contribution with the appropriate weight
    for (auto process : processes) {
        auto gaussian = process.compute_gaussian() * P3_factor;
        auto symmetry_weight = grid.get_cardinal(process.c);
        if (process.type == alma::threeph_type::absorption) {
            auto p3 = plus_factor * gaussian;
            nruter.first += p3 * symmetry_weight;
            my_P3plus(process.alpha[0], process.q[0]) += p3;
            auto wp3 = process.compute_weighted_gaussian(grid, T) * P3_factor;
            nruter.second += wp3 * symmetry_weight;
            my_WP3plus(process.alpha[0], process.q[0]) += wp3;
        }
        else {
            auto p3 = minus_factor * gaussian;
            nruter.first += p3 * symmetry_weight;
            my_P3minus(process.alpha[0], process.q[0]) += p3;
            auto wp3 = process.compute_weighted_gaussian(grid, T) * P3_factor;
            nruter.second += wp3 * symmetry_weight;
            my_WP3minus(process.alpha[0], process.q[0]) += wp3;
        }
    }

    // Regenerate values in the full BZ using symmetry
    auto nequivalences = grid.get_nequivalences();
    for (decltype(nequivalences) i = 0; i < nequivalences; ++i) {
        auto eq = grid.get_equivalence(i);

        for (std::size_t iq = 1; iq < eq.size(); ++iq) {
            my_P3plus.col(eq[iq]) = my_P3plus.col(eq[0]);
            my_WP3plus.col(eq[iq]) = my_WP3plus.col(eq[0]);
            my_P3minus.col(eq[iq]) = my_P3minus.col(eq[0]);
            my_WP3minus.col(eq[iq]) = my_WP3minus.col(eq[0]);
        }
    }

    // In place reduction of the nruter components
    nruter.first =
        boost::mpi::all_reduce(world, nruter.first, std::plus<double>());
    nruter.second =
        boost::mpi::all_reduce(world, nruter.second, std::plus<double>());

    P3plus.resize(nmodes, nqpoints);
    P3plus.setZero();
    P3minus.resize(nmodes, nqpoints);
    P3minus.setZero();
    WP3plus.resize(nmodes, nqpoints);
    WP3plus.setZero();
    WP3minus.resize(nmodes, nqpoints);
    WP3minus.setZero();

    // Reduction over all processes
    boost::mpi::all_reduce(world,
                           my_P3plus.data(),
                           my_P3plus.size(),
                           P3plus.data(),
                           std::plus<double>());
    boost::mpi::all_reduce(world,
                           my_WP3plus.data(),
                           my_WP3plus.size(),
                           WP3plus.data(),
                           std::plus<double>());
    boost::mpi::all_reduce(world,
                           my_P3minus.data(),
                           my_P3minus.size(),
                           P3minus.data(),
                           std::plus<double>());
    boost::mpi::all_reduce(world,
                           my_WP3minus.data(),
                           my_WP3minus.size(),
                           WP3minus.data(),
                           std::plus<double>());

    return nruter;
}


double calc_anharmonicity(const alma::Crystal_structure& poscar,
                          const alma::Gamma_grid& grid,
                          alma::Threeph_processes& processes,
                          Eigen::MatrixXd& vp2,
                          const boost::mpi::communicator& world) {
    const auto nqpoints = grid.nqpoints;
    const auto nmodes = grid.get_spectrum_at_q(0).omega.size();

    // constexpr double plus_factor  = 1.0;
    // constexpr double minus_factor = 1.0 / 2.0;

    double my_vp2_total = 0.0;
    std::size_t nplus = 0;
    std::size_t nminus = 0;
    Eigen::MatrixXd my_vp2(nmodes, nqpoints);
    my_vp2.setZero();

    /// Get the number of processes
    for (auto process : processes) {
        auto symmetry_weight = grid.get_cardinal(process.c);
        if (process.type == alma::threeph_type::absorption) {
            nplus += symmetry_weight;
        }
        else {
            nminus += symmetry_weight;
        }
    }

    nplus = boost::mpi::all_reduce(world, nplus, std::plus<std::size_t>());
    nminus = boost::mpi::all_reduce(world, nminus, std::plus<std::size_t>());
    auto ntotal = nplus + nminus;
    if (ntotal == 0) {
        vp2 = my_vp2;
        return 0.;
    }

    // Compute the contribution to the mean
    for (auto process : processes) {
        auto matel = process.get_vp2() / ntotal;
        auto symmetry_weight = grid.get_cardinal(process.c);
        my_vp2_total += symmetry_weight * matel;
        my_vp2(process.alpha[0], process.q[0]) += matel;
    }

    // Regenerate values in the full BZ using symmetry
    auto nequivalences = grid.get_nequivalences();
    for (decltype(nequivalences) i = 0; i < nequivalences; ++i) {
        auto eq = grid.get_equivalence(i);
        for (std::size_t iq = 1; iq < eq.size(); ++iq)
            my_vp2.col(eq[iq]) = my_vp2.col(eq[0]);
    }

    // In place reduction of the nruter components
    my_vp2_total =
        boost::mpi::all_reduce(world, my_vp2_total, std::plus<double>());

    vp2.resize(nmodes, nqpoints);
    vp2.setZero();

    // Reduction over all processes
    boost::mpi::all_reduce(
        world, my_vp2.data(), my_vp2.size(), vp2.data(), std::plus<double>());

    return my_vp2_total;
}

} // namespace alma
