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
/// Definitions corresponding to bulk_properties.hpp

#include <bulk_properties.hpp>
#include <analytic1d.hpp>
#include <boost/math/special_functions/pow.hpp>

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

    // The Gamma point is ignored.
    for (decltype(nequiv) iequiv = 1; iequiv < nequiv; ++iequiv) {
        auto iq0 = grid.get_representative(iequiv);
        auto sp0 = grid.get_spectrum_at_q(iq0);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double tau = (w(im, iq0) == 0.) ? 0. : (1. / w(im, iq0));
            Eigen::MatrixXd outer(3, 3);
            outer.fill(0.);

            for (auto iq : grid.get_equivalence(iequiv)) {
                auto sp = grid.get_spectrum_at_q(iq);
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

Eigen::MatrixXd calc_kappa_coherence(const alma::Crystal_structure& poscar,
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
    Eigen::MatrixXcd nruter(3, 3);
    nruter.fill(std::complex<double>(0.,0.));

    for (decltype(nequiv) iequiv = 0; iequiv < nequiv; ++iequiv) {
        
        auto iq0 = grid.get_representative(iequiv);
        auto sp0 = grid.get_spectrum_at_q(iq0);

        // Modes coupling with 0 frequency are ignored (i.e. acoustic modes at Gamma).
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {

            if (alma::almost_equal(sp0.omega(im),0.)) continue;
            double Gamma = w(im, iq0);
            
            for (decltype(nmodes) imp = 0; imp < nmodes; ++im) {

                if (alma::almost_equal(sp0.omega(imp),0.) || im == imp) continue;
                double Gammap = w(imp, iq0);

                // The complex elements of the matrix should vanish in the total summation
                // I keep them until the end for a last sanity check
                Eigen::Matrix3cd wigner_vvp = (
                    sp0.wigner_v.col(im+nmodes*imp).matrix() * 
                    sp0.wigner_v.col(imp+nmodes*im).matrix().transpose());

                auto Gamma_factor  = 0.5 * (Gamma + Gammap) / (
                    boost::math::pow<2>(sp0.omega(imp) - sp0.omega(im)) +
                    0.25 * boost::math::pow<2>(  Gamma + Gammap ));

                auto Cfactor = alma::bose_einstein_kernel(sp0.omega[im], T) / sp0.omega[im] +
                               alma::bose_einstein_kernel(sp0.omega[imp], T) / sp0.omega[imp];

                nruter += 0.25 * (sp0.omega(imp) + sp0.omega(im)) * Cfactor *
                        Gamma_factor * wigner_vvp;

            }
        }
    }

    nruter = (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;

    /// Symmetrise the complex kappa tensor

    Eigen::Matrix3cd nruter_accumulated;
    nruter_accumulated.fill(0.0);

    for (std::size_t nsymm = 0; nsymm < syms.get_nsym(); nsymm++) {
        nruter_accumulated += syms.rotate_m<std::complex<double>>(nruter, nsymm, true);
    }

    nruter = nruter_accumulated / static_cast<double>(syms.get_nsym());

    /// Check that the imaginary part is small
    if (!alma::almost_equal(nruter.imag().maxCoeff(),0.)){
        std::cerr << nruter << std::endl;
        throw alma::value_error("Imaginary terms of the coherence contribution are not null.");
    }

    // Return the real part
    return nruter.real();
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
        auto sp0 = grid.get_spectrum_at_q(iq0);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double modv = sp0.vg.col(im).matrix().norm();

            if (!almost_equal(0., modv)) {
                Eigen::MatrixXd outer(3, 3);
                outer.fill(0.);

                for (auto iq : grid.get_equivalence(iequiv)) {
                    auto sp = grid.get_spectrum_at_q(iq);
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


    return nruter;
}


double calc_kappa_1d(const alma::Crystal_structure& poscar,
                     const alma::Gamma_grid& grid,
                     const Eigen::Ref<const Eigen::ArrayXXd>& w,
                     double T,
                     const Eigen::Ref<const Eigen::Vector3d>& direction) {
    auto nqpoints = grid.nqpoints;
    auto nmodes =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());

    if ((static_cast<std::size_t>(w.rows()) != nmodes) ||
        (static_cast<std::size_t>(w.cols()) != grid.nqpoints))
        throw alma::value_error("inconsistent dimensions");
    double unorm = direction.norm();

    if (unorm == 0.)
        throw alma::value_error("invalid direction");
    Eigen::Vector3d u = direction / unorm;
    double nruter = 0.;

    // The Gamma point is ignored.
    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto spectrum = grid.get_spectrum_at_q(iq);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double tau = (w(im, iq) == 0.) ? 0. : (1. / w(im, iq));
            double v = u.dot(spectrum.vg.matrix().col(im));
            nruter +=
                alma::bose_einstein_kernel(spectrum.omega[im], T) * tau * v * v;
        }
    }
    return (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;
}
} // namespace alma
