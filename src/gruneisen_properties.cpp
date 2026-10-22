// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2015-2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <gruneisen.hpp>
namespace alma {
/// Compute the Grüneisen parameter for each mode on a grid.
///
/// @param[in] cell - description of the unit cell
/// @param[in] grid - regular grid with phonon spectrum
/// @param[in] fc3 - set of third-order force constants
/// @return an array with the mode Grüneisen parameters for each q point
/// and each branch.
Eigen::ArrayXXd calc_mode_gruneisen(
    const alma::Crystal_structure& cell,
    const alma::Gamma_grid& grid,
    const std::vector<alma::Thirdorder_ifcs>& fc3) {
    // Factor to transform from nm*eV/(amu * A ** 3 * THz ** 2) to unitless
    double prefactor = alma::constants::e * 1e-3 / alma::constants::amu;
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    Eigen::ArrayXXd nruter(Eigen::ArrayXXd::Zero(nmodes, grid.nqpoints));

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        const auto& spectrum = grid.get_spectrum_at_q(iq);
        for (std::size_t im = 0; im < nmodes; ++im) {
            if (spectrum.omega(im) == 0.) {
                continue;
            }
            for (const auto& tri : fc3) {
                double massfactor =
                    std::sqrt(cell.get_mass(tri.i) * cell.get_mass(tri.j));
                double arg = grid.get_q(iq).dot(tri.rj);
                std::complex<double> factor1 =
                    (std::cos(arg) + alma::constants::imud * std::sin(arg)) /
                    massfactor;
                Eigen::Vector3d cart =
                    cell.lattvec * cell.positions.col(tri.k) + tri.rk;
                for (auto rr = 0; rr < 3; ++rr) {
                    std::complex<double> factor2 =
                        factor1 * std::conj(spectrum.wfs(3 * tri.i + rr, im));
                    for (auto ss = 0; ss < 3; ++ss) {
                        std::complex<double> factor3 =
                            factor2 * spectrum.wfs(3 * tri.j + ss, im);
                        for (auto tt = 0; tt < 3; ++tt) {
                            nruter(im, iq) += std::real(
                                factor3 * tri.ifc(rr, ss, tt) * cart(tt));
                        }
                    }
                }
            }
            nruter(im, iq) *=
                -prefactor / 6. / boost::math::pow<2>(spectrum.omega(im));
        }
    }
    return nruter;
}

/// Compute the total Grüneisen parameter as a weighted average of mode
/// Grüneisen parameters.
///
/// @param[in] grid - regular grid with phonon spectrum
/// @param[in] modegrun - Grüneisen parameters for each q point and each branch
/// @param[in] T - temperature in K
/// @return the total Grüneisen parameter at the given temperature
double calc_total_gruneisen(const alma::Gamma_grid& grid,
                            const Eigen::Ref<const Eigen::ArrayXXd>& modegrun,
                            double T) {
    if (!std::isfinite(T) || T <= 0. ||
        modegrun.rows() != grid.get_spectrum_at_q(0).omega.size() ||
        modegrun.cols() != static_cast<Eigen::Index>(grid.nqpoints) ||
        !modegrun.isFinite().all())
        throw value_error("invalid Gruneisen parameters or temperature");
    double nruter = 0.;
    double denominator = 0.;
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    // The weight of each mode is proportional to its contribution to the heat
    // capacity.
    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        const auto& spectrum = grid.get_spectrum_at_q(iq);
        for (std::size_t im = 0; im < nmodes; ++im) {
            double weight = alma::bose_einstein_kernel(spectrum.omega(im), T);
            nruter += weight * modegrun(im, iq);
            denominator += weight;
        }
    }
    if (denominator <= 0.)
        throw alma::value_error(
            "Gruneisen average has no heat-capacity weight");
    nruter /= denominator;
    return nruter;
}


} // namespace alma
