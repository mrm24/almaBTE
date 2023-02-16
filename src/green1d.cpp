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

#include <cmath>
#include <tuple>
#include <vector>
#include <array>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <utilities.hpp>
#include <constants.hpp>
#include <green1d.hpp>
#include <aux_cubic.hpp>

/// @file
/// Definitions corresponding to green1d.hpp.

namespace alma {
Green1d_factory::Green1d_factory(
    const Crystal_structure& structure,
    const Eigen::Ref<const Eigen::VectorXd>& q0,
    const Eigen::Ref<const Eigen::VectorXi>& normal,
    const std::size_t nqpoints_,
    const Dynamical_matrix_builder& builder)
    : nqpoints{nqpoints_}, ndof{3 * static_cast<std::size_t>(
                                        structure.get_natoms())},
      offset{2. * constants::pi *
             (structure.rlattvec * normal.cast<double>()).dot(q0) /
             (structure.rlattvec * normal.cast<double>()).squaredNorm()},
      qgrid{nqpoints}, Egrid{ndof, nqpoints}, dEgrid{ndof, nqpoints} {
    Eigen::VectorXd bz{structure.rlattvec * normal.cast<double>() / 2. /
                       alma::constants::pi};
    this->wfgrid.reserve(this->nqpoints);

    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        double q{2. * alma::constants::pi * iq / this->nqpoints};
        auto spectrum = builder.get_spectrum(q0 + q * bz);
        this->qgrid(iq) = q;
        this->Egrid.col(iq) = spectrum->omega * spectrum->omega;
        this->dEgrid.col(iq) = 2. * spectrum->omega *
                               (spectrum->vg.matrix().transpose() * bz).array();
        this->wfgrid.emplace_back(spectrum->wfs);
    }
}

Eigen::MatrixXcd Green1d_factory::build(double omega,
                                        std::size_t ncells) const {
    auto E = omega * omega;
    std::size_t sndof = this->ndof * ncells;
    Eigen::MatrixXcd nruter{Eigen::MatrixXcd::Zero(sndof, sndof)};
    // Build the Green's function by adding all the contributions
    // from all q points and all branches.
    auto weights = this->compute_weights(E);

    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        auto swfs = this->get_superwfs(iq, ncells);

        for (std::size_t im = 0; im < this->ndof; ++im)
            nruter += weights(im, iq) * (swfs.col(im) * swfs.col(im).adjoint());
    }
    return nruter;
}


double Green1d_factory::calc_w0_dmass(
    double q,
    double omega,
    const Eigen::Ref<const Eigen::VectorXcd>& wfin,
    const Eigen::Ref<const Eigen::VectorXd>& factors) const {
    auto E = omega * omega;
    auto natoms = this->ndof / 3;

    // Number of unit cells spanned by the perturbation.
    if (factors.size() % natoms)
        throw value_error("the perturbation must span an"
                          "integer number of unit cells");
    auto ncells = factors.size() / natoms;
    std::size_t sndof{ncells * this->ndof};
    // Perturbation matrix.
    Eigen::MatrixXcd V{Eigen::MatrixXcd::Zero(sndof, sndof)};

    for (std::size_t i = 0; i < sndof; ++i)
        V(i, i) = E * factors(i / 3);
    // Green's function.
    auto gf = this->build(omega, ncells);
    // t matrix
    Eigen::MatrixXcd imvg{Eigen::MatrixXcd::Identity(sndof, sndof) - V * gf};
    Eigen::MatrixXcd t{imvg.colPivHouseholderQr().solve(V)};
    // Extend the incident wave function to the supercell.
    Eigen::VectorXcd swf{sndof};

    for (std::size_t ic = 0; ic < ncells; ++ic) {
        double arg{ic * q};
        std::complex<double> expfactor{std::cos(arg), -std::sin(arg)};
        swf.segment(ndof * ic, ndof) = wfin * expfactor;
    }
    // Get the scattering rate from the optical theorem.
    return std::max(0., -swf.dot(t * swf).imag() / omega) / ncells;
}


Eigen::MatrixXcd Green1d_factory::get_superwfs(std::size_t iq,
                                               std::size_t ncells) const {
    std::size_t sndof = this->ndof * ncells;
    Eigen::MatrixXcd nruter{sndof, this->ndof};
    double q{this->qgrid(iq) + this->offset};

    for (std::size_t ic = 0; ic < ncells; ++ic) {
        double arg{ic * q};
        std::complex<double> expfactor{std::cos(arg), -std::sin(arg)};
        nruter.middleRows(ndof * ic, ndof) = this->wfgrid[iq] * expfactor;
    }
    return nruter;
}


Eigen::MatrixXcd Green1d_factory::compute_weights(double omega2) const {
    Eigen::MatrixXcd nruter{Eigen::MatrixXcd::Zero(this->ndof, this->nqpoints)};

    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        std::size_t iq1;
        double dq;

        if (iq == nqpoints - 1) {
            iq1 = 0;
            dq = 1. + (this->qgrid(iq1) - this->qgrid(iq)) / 2. / constants::pi;
        }
        else {
            iq1 = iq + 1;
            dq = (this->qgrid(iq1) - this->qgrid(iq)) / 2. / constants::pi;
        }
        double dq2pi{2. * constants::pi * dq};

        for (std::size_t im = 0; im < this->ndof; ++im) {
            double E0{this->Egrid(im, iq)};
            double E1{this->Egrid(im, iq1)};
            double dE0{this->dEgrid(im, iq) * dq2pi};
            double dE1{this->dEgrid(im, iq1) * dq2pi};
            aux_cubic::Cubic_segment segment(E0, E1, dE0, dE1);
            auto integrals = segment.calc_integrals(omega2);
            nruter(im, iq) +=
                std::complex<double>((integrals[0] - integrals[1]) * dq,
                                     -(integrals[2] - integrals[3]) * dq);
            nruter(im, iq1) +=
                std::complex<double>(integrals[1] * dq, -integrals[3] * dq);
        }
    }
    return nruter;
}
} // namespace alma
