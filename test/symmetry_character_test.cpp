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
/// Test the representation matrices of the symmetry operations on the
/// phonon eigenvectors, and the characters derived from them.
///
/// Silicon (Fd-3m, 48 operations) is used. At Gamma the six modes
/// split into the acoustic T1u irrep and the optic T2g irrep of Oh.

#include <cmath>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <constants.hpp>
#include <vasp_io.hpp>
#include <symmetry.hpp>
#include <dynamical_matrix.hpp>
#include <qpoint_grid.hpp>

namespace {
/// Conjugacy classes of Oh, labelled by the class of the proper rotation
/// obtained by multiplying the operation by its determinant.
enum class Oh_class { E, C3, C2, C4, C2p };

Oh_class classify(const Eigen::MatrixXd& crot) {
    const double det = crot.determinant() > 0. ? 1. : -1.;
    const Eigen::MatrixXd proper = det * crot;
    const long tr = std::lround(proper.trace());

    if (tr == 3)
        return Oh_class::E;
    if (tr == 0)
        return Oh_class::C3;
    if (tr == 1)
        return Oh_class::C4;
    // tr == -1: C2 around a cubic axis (diagonal matrix) or around a
    // face diagonal (C2').
    Eigen::MatrixXd offdiag = proper;
    offdiag.diagonal().setZero();
    return offdiag.norm() < 1e-6 ? Oh_class::C2 : Oh_class::C2p;
}

/// Characters of the T2g irrep (optic modes of Si at Gamma) for the
/// proper rotations. Even parity: improper operations have the same
/// character as their proper counterpart.
double chi_T2g(Oh_class c) {
    switch (c) {
    case Oh_class::E:
        return 3.;
    case Oh_class::C3:
        return 0.;
    case Oh_class::C2:
        return -1.;
    case Oh_class::C4:
        return -1.;
    case Oh_class::C2p:
        return 1.;
    }
    return 0.;
}

/// Characters of the T1u irrep (acoustic modes at Gamma). Odd parity:
/// improper operations have minus the character of their proper
/// counterpart.
double chi_T1u(Oh_class c, bool improper) {
    double proper = 0.;
    switch (c) {
    case Oh_class::E:
        proper = 3.;
        break;
    case Oh_class::C3:
        proper = 0.;
        break;
    case Oh_class::C2:
        proper = -1.;
        break;
    case Oh_class::C4:
        proper = 1.;
        break;
    case Oh_class::C2p:
        proper = -1.;
        break;
    }
    return improper ? -proper : proper;
}
} // namespace


class Silicon_character_test : public ::testing::Test {
public:
    std::unique_ptr<alma::Crystal_structure> poscar;
    std::unique_ptr<alma::Symmetry_operations> syms;
    std::unique_ptr<alma::Dynamical_matrix_builder> factory;

    virtual void SetUp() {
        auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
        auto si_dir = basedir / boost::filesystem::path("Si");
        auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
        auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");

        poscar = alma::load_POSCAR(poscar_path.string().c_str());
        syms = alma::make_unique<alma::Symmetry_operations>(*poscar);
        auto force_constants = alma::load_FORCE_CONSTANTS(
            ifc_path.string().c_str(), *poscar, 5, 5, 5);
        factory = alma::make_unique<alma::Dynamical_matrix_builder>(
            *poscar, *syms, *force_constants);
    }
};


TEST_F(Silicon_character_test, optic_characters_at_gamma) {
    Eigen::Vector3d gamma(Eigen::Vector3d::Zero());
    auto spectrum = factory->get_spectrum(gamma);

    // The optic modes are the three highest ones.
    Eigen::MatrixXcd optic = spectrum->wfs.block(0, 3, 6, 3);

    // Every operation of the crystal belongs to the little group of Gamma.
    auto little = syms->get_little_group(*poscar, gamma);
    ASSERT_EQ(syms->get_nsym(), little.size());
    ASSERT_EQ(48u, little.size());

    for (auto iop : little) {
        auto chi = syms->get_character(*poscar, iop, gamma, optic);
        auto expected = chi_T2g(classify(syms->get_crotation(iop)));
        EXPECT_NEAR(expected, chi.real(), 1e-6) << "operation " << iop;
        EXPECT_NEAR(0., chi.imag(), 1e-6) << "operation " << iop;
    }

    // Sum of |chi|^2 over the group equals the group order for an
    // irreducible representation.
    double norm = 0.;

    for (auto iop : little)
        norm += std::norm(syms->get_character(*poscar, iop, gamma, optic));
    EXPECT_NEAR(48., norm, 1e-5);
}


TEST_F(Silicon_character_test, acoustic_characters_at_gamma) {
    Eigen::Vector3d gamma(Eigen::Vector3d::Zero());
    auto spectrum = factory->get_spectrum(gamma);
    Eigen::MatrixXcd acoustic = spectrum->wfs.block(0, 0, 6, 3);

    for (std::size_t iop = 0; iop < syms->get_nsym(); ++iop) {
        auto chi = syms->get_character(*poscar, iop, gamma, acoustic);
        auto expected = chi_T1u(classify(syms->get_crotation(iop)),
                                syms->is_inversion(iop));
        EXPECT_NEAR(expected, chi.real(), 1e-6) << "operation " << iop;
        EXPECT_NEAR(0., chi.imag(), 1e-6) << "operation " << iop;
    }
}


TEST_F(Silicon_character_test, representation_commutes_with_dynamical_matrix) {
    const double a = 0.5431; // nm
    const double twopi = 2. * alma::constants::pi;
    std::vector<Eigen::Vector3d> qs;
    qs.emplace_back(Eigen::Vector3d(0.3, 0., 0.) * twopi / a);
    qs.emplace_back(Eigen::Vector3d(0.2, 0.2, 0.) * twopi / a);
    qs.emplace_back(Eigen::Vector3d(0.25, 0.25, 0.25) * twopi / a);
    // A q point on the edge of the zone, where Rq = q + G with G != 0.
    qs.emplace_back(Eigen::Vector3d(1., 0., 0.) * twopi / a);

    for (const auto& q : qs) {
        auto little = syms->get_little_group(*poscar, q);
        EXPECT_GT(little.size(), 1u);
        Eigen::MatrixXcd dyn = factory->build(q)[0];

        for (auto iop : little) {
            Eigen::MatrixXcd S = syms->get_representation(*poscar, iop, q);
            // Unitary.
            EXPECT_NEAR(
                0.,
                (S.adjoint() * S - Eigen::MatrixXcd::Identity(6, 6)).norm(),
                1e-8);
            // Commutes with the dynamical matrix at q.
            EXPECT_NEAR(0., (S * dyn - dyn * S).norm(), 1e-6)
                << "operation " << iop;
        }
    }
}


TEST_F(Silicon_character_test, copy_symmetry_with_character_is_a_projector) {
    static boost::mpi::environment env;
    auto force_constants = alma::load_FORCE_CONSTANTS(
        (boost::filesystem::path(TEST_RESOURCE_DIR) / "Si" / "FORCE_CONSTANTS")
            .string()
            .c_str(),
        *poscar,
        5,
        5,
        5);
    alma::Gamma_grid grid(*poscar, *syms, *force_constants, 5, 5, 5);
    grid.enforce_asr();

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        const auto& sp = grid.get_spectrum_at_q(iq);
        Eigen::ArrayXXcd once = grid.copy_symmetry_with_character(
            iq, *poscar, *syms, sp.wigner_v, sp.wfs);
        Eigen::ArrayXXcd twice =
            grid.copy_symmetry_with_character(iq, *poscar, *syms, once, sp.wfs);
        EXPECT_NEAR(0., (once - twice).abs().maxCoeff(), 1e-8)
            << "q point " << iq;
    }
}

/// The tensors T^{ab}_{AB} = sum_{s in A, s' in B} v^a_{ss'} v^b_{s's}, where
/// A and B are different degenerate subspaces, do not depend on the basis
/// chosen inside the subspaces, and they are what enters the coherence
/// conductivity. They must therefore be identical before and after
/// imposing the point group.
TEST_F(Silicon_character_test, symmetrization_preserves_basis_invariant_tensors) {
    static boost::mpi::environment env;
    auto force_constants = alma::load_FORCE_CONSTANTS(
        (boost::filesystem::path(TEST_RESOURCE_DIR) / "Si" / "FORCE_CONSTANTS")
            .string()
            .c_str(),
        *poscar,
        5,
        5,
        5);
    alma::Gamma_grid grid(*poscar, *syms, *force_constants, 5, 5, 5);

    auto invariants = [](const alma::Spectrum_at_point& sp) {
        const int n = sp.omega.size();
        std::array<Eigen::MatrixXcd, 3> M;

        for (int a = 0; a < 3; ++a) {
            M[a].resize(n, n);

            for (int t = 0; t < n; ++t)
                for (int s = 0; s < n; ++s)
                    M[a](s, t) = sp.wigner_v(a, s + n * t);
        }
        // Label the degenerate subspaces.
        std::vector<int> label(n, 0);

        for (int s = 1; s < n; ++s)
            label[s] = alma::almost_equal(sp.omega(s), sp.omega(s - 1))
                           ? label[s - 1]
                           : label[s - 1] + 1;
        const int nblocks = label[n - 1] + 1;
        Eigen::MatrixXcd nruter = Eigen::MatrixXcd::Zero(9, nblocks * nblocks);

        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int s = 0; s < n; ++s)
                    for (int t = 0; t < n; ++t)
                        if (label[s] != label[t])
                            nruter(3 * a + b, label[s] * nblocks + label[t]) +=
                                M[a](s, t) * M[b](t, s);
        return nruter;
    };

    for (std::size_t iq = 1; iq < grid.nqpoints; ++iq) {
        auto raw = factory->get_spectrum(grid.get_q(iq));
        auto ref = invariants(*raw);
        auto sym = invariants(grid.get_spectrum_at_q(iq));
        const double scale = std::max(1e-12, ref.cwiseAbs().maxCoeff());
        EXPECT_LT((ref - sym).cwiseAbs().maxCoeff() / scale, 1e-6)
            << "q point " << iq;
    }
}
