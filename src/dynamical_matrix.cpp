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
/// Definitions corresponding to dynamical_matrix.hpp.

#include <iostream>
#include <complex>
#include <cmath>
#include <constants.hpp>
#include <utilities.hpp>
#include <periodic_table.hpp>
#include <dynamical_matrix.hpp>
#include <boost/math/special_functions/pow.hpp>

namespace alma {

/// @brief Overload of the stream insertion operator for nonanalytic_treatment
/// enum.
///
/// This function enables printing of alma::nonanalytic_treatment values
/// using std::ostream (e.g., std::cout) in human-readable form ("gonze",
/// "wang").
///
/// @param os The output stream to which the enum will be written.
/// @param method The nonanalytic_treatment enum value to print.
/// @return A reference to the modified output stream.
std::ostream& operator<<(std::ostream& os,
                         const nonanalytic_treatment& method) {
    switch (method) {
        case alma::nonanalytic_treatment::gonze:
            os << "gonze";
            break;
        case alma::nonanalytic_treatment::wang:
            os << "wang";
            break;
        default:
            os << "unknown";
    }
    return os;
}

/// Return a square matrix with 3 * natoms rows where
/// element ij is equal to the square root of the products
/// of the masses of atom i/3 and atom j/3.
///
/// @param[in] structure - structure of the unit cell
/// @return - the aforementioned square matrix
Eigen::ArrayXXd build_mass_matrix(const Crystal_structure& structure) {
    auto natoms = structure.get_natoms();
    auto ndof = 3 * natoms;
    Eigen::ArrayXd m(ndof);

    for (auto i = 0; i < natoms; ++i)
        m.segment<3>(3 * i).setConstant(std::sqrt(structure.get_mass(i)));
    return m.matrix() * m.matrix().transpose();
}

/// Find all atom pairs in a Wigner-Seitz representation
/// of an na x nb x nc supercell.
///
/// @param[in] structure - description of the unit cell
/// @param[in] fcs - Harmomic_ifcs object adapted to the supercell
/// @return - a vector of Atom_pair
std::vector<Atom_pair> get_normal_pairs(const Crystal_structure& structure,
                                        const Harmonic_ifcs& fcs,
                                        int na,
                                        int nb,
                                        int nc) {
    std::vector<Atom_pair> nruter;
    auto natoms = structure.get_natoms();
    Eigen::Vector3d delta;
    Eigen::Vector3d sdelta;

    for (auto iatom1 = 0; iatom1 < natoms; ++iatom1)
        for (auto iatom2 = 0; iatom2 < natoms; ++iatom2)
            for (auto p : fcs.pos) {
                auto ia = p[0];
                auto ib = p[1];
                auto ic = p[2];
                delta << ia, ib, ic;
                delta += structure.positions.col(iatom1).transpose();
                delta -= structure.positions.col(iatom2).transpose();
                auto dmin = Min_keeper<Triple_int>();

                for (auto sa = -2; sa < 3; ++sa)
                    for (auto sb = -2; sb < 3; ++sb)
                        for (auto sc = -2; sc < 3; ++sc) {
                            sdelta << sa * na, sb * nb, sc * nc;
                            sdelta = structure.lattvec * (delta + sdelta);
                            dmin.update(Triple_int({{ia + sa * na,
                                                     ib + sb * nb,
                                                     ic + sc * nc}}),
                                        sdelta.squaredNorm());
                        }
                nruter.emplace_back(Atom_pair{
                    iatom1, iatom2, {{ia, ib, ic}}, dmin.get_vector()});
            }
    return nruter;
}


void Dynamical_matrix_builder::copy_blocks(const Harmonic_ifcs& fcs) {
    auto natoms = this->structure.get_natoms();
    auto ndof = 3 * natoms;

    this->massmatrix = build_mass_matrix(this->structure);

    auto pairs =
        get_normal_pairs(this->structure, fcs, this->na, this->nb, this->nc);

    // "Unfold" the blocks in fcs to obtain the blocks of the
    // dynamical matrix.
    Triple_int_map<Eigen::MatrixXd> blocks;
    Triple_int_map<Eigen::MatrixXd> masks;

    for (auto p : pairs) {
        auto ip = std::distance(
            fcs.pos.begin(), std::find(fcs.pos.begin(), fcs.pos.end(), p.cj));
        // The transposition is required by Phonopy's conventions
        // about indices.
        Eigen::MatrixXd block =
            ((fcs.ifcs[ip] / p.cjp.size()).array() / this->massmatrix)
                .transpose();
        Eigen::MatrixXd mask = Eigen::MatrixXd::Constant(
            fcs.ifcs[ip].rows(), fcs.ifcs[ip].cols(), 1. / p.cjp.size());

        for (auto pp : p.cjp) {
            if (blocks.find(pp) == blocks.end()) {
                blocks[pp] = Eigen::MatrixXd::Zero(ndof, ndof);
                masks[pp] = Eigen::MatrixXd::Zero(ndof, ndof);
            }
            blocks[pp].block<3, 3>(3 * p.i, 3 * p.j) =
                block.block<3, 3>(3 * p.i, 3 * p.j);
            masks[pp].block<3, 3>(3 * p.i, 3 * p.j) =
                mask.block<3, 3>(3 * p.i, 3 * p.j);
        }
    }


    auto kav = split_keys_and_values(blocks);
    this->pos.swap(std::get<0>(kav));
    this->blocks.swap(std::get<1>(kav));
    this->mpos.resize(3, this->blocks.size());
    kav = split_keys_and_values(masks);
    this->masks.swap(std::get<1>(kav));

    for (decltype(this->blocks.size()) i = 0; i < this->blocks.size(); ++i)
        for (auto j = 0; j < 3; ++j)
            this->mpos(j, i) = this->pos[i][j];
    this->cpos = this->structure.lattvec * this->mpos;

    // Convert from eV / A^2 / amu to (rad / ps)^2.
    for (auto& p : this->blocks)
        p *= constants::e / constants::amu * 1e-4;


    // When Gonze's NAC is used one needs to substract the dipole-dipole
    // interaction present in the supercell force constants.
    //
    // See Eqs. 52-54 of 10.1088/1361-648X/acd831
    //
    if (this->nonanalytic &&
        (this->nonanalytic_method == nonanalytic_treatment::gonze))
        remove_dipole_dipole(fcs, pairs);
}

void Dynamical_matrix_builder::remove_dipole_dipole(
    const Harmonic_ifcs& fcs,
    std::vector<Atom_pair>& pairs) {
    auto natoms = this->structure.get_natoms();
    auto ndof = 3 * natoms;

    // Compute the number of conmensurate points
    int n_commensurate = this->na * this->nb * this->nc;

    // Get the conmensurate points
    Eigen::MatrixXd conmensurate_points(3, n_commensurate);

    int idx = 0;
    for (int a = 0; a < this->na; a++)
        for (int b = 0; b < this->nb; b++)
            for (int c = 0; c < this->nc; c++) {
                conmensurate_points(0, idx) = static_cast<double>(a) / this->na;
                conmensurate_points(1, idx) = static_cast<double>(b) / this->nb;
                conmensurate_points(2, idx) = static_cast<double>(c) / this->nc;
                idx++;
            }


    // Obtain them in Cartesian coordinates
    conmensurate_points = this->structure.rlattvec * conmensurate_points;

    // Map to the first bz
    for (int i = 0; i < n_commensurate; i++)
        conmensurate_points.col(i) =
            this->structure.map_to_firstbz(conmensurate_points.col(i)).col(0);

    // Now compute the dipole-dipole dynamical matrices and substract it from
    // the supercell one
    std::vector<Eigen::MatrixXcd> dynamical_matrix_sr(
        n_commensurate, Eigen::MatrixXcd::Zero(ndof, ndof));
    for (int i = 0; i < n_commensurate; i++) {
        auto nac = this->build_nac_gonze(conmensurate_points.col(i));
        auto dyn = this->build_interpolated(conmensurate_points.col(i));
        dynamical_matrix_sr[i] = dyn[0] - nac[0].matrix();
    }

    // Compute the correct short range force constants
    // This is done by inverse Fourier transforming the
    // short-range dynamical matrix
    Triple_int_map<Eigen::MatrixXd> blocks;

    for (auto p : pairs) {
        Eigen::MatrixXcd fc_ij = Eigen::MatrixXcd::Zero(3, 3);

        for (int i = 0; i < n_commensurate; i++) {
            // We average over equivalent pairs
            std::complex<double> phase(0., 0.);
            for (auto pp : p.cjp) {
                Eigen::Vector3d Rb = pp[0] * this->structure.lattvec.col(0) +
                                     pp[1] * this->structure.lattvec.col(1) +
                                     pp[2] * this->structure.lattvec.col(2);
                phase += std::exp(constants::imud *
                                  Rb.dot(conmensurate_points.col(i)));
            }
            fc_ij += phase *
                     dynamical_matrix_sr[i].block<3, 3>(3 * p.i, 3 * p.j) /
                     p.cjp.size();
        }

        fc_ij /= n_commensurate;

        for (auto pp : p.cjp) {
            if (blocks.find(pp) == blocks.end()) {
                blocks[pp] = Eigen::MatrixXd::Zero(ndof, ndof);
            }
            // Note that the mass factor is not needed
            blocks[pp].block<3, 3>(3 * p.i, 3 * p.j) =
                fc_ij.real().array() / p.cjp.size();
        }
    }


    // Overwrite the original force constants
    // with the short range ones
    auto kav = split_keys_and_values(blocks);
    this->pos.swap(std::get<0>(kav));
    this->blocks.swap(std::get<1>(kav));
    this->mpos.resize(3, this->pos.size());
    for (std::size_t k = 0; k < pos.size(); ++k)
        for (int axis = 0; axis < 3; ++axis)
            mpos(axis, k) = pos[k][axis];
    this->cpos = structure.lattvec * mpos;
}


Dynamical_matrix_builder::Dynamical_matrix_builder(
    const Crystal_structure& _structure,
    const Symmetry_operations& syms,
    const Harmonic_ifcs& fcs)
    : na(fcs.na), nb(fcs.nb), nc(fcs.nc), V(_structure.V),
      structure(_structure), rlattvec(_structure.rlattvec), symmetries(syms),
      nonanalytic(false), nonanalytic_method(nonanalytic_treatment::none) {
    this->copy_blocks(fcs);
}


Dynamical_matrix_builder::Dynamical_matrix_builder(
    const Crystal_structure& _structure,
    const Symmetry_operations& syms,
    const Harmonic_ifcs& fcs,
    const Dielectric_parameters& _born,
    const nonanalytic_treatment _nonanalytic_method)
    : na(fcs.na), nb(fcs.nb), nc(fcs.nc), V(_structure.V),
      structure(_structure), rlattvec(_structure.rlattvec), symmetries(syms),
      nonanalytic(true), nonanalytic_method(_nonanalytic_method), born(_born) {
    if (this->born.born.size() !=
        static_cast<std::size_t>(this->structure.get_natoms()))
        throw value_error("wrong number of Born charges");
    if (nonanalytic_method != nonanalytic_treatment::none &&
        nonanalytic_method != nonanalytic_treatment::wang &&
        nonanalytic_method != nonanalytic_treatment::gonze)
        throw value_error("unsupported nonanalytic correction method");
    if (born.epsilon.rows() != 3 || born.epsilon.cols() != 3 ||
        !born.epsilon.allFinite() ||
        !born.epsilon.isApprox(born.epsilon.transpose()))
        throw value_error("dielectric tensor must be finite and symmetric 3x3");
    Eigen::LLT<Eigen::MatrixXd> dielectric(born.epsilon);
    if (dielectric.info() != Eigen::Success)
        throw value_error("dielectric tensor must be positive definite");
    for (const auto& charge : born.born)
        if (charge.rows() != 3 || charge.cols() != 3 || !charge.allFinite())
            throw value_error(
                "Born charge tensors must be finite 3x3 matrices");
    this->copy_blocks(fcs);
}


Eigen::ArrayXcd Dynamical_matrix_builder::get_exponentials(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    auto args = (q.transpose() * this->cpos).array();

    return args.cos().cast<std::complex<double>>() -
           constants::imud * args.sin().cast<std::complex<double>>();
}


std::array<Eigen::ArrayXXcd, 4> Dynamical_matrix_builder::build_nac_wang(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    constexpr double prefactor = constants::e * constants::e /
                                 constants::epsilon0 / constants::amu * 1e3;
    // The vector is reduced to the first BZ and normalized, since only
    // its direction matters.
    Eigen::Vector3d uq{this->structure.map_to_firstbz(q).col(0)};
    const double qnorm = uq.norm();
    uq /= qnorm;
    auto ndof = this->blocks[0].cols();
    auto natoms = ndof / 3;
    double epsilon = uq.dot(this->born.epsilon * uq);

    std::array<Eigen::ArrayXXcd, 4> nruter;

    for (auto i = 0; i < 4; ++i)
        nruter[i].setZero(ndof, ndof);

    for (decltype(natoms) iatom1 = 0; iatom1 < natoms; ++iatom1) {
        Eigen::MatrixXd z1{uq.transpose() * this->born.born[iatom1]};
        for (decltype(natoms) iatom2 = 0; iatom2 < natoms; ++iatom2) {
            Eigen::MatrixXd z2{uq.transpose() * this->born.born[iatom2]};
            Eigen::MatrixXd zz{z1.transpose() * z2};
            nruter[0].block<3, 3>(3 * iatom1, 3 * iatom2) = zz;

            for (auto ip = 0; ip < 3; ++ip) {
                nruter[ip + 1].block<3, 3>(3 * iatom1, 3 * iatom2) =
                    this->born.born[iatom1].row(ip).transpose() * z2 +
                    z1.transpose() * this->born.born[iatom2].row(ip) +
                    -2. * zz * this->born.epsilon.row(ip).dot(uq) / epsilon;
            }
        }
    }
    for (auto i = 0; i < 4; ++i) {
        nruter[i] /= this->massmatrix;
        nruter[i] *=
            prefactor / epsilon / this->V / this->na / this->nb / this->nc;
        if (i > 0)
            nruter[i] /= qnorm;
    }
    return nruter;
}

std::array<Eigen::ArrayXXcd, 4> Dynamical_matrix_builder::build_nac_gonze(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    // The dipole-dipole contribution using alma units
    constexpr double prefactor = constants::e * constants::e /
                                 constants::epsilon0 / constants::amu * 1e3;


    auto ndof = this->blocks[0].cols();
    auto natoms = ndof / 3;
    // We need the 1st BZ q-point (Should we average over equivalent points)
    Eigen::Vector3d uq = this->structure.map_to_firstbz(q).col(0);

    std::array<Eigen::ArrayXXcd, 4> nruter;

    for (auto i = 0; i < 4; ++i)
        nruter[i].setZero(ndof, ndof);

    const auto G_cutoff = 2.0 * constants::pi *
                          std::cbrt(900.0 / (4.0 * constants::pi) / this->V);
    constexpr double exp_cutoff = 1e-10;
    double GepsilonG = G_cutoff * G_cutoff * this->born.epsilon.trace() / 3.0;
    const double alpha = -GepsilonG / 4.0 / std::log(exp_cutoff);

    /// Get the G-mesh size, only those dimensions with periodicity are
    /// considered
    int shell = 0;
    while (true) {
        bool inside = false;
        for (auto iga : {-shell, 0, shell})
            for (auto igb : {-shell, 0, shell})
                for (auto igc : {-shell, 0, shell}) {
                    if (iga == 0 && igb == 0 && igc == 0 && shell != 0)
                        continue;
                    Eigen::Vector3d G = iga * this->structure.rlattvec.col(0) +
                                        igb * this->structure.rlattvec.col(1) +
                                        igc * this->structure.rlattvec.col(2);
                    if (G.norm() < G_cutoff)
                        inside = true;
                }
        if (!inside)
            break;
        shell++;
    }

    /// Iterate over the elements in the shell to get the proper result
    for (auto iga = -shell; iga <= shell; iga++)
        for (auto igb = -shell; igb <= shell; igb++)
            for (auto igc = -shell; igc <= shell; igc++) {
                Eigen::Vector3d G = iga * this->structure.rlattvec.col(0) +
                                    igb * this->structure.rlattvec.col(1) +
                                    igc * this->structure.rlattvec.col(2);

                GepsilonG = G.dot(this->born.epsilon * G);
                // We substract the q=0 term to impose the ASR
                if (!almost_equal(G.norm(), 0.0)) {
                    double decay =
                        std::exp(-GepsilonG / alpha / 4.0) / GepsilonG;
                    for (auto iatom = 0; iatom < natoms; iatom++) {
                        Eigen::MatrixXd zi{G.transpose() *
                                           this->born.born[iatom]};
                        for (auto jatom = 0; jatom < natoms; jatom++) {
                            Eigen::MatrixXd zj{G.transpose() *
                                               this->born.born[jatom]};
                            Eigen::Vector3d taudiff =
                                this->structure.lattvec *
                                (this->structure.positions.col(iatom) -
                                 this->structure.positions.col(jatom));
                            Eigen::MatrixXd zij{zi.transpose() * zj};
                            auto phase =
                                std::exp(constants::imud * G.dot(taudiff));
                            // We impose Hermiticity on the long-range part of
                            // the dynamical matrix
                            nruter[0].block<3, 3>(3 * iatom, 3 * iatom) -=
                                0.5 * (zij * phase * decay).array();
                            nruter[0].block<3, 3>(3 * iatom, 3 * iatom) -=
                                0.5 *
                                (zij.transpose() * std::conj(phase) * decay)
                                    .array();
                        }
                    }
                }

                Eigen::Vector3d Gq = G + uq;
                GepsilonG = Gq.dot(this->born.epsilon * Gq);
                if (!almost_equal(Gq.norm(), 0.0)) {
                    double decay =
                        std::exp(-GepsilonG / alpha / 4.0) / GepsilonG;
                    Eigen::Vector3d dGepsilonG =
                        (this->born.epsilon + this->born.epsilon.transpose()) *
                        Gq;
                    for (auto iatom = 0; iatom < natoms; iatom++) {
                        Eigen::MatrixXd zi{Gq.transpose() *
                                           this->born.born[iatom]};
                        for (auto jatom = 0; jatom < natoms; jatom++) {
                            Eigen::MatrixXd zj{Gq.transpose() *
                                               this->born.born[jatom]};
                            Eigen::Vector3d taudiff =
                                this->structure.lattvec *
                                (this->structure.positions.col(iatom) -
                                 this->structure.positions.col(jatom));
                            Eigen::MatrixXd zij{zi.transpose() * zj};
                            auto phase =
                                std::exp(constants::imud * Gq.dot(taudiff));
                            nruter[0].block<3, 3>(3 * iatom, 3 * jatom) +=
                                (zij * phase * decay).array();
                            for (int axis = 0; axis < 3; axis++) {
                                nruter[axis + 1].block<3, 3>(3 * iatom,
                                                             3 * jatom) +=
                                    (decay * phase *
                                     (this->born.born[iatom]
                                              .row(axis)
                                              .transpose() *
                                          zj +
                                      zi.transpose() *
                                          this->born.born[jatom].row(axis) +
                                      zij * constants::imud * taudiff(axis) -
                                      zij * (dGepsilonG(axis) / alpha / 4.0 +
                                             dGepsilonG(axis) / GepsilonG)))
                                        .array();
                            }
                        }
                    }
                }
            }

    for (auto i = 0; i < 4; ++i) {
        nruter[i] *= prefactor / this->V;
        nruter[i] /= this->massmatrix;
    }

    return nruter;
}

/// Pairwise summation of a vector of matrices or arrays. This reduces the
/// expected error versus a standard summation.
///
/// @param[inout] v - vector of operands. All of them must have the same size.
/// The vector will be overwritten.
/// @return the result of the sum
template <typename T> T eigen_pairwise_sum(std::vector<T>& v) {
    std::size_t last = v.size();
    T nruter;
    if (last == 0) {
        nruter.fill(0.);
    }
    else {
        while (last != 1) {
            std::size_t half = last / 2;
            for (std::size_t i = 0; i < half; ++i) {
                v[i] += v[i + half];
            }
            if (last % 2 == 0) {
                last = half;
            }
            else {
                v[half] = v[last - 1];
                last = half + 1;
            }
        }
        nruter = v[0];
    }
    return nruter;
}


std::array<Eigen::MatrixXcd, 4> Dynamical_matrix_builder::build_interpolated(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    Eigen::MatrixXd qbzs = this->structure.map_to_firstbz(q);
    Eigen::Vector3d qbz{qbzs.col(0)};
    auto nblocks = this->blocks.size();
    auto ndof = this->blocks[0].cols();
    auto coefficients = this->get_exponentials(q);
    // The nonanalytic correction is never applied at Gamma or at points
    // on the surface of the BZ. Moreover, in this function we only
    // correct using Wang's method.
    const bool nonanalytic =
        this->nonanalytic && !almost_equal(0., qbz.norm()) &&
        qbzs.cols() == 1 &&
        this->nonanalytic_method == nonanalytic_treatment::wang;

    std::array<Eigen::ArrayXXcd, 4> nac;
    if (nonanalytic) {
        nac = this->build_nac_wang(q);
    }

    std::array<Eigen::MatrixXcd, 4> nruter;
    std::vector<Eigen::MatrixXcd> terms;
    Eigen::MatrixXcd term;

    for (auto i = 0; i < 4; i++) {
        nruter[i].setZero(ndof, ndof);
    }
    for (decltype(nblocks) i = 0; i < nblocks; ++i) {
        term.array() = coefficients(i) * this->blocks[i];
        if (nonanalytic) {
            term.array() += coefficients(i) * this->masks[i].array() * nac[0];
        }
        terms.emplace_back(term);
    }
    nruter[0] = eigen_pairwise_sum<Eigen::MatrixXcd>(terms);
    for (auto j = 0; j < 3; ++j) {
        terms.clear();
        for (decltype(nblocks) i = 0; i < nblocks; ++i) {
            term.array() = -this->cpos(j, i) * coefficients(i) *
                           this->blocks[i] * constants::imud;
            if (nonanalytic) {
                term.array() -= this->cpos(j, i) * coefficients(i) *
                                this->masks[i].array() * nac[0].array() *
                                constants::imud;
                term.array() += coefficients(i) * this->masks[i].array() *
                                nac[j + 1].array();
            }
            terms.emplace_back(term);
        }
        nruter[j + 1] = eigen_pairwise_sum<Eigen::MatrixXcd>(terms);
    }

    return nruter;
}


std::array<Eigen::MatrixXcd, 4> Dynamical_matrix_builder::build(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    auto result = build_interpolated(q);
    if (nonanalytic && nonanalytic_method == nonanalytic_treatment::gonze) {
        const auto long_range = build_nac_gonze(q);
        for (int component = 0; component < 4; ++component)
            result[component] += long_range[component].matrix();
    }
    return result;
}

/// Choose a unique base of eigenvectors in a degenerate subspace using
/// perturbation theory.
///
/// Given an arbitrary basis of a degenerate eigenvector space and a suitable
/// perturbation matrix, this function applies an orthogonal transformation
/// to the basis so that the perturbation does not mix the states of the final
/// basis.
/// @param[in] dDdq - perturbation matrix (normally related to a group velocity
/// operator)
/// @param[inout] eigvecs - original basis
/// @return the new basis
Eigen::MatrixXcd solve_degeneracy(
    const Eigen::Ref<const Eigen::MatrixXcd>& pert,
    const Eigen::Ref<const Eigen::MatrixXcd>& eigvecs) {
    Eigen::MatrixXcd confusion = eigvecs.adjoint() * pert * eigvecs;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(confusion);
    return eigvecs * solver.eigenvectors();
}


std::unique_ptr<Spectrum_at_point> Dynamical_matrix_builder::get_spectrum(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    auto ndof = this->blocks[0].cols();
    int natoms = ndof / 3;

    // Build the dynamical matrix and its derivatives, including the selected
    // nonanalytic correction.
    auto matrices = this->build(q);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(matrices[0]);
    auto omega2 = solver.eigenvalues();
    auto wfs = solver.eigenvectors();
    Eigen::MatrixXd vg(Eigen::MatrixXd::Zero(3, ndof));
    Eigen::MatrixXcd wigner_v(Eigen::MatrixXcd::Zero(3, ndof * ndof));
    Eigen::ArrayXd omega(ndof);

    for (auto i = 0; i < omega.size(); ++i) {
        omega(i) = alma::ssqrt(omega2(i));
    }

    Eigen::MatrixXd qbzs = this->structure.map_to_firstbz(q);


    auto start = 0;
    // Degenerate subspaces are treated together in order to have a univocal,
    // and hopefully physically correct, estimate of the group velocities.
    // The x component of the group velocity is estimated using the set of
    // eigenvectors that diagonalize d D / d q_x over the degenerate subspace,
    // and so on.
    for (auto i = 1; i <= omega.size(); ++i) {
        if (i == omega.size() || !almost_equal(omega(i), omega(start))) {
            int dim = i - start;
            if (!almost_equal(omega(start), 0.)) {
                // Shortcut for non-degenerate cases.
                if (dim == 1) {
                    for (auto j = 0; j < 3; ++j) {
                        vg(j, start) =
                            wfs.col(start)
                                .dot(matrices[j + 1] * wfs.col(start))
                                .real();
                    }

                    vg.col(start) /= (2. * omega(start));
                }
                // General implementation.
                else {
                    for (auto j = 0; j < 3; ++j) {
                        Eigen::MatrixXcd vectors = solve_degeneracy(
                            matrices[j + 1], wfs.block(0, start, ndof, dim));
                        for (auto l = 0; l < dim; ++l) {
                            vg(j, start + l) =
                                vectors.col(l)
                                    .dot(matrices[j + 1] * vectors.col(l))
                                    .real();
                        }
                    }
                    vg.block(0, start, 3, dim) /= (2. * omega(start));
                    // Finally, choose an arbitrary but unique set of
                    // eigenvectors, to reduce the variability of the results
                    // across platforms.
                    Eigen::MatrixXcd directional =
                        1. * matrices[1] + 2. * matrices[2] + 3. * matrices[3];
                    wfs.block(0, start, ndof, dim) =
                        solve_degeneracy(directional,
                                         wfs.block(0, start, ndof, dim))
                            .eval();
                }
            }
            start = i;
        }
    }

    // Store every component in the final eigenvector basis. Rotating each
    // component independently inside a degenerate subspace would make the
    // off-diagonal operator inconsistent with the stored eigenvectors.
    // The position commutator converts our cell-only Fourier phases to the
    // smooth convention (Phys. Rev. X 12, 041011, Eq. 50).
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd positions(ndof);
        for (int atom = 0; atom < natoms; ++atom) {
            const Eigen::Vector3d tau =
                structure.lattvec * structure.positions.col(atom);
            positions.segment<3>(3 * atom).setConstant(tau(axis));
        }
        const Eigen::MatrixXcd derivative =
            wfs.adjoint() * matrices[axis + 1] * wfs;
        const Eigen::MatrixXcd position =
            wfs.adjoint() * positions.asDiagonal() * wfs;
        for (Eigen::Index j = 0; j < ndof; ++j)
            for (Eigen::Index i = 0; i < ndof; ++i) {
                if (omega(i) <= 0. || omega(j) <= 0.)
                    continue;
                wigner_v(axis, i + ndof * j) =
                    derivative(i, j) / (omega(i) + omega(j)) -
                    constants::imud * (omega(j) - omega(i)) * position(i, j);
            }
    }
    return std::make_unique<Spectrum_at_point>(omega, wfs, vg, wigner_v);
}
} // namespace alma
