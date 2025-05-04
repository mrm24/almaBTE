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
/// Definitions corresponding to dynamical_matrix.hpp.

#include <iostream>
#include <complex>
#include <cmath>
#include <constants.hpp>
#include <utilities.hpp>
#include <periodic_table.hpp>
#include <dynamical_matrix.hpp>

namespace alma {

/// @brief Overload of the stream insertion operator for nonanalytic_treatment enum.
///
/// This function enables printing of alma::nonanalytic_treatment values
/// using std::ostream (e.g., std::cout) in human-readable form ("gonze", "wang").
///
/// @param os The output stream to which the enum will be written.
/// @param method The nonanalytic_treatment enum value to print.
/// @return A reference to the modified output stream.
std::ostream& operator<<(std::ostream& os, const nonanalytic_treatment& method) {
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


/// POD class representing a pair of atoms - one in unit cell (0, 0,
/// 0) the other in an arbitrary unit cell cj, and the image of the
/// latter in a number of unit cells cjp.
class Atom_pair {
public:
    /// Index of the first atom in its unit cell.
    int i;
    /// Index of the second atom in its unit cell.
    int j;
    /// Unit cell the second atom belongs to in a regular
    /// supercell representation.
    Triple_int cj;
    /// All unit cells that the image of the second atom
    /// belongs to in a Wigner-Seitz supercell representation.
    std::vector<Triple_int> cjp;
};


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
    const nonanalytic_treatment _nonanalytic_method
    )
    : na(fcs.na), nb(fcs.nb), nc(fcs.nc), V(_structure.V),
      structure(_structure), rlattvec(_structure.rlattvec), symmetries(syms),
      nonanalytic(true), born(_born), nonanalytic_method(_nonanalytic_method){
    if (this->born.born.size() !=
        static_cast<std::size_t>(this->structure.get_natoms()))
        throw value_error("wrong number of Born charges");
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
    Eigen::Vector3d uq{this->structure.map_to_firstbz(q).col(0).normalized()};
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
    }
    return nruter;
}

std::array<Eigen::ArrayXXcd, 4> Dynamical_matrix_builder::build_nac_gonze(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    constexpr double prefactor = constants::e * constants::e /
                                 constants::epsilon0 / constants::amu * 1e3;
    
    auto ndof = this->blocks[0].cols();
    auto natoms = ndof / 3;
    // We need the 1st BZ q-point
    Eigen::Vector3d uq = this->structure.map_to_firstbz(q).col(0);

    std::array<Eigen::ArrayXXcd, 4> nruter;

    for (auto i = 0; i < 4; ++i)
        nruter[i].setZero(ndof, ndof);


    auto Gmax  = 14.0;
    auto alpha = std::max({
            this->structure.rlattvec.col(0).squaredNorm(),
            this->structure.rlattvec.col(1).squaredNorm(),
            this->structure.rlattvec.col(2).squaredNorm()});
    
    /// Get the G-mesh size, only those dimensions with periodicity are considered
    std::array<double,3> cell_g;
    cell_g[0] = this->na == 1 ? 0 : int( std::sqrt(Gmax * 4.0 * alpha) / this->structure.rlattvec.col(0).norm()) + 1;
    cell_g[1] = this->nb == 1 ? 0 : int( std::sqrt(Gmax * 4.0 * alpha) / this->structure.rlattvec.col(1).norm()) + 1;
    cell_g[2] = this->nc == 1 ? 0 : int( std::sqrt(Gmax * 4.0 * alpha) / this->structure.rlattvec.col(2).norm()) + 1;

    for (auto iga = -cell_g[0]; iga <= cell_g[0]; iga++)
        for (auto igb = -cell_g[1]; igb <= cell_g[1]; igb++)
            for (auto igc = -cell_g[2]; igc <= cell_g[2]; igc++) {
                Eigen::Vector3d G = iga * this->structure.rlattvec.col(0) +
                                    igb * this->structure.rlattvec.col(1) +
                                    igc * this->structure.rlattvec.col(2);

                double GepsilonG = G.dot(this->born.epsilon * G);

                if (!almost_equal(GepsilonG,0.0) && GepsilonG / alpha / 4.0 < Gmax) {
                    auto decay = std::exp(- GepsilonG / alpha / 4.0) / GepsilonG;
                    for (auto iatom = 0; iatom < natoms; iatom++) {
                        Eigen::MatrixXd zi{G.transpose() * this->born.born[iatom]};
                        for (auto jatom = 0; jatom < natoms; jatom++) {
                            Eigen::MatrixXd zj{G.transpose() * this->born.born[iatom]};
                            Eigen::Vector3d taudiff = this->structure.lattvec * (
                                this->structure.positions.col(iatom) -  this->structure.positions.col(jatom));
                            Eigen::MatrixXd zij{zi.transpose() * zj};
                            auto phase = std::exp(constants::imud * G.dot(taudiff));
                            nruter[0].block<3, 3>(3 * iatom, 3 * iatom) -= (zij * phase * decay).array();
                        }
                    }
                }

		
                Eigen::Vector3d Gq = G + uq;
                GepsilonG = Gq.dot(this->born.epsilon * Gq);
                if (!almost_equal(GepsilonG,0.0) && GepsilonG / alpha / 4.0 < Gmax) {
                    double decay = std::exp(- GepsilonG / alpha / 4.0) / GepsilonG;
                    Eigen::Vector3d dGepsilonG = (this->born.epsilon + this->born.epsilon.transpose()) * Gq;
                    for (auto iatom = 0; iatom < natoms; iatom++) {
                        Eigen::MatrixXd zi{Gq.transpose() * this->born.born[iatom]};
                        for (auto jatom = 0; jatom < natoms; jatom++) {
                            Eigen::MatrixXd zj{Gq.transpose() * this->born.born[iatom]};
                            Eigen::Vector3d taudiff = this->structure.lattvec * (
                                this->structure.positions.col(iatom) -  this->structure.positions.col(jatom));
                            Eigen::MatrixXd zij{zi.transpose() * zj};
                            auto phase = std::exp(constants::imud * Gq.dot(taudiff));
                            nruter[0].block<3, 3>(3 * iatom, 3 * jatom) += (zij * phase * decay).array();
                            for (int axis = 0; axis < 3; axis++) {
                                nruter[axis+1].block<3, 3>(3 * iatom, 3 * jatom) += (decay * phase * (
                                    this->born.born[iatom].row(axis).transpose() * zj +
                                    zi.transpose() * this->born.born[jatom].row(axis) +
                                    zij * constants::imud * taudiff(axis) -
                                    zij * (dGepsilonG(axis) / alpha / 4.0 +  dGepsilonG(axis) / GepsilonG))).array();
                            }
                        }
                    }
                }
    }

    for (auto i = 0; i < 4; ++i) {
        nruter[i] *= 8 * prefactor * constants::pi;
        nruter[i] /= this->V * this->massmatrix;
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


std::array<Eigen::MatrixXcd, 4> Dynamical_matrix_builder::build(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    Eigen::MatrixXd qbzs = this->structure.map_to_firstbz(q);
    Eigen::Vector3d qbz{qbzs.col(0)};
    auto nblocks = this->blocks.size();
    auto ndof = this->blocks[0].cols();
    auto coefficients = this->get_exponentials(q);
    // The nonanalytic correction is never applied at Gamma or at points
    // on the surface of the BZ.
    const bool nonanalytic =
        this->nonanalytic && !almost_equal(0., qbz.norm()) && qbzs.cols() == 1;

    std::array<Eigen::ArrayXXcd, 4> nac;
    if (nonanalytic && (this->nonanalytic_method == nonanalytic_treatment::wang)) {
        nac = this->build_nac_wang(q);
    }
    else if (nonanalytic && (this->nonanalytic_method == nonanalytic_treatment::gonze)) {
	nac = this->build_nac_gonze(q);
    }
    else if (nonanalytic && (this->nonanalytic_method == nonanalytic_treatment::none)) {
	throw value_error("nonanalytic_treatment::none is selected but NAC is required");
    }
    else if (nonanalytic){
	throw value_error("Unkown NAC treatment");
    }

    std::array<Eigen::MatrixXcd, 4> nruter;
    std::vector<Eigen::MatrixXcd> terms;
    Eigen::MatrixXcd term;

    for (auto i = 0; i < 4; i++) {
        nruter[i].setZero(ndof, ndof);
    }
    for (decltype(nblocks) i = 0; i < nblocks; ++i) {
        term.array() = coefficients(i) * this->blocks[i];
	/// The Wang method for the NACs provides IFCs, so it requires the 
	/// phases and the mask (i.e. the weights).
        if (nonanalytic && (this->nonanalytic_method == nonanalytic_treatment::wang)) {
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
            if (nonanalytic && (this->nonanalytic_method == nonanalytic_treatment::wang)) {
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

    // Gonze's method for the NAC is directly giving the proper
    // dynamical matrix contribution, not the IFCs.
    if (nonanalytic && (this->nonanalytic_method == nonanalytic_treatment::gonze)) {
	for (int i = 0; i < 4; i++) 
		nruter[i] += nac[i].matrix();
    }

    return nruter;
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
    auto matrices = this->build(q);
    int natoms = ndof/3;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(matrices[0]);
    auto omega2 = solver.eigenvalues();
    auto wfs = solver.eigenvectors();
    Eigen::MatrixXd vg(Eigen::MatrixXd::Zero(3, ndof));
    Eigen::MatrixXcd wigner_v(Eigen::MatrixXcd::Zero(3, ndof*ndof));
    Eigen::ArrayXd omega(ndof);

    for (auto i = 0; i < omega.size(); ++i) {
        omega(i) = alma::ssqrt(omega2(i));
    }

    /// Build Wigner velocities. See 10.1103/PhysRevX.12.041011 
    /// for the theoretical framework.
    
    /// Build matrices needed to account for the phase choice within almaBTE.
    /// In particular, in almaBTE we do not include atomic positions within the phase (henceforth steplike convention).
    /// While for LBTE quantities this is not relevant, in LWTE the phase choice is important
    /// and the atomic positions need to be included in the phase (henceforth smooth convention).
    /// Here, we use unitary transformations for accounting our phase choice.
    
    /// Build unitary transform between smooth and steplike convetion for the eigenvectors
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Zero(ndof,ndof);
    for (auto id_atom = 0; id_atom < natoms; ++id_atom) {
        Eigen::Vector3d tau_ = structure.positions.col(id_atom).transpose();
	tau_ = structure.lattvec * tau_;
        auto phase = std::exp(alma::constants::imud * q.dot(tau_));
        for (auto cartesian_dir = 0; cartesian_dir < 3; ++cartesian_dir) {
            U(3*id_atom + cartesian_dir, 3*id_atom + cartesian_dir) = phase;
        }
    }

    // We need the inverse to obtain the vectors in the smooth convetion.
    // We directly invert it; as it is not a big matrix, and it pays the price of solving N times
    // a linear system.
    Eigen::MatrixXcd invU = U.inverse();

    /// Build the positions vectors for the phase correction in the wigner velocities.
    Eigen::ArrayXXd tau(3,ndof);

    for (auto id_atom = 0; id_atom < natoms; ++id_atom) {
	Eigen::Vector3d tau_ = structure.lattvec * structure.positions.col(id_atom);
	for (auto cartesian_dir = 0; cartesian_dir < 3; ++cartesian_dir) {
            tau.block(cartesian_dir, 3*id_atom, 1, 3).setConstant(tau_(cartesian_dir));
        }
    }

    /// Second, we create a list of the degenerate subspaces
    /// to properly tackle the degeneracy
    std::vector<std::pair<std::size_t,std::size_t>> subspaces;
    auto subspace_init = 0;
    for (size_t i = 1; i < omega.size(); ++i) {
        if (!almost_equal(omega(i),omega(i - 1))) {
            subspaces.emplace_back(subspace_init, i - 1);
            subspace_init = i;
        }
    }
    subspaces.emplace_back(subspace_init, omega.size() - 1);

    /// Iterate over the degenerate subspaces and 
    /// treat them in the appropiate way
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (auto subspace_left : subspaces) {

            // Build left degenerate safe eigenvectors
            auto dim_left = subspace_left.second - subspace_left.first + 1;
            Eigen::MatrixXcd vectors_left = (subspace_left.first == subspace_left.second) ?
                                            wfs.block(0, subspace_left.first, ndof, dim_left) :
                                            solve_degeneracy(matrices[axis + 1], wfs.block(0, subspace_left.first, ndof, dim_left));
            /// Obtain the left eignevectors in the smooth convention
            Eigen::MatrixXcd vectors_left_smooth = invU * vectors_left;

            for (auto subspace_right : subspaces){

                auto dim_right = subspace_right.second - subspace_right.first + 1;
                // Build right degenerate safe eigenvector
                Eigen::MatrixXcd vectors_right = (subspace_right.first == subspace_right.second) ?
                                            wfs.block(0, subspace_right.first, ndof, dim_right) :
                                            solve_degeneracy(matrices[axis + 1], wfs.block(0, subspace_right.first, ndof, dim_right));
                /// Obtain the right eignevectors in the smooth convention
                Eigen::MatrixXcd vectors_right_smooth = invU * vectors_right;

                for (auto i = 0; i < dim_left; ++i)
                    for (auto j = 0; j < dim_right; ++j) {
                        auto istate = i + subspace_left.first;
                        auto jstate = j + subspace_right.first;
                        wigner_v(axis,istate+ndof*jstate) = vectors_left.col(i)
                                                            .dot(matrices[axis + 1] * vectors_right.col(j));

                        if (almost_equal(omega(istate),0.) or almost_equal(omega(jstate),0.)) {
                            wigner_v(axis,istate+ndof*jstate) = std::complex<double>(0.0,0.0);
                        }
                        else{
                            wigner_v(axis,istate+ndof*jstate) /= omega(istate) + omega(jstate);
			    /// Now account by the fact that in almaBTE we are using the dynamical matrix convention
                            /// that does not have the atomic positions in the phase. See Eq. 50 in 10.1103/PhysRevX.12.041011.
                            auto phase_correction = -alma::constants::imud * ( omega(jstate) - omega(istate) ) *
                                                     vectors_left_smooth.col(i).dot(
                                                     (tau.transpose().col(axis).array() * vectors_right_smooth.col(j).array()).matrix());
			    wigner_v(axis,istate+ndof*jstate) += phase_correction; 
                        }
                    }
            }
        }
    }

    auto start = 0;
    // Degenerate subspaces are treated together in order to have a univocal,
    // and hopefully physically correct, estimate of the group velocities.
    // The x component of the group velocity is estimated using the set of
    // eigenvectors that diagonalize d D / d q_x over the degenerate subspace,
    // and so on.
    for (auto i = 1; i <= omega.size(); ++i) {
        if (i == omega.size() || !almost_equal(omega(i), omega(start))||true) {
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

    return alma::make_unique<Spectrum_at_point>(omega, wfs, vg, wigner_v);
}
} // namespace alma

