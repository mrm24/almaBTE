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
/// Definitions corresponding to vasp_io.hpp.

#include <utility>
#include <iostream>
#include <fstream>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <vasp_io.hpp>
#include <utilities.hpp>
#include <exceptions.hpp>

namespace alma {
std::unique_ptr<Crystal_structure> load_POSCAR(const char* filename) {
    std::ifstream f(filename);

    if (!f) {
        throw value_error("could not open file");
    }
    // The first line in the file can be either a description or an
    // element list.
    std::string firstline;
    std::getline(f, firstline);
    // The next line contains a multiplicative factor.
    double factor;
    f >> factor;
    factor /= 10.;
    // The next three lines contain the lattice vectors, one per
    // line.
    Eigen::Matrix3d lattvec;

    for (auto i = 0; i < 3; ++i)
        for (auto j = 0; j < 3; ++j)
            f >> lattvec(j, i);
    lattvec *= factor;
    // The next line can either contain an element list or a list of
    // integers.
    std::string line;
    flush_istream(f);
    std::getline(f, line);
    std::vector<std::string> elements;
    std::vector<int> numbers;
    try {
        numbers = tokenize_homogeneous_line<int>(line);
        // The list of elements must have been in the first line.
        elements = tokenize_homogeneous_line<std::string>(firstline);
    }
    catch (boost::bad_lexical_cast& e) {
        elements = tokenize_homogeneous_line<std::string>(line);
        // The list of integers must come next.
        std::getline(f, line);
        numbers = tokenize_homogeneous_line<int>(line);
    }
    // Next we can have "Selective dynamics", which we ignore.
    std::getline(f, line);

    if (starts_with_character(line, "Ss"))
        std::getline(f, line);
    // The next line tells us about the type of coordinates contained
    // in the file: Direct or Cartesian.
    auto cartesian = starts_with_character(line, "CcKk");
    // Finally, read the coordinates.
    auto natoms = std::accumulate(numbers.begin(), numbers.end(), 0);
    Eigen::Matrix<double, 3, Eigen::Dynamic> positions(3, natoms);

    for (int i = 0; i < natoms; ++i)
        for (auto j = 0; j < 3; ++j)
            f >> positions(j, i);

    // Make sure that we store lattice coordinates.
    if (cartesian)
        positions = lattvec.colPivHouseholderQr().solve(positions / 10.);

    // The code expects direct coordinates to lie in the
    // [0.,1.) range.
    for (int i = 0; i < natoms; ++i)
        for (auto j = 0; j < 3; ++j) {
            positions(j, i) = std::fmod(positions(j, i), 1.);

            if (positions(j, i) < 0.)
                positions(j, i) += 1.;
        }
    // Build and return the object.
    return alma::make_unique<Crystal_structure>(
        lattvec, positions, elements, numbers);
}


std::unique_ptr<Harmonic_ifcs> load_FORCE_CONSTANTS(
    const char* filename,
    const Crystal_structure& cell,
    const int na,
    const int nb,
    const int nc) {
    if (std::min({na, nb, nc}) <= 0)
        throw value_error("na, nb and nc must be positive");
    std::ifstream f(filename);

    if (!f) {
        throw value_error("could not open file");
    }
    // The first line of the file contains the number of atoms in the
    // supercell.
    //
    // However we have several formats:
    //   -old: natoms_supercell
    //   -new-full:  natoms_supercell natoms_supercell
    //   -new-compact: natoms_unitcell natoms_supercell
    force_constants_format format;

    std::string first_line;
    std::getline(f, first_line);
    std::vector<int> fc_sizes = tokenize_homogeneous_line<int>(first_line);

    if (fc_sizes.size() == 1) {
	format = force_constants_format::old;
    }
    else if (fc_sizes.size() == 2 and fc_sizes[0] == fc_sizes[1]) {
    	format = force_constants_format::new_full;
    }
    else if (fc_sizes.size() == 2 and fc_sizes[0] != fc_sizes[1]) {
        format = force_constants_format::new_compact;
    }
    else {
	throw value_error("format of the FORCE_CONSTANTS is not recognized");
    }

    std::array<int,2> ntot = {fc_sizes[0], fc_sizes[0]};
    if ( format != force_constants_format::old) ntot[1] = fc_sizes[1];
    auto natoms = cell.get_natoms();
    auto ndof = 3 * natoms;
    auto nexpected = na * nb * nc * natoms;

    if (ntot[1] != nexpected)
        throw value_error(boost::str(
            boost::format("expected %1% atoms, got %2%") % nexpected % ntot[1]));
    // Then comes one block for each atom pair.
    // This means that the file is either highly redundant or
    // inconsistent. Here we assume the former.
    int field;
    std::string line;
    Triple_int_map<Eigen::MatrixXd> matrices;
    std::cout <<  ntot[0] << '\t' << ntot[1] << std::endl;
    auto builder1 = format == force_constants_format::new_compact ? 
	    	    Supercell_index_builder(1, 1, 1, natoms) : 
		    Supercell_index_builder(na, nb, nc, natoms);
    auto builder2 = Supercell_index_builder(na, nb, nc, natoms);

    for (auto i = 0; i < ntot[0]; ++i) {
        for (auto j = 0; j < ntot[1]; ++j) {
            f >> field;
            auto index1 = builder1.create_index_safely(field - 1);
            f >> field;
            auto index2 = builder2.create_index_safely(field - 1);

            if ((index1.index != i) || (index2.index != j))
                throw input_error("unexpected cell indices");
            // If there is anything else in the line, discard it.
            std::string tmp;
            std::getline(f, tmp);

            std::cout << i << '\t' << j << std::endl;
            std::cout << index1.ia << '\t' << index1.ib << '\t' << index1.ic << std::endl;
            std::cout << index2.ia << '\t' << index2.ib << '\t' << index2.ic << std::endl;
            std::cout << "IDXS: " << index1.iatom << '\t' << index2.iatom << std::endl;

            if ((index1.ia == 0) && (index1.ib == 0) && (index1.ic == 0)) {
                // The first unit cell is (0, 0, 0). Read and store
                // the data.
                auto key = index2.get_pos();

                if (matrices.find(key) == matrices.end())
                    // If the submatrix does not exist, create it.
                    matrices[key] = Eigen::MatrixXd(ndof, ndof);

                for (auto k = 0; k < 3; ++k)
                    for (auto l = 0; l < 3; ++l)
                        f >> matrices[key](3 * index1.iatom + k,
                                           3 * index2.iatom + l);
            }
            else {
                // Redundant block. Skip those lines.
                flush_istream(f);

                for (auto k = 0; k < 3; ++k)
                    std::getline(f, line);
            }
        }
    }
    // Put the results into data structures that can be passed to the
    // constructor.
    std::vector<Triple_int> pos;
    std::vector<Eigen::MatrixXd> ifcs;
    std::tie(pos, ifcs) = split_keys_and_values(matrices);
    return alma::make_unique<Harmonic_ifcs>(pos, ifcs, na, nb, nc);
}


Eigen::ArrayXXd load_FORCE_CONSTANTS_raw(const char* filename) {
    std::ifstream f(filename);

    if (!f) {
        throw value_error("could not open file");
    }
    std::size_t ntot;
    f >> ntot;
    std::size_t ndof = 3 * ntot;
    Eigen::ArrayXXd nruter{ndof, ndof};

    for (std::size_t i = 0; i < ntot; ++i) {
        for (std::size_t j = 0; j < ntot; ++j) {
            std::size_t index;
            f >> index;

            if (index - 1 != i)
                throw input_error("unexpected index");
            f >> index;

            if (index - 1 != j)
                throw input_error("unexpected index");
            std::string tmp;
            std::getline(f, tmp);

            for (auto k = 0; k < 3; ++k)
                for (auto l = 0; l < 3; ++l)
                    f >> nruter(3 * i + k, 3 * j + l);
        }
    }
    return nruter;
}


std::unique_ptr<Dielectric_parameters> load_BORN(const char* filename) {
    std::ifstream f(filename);

    if (!f) {
        throw value_error("could not open file");
    }
    // The first line of the file contains a unit conversion factor
    // employed by Phonopy. We ignore it.
    std::string line;
    std::getline(f, line);
    // Then come the nine components of the dielectric tensor.
    Eigen::Matrix3d epsilon;

    for (auto k = 0; k < 3; ++k)
        for (auto l = 0; l < 3; ++l)
            f >> epsilon(k, l);
    // And finally the born charge tensors for each atom.
    flush_istream(f);
    std::vector<Eigen::MatrixXd> born;

    do {
        std::getline(f, line);
        auto fields = tokenize_homogeneous_line<double>(line);

        if (fields.size() == 0) {
            continue;
        }
        else if (fields.size() == 9) {
            auto tensor = Eigen::MatrixXd(3, 3);

            for (auto k = 0; k < 3; ++k)
                for (auto l = 0; l < 3; ++l)
                    tensor(k, l) = fields[3 * k + l];
            born.emplace_back(tensor);
        }
        else {
            throw input_error("wrong number of fields in a line");
        }
    } while (!f.eof());
    return alma::make_unique<Dielectric_parameters>(born, epsilon);
}


std::unique_ptr<std::vector<Thirdorder_ifcs>> load_FORCE_CONSTANTS_3RD(
    const char* filename,
    const Crystal_structure& cell) {
    std::ifstream f(filename);

    if (!f) {
        throw value_error("could not open file");
    }
    auto solver = cell.lattvec.colPivHouseholderQr();
    // The first line is the number of blocks in the file.
    std::size_t nblocks;
    f >> nblocks;
    auto nruter = make_unique<std::vector<Thirdorder_ifcs>>();
    nruter->reserve(nblocks);

    // For each block.
    for (decltype(nblocks) iblock = 0; iblock < nblocks; ++iblock) {
        std::size_t tmp;
        // Read the block number and complain if it does not
        // match our expectations.
        f >> tmp;

        if (tmp != iblock + 1)
            throw value_error("wrong block number");
        // Read the Cartesian coordinates of the second unit cell.
        Eigen::VectorXd rj(3);
        f >> rj(0) >> rj(1) >> rj(2);
        // Read the Cartesian coordinates of the third unit cell.
        Eigen::VectorXd rk(3);
        f >> rk(0) >> rk(1) >> rk(2);
        // Convert these coordinates to nm and
        // round each vector to the closest unit cell.
        Eigen::VectorXd solj = solver.solve(rj / 10.);
        Eigen::VectorXd solk = solver.solve(rk / 10.);

        for (auto ic = 0; ic < 3; ++ic) {
            solj(ic) = std::round(solj(ic));
            solk(ic) = std::round(solk(ic));
        }
        rj = cell.lattvec * solj;
        rk = cell.lattvec * solk;
        // The next line contains the three atom indices.
        std::size_t i;
        std::size_t j;
        std::size_t k;
        f >> i >> j >> k;
        // Create an empty Thirdorder_ifcs object.
        Thirdorder_ifcs block(rj, rk, i - 1, j - 1, k - 1);

        // And read the contents of the 27 remaining lines to fill
        // in the values of the ifcs.
        for (auto ic = 0; ic < 27; ++ic) {
            int p1;
            int p2;
            int p3;
            f >> p1 >> p2 >> p3;
            f >> block.ifc(p1 - 1, p2 - 1, p3 - 1);
        }
        // Add the new block to the vector.
        nruter->emplace_back(block);
    }
    return nruter;
}


std::unique_ptr<std::vector<Fourthorder_ifcs>> load_FORCE_CONSTANTS_4TH(
    const char* filename,
    const Crystal_structure& cell) {
    std::ifstream f(filename);

    if (!f) {
        throw value_error("could not open file");
    }
    auto solver = cell.lattvec.colPivHouseholderQr();
    // The first line is the number of blocks in the file.
    std::size_t nblocks;
    f >> nblocks;
    auto nruter = make_unique<std::vector<Fourthorder_ifcs>>();
    nruter->reserve(nblocks);

    // For each block.
    for (decltype(nblocks) iblock = 0; iblock < nblocks; ++iblock) {
        std::size_t tmp;
        // Read the block number and complain if it does not
        // match our expectations.
        f >> tmp;

        if (tmp != iblock + 1)
            throw value_error("wrong block number");
        // Read the Cartesian coordinates of the second unit cell.
        Eigen::VectorXd rj(3);
        f >> rj(0) >> rj(1) >> rj(2);
        // Read the Cartesian coordinates of the third unit cell.
        Eigen::VectorXd rk(3);
        f >> rk(0) >> rk(1) >> rk(2);
        // Read the Cartesian coordinates of the fourth unit cell.
        Eigen::VectorXd rl(3);
        f >> rl(0) >> rl(1) >> rl(2);
        // Convert these coordinates to nm and
        // round each vector to the closest unit cell.
        Eigen::VectorXd solj = solver.solve(rj / 10.);
        Eigen::VectorXd solk = solver.solve(rk / 10.);
        Eigen::VectorXd soll = solver.solve(rl / 10.);

        for (auto ic = 0; ic < 3; ++ic) {
            solj(ic) = std::round(solj(ic));
            solk(ic) = std::round(solk(ic));
            soll(ic) = std::round(soll(ic));
        }
        rj = cell.lattvec * solj;
        rk = cell.lattvec * solk;
        rl = cell.lattvec * soll;
        // The next line contains the four atom indices.
        std::size_t i;
        std::size_t j;
        std::size_t k;
        std::size_t l;
        f >> i >> j >> k >> l;
        // Create an empty Thirdorder_ifcs object.
        Fourthorder_ifcs block(rj, rk, rl, i - 1, j - 1, k - 1, l -1);

        // And read the contents of the 81 remaining lines to fill
        // in the values of the ifcs.
        for (auto ic = 0; ic < 81; ++ic) {
            int p1;
            int p2;
            int p3;
            int p4;
            f >> p1 >> p2 >> p3 >> p4;
            f >> block.ifc(p1 - 1, p2 - 1, p3 - 1, p4 - 1);
        }
        // Add the new block to the vector.
        nruter->emplace_back(block);
    }
    return nruter;
}

} // namespace alma
