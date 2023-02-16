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

#pragma once

/// @file
///
/// C++ interface to Atsushi Togo's spglib.

#include <structures.hpp>

namespace alma {
/// POD class used to hold a pair of atom indices and a symmetry
/// operation. Intended to be used for classifying pairs of atoms
/// in a supercell.
class Transformed_pair {
public:
    /// Pair of atom indices.
    const std::array<std::size_t, 2> pair;
    /// Index of a symmetry operation.
    const std::size_t operation;
    /// True if an additional exchange of indices is necessary.
    const bool exchange;
    /// Basic constructor.
    Transformed_pair(std::size_t a,
                     std::size_t b,
                     std::size_t _operation,
                     bool _exchange)
        : pair({{a, b}}), operation(_operation), exchange(_exchange) {
    }
};


/// Objects of this class hold a subset of the information
/// provided by spg_get_dataset().
class Symmetry_operations {
public:
    /// Tolerance for the symmetry search.
    const double symprec;
    /// Constructor that calls spglib to analyze the symmetries
    /// of the provided structure.
    ///
    /// @param[in] structure - a description of the crystal
    /// @param[in] _symprec - tolerance for the symmetry search
    Symmetry_operations(const Crystal_structure& structure,
                        double _symprec = 1e-5);
    /// @return the number of symmetry operations.
    std::size_t get_nsym() const {
        return this->translations.size();
    }
    /// @return the international space group number.
    int get_spacegroup_number() const {
        return this->sg_number;
    }
    /// @return the space group symbol.
    std::string get_spacegroup_symbol() const {
        return this->sg_symbol;
    }


    /// @return the Wyckoff positions of each atom.
    std::string get_wyckoff() const {
        return this->wyckoff;
    }
    /// @return the equivalence class of each atom.
    std::vector<int> get_equivalences() const {
        return this->equivalences;
    }


    /// Transform a vector according to one
    /// of the symmetry operations.
    ///
    /// @param[in] vector - the input vector or matrix of vectors
    /// @param[in[ index - the operation number
    /// @param[in] cartesian - is the input in
    /// Cartesian coordinates?
    /// @return the transformed vector or matrix of vectors
    template <typename T>
    auto transform_v(const Eigen::MatrixBase<T>& vector,
                     std::size_t index,
                     bool cartesian = false) const
        -> Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, Eigen::Dynamic> {
        if (index >= this->translations.size())
            throw alma::value_error("invalid operation index");

        if (cartesian)
            return this->crotations[index] * vector +
                   this->ctranslations[index];
        else
            return this->rotations[index] * vector + this->translations[index];
    }


    /// Rotate (but do not translate) a vector according to one
    /// of the symmetry operations.
    ///
    /// @param[in] vector - the input vector or matrix of vectors
    /// @param[in[ index - the operation number
    /// @param[in] cartesian - is the input in
    /// Cartesian coordinates?
    /// @return the transformed vector or matrix of vectors
    template <typename T>
    auto rotate_v(const Eigen::MatrixBase<T>& vector,
                  std::size_t index,
                  bool cartesian = false) const
        -> Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, Eigen::Dynamic> {
        if (index >= this->translations.size())
            throw alma::value_error("invalid operation index");

        if (cartesian)
            return this->crotations[index] * vector;
        else
            return this->rotations[index] * vector;
    }


    /// Rotate a 3 x 3 matrix according to one of the symmetry
    /// operations.
    ///
    /// @param[in] matrix - the input matrix
    /// @param[in] index - the operation number
    /// @param[in] cartesian - is the input in
    /// Cartesian coordinates?
    /// @return the rotated matrix
    template <typename T>
    Eigen::Matrix<T, 3, 3> rotate_m(
        const Eigen::Ref<const Eigen::Matrix<T, 3, 3>>& matrix,
        std::size_t index,
        bool cartesian = false) const {
        if (index >= this->translations.size())
            throw alma::value_error("invalid operation index");

        if (cartesian)
            return this->crotations[index] * matrix *
                   this->crotations[index].transpose();
        else
            return this->rotations[index] * matrix *
                   this->rotations[index].transpose();
    }


    /// Rotate a 3 x 3 matrix according to the inverse of one of the
    /// symmetry operations.
    ///
    /// @param[in] matrix - the input matrix
    /// @param[in] index - the operation number
    /// @param[in] cartesian - is the input in
    /// Cartesian coordinates?
    /// @return the rotated matrix
    template <typename T>
    Eigen::Matrix<T, 3, 3> unrotate_m(
        const Eigen::Ref<const Eigen::Matrix<T, 3, 3>>& matrix,
        std::size_t index,
        bool cartesian = false) const {
        if (index >= this->translations.size())
            throw alma::value_error("invalid operation index");

        if (cartesian)
            return this->crotations[index].transpose() * matrix *
                   this->crotations[index];
        else
            return this->rotations[index].transpose() * matrix *
                   this->rotations[index];
    }


    /// Get the atom index that an atom is mapped to through
    /// a symmetry operation.
    ///
    /// @param[in] original - the input atom index
    /// @param[in] index - the operation number
    /// @return the new atom index
    std::size_t map_atom(std::size_t original, std::size_t index) const {
        if (original >= this->symmetry_map.size())
            throw alma::value_error("invalid atom index");

        if (index >= this->translations.size())
            throw alma::value_error("invalid operation index");
        return this->symmetry_map[original][index];
    }


    /// Find out if a symmetry operation involves an inversion
    ///
    /// @param[in] index - the operation number
    /// @return True if the operation does not preserve the
    /// orientation of the axes
    bool is_inversion(std::size_t index) const {
        return this->determinants[index] < 0.;
    }


    /// Rotate a vector expressed in direct reciprocal coordinates
    /// according to one of the symmetry operations.
    ///
    /// @param[in] vector - the input vector or matrix of vectors
    /// @param[in] index - the operation number
    /// @return the transformed vector or matrix of vectors
    template <typename T>
    auto rotate_q(const Eigen::MatrixBase<T>& vector, std::size_t index) const
        -> Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, Eigen::Dynamic> {
        if (index >= this->translations.size())
            throw alma::value_error("invalid operation index");
        return this->qrotations[index] * vector;
    }

    /// Symmetrize a 3x3 matrix using all the operations in the group.
    ///
    /// @param[in] matrix - the input matrix
    /// @param[in] cartesian - is the input in
    /// Cartesian coordinates?
    /// @return the symmetrized matrix
    template <typename T>
    Eigen::Matrix<T, 3, 3> symmetrize_m(
        const Eigen::Ref<const Eigen::Matrix<T, 3, 3>>& matrix,
        bool cartesian = false) const {
        Eigen::Matrix<T, 3, 3> nruter;
        nruter.fill(0.);
        const std::vector<Eigen::MatrixXd>* rots;
        if (cartesian) {
            rots = &this->crotations;
        }
        else {
            rots = &this->rotations;
        }
        for (const auto& r : *rots) {
            nruter += r.transpose() * matrix * r;
        }
        nruter /= rots->size();
        return nruter;
    }

    /// Take all pairs of atoms in the structure and divide them
    /// among equivalence classes defined by the internal
    /// translations.
    ///
    /// @return a vector of vectors of Transformed_Pairs
    std::vector<std::vector<Transformed_pair>> get_pair_classes() const;

private:
    /// Translation vectors in direct coordinates.
    std::vector<Eigen::VectorXd> translations;
    /// Translation vectors in Cartesian coordinates.
    std::vector<Eigen::VectorXd> ctranslations;
    /// Rotation matrices in direct coordinates.
    std::vector<Eigen::MatrixXd> rotations;
    /// Rotation matrices in Cartesian coordinates.
    std::vector<Eigen::MatrixXd> crotations;
    /// Rotation matrices in direct reciprocal coordinates.
    std::vector<Eigen::MatrixXd> qrotations;
    /// Determinants of the rotation matrices in direct coordinates.
    std::vector<double> determinants;
    /// Displacements of all atoms due to each symmetry operation,
    /// in direct coordinates, without periodic boundary conditions.
    std::vector<Eigen::MatrixXd> displacements;
    /// Space group number.
    int sg_number;
    /// International symbol.
    std::string sg_symbol;
    /// Wyckoff positions of each atom.
    std::string wyckoff;
    /// Equivalence class of each atom, as determined by the
    /// symmetry operations.
    std::vector<int> equivalences;
    /// Detailed map between atoms through the symmetry operations.
    std::vector<std::vector<std::size_t>> symmetry_map;
    /// Fill the symmetry_map attribute.
    /// @param[in] structure - a description of the crystal
    void fill_map(const Crystal_structure& structure);
};
} // namespace alma
