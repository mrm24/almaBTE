// Copyright 2022 Martí Raya Moreno
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
/// Code to interface GPU solvers with eigen containers
//  the purpose is to allow MAGMA to be used with eigen solvers
//  NOTE that matrix ordering in Eigen must be C++-like (i.e. row-major)
//  and not Fortran default.
//


/// Eigen containers

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace alma {

/// Solves the A*x = b iteratively using GMRES with JACOBI preconditioner
/// using GPU
/// Eigen Matrix/Vectors
/// @param[in]     A_ - A matrix (sparse)
/// @param[in,out] b_ - b vector
/// @param[in]     x_  - Initial guess for x
double linear_iterative_solver_gpu(Eigen::SparseMatrix<double,Eigen::RowMajor>& A_,
                  Eigen::VectorXd& b_,
                  Eigen::VectorXd& x_);

}//namespace alma
