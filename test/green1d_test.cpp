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
/// Test if the implementation of the 1D Green's function works.

#include <iostream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <randutils.hpp>
#pragma GCC diagnostic pop
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <dynamical_matrix.hpp>
#include <qpoint_grid.hpp>
#include <green1d.hpp>

constexpr std::size_t nqpoints{101};
constexpr std::size_t nE{10000};
constexpr double maxomega{140.};

TEST(green1d_case, green1d_test) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    auto nprocs = world.size();
    auto my_id = world.rank();

    // Load the data for pure Si.
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto chain_dir = basedir / boost::filesystem::path("1d_chain");
    auto poscar_path = chain_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = chain_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants = alma::load_FORCE_CONSTANTS(
        ifc_path.string().c_str(), *poscar, 10, 1, 1);

    randutils::mt19937_rng rng;
    // Generate a random offset for the sampling.
    double deltaq;

    if (my_id == 0)
        deltaq = rng.uniform(0., 2. * alma::constants::pi);
    boost::mpi::broadcast(world, deltaq, 0);

    Eigen::VectorXd q0(3);
    q0.fill(0.);
    Eigen::VectorXi normal(3);
    normal(0) = 1;
    normal(1) = 0;
    normal(2) = 0;
    Eigen::VectorXi rnormal(alma::reduce_integers(normal));
    Eigen::VectorXd bz{rnormal.cast<double>() / 2. / alma::constants::pi};
    Eigen::Vector3d origin = q0 - deltaq * bz;

    // Create the D(q) factory and the Green's function factory.
    auto builder =
        alma::Dynamical_matrix_builder(*poscar, syms, *force_constants);
    auto gff =
        alma::Green1d_factory(*poscar, origin, normal, nqpoints, builder);

    auto limits = alma::my_jobs(nE, nprocs, my_id);
    Eigen::ArrayXcd my_gf{Eigen::ArrayXcd::Zero(nE)};

    for (std::size_t i = limits[0]; i < limits[1]; ++i) {
        double omega = maxomega * std::sqrt(static_cast<double>(i) / nE);
        my_gf(i) = gff.build(omega, 1)(0, 0);
    }
    Eigen::ArrayXcd total_gf{Eigen::ArrayXcd::Zero(nE)};
    boost::mpi::all_reduce(world,
                           my_gf.data(),
                           my_gf.size(),
                           total_gf.data(),
                           std::plus<std::complex<double>>());

    if (my_id == 0) {
        // Load the reference data and compare each element.
        auto ref_path = chain_dir / "g00.txt";
        std::ifstream f(ref_path.c_str());
        std::size_t nfailures = 0;

        for (std::size_t i = 0; i < nE; ++i) {
            double field;
            f >> field;
            f >> field;

            if (!alma::almost_equal(field, total_gf(i).real(), 1e-5))
                ++nfailures;
            f >> field;

            if (!alma::almost_equal(field, total_gf(i).imag(), 1e-5))
                ++nfailures;
        }
        // A limited number of failures due to numerical noise close
        // to the divergences can be tolerated.
        EXPECT_LE(nfailures, 5 * nE / 1000);
    }
}
