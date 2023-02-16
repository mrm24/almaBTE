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
/// Verify that the superlattice_builder executable works correctly.

#include <cstdlib>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <io_utils.hpp>
#include <Eigen/Dense>
#include <cmakevars.hpp>
#include <utilities.hpp>
#include <bulk_hdf5.hpp>
#include <bulk_properties.hpp>
#include <where_is_superlattice_builder.hpp>

// Randomly-named directory to store the output of the program.
std::string targetdir;

class G_Environment : public ::testing::Environment {
public:
    virtual ~G_Environment() {
    }
    virtual void SetUp() {
        std::string path_prefix(boost::filesystem::path(superlattice_builder)
                                    .parent_path()
                                    .string());

        // BUILD XML INPUT FILE
        std::string xmlfilename =
            (boost::filesystem::path(path_prefix) /
             boost::filesystem::path("test_superlattice_builder.xml"))
                .string();
        std::ofstream xmlwriter(xmlfilename);
        targetdir = (boost::filesystem::path("test_results") /
                     boost::filesystem::unique_path(
                         "superlattice_builder_%%%%-%%%%-%%%%-%%%%"))
                        .string();

        xmlwriter << "<superlattice>" << std::endl;
        xmlwriter << "  <materials_repository root_directory=\"";
        auto repository_root = boost::filesystem::path(TEST_RESOURCE_DIR);
        xmlwriter << repository_root.string() << "\"/>" << std::endl;
        xmlwriter << "  <gridDensity A=\"8\" B=\"8\" C=\"8\"/>" << std::endl;
        xmlwriter << "  <normal na=\"0\" nb=\"0\" nc=\"1\" nqline=\"501\"/>"
                  << std::endl;
        xmlwriter << "  <compound name=\"Si\"/>" << std::endl;
        xmlwriter << "  <compound name=\"Ge\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.0\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.0\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.1\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.2\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.3\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.2\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.1\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.0\"/>" << std::endl;
        xmlwriter << "  <layer mixfraction=\"0.0\"/>" << std::endl;
        xmlwriter << "  <target directory=\"" << targetdir << "\"/>"
                  << std::endl;
        xmlwriter << "</superlattice>" << std::endl;
        xmlwriter.close();

        // RUN EXECUTABLE
        std::string command = (boost::filesystem::path(path_prefix) /
                               boost::filesystem::path("superlattice_builder"))
                                  .string() +
                              " " + xmlfilename;
        ASSERT_EQ(std::system(command.c_str()), 0);
    }
    virtual void TearDown() {
    }
};

TEST(filecreation_case, superlattice_builder_test) {
    EXPECT_TRUE(boost::filesystem::exists(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("superlattice_Si0.9_Ge0.1_LXXTKLO2_8_8_8.h5"))
            .string()));
}

TEST(conductivity_value_case, superlattice_builder_test) {
    // set up MPI environment
    boost::mpi::environment env;
    boost::mpi::communicator world;

    // obtain phonon data from HDF5 file
    auto hdf5_path = boost::filesystem::path(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("superlattice_Si0.9_Ge0.1_LXXTKLO2_8_8_8.h5"))
            .string());

    auto hdf5_data = alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
    auto description = std::get<0>(hdf5_data);
    auto poscar = std::move(std::get<1>(hdf5_data));
    auto syms = std::move(std::get<2>(hdf5_data));
    auto grid = std::move(std::get<3>(hdf5_data));
    auto processes = std::move(std::get<4>(hdf5_data));

    // Check that the file contains valid superlattice data.

    auto subgroups =
        alma::list_scattering_subgroups(hdf5_path.string().c_str(), world);

    int superlattice_count = 0;

    for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
        if (subgroups.at(ngroup).find("superlattice") != std::string::npos) {
            superlattice_count++;
        }
    }

    superlattice_count /= 2;

    EXPECT_TRUE(superlattice_count > 0);

    // Check that the file contains the specific superlattice we are dealing
    // with.

    std::string superlattice_UID = "LXXTKLO2";
    int UIDcount = 0;

    for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
        if (subgroups.at(ngroup).find(superlattice_UID) != std::string::npos) {
            UIDcount++;
        }
    }

    EXPECT_TRUE(UIDcount == 2);

    // Load the scattering rates from the H5 file

    Eigen::ArrayXXd w0_SLdisorder;
    Eigen::ArrayXXd w0_SLbarriers;


    for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
        bool contains_SLdisorder =
            (subgroups.at(ngroup).find("superlattice") != std::string::npos) &&
            (subgroups.at(ngroup).find("disorder") != std::string::npos);

        bool contains_SLbarriers =
            (subgroups.at(ngroup).find("superlattice") != std::string::npos) &&
            (subgroups.at(ngroup).find("barriers") != std::string::npos);

        bool UIDmatch =
            (subgroups.at(ngroup).find(superlattice_UID) != std::string::npos);

        if (contains_SLdisorder && UIDmatch) {
            auto mysubgroup = alma::load_scattering_subgroup(
                hdf5_path.string().c_str(), subgroups.at(ngroup), world);
            w0_SLdisorder = mysubgroup.w0;
        }

        if (contains_SLbarriers && UIDmatch) {
            auto mysubgroup = alma::load_scattering_subgroup(
                hdf5_path.string().c_str(), subgroups.at(ngroup), world);
            w0_SLbarriers = mysubgroup.w0;
        }
    }

    // RUN CALCULATIONS

    double Tref = 300.0;

    Eigen::ArrayXXd w_elastic = w0_SLbarriers.array() + w0_SLdisorder.array();
    Eigen::ArrayXXd w3(alma::calc_w0_threeph(*grid, *processes, Tref, world));
    Eigen::ArrayXXd w(w3 + w_elastic);
    Eigen::Matrix3d kappa_RTA = alma::calc_kappa(*poscar, *grid, w, Tref);

    constexpr double kappa_target = 2.79990804;
    EXPECT_NEAR(kappa_RTA(2, 2), kappa_target, 5e-3 * kappa_target);
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new G_Environment);
    return RUN_ALL_TESTS();
}
