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
/// Verify that the kappa_crossplanefilms executable works correctly.

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
#include <where_is_kappa_crossplanefilms.hpp>

// Randomly-named directory to store the output of the program.
std::string targetdir;

class G_Environment : public ::testing::Environment {
public:
    virtual ~G_Environment() {
    }
    virtual void SetUp() {
        std::string path_prefix(boost::filesystem::path(kappa_crossplanefilms)
                                    .parent_path()
                                    .string());

        // BUILD XML INPUT FILE
        std::string xmlfilename =
            (boost::filesystem::path(path_prefix) /
             boost::filesystem::path("test_kappa_crossplanefilms.xml"))
                .string();
        std::ofstream xmlwriter(xmlfilename);

        targetdir = (boost::filesystem::path("test_results") /
                     boost::filesystem::unique_path(
                         "kappa_crossplanefilms_%%%%-%%%%-%%%%-%%%%"))
                        .string();

        xmlwriter << "<crossplanefilmsweep>" << std::endl;
        xmlwriter << "  <H5repository root_directory=\"";
        auto repository_root = boost::filesystem::path(TEST_RESOURCE_DIR);
        xmlwriter << repository_root.string() << "\"/>" << std::endl;
        xmlwriter << "  <compound directory=\"Si\" base=\"Si\" gridA=\"12\" "
                     "gridB=\"12\" gridC=\"12\"/>"
                  << std::endl;
        xmlwriter << "  <sweep type=\"log\" start=\"1e-9\" stop=\"1e-4\" "
                     "points=\"6\"/>"
                  << std::endl;
        xmlwriter << "  <transportAxis x=\"0\" y=\"0\" z=\"1\"/>" << std::endl;
        xmlwriter << "  <obtainCompactModel/>" << std::endl;
        xmlwriter << "  <target directory=\"" << targetdir
                  << "\" "
                     "file=\"AUTO\"/>"
                  << std::endl;
        xmlwriter << "</crossplanefilmsweep>" << std::endl;
        xmlwriter.close();

        // RUN EXECUTABLE
        std::string command = (boost::filesystem::path(path_prefix) /
                               boost::filesystem::path("kappa_crossplanefilms"))
                                  .string() +
                              " " + xmlfilename;
        ASSERT_EQ(std::system(command.c_str()), 0);
    }
    virtual void TearDown() {
    }
};

TEST(conductivityvalues_case, kappa_crossplanefilms_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path(
             "Si_12_12_12_0,0,1_300K_1nm_100um.crossplanefilms"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(6, 3);
    data_ref << 1e-09, 1.00219, 0.00740814, 1e-08, 7.9318, 0.0586315, 1e-07,
        37.8199, 0.279562, 1e-06, 84.2131, 0.622499, 1e-05, 120.888, 0.893597,
        0.0001, 133.402, 0.986104;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(parametricmodel_case, kappa_crossplanefilms_test) {
    std::ifstream modelreader;
    modelreader.open(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacrossplanemodel"))
            .string());
    std::string dataline;

    getline(modelreader, dataline, '\n');
    getline(modelreader, dataline, '\n');
    EXPECT_TRUE(dataline.find("135.282") != std::string::npos);
    getline(modelreader, dataline, '\n');
    EXPECT_TRUE(dataline.find("Lb = 23.3125nm") != std::string::npos);
    EXPECT_TRUE(dataline.find("Ld = 8.76258um") != std::string::npos);
    getline(modelreader, dataline, '\n');
    EXPECT_TRUE(dataline.find("0.902486%") != std::string::npos);
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new G_Environment);
    return RUN_ALL_TESTS();
}
