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

/// @file
/// Computes the interface conductance between two materials. The 
/// transmission coefficients can either be set to a constant, or computed
/// using DMM or AMM

/// STL headers
#include <iostream>
#include <fstream>
#include <functional>
#include <vector>
#include <string>
#include <chrono>

///External libraries headers
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <Eigen/Dense>

///almaBTE headers
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <isotopic_scattering.hpp>
#include <bulk_hdf5.hpp>
#include <analytic1d.hpp>
#include <io_utils.hpp>
#include <bulk_properties.hpp>
#include <interfaces.hpp>


/// This boost specialization of serialization is required to
/// perform MPI operations of the matrix stored quantities
namespace boost {
namespace serialization {
/// Eigen Array serialization:
template <class Archive>
void serialize(Archive& ar,
               Eigen::Array<double, -1, 1>& t,
               const unsigned int version) {
    Eigen::ArrayXd::Index rows = t.rows();
    Eigen::ArrayXd::Index cols = t.cols();

    ar& rows;
    ar& cols;
    // Because our array is dynamic we need to ensure resizing
    if (rows * cols != t.size())
        t.resize(rows, cols);

    ar& boost::serialization::make_array(t.data(), rows * cols);
}
template <class Archive>
void serialize(Archive& ar,
               Eigen::Vector3d& t,
               const unsigned int version) {
    Eigen::ArrayXd::Index rows = t.rows();
    Eigen::ArrayXd::Index cols = t.cols();
    ar& boost::serialization::make_array(t.data(), rows * cols);
}

template <class Archive>
void serialize(Archive& ar,
               Eigen::Array<double, -1, -1>& t,
               const unsigned int version) {
    Eigen::ArrayXd::Index rows = t.rows();
    Eigen::ArrayXd::Index cols = t.cols();

    ar& rows;
    ar& cols;
    // Because our array is dynamic we need to ensure resizing
    if (rows * cols != t.size())
        t.resize(rows, cols);

    ar& boost::serialization::make_array(t.data(), rows * cols);
}
} // namespace serialization
} // namespace boost

/// Inline function for the dump of the transmission coefficient
inline void dump_transmission_coefficient(const std::string& interface_model, 
                                          const std::string& h5_repository_A, 
                                          const Eigen::Vector3d& uvector_A, 
                                          const std::string& h5_repository_B, 
                                          const Eigen::Vector3d& uvector_B, 
                                          const Eigen::ArrayXXd& alpha) {

    std::ofstream ofile;
    ofile.open("alpha."+interface_model+".dat");
    boost::archive::text_oarchive oarchive(ofile);
    oarchive << interface_model;
    oarchive << h5_repository_A;
    oarchive << uvector_A;
    oarchive << h5_repository_B;
    oarchive << uvector_B;
    oarchive << alpha;
    ofile.close();

    return;
}



/// Aliases
using dynmat_info = std::tuple<std::string,int,int,int,bool,std::string,alma::nonanalytic_treatment>;

int main(int argc, char** argv) {
    // set up MPI environment
    boost::mpi::environment env;
    boost::mpi::communicator world;

    if (argc < 2) {
        if (world.rank() == 0)
            std::cout << "USAGE: interface_conductance <inputfile.xml>" << std::endl;
        return 1;
    }
    else {
        // define variables
        std::string h5_repository_A    = "None";
        std::string h5_repository_B   = "None";
        std::string mat_directory_A    = ".";
        std::string mat_directory_B   = ".";
        Eigen::Vector3d uvector_A(0.0, 0.0, 0.0);
        Eigen::Vector3d uvector_B(0.0, 0.0, 0.0);

        double Temperature = -1.; // K
        std::string interface_model = "constant";
        std::string alpha_file = "None";
        double trans_const = 0.5;


        /// Information for AMM
        dynmat_info dynmat_info_B, dynmat_info_A;

        if (world.rank()==0) {
            std::cout << "*************************************" << std::endl;
            std::cout << "This is almaBTE/interface_conductance version " << ALMA_VERSION_MAJOR
                    << "." << ALMA_VERSION_MINOR << std::endl;
            std::cout << "*************************************" << std::endl;
        }

        // verify that input file exists.
        if (!boost::filesystem::exists(boost::filesystem::path{argv[1]}) and world.rank()==0) {
            std::cout << "ERROR: input file " << argv[1] << " does not exist."
                      << std::endl;
            exit(1);
        }

        /////////////////////////
        /// PARSE INPUT FILE  ///
        /////////////////////////

        std::string xmlfile(argv[1]);
        if (world.rank()==0)
            std::cout << "PARSING " << xmlfile << " ..." << std::endl;

        // Create empty property tree object
        boost::property_tree::ptree tree;

        // Parse XML input file into the tree
        boost::property_tree::read_xml(xmlfile, tree);

        for (const auto& v : tree.get_child("interface_conductance")) {

            ///Read readout material information

            if (v.first == "materialA") {
                for (auto it = v.second.begin(); it != v.second.end(); it++) {
                    if (it->first == "material") {

                        if (alma::probeXMLfield<std::string>(*it,"directory"))
                            mat_directory_A =
                                alma::parseXMLfield<std::string>(*it, "directory");

                        h5_repository_A =
                            alma::parseXMLfield<std::string>(*it, "database");
                    }
                    if (it->first == "axis") {
                        uvector_A(0) =
                            alma::parseXMLfield<double>(*it, "x");
                        uvector_A(1) =
                            alma::parseXMLfield<double>(*it, "y");
                        uvector_A(2) =
                            alma::parseXMLfield<double>(*it, "z");

                        uvector_A = uvector_A.normalized();

                    }
                }
            }


            ///Read readout material information

            if (v.first == "materialB") {
                for (auto it = v.second.begin(); it != v.second.end(); it++) {
                    if (it->first == "material") {

                        if (alma::probeXMLfield<std::string>(*it,"directory"))
                            mat_directory_B =
                                alma::parseXMLfield<std::string>(*it, "directory");

                        h5_repository_B =
                            alma::parseXMLfield<std::string>(*it, "database");
                    }
                    if (it->first == "axis") {
                        uvector_B(0) =
                            alma::parseXMLfield<double>(*it, "x");
                        uvector_B(1) =
                            alma::parseXMLfield<double>(*it, "y");
                        uvector_B(2) =
                            alma::parseXMLfield<double>(*it, "z");

                        uvector_B = uvector_B.normalized();

                    }
                }
            }

            if (v.first == "Temperature")
                Temperature =
                	alma::parseXMLfield<double>(v, "T");

            if (v.first == "Interface") {
                interface_model =
                    alma::parseXMLfield<std::string>(v,"model");

                if (alma::probeXMLfield<double>(v,"alpha"))
                            trans_const =
                                alma::parseXMLfield<double>(v, "alpha");

                if (alma::probeXMLfield<std::string>(v,"file"))
                            alpha_file =
                                alma::parseXMLfield<std::string>(v, "file");


                for (auto it = v.second.begin(); it != v.second.end(); it++) {
                   if (it->first == "materialA") {
                       auto fname = alma::parseXMLfield<std::string>(*it,"ifc");
                       auto sA    = alma::parseXMLfield<int>(*it,"sA");
                       auto sB    = alma::parseXMLfield<int>(*it,"sB");
                       auto sC    = alma::parseXMLfield<int>(*it,"sC");

                       bool born = false;
                       std::string fborn = "";

                       if (alma::probeXMLfield<std::string>(*it,"born")) {
                           born = true;
                           fborn = alma::parseXMLfield<std::string>(*it,"born");
                       }

                       alma::nonanalytic_treatment nonanalytic_method = alma::nonanalytic_treatment::none;
                       if (v.first == "nonanalytic_treatment") {
                           std::string my_nac = alma::parseXMLfield<std::string>(v, "method");
                           alma::string_to_lower(my_nac);
                           if (my_nac.find("gonze") != std::string::npos) {
                                   nonanalytic_method = alma::nonanalytic_treatment::gonze;
                           }
                           else if (my_nac.find("wang") != std::string::npos) {
                                   nonanalytic_method = alma::nonanalytic_treatment::wang;
                           }
                           else {
                                   throw alma::value_error("Unrecognized NAC treatment: only Gonze and Wang are supported.");
                           }
                       }

                       dynmat_info_A = {fname,sA,sB,sC,born,fborn,nonanalytic_method};

                   }
                   if (it->first == "materialB") {
                       auto fname = alma::parseXMLfield<std::string>(*it,"ifc");
                       auto sA    = alma::parseXMLfield<int>(*it,"sA");
                       auto sB    = alma::parseXMLfield<int>(*it,"sB");
                       auto sC    = alma::parseXMLfield<int>(*it,"sC");

                       bool born = false;
                       std::string fborn = "";

                       if (alma::probeXMLfield<std::string>(*it,"born")) {
                           born = true;
                           fborn = alma::parseXMLfield<std::string>(*it,"born");
                       }

                       alma::nonanalytic_treatment nonanalytic_method = alma::nonanalytic_treatment::none;
                       if (v.first == "nonanalytic_treatment") {
                           std::string my_nac = alma::parseXMLfield<std::string>(v, "method");
                           alma::string_to_lower(my_nac);
                           if (my_nac.find("gonze") != std::string::npos) {
                                   nonanalytic_method = alma::nonanalytic_treatment::gonze;
                           }
                           else if (my_nac.find("wang") != std::string::npos) {
                                   nonanalytic_method = alma::nonanalytic_treatment::wang;
                           }
                           else {
                                   throw alma::value_error("Unrecognized NAC treatment: only Gonze and Wang are supported.");
                           }
                       }

                       dynmat_info_B = {fname,sA,sB,sC,born,fborn,nonanalytic_method};
                   }

                }


            }

            
        } // end XML parsing

        // Ensure that provided information is within expected bounds

        bool badinput = false;

        if (Temperature <= 0.0 and world.rank()==0) {
            std::cout << "ERROR: provided temperature is " << Temperature
                      << " K" << std::endl;
            std::cout << "Value must be positive." << std::endl;
            badinput = true;
        }

        if (uvector_A.norm() < 1e-12 and world.rank()==0) {
            std::cout << "ERROR: target provided transport axis vector has zero norm."
                      << std::endl;
            badinput = true;
        }


        if (uvector_B.norm() < 1e-12 and world.rank()==0) {
            std::cout << "ERROR: readout provided transport axis vector has zero norm."
                      << std::endl;
            badinput = true;
        }

        if (badinput and world.rank()==0) {
            world.abort(1);
        }


        // Initialise file system and verify that directories actually exist

        const auto hdf5_path_A = boost::filesystem::current_path() /
                        boost::filesystem::path(mat_directory_A) /
                        boost::filesystem::path(h5_repository_A);

        const auto hdf5_path_B = boost::filesystem::current_path() /
                        boost::filesystem::path(mat_directory_B) /
                        boost::filesystem::path(h5_repository_B);

        if (!(boost::filesystem::exists(hdf5_path_A)) and world.rank()==0) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Target H5 file " << h5_repository_A
                      << " does not exist within the material directory."
                      << std::endl;
            world.abort(1);
        }

        if (!(boost::filesystem::exists(hdf5_path_B)) and world.rank()==0) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Readout H5 file " << h5_repository_B
                      << " does not exist within the material directory."
                      << std::endl;
            world.abort(1);
        }

        // Opening HDF5 files
        if (world.rank()==0)
            std::cout << "Opening material A HDF5 file " << hdf5_path_A << std::endl;

        auto hdf5_data_A =
            alma::load_bulk_hdf5(hdf5_path_A.string().c_str(), world);

        /// Defined as shared_ptr through movement operation from unique_ptr
        /// as our routines want to internally share copies of the ptr
        std::shared_ptr<alma::Crystal_structure> poscar_A =
            std::move(std::get<1>(hdf5_data_A));
        std::shared_ptr<alma::Symmetry_operations> syms_A =
            std::move(std::get<2>(hdf5_data_A));
        std::shared_ptr<alma::Gamma_grid> grid_A =
            std::move(std::get<3>(hdf5_data_A));
        std::shared_ptr<std::vector<alma::Threeph_process>> processes_A =
            std::move(std::get<4>(hdf5_data_A));

        if (processes_A->size() == 0) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "List of 3-phonon processes is missing in H5 target file."
                      << std::endl;
            world.abort(1);
        }

        if (world.rank()==0)
            std::cout << "Opening material B HDF5 file " << hdf5_path_B << std::endl;

        auto hdf5_data_B =
            alma::load_bulk_hdf5(hdf5_path_B.string().c_str(), world);

        /// Defined as shared_ptr through movement operation from unique_ptr
        /// as our routines want to internally share copies of the ptr
        std::shared_ptr<alma::Crystal_structure> poscar_B =
            std::move(std::get<1>(hdf5_data_B));
        std::shared_ptr<alma::Symmetry_operations> syms_B =
            std::move(std::get<2>(hdf5_data_B));
        std::shared_ptr<alma::Gamma_grid> grid_B =
            std::move(std::get<3>(hdf5_data_B));
        std::shared_ptr<std::vector<alma::Threeph_process>> processes_B =
            std::move(std::get<4>(hdf5_data_B));


        /// Computing interface
        if (world.rank()==0) {
            std::cout << "*Computing transmission coefficient:" << std::endl;
            std::cout << "   -Model: " << interface_model << std::endl;
        }

        Eigen::ArrayXXd alpha(grid_A->get_spectrum_at_q(0).omega.size(),grid_A->nqpoints);
        alpha.setZero();

        if (interface_model=="constant") {
            alpha.setConstant(trans_const);
        }
        else if (interface_model == "DMM") {

            /// Only master calculates the DMM
            if (world.rank() == 0) {

                alma::DMM_interface myinterface(*poscar_A,*grid_A,*syms_A,uvector_A,
                    *poscar_B,*grid_B,*syms_B,uvector_B);

                auto nbands_ = grid_A->get_spectrum_at_q(0).omega.size();

                for (std::size_t iq = 0; iq < grid_A->nqpoints; iq++) {
                    auto sp = grid_A->get_spectrum_at_q(iq);
                    for (decltype(nbands_) ib = 0; ib < nbands_; ib++) {
                        /// Remove acoustic modes at Gamma and ignore non-incident modes
                        double vproj = uvector_A.dot(sp.vg.col(ib).matrix());
                        if (alma::almost_equal(sp.omega(ib),0.) or vproj < 0. or alma::almost_equal(vproj,0.))
                            continue;
                        alpha(ib,iq) = myinterface.get_transmission(iq,ib);
                    }
                }

            }
            /// Sharing to all procs
            broadcast(world,alpha.data(),alpha.size(),0);

        }
        else if (interface_model == "AMM") {

            /// AMM is experimental
	    std::cerr << "AMM model is experimental: do not use for production runs" << std::endl;

            ///Compute the dynamical matrices
            auto ifcs_A = alma::load_FORCE_CONSTANTS(std::get<0>(dynmat_info_A).c_str(),
                                                         *poscar_A,
                                                         std::get<1>(dynmat_info_A),
                                                         std::get<2>(dynmat_info_A),
                                                         std::get<3>(dynmat_info_A));

            std::unique_ptr<alma::Dielectric_parameters> born_A;
	    alma::nonanalytic_treatment nac_A = std::get<6>(dynmat_info_A);

            if (std::get<4>(dynmat_info_A))
                born_A = alma::load_BORN(std::get<5>(dynmat_info_A).c_str());

            auto dyn_A = (std::get<4>(dynmat_info_A)) ? alma::Dynamical_matrix_builder(*poscar_A,*syms_A,*ifcs_A,*born_A,nac_A) :
                alma::Dynamical_matrix_builder(*poscar_A,*syms_A,*ifcs_A);

            auto ifcs_B = alma::load_FORCE_CONSTANTS(std::get<0>(dynmat_info_B).c_str(),
                                                         *poscar_B,
                                                         std::get<1>(dynmat_info_B),
                                                         std::get<2>(dynmat_info_B),
                                                         std::get<3>(dynmat_info_B));

            std::unique_ptr<alma::Dielectric_parameters> born_B;
	    alma::nonanalytic_treatment nac_B = std::get<6>(dynmat_info_B);

            if (std::get<4>(dynmat_info_B))
                born_B = alma::load_BORN(std::get<5>(dynmat_info_B).c_str());

            auto dyn_B = (std::get<4>(dynmat_info_B)) ? alma::Dynamical_matrix_builder(*poscar_B,*syms_B,*ifcs_B,*born_B,nac_B) :
                alma::Dynamical_matrix_builder(*poscar_B,*syms_B,*ifcs_B);

            alma::AMM_interface myinterface(*poscar_A,*grid_A,*syms_A,uvector_A,dyn_A,
                *poscar_B,*grid_B,*syms_B,uvector_B,dyn_B,world);

            auto nbands_ = grid_A->get_spectrum_at_q(0).omega.size();


            for (std::size_t iq = 0; iq < grid_A->nqpoints; iq++) {
                auto sp = grid_A->get_spectrum_at_q(iq);
                for (decltype(nbands_) ib = 0; ib < nbands_; ib++) {
                    // Remove acoustic modes at Gamma and ignore non-incident modes
                    double vproj = uvector_A.dot(sp.vg.col(ib).matrix());
                    if (alma::almost_equal(sp.omega(ib),0.) or vproj < 0. or alma::almost_equal(vproj,0.))
                        continue;
                    alpha(ib,iq) = myinterface.get_transmission(iq,ib);
                }
            }
        }
        else if (interface_model == "load_from_file"){
            /// Only the master reads the file
            if (world.rank() == 0) {
                std::ifstream ifile;
                std::string i_model, i_A_name, i_B_name;
                Eigen::Vector3d i_A_axis, i_B_axis;
                ifile.open(alpha_file);
                boost::archive::text_iarchive iarchive(ifile);
                /// Getting information from the archive
                iarchive >> i_model;
                iarchive >> i_A_name;
                iarchive >> i_A_axis;
                iarchive >> i_B_name;
                iarchive >> i_B_axis;
                iarchive >> alpha;

                /// Make some check:
                if (h5_repository_A != i_A_name) {
                    std::cerr << "Error : A from file and from run differ\n";
                    world.abort(1);
                }
                if (h5_repository_B != i_B_name) {
                    std::cerr << "Error : B from file and from run differ\n";
                    world.abort(1);
                }
                if ( !alma::almost_equal((i_A_axis-uvector_A).norm(),0.)) {
                    std::cerr << "Error : A axis from file and from run differ\n";
                    world.abort(1);
                }
                if ( !alma::almost_equal((i_B_axis-uvector_B).norm(),0.)) {
                    std::cerr << "Error : B axis from file and from run differ\n";
                    world.abort(1);
                }


                std::cout << "   -File model : " << i_model << std::endl;
                ifile.close();
            }
            /// Sharing to all procs
            broadcast(world,alpha.data(),alpha.size(),0);
        }
        else {
            if (world.rank() == 0 )    {
                std::cerr << "ERROR: " << interface_model << " is not a valid interface model" << std::endl;
            }
            world.barrier();
            world.abort(1);
        }

        if (interface_model != "load_from_file" && world.rank() == 0) {
            dump_transmission_coefficient(interface_model, h5_repository_A, 
                                          uvector_A, h5_repository_B, uvector_B, alpha);
        }

        if (world.rank() == 0) {
            std::cout << "   [DONE]" << std::endl;
            std::cout << "*Computing the conductance:\n";
	}

	double G = interface_conductance(*poscar_A,*grid_A,uvector_A,alpha,Temperature,world);

	if (world.rank() == 0) {
 	    std::cout << "G : " << G << " W/(m^2·K)" << std::endl;
	    std::cout << "   [DONE]" << std::endl;
	}

        return EXIT_SUCCESS;
    }
}
