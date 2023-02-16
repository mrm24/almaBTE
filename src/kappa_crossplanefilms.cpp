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
// WITHOUT WARRANLIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

/// @file
/// EXECUTABLE THAT COMPUTES RTA CONDUCTIVITIES VS FILM THICKNESS.
/// If parametric option is turned on, the program will also determine
/// a compact parametric form for the cross-plane conductivity.
/// For technical details see
/// B. Vermeersch, J. Carrete, N. Mingo
/// Applied Physics Letters 108, 193104 (2016)
/// http://dx.doi.org/10.1063/1.4948968

#include <iostream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <isotopic_scattering.hpp>
#include <bulk_hdf5.hpp>
#include <analytic1d.hpp>
#include <io_utils.hpp>
#include <bulk_properties.hpp>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <unsupported/Eigen/NonLinearOptimization>

// functor for parametric model fitting in single crystals

struct singlecrystal_functor {
    int operator()(const Eigen::VectorXd& parlist, Eigen::VectorXd& FUN) const {
        if (parlist(0) <= 0.0) {
            std::cout
                << "kappa_crossplanefilms > parametric model fitting > WARNING:"
                << std::endl;
            std::cout << "Encountered Lb <= 0. Least-square optimiser likely "
                         "becoming unstable."
                      << std::endl;
        }

        if (parlist(1) <= 0.0) {
            std::cout
                << "kappa_crossplanefilms > parametric model fitting > WARNING:"
                << std::endl;
            std::cout << "Encountered Ld <= 0. Least-square optimiser likely "
                         "becoming unstable."
                      << std::endl;
        }

        FUN = (this->computeModel(parlist)).array() / (this->target).array() -
              1.0;

        return 0;
    }


    int df(const Eigen::VectorXd& parlist, Eigen::MatrixXd& dfdpar) const {
        double rel_change = 1e-6;

        for (int ncol = 0; ncol < this->inputs(); ncol++) {
            double deltapar = 0.5 * rel_change * parlist(ncol);

            if (deltapar == 0.0) {
                deltapar = rel_change;
            }

            Eigen::VectorXd parleft = parlist;
            parleft(ncol) -= deltapar;
            Eigen::VectorXd fleft(this->values());
            this->operator()(parleft, fleft);

            Eigen::VectorXd parright = parlist;
            parright(ncol) += deltapar;
            Eigen::VectorXd fright(this->values());
            this->operator()(parright, fright);

            dfdpar.col(ncol) =
                (fright.array() - fleft.array()) / (2.0 * deltapar);
        }

        return 0;
    }


    Eigen::VectorXd target;
    Eigen::VectorXd thicknesslist;
    double kappa_bulk;

    int inputs() const {
        return 2;
    } // Number of fitting parameters


    int values() const {
        return this->thicknesslist.size();
    } // Number of entries in FUN


    Eigen::VectorXd computeModel(const Eigen::VectorXd& parlist) const {
        double Lb = parlist(0) * 1e-9;
        double Ld = parlist(1) * 1e-6;

        return (this->kappa_bulk / std::log(Ld / Lb)) *
               ((this->thicknesslist.array() / Lb + 1.0) /
                (this->thicknesslist.array() / Ld + 1.0))
                   .log();
    }
};

struct alloy_functor {
    int operator()(const Eigen::VectorXd& parlist, Eigen::VectorXd& FUN) const {
        if (parlist(0) <= 1.0) {
            std::cout
                << "kappa_crossplanefilms > parametric model fitting > WARNING:"
                << std::endl;
            std::cout << "Encountered alpha <= 1. Least-square optimiser "
                         "likely becoming unstable."
                      << std::endl;
        }

        if (parlist(0) >= 2.0) {
            std::cout
                << "kappa_crossplanefilms > parametric model fitting > WARNING:"
                << std::endl;
            std::cout << "Encountered alpha >= 2. Least-square optimiser "
                         "likely becoming unstable."
                      << std::endl;
        }

        if (parlist(1) <= 0.0) {
            std::cout
                << "kappa_crossplanefilms > parametric model fitting > WARNING:"
                << std::endl;
            std::cout << "Encountered Lb <= 0. Least-square optimiser likely "
                         "becoming unstable."
                      << std::endl;
        }

        if (parlist(2) <= 0.0) {
            std::cout
                << "kappa_crossplanefilms > parametric model fitting > WARNING:"
                << std::endl;
            std::cout << "Encountered Ld <= 0. Least-square optimiser likely "
                         "becoming unstable."
                      << std::endl;
        }

        FUN = (this->computeModel(parlist)).array() / (this->target).array() -
              1.0;

        return 0;
    }


    int df(const Eigen::VectorXd& parlist, Eigen::MatrixXd& dfdpar) const {
        double rel_change = 1e-6;

        for (int ncol = 0; ncol < this->inputs(); ncol++) {
            double deltapar = 0.5 * rel_change * parlist(ncol);

            if (deltapar == 0.0) {
                deltapar = rel_change;
            }

            Eigen::VectorXd parleft = parlist;
            parleft(ncol) -= deltapar;
            Eigen::VectorXd fleft(this->values());
            this->operator()(parleft, fleft);

            Eigen::VectorXd parright = parlist;
            parright(ncol) += deltapar;
            Eigen::VectorXd fright(this->values());
            this->operator()(parright, fright);

            dfdpar.col(ncol) =
                (fright.array() - fleft.array()) / (2.0 * deltapar);
        }

        return 0;
    }


    Eigen::VectorXd target;
    Eigen::VectorXd thicknesslist;
    double kappa_bulk;

    int inputs() const {
        return 3;
    } // Number of fitting parameters


    int values() const {
        return this->thicknesslist.size();
    } // Number of entries in FUN


    Eigen::VectorXd computeModel(const Eigen::VectorXd& parlist) const {
        double alpha = parlist(0);
        double Lb = parlist(1) * 1e-9;
        double Ld = parlist(2) * 1e-6;

        Eigen::VectorXd inverseL =
            Eigen::pow(this->thicknesslist.array(), -1.0);

        Eigen::VectorXd factor1 =
            Eigen::pow(1.0 + Ld * inverseL.array(), alpha - 2.0);
        Eigen::VectorXd factor2 =
            Eigen::pow(1.0 + Lb * inverseL.array(), 1.0 - alpha);

        return this->kappa_bulk * factor1.array() * factor2.array();
    }
};

int main(int argc, char** argv) {
    // set up MPI environment
    boost::mpi::environment env;
    boost::mpi::communicator world;

    if (world.size() > 1) {
        std::cout << "*** ERROR: kappa_crossplanefilms does not run in "
                     "parallel mode. ***"
                  << std::endl;
        world.abort(1);
    }

    if (argc < 2) {
        std::cout << "USAGE: kappa_crossplanefilms <inputfile.xml> "
                     "<OPTIONAL:Tambient>"
                  << std::endl;
        return 1;
    }

    else {
        // define variables
        std::string target_directory = "AUTO";
        std::string target_filename = "AUTO";
        std::string h5_repository = ".";
        std::string mat_directory;
        std::string mat_base;
        int gridDensityA = -1;
        int gridDensityB = -1;
        int gridDensityC = -1;
        bool logsweep = false;
        double Lmin = -1.0;
        double Lmax = -1.0;
        int NL = -1;
        Eigen::Vector3d uvector(0.0, 0.0, 0.0);
        bool superlattice = false;
        std::string superlattice_UID = "NULL";

        double T = 300.0;

        if (argc == 3) {
            T = atof(argv[2]);
        }

        bool parametric = false; // determine parametric formula?

        std::cout << "**********************************************"
                  << std::endl;
        std::cout << "This is ALMA/kappa_crossplanefilms version "
                  << ALMA_VERSION_MAJOR << "." << ALMA_VERSION_MINOR
                  << std::endl;
        std::cout << "**********************************************"
                  << std::endl;

        // verify that input file exists.
        if (!boost::filesystem::exists(boost::filesystem::path{argv[1]})) {
            std::cout << "ERROR: input file " << argv[1] << " does not exist."
                      << std::endl;
            exit(1);
        }

        /////////////////////////
        /// PARSE INPUT FILE  ///
        /////////////////////////

        std::string xmlfile(argv[1]);
        std::cout << "PARSING " << xmlfile << " ..." << std::endl;

        // Create empty property tree object
        boost::property_tree::ptree tree;

        // Parse XML input file into the tree
        boost::property_tree::read_xml(xmlfile, tree);

        for (const auto& v : tree.get_child("crossplanefilmsweep")) {
            if (v.first == "H5repository") {
                h5_repository =
                    alma::parseXMLfield<std::string>(v, "root_directory");
            }

            if (v.first == "target") {
                target_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
                target_filename = alma::parseXMLfield<std::string>(v, "file");
            }

            if (v.first == "compound") {
                mat_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
                mat_base = alma::parseXMLfield<std::string>(v, "base");

                gridDensityA = alma::parseXMLfield<int>(v, "gridA");
                gridDensityB = alma::parseXMLfield<int>(v, "gridB");
                gridDensityC = alma::parseXMLfield<int>(v, "gridC");
            }

            if (v.first == "superlattice") {
                superlattice_UID = alma::parseXMLfield<std::string>(v, "UID");
            }

            if (v.first == "sweep") {
                std::string sweepID =
                    alma::parseXMLfield<std::string>(v, "type");

                if (sweepID.compare("log") == 0) {
                    logsweep = true;
                }

                Lmin = alma::parseXMLfield<double>(v, "start");
                Lmax = alma::parseXMLfield<double>(v, "stop");
                NL = alma::parseXMLfield<int>(v, "points");
            }

            if (v.first == "transportAxis") {
                double ux = alma::parseXMLfield<double>(v, "x");
                double uy = alma::parseXMLfield<double>(v, "y");
                double uz = alma::parseXMLfield<double>(v, "z");

                uvector << ux, uy, uz;
            }

            if (v.first == "obtainCompactModel") {
                parametric = true;
            }
        } // end XML parsing

        // Ensure that provided information is within expected bounds

        bool badinput = false;

        if (gridDensityA < 1) {
            std::cout << "ERROR: provided gridA is " << gridDensityA
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (gridDensityB < 1) {
            std::cout << "ERROR: provided gridB is " << gridDensityB
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (gridDensityC < 1) {
            std::cout << "ERROR: provided gridC is " << gridDensityC
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (Lmin <= 0.0) {
            std::cout << "ERROR: provided start thickness is " << Lmin
                      << std::endl;
            std::cout << "Value must be positive." << std::endl;
            badinput = true;
        }

        if ((Lmax < Lmin) || (Lmax <= 0.0)) {
            std::cout << "ERROR: provided end thickness is " << Lmax
                      << std::endl;
            std::cout << "Value must be positive and >= start thickness."
                      << std::endl;
            badinput = true;
        }

        if (NL <= 0) {
            std::cout << "ERROR: provided number of thickness values is " << NL
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (uvector.norm() < 1e-12) {
            std::cout << "ERROR: provided transport axis vector has zero norm."
                      << std::endl;
            badinput = true;
        }

        if (T <= 0) {
            std::cout << "ERROR: provided ambient temperature is " << T
                      << std::endl;
            std::cout << "Value must be postive." << std::endl;
            badinput = true;
        }

        if (badinput) {
            exit(1);
        }

        // Initialise file system and verify that directories actually
        // exist
        auto launch_path = boost::filesystem::current_path();
        auto basedir = boost::filesystem::path(h5_repository);

        if (!(boost::filesystem::exists(boost::filesystem::path(basedir)))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Repository directory " << basedir
                      << " does not exist." << std::endl;
            exit(1);
        }

        if (!(boost::filesystem::exists(
                boost::filesystem::path(basedir / mat_directory)))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Material directory " << mat_directory
                      << " does not exist within the HDF5 repository."
                      << std::endl;
            exit(1);
        }

        // Resolve name of HDF5 file
        std::stringstream h5namebuilder;
        h5namebuilder << mat_base << "_";
        h5namebuilder << gridDensityA << "_" << gridDensityB << "_"
                      << gridDensityC << ".h5";
        std::string h5filename = h5namebuilder.str();

        // obtain phonon data from HDF5 file
        auto hdf5_path = basedir / boost::filesystem::path(mat_directory) /
                         boost::filesystem::path(h5filename);

        if (!(boost::filesystem::exists(hdf5_path))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "H5 file " << h5filename
                      << " does not exist within the material directory."
                      << std::endl;
            exit(1);
        }

        std::cout << "Opening HDF5 file " << hdf5_path << std::endl;

        auto hdf5_data =
            alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
        auto description = std::get<0>(hdf5_data);
        auto poscar = std::move(std::get<1>(hdf5_data));
        auto syms = std::move(std::get<2>(hdf5_data));
        auto grid = std::move(std::get<3>(hdf5_data));
        auto processes = std::move(std::get<4>(hdf5_data));

        if (processes->size() == 0) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "List of 3-phonon processes is missing in H5 file."
                      << std::endl;
            exit(1);
        }

        // Check if we are dealing with a superlattice.
        // If so, load the applicable scattering data.

        auto subgroups =
            alma::list_scattering_subgroups(hdf5_path.string().c_str(), world);

        int superlattice_count = 0;

        for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
            if (subgroups.at(ngroup).find("superlattice") !=
                std::string::npos) {
                superlattice_count++;
            }
        }

        superlattice_count /= 2;

        if (superlattice_count > 0) {
            superlattice = true;
        }

        Eigen::ArrayXXd w0_SLdisorder;
        Eigen::ArrayXXd w0_SLbarriers;

        if (superlattice) {
            // complain if there are multiple possibilities

            if ((superlattice_count > 1) && (superlattice_UID == "NULL")) {
                std::cout << "ERROR:" << std::endl;
                std::cout << "H5 file contains scattering information for "
                             "multiple superlattices."
                          << std::endl;
                std::cout << "Must provide the superlattice UID via the "
                             "<superlattice> XML tag."
                          << std::endl;
                exit(1);
            }

            // if the user provided a UID, verify that corresponding data exists

            if (superlattice_UID != "NULL") {
                int UIDcount = 0;

                for (std::size_t ngroup = 0; ngroup < subgroups.size();
                     ngroup++) {
                    if (subgroups.at(ngroup).find(superlattice_UID) !=
                        std::string::npos) {
                        UIDcount++;
                    }
                }

                if (UIDcount != 2) {
                    std::cout << "ERROR:" << std::endl;
                    std::cout << "H5 file does not contain any superlattice "
                                 "data with provided UID "
                              << superlattice_UID << "." << std::endl;
                    exit(1);
                }
            }

            // load the scattering rates from the H5 file

            bool UIDmatch = true;

            for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
                bool contains_SLdisorder =
                    (subgroups.at(ngroup).find("superlattice") !=
                     std::string::npos) &&
                    (subgroups.at(ngroup).find("disorder") !=
                     std::string::npos);

                bool contains_SLbarriers =
                    (subgroups.at(ngroup).find("superlattice") !=
                     std::string::npos) &&
                    (subgroups.at(ngroup).find("barriers") !=
                     std::string::npos);

                if (superlattice_count > 1) {
                    UIDmatch = (subgroups.at(ngroup).find(superlattice_UID) !=
                                std::string::npos);
                }

                if (contains_SLdisorder && UIDmatch) {
                    auto mysubgroup = alma::load_scattering_subgroup(
                        hdf5_path.string().c_str(),
                        subgroups.at(ngroup),
                        world);
                    w0_SLdisorder = mysubgroup.w0;
                }

                if (contains_SLbarriers && UIDmatch) {
                    auto mysubgroup = alma::load_scattering_subgroup(
                        hdf5_path.string().c_str(),
                        subgroups.at(ngroup),
                        world);
                    w0_SLbarriers = mysubgroup.w0;
                }
            }
        }

        // calculate scattering rates at the chosen temperature
        Eigen::ArrayXXd w3(alma::calc_w0_threeph(*grid, *processes, T, world));
        Eigen::ArrayXXd w_elastic;

        if (superlattice) {
            w_elastic = w0_SLdisorder.array() + w0_SLbarriers.array();
        }
        else {
            auto twoph_processes = alma::find_allowed_twoph(*grid, world);
            w_elastic =
                alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world);
        }

        Eigen::ArrayXXd w(w3 + w_elastic);

        // Build list of thickness values

        Eigen::VectorXd Llist;

        if (logsweep) {
            Llist = alma::logSpace(Lmin, Lmax, NL);
        }

        else {
            Llist.setLinSpaced(NL, Lmin, Lmax);
        }

        // Create output writer
        std::stringstream outputbuffer;

        // Write file header

        outputbuffer << "FilmThickness[m],";
        outputbuffer << "kappa<" << uvector(0) << "," << uvector(1) << ","
                     << uvector(2) << ">[W/m-K],";
        outputbuffer << "kappa<" << uvector(0) << "," << uvector(1) << ","
                     << uvector(2) << ">/kappa_bulk<" << uvector(0) << ","
                     << uvector(1) << "," << uvector(2) << ">[-]" << std::endl;

        std::cout << std::endl
                  << "Computing cross-plane film conductivities for "
                  << mat_base << " at " << T << "K" << std::endl;

        // RUN CALCULATIONS

        alma::analytic1D::BasicProperties_calculator propCalc(
            poscar.get(), grid.get(), &w, T);
        propCalc.setDirection(uvector);
        propCalc.setBulk();
        double kappa_bulk = propCalc.getConductivity();

        for (int nL = 0; nL < NL; nL++) {
            double L = Llist(nL);
            propCalc.setCrossPlaneFilm(L);

            double kappa_film = propCalc.getConductivity();

            outputbuffer << L << "," << kappa_film << ","
                         << kappa_film / kappa_bulk << std::endl;
        }

        std::stringstream modelheader;
        Eigen::VectorXd Lmodel;
        Eigen::VectorXd kappa_actual;
        Eigen::VectorXd kappa_model;
        double Lb, Ld, alpha;
        double avgdev;
        int Nmodel = 301;

        if (parametric) { // determine compact parametric model
            std::cout << "Determining compact parametric model for cross-plane "
                         "conductivity"
                      << std::endl;

            // create list of film thicknesses to be used for model fitting

            double Lmin = 1e-9;
            double Lmax = 1e-3;

            Eigen::VectorXd logL;
            logL.setLinSpaced(Nmodel, std::log10(Lmin), std::log10(Lmax));
            Lmodel = Eigen::pow(10.0, logL.array());

            // obtain actual film conductivity

            kappa_actual.resize(Nmodel);

            for (int nL = 0; nL < Nmodel; nL++) {
                propCalc.setCrossPlaneFilm(Lmodel(nL));
                kappa_actual(nL) = propCalc.getConductivity();
            }

            // determine type of compound
            bool alloy = poscar->is_alloy();

            // perform least square fitting

            if (!alloy) {
                // create least-square solver

                singlecrystal_functor SCF;
                SCF.kappa_bulk = kappa_bulk;
                SCF.thicknesslist = Lmodel;
                SCF.target = kappa_actual;

                Eigen::LevenbergMarquardt<singlecrystal_functor> LM(SCF);

                // initialise solver and find optimal parameter values

                Eigen::VectorXd parlist(2);
                parlist << 1.0, 1.0;

                LM.minimize(parlist);

                Lb = parlist(0);
                Ld = parlist(1);

                kappa_model = SCF.computeModel(parlist);
            }

            if (alloy) {
                // create least-square solver

                alloy_functor AF;
                AF.kappa_bulk = kappa_bulk;
                AF.thicknesslist = Lmodel;
                AF.target = kappa_actual;

                Eigen::LevenbergMarquardt<alloy_functor> LM(AF);

                // initialise solver and find optimal parameter values

                Eigen::VectorXd parlist(3);
                parlist << 1.7, 1.0, 1.0;

                LM.minimize(parlist);

                alpha = parlist(0);
                Lb = parlist(1);
                Ld = parlist(2);

                kappa_model = AF.computeModel(parlist);
            }

            avgdev =
                (kappa_model.array() / kappa_actual.array() - 1.0).abs().sum() /
                static_cast<double>(kappa_model.size());

            if (!alloy) {
                modelheader << "PARAMETRIC FORM: kappa(L) = "
                               "[kappa_bulk/ln(Ld/Lb)]*ln[(1+L/Lb)/(1+L/Ld)]"
                            << std::endl;
            }

            else {
                modelheader
                    << "PARAMETRIC FORM: kappa(L) = "
                       "kappa_bulk/[(1+Ld/L)^(2-alpha)*(1+Lb/L)^(alpha-1)]"
                    << std::endl;
            }

            modelheader << "FIXED PARAMETER: kappa_bulk = " << kappa_bulk
                        << " W/m-K" << std::endl;
            modelheader << "BEST FITTING PARAMETERS: ";

            if (alloy) {
                modelheader << "alpha = " << alpha << " , ";
            }
            modelheader << "Lb = " << alma::engineer_format(1e-9 * Lb)
                        << "m , Ld = " << alma::engineer_format(1e-6 * Ld)
                        << "m" << std::endl;
            modelheader << "Average model tolerance = " << 100.0 * avgdev << "%"
                        << std::endl;
            modelheader << "FilmThickness[m],kappa_actual[W/"
                           "m-K],kappa_parametric[W/m-K]"
                        << std::endl;
        } // end parametric model fitting

        // END CALCULATIONS

        // WRITE FILE(S)

        // go to the launch directory
        boost::filesystem::current_path(launch_path);

        // resolve output directory if AUTO is selected
        if (target_directory.compare("AUTO") == 0) {
            target_directory = "output/kappa_crossplanefilms";
        }

        // create output directory if it doesn't exist yet
        auto outputfolder = boost::filesystem::path(target_directory);

        if (!(boost::filesystem::exists(outputfolder))) {
            boost::filesystem::create_directories(outputfolder);
        }

        boost::filesystem::current_path(launch_path);

        // resolve file name if AUTO selected

        if (target_filename.compare("AUTO") == 0) {
            std::stringstream filenamebuilder;
            filenamebuilder << mat_base << "_" << gridDensityA << "_"
                            << gridDensityB << "_" << gridDensityC;

            filenamebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                            << uvector(2);
            filenamebuilder << "_" << T << "K";
            filenamebuilder << "_" << alma::engineer_format(Lmin) << "m_"
                            << alma::engineer_format(Lmax)
                            << "m.crossplanefilms";

            target_filename = filenamebuilder.str();
        }

        // save file

        std::cout << std::endl;
        std::cout << "Writing film conductivities to file " << target_filename
                  << std::endl;
        std::cout << "in directory " << target_directory << std::endl;
        std::cout << "under base directory " << launch_path << std::endl;

        std::ofstream outputwriter;
        outputwriter.open("./" + target_directory + "/" + target_filename);
        outputwriter << outputbuffer.str();
        outputwriter.close();

        // save parametric model if applicable

        if (parametric) {
            std::stringstream filenamebuilder;
            filenamebuilder << mat_base << "_" << gridDensityA << "_"
                            << gridDensityB << "_" << gridDensityC;

            filenamebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                            << uvector(2);
            filenamebuilder << "_" << T << "K";
            filenamebuilder << ".kappacrossplanemodel";

            std::string modelfilename = filenamebuilder.str();
            std::cout << std::endl;
            std::cout << "Writing parametric model to file " << modelfilename
                      << std::endl;
            std::cout << "in directory " << target_directory << std::endl;
            std::cout << "under base directory " << launch_path << std::endl;

            modelfilename = "./" + target_directory + "/" + modelfilename;
            Eigen::MatrixXd output(Nmodel, 3);
            output.col(0) = Lmodel;
            output.col(1) = kappa_actual;
            output.col(2) = kappa_model;

            alma::write_to_csv(
                modelfilename, output, ',', false, modelheader.str());
        }

        std::cout << std::endl << "[DONE.]" << std::endl;

        return 0;
    }
}
