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
/// Builds HDF5 files describing binary superlattices, based on
/// XML input.

#include <map>
#include <set>
#include <string>
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/serialization/map.hpp>
#include <boost/mpi.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/uuid/sha1.hpp>
#include <boost/endian/conversion.hpp>
#include <Eigen/Dense>
#include <basen.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <randutils.hpp>
#pragma GCC diagnostic pop
#include <cmakevars.hpp>
#include <constants.hpp>
#include <utilities.hpp>
#include <io_utils.hpp>
#include <structures.hpp>
#include <vc.hpp>
#include <vasp_io.hpp>
#include <bulk_hdf5.hpp>
#include <sampling.hpp>
#include <bulk_properties.hpp>
#include <deviational_particle.hpp>
#include <isotopic_scattering.hpp>
#include <superlattices.hpp>

// "Black hole" for output redirection.
boost::iostreams::stream_buffer<boost::iostreams::null_sink> null_buf{
    boost::iostreams::null_sink()};

/////////// GLOBAL PLACEHOLDERS ////////////
// list of compound names
std::vector<std::string> materialBase;
// number of wavevector points along A-axis
int gridDensityA;
// number of wavevector points along B-axis
int gridDensityB;
// number of wavevector points along C-axis
int gridDensityC;
// root directory
std::string materials_repository = ".";
// target directory
std::string target_directory = "AUTO";
// target filename
std::string target_filename = "AUTO";
// type of material
int materialType = -1;
// list of layer compositions. Only the atomic fraction of compound 1
// is stored.
std::vector<double> mixfractions;
// orientation of the main superlattice axis
Eigen::Vector3i normal;
// number of q points in the 1D Brillouin zone used to build the Green's
// Function
std::size_t nqline;
// deal with three phonon processed?
bool do3ph = true;
// overwrite H5 if file already exists?
bool overwrite = false;
////////////////////////////////////////////


/// Return a single string describing a composition profile.
///
/// @param[in] input - vector of doubles
/// @return the string built by representing each concentration as
/// a string with %.6f and separating them by commas.
std::string get_strprofile(const std::vector<double>& input) {
    auto n = input.size();
    std::string nruter;

    for (decltype(n) i = 0; i < n; ++i) {
        nruter += (boost::format("%|1$.6f|") % mixfractions[i]).str();

        if (i != n - 1) {
            nruter += ",";
        }
    }
    return nruter;
}

/// Return a reproducible short identifier obtained from a set of
/// doubles.
///
/// The identifier is defined as the first 8 characters of the
/// base32-encoded SHA1 hash of the string built by representing each
/// concentration as a string with %.6f and separating them by commas.
/// @param[in] input - vector of doubles
/// @return the identifier as a string
std::string get_uid(const std::vector<double>& input) {
    std::remove_reference<boost::uuids::detail::sha1::digest_type>::type digest;
    static_assert(sizeof(digest[0]) == 4u,
                  "expected each element of "
                  "boost::uuids::detail::sha1::digest_"
                  "type to take 4 bytes");
    // Convert the input to a comma-separated list.
    auto strprofile = get_strprofile(input);
    // Compute a binary version of its SHA1 hash.
    boost::uuids::detail::sha1 hasher;
    hasher.process_bytes(strprofile.c_str(), strprofile.size());
    hasher.get_digest(digest);
// Force a single standard endianess.
#ifdef BOOST_LITTLE_ENDIAN

    for (auto i = 0; i < 5; ++i) {
        digest[i] = boost::endian::endian_reverse(digest[i]);
    }
#endif

    // Convert the hash to a string of (possibly non printable) bytes.
    strprofile = "";

    for (auto i = 0; i < 5; ++i) {
        char* tmp = reinterpret_cast<char*>(digest + i);

        for (auto j = 0; j < 4; j++) {
            strprofile += tmp[j];
        }
    }
    // Encode the result in base 32.
    std::string encoded;
    bn::encode_b32(
        strprofile.begin(), strprofile.end(), std::back_inserter(encoded));
    // And return a subset of that hash.
    return encoded.substr(0, 8);
}


int main(int argc, char** argv) {
    // Initialize MPI.
    boost::mpi::environment env;
    boost::mpi::communicator world;
    std::size_t my_id = world.rank();

    // Create and seed a random number generator.
    randutils::mt19937_rng rng;

    // Only print out messages from the master process
    if (my_id != 0) {
        std::cout.rdbuf(&null_buf);
    }

    // Check the sanitiy of the command line.
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <inputfile.xml>" << std::endl;
        world.abort(1);
    }

    std::cout << "*********************************************" << std::endl;
    std::cout << "This is ALMA/superlattice_builder version "
              << ALMA_VERSION_MAJOR << "." << ALMA_VERSION_MINOR << std::endl;
    std::cout << "*********************************************" << std::endl;

    // verify that input file exists.
    if (!boost::filesystem::exists(boost::filesystem::path{argv[1]})) {
        std::cout << "ERROR: input file " << argv[1] << " does not exist."
                  << std::endl;
        world.abort(1);
    }


    // Create an empty property tree.
    boost::property_tree::ptree tree;
    // And fill it with input from an XML file.
    std::string xmlfile(argv[1]);
    boost::property_tree::read_xml(xmlfile, tree);

    // Check that the input describes a superlattice.
    if (my_id == 0) {
        try {
            tree.get_child("superlattice");
        }
        catch (boost::property_tree::ptree_bad_path) {
            std::cout << "ERROR: file " << xmlfile
                      << " does not contain a description of a superlattice"
                      << std::endl;
            world.abort(1);
        }
    }

    // Iterate over the children nodes to extract the relevant information.
    // TODO: add information about crystallographic orientations.
    bool normalfound = false;
    bool gridfound = false;

    for (const auto& v : tree.get_child("superlattice")) {
        if (v.first == "materials_repository") {
            materials_repository =
                alma::parseXMLfield<std::string>(v, "root_directory");
        }
        else if (v.first == "gridDensity") {
            gridDensityA = alma::parseXMLfield<int>(v, "A");
            gridDensityB = alma::parseXMLfield<int>(v, "B");
            gridDensityC = alma::parseXMLfield<int>(v, "C");
            gridfound = true;
        }
        else if (v.first == "compound") {
            materialBase.emplace_back(
                alma::parseXMLfield<std::string>(v, "name"));
        }
        else if (v.first == "target") {
            target_directory = alma::parseXMLfield<std::string>(v, "directory");
        }
        else if (v.first == "layer") {
            mixfractions.emplace_back(
                alma::parseXMLfield<double>(v, "mixfraction"));
        }
        else if (v.first == "skip3ph") {
            do3ph = false;
        }
        else if (v.first == "normal") {
            normal << alma::parseXMLfield<int>(v, "na"),
                alma::parseXMLfield<int>(v, "nb"),
                alma::parseXMLfield<int>(v, "nc");
            nqline = alma::parseXMLfield<std::size_t>(v, "nqline");
            normalfound = true;
        }
        else if (v.first == "overwrite") {
            overwrite = true;
        }

        else if (v.first != "<xmlcomment>") {
            std::cout << "ERROR: unknown tag '" << v.first << "' in file "
                      << xmlfile << std::endl;
            world.abort(1);
        }
    }

    // Check that all required information has been read.
    if (!gridfound) {
        std::cout << "ERROR: <gridDensity> not found" << std::endl;
        world.abort(1);
    }

    if (!normalfound) {
        std::cout << "ERROR: <normal> not found" << std::endl;
        world.abort(1);
    }

    // Perform some extra sanity checks.
    if (mixfractions.size() < 2) {
        std::cout << "ERROR: each period must contain at least two layers"
                  << std::endl;
        world.abort(1);
    }

    if (materialBase.size() != 2) {
        std::cout << "ERROR: only binary superlattices are supported"
                  << std::endl;
        world.abort(1);
    }

    for (const auto& f : mixfractions)
        if ((f < 0.) || (f > 1.)) {
            std::cout
                << "ERROR: atomic fractions must lie in the [0, 1] interval"
                << std::endl;
            world.abort(1);
        }

    // Initialise file system and verify that directories actually exist
    auto launch_path = boost::filesystem::current_path();
    auto basedir = boost::filesystem::path(materials_repository);

    bool base_error = false;

    for (const auto& m : materialBase) {
        if (!(boost::filesystem::exists(basedir / m))) {
            std::cout << "ERROR: base directory " << m << " does not exist"
                      << std::endl;
            base_error = true;
        }
    }

    if (base_error) {
        world.abort(1);
    }

    // Check consistency of input data.
    std::set<bool> polarity_checker;
    std::set<int> scA_checker;
    std::set<int> scB_checker;
    std::set<int> scC_checker;

    boost::filesystem::current_path(basedir);

    for (const auto& m : materialBase) {
        // Automatically detect whether the compound is polar or not,
        // and act accordingly.
        auto compound_dir = boost::filesystem::path(m);
        auto born_path = compound_dir / boost::filesystem::path("BORN");
        try {
            alma::load_BORN(born_path.string().c_str());
            polarity_checker.insert(true);
        }
        catch (alma::value_error) {
            polarity_checker.insert(false);
        }
        // Retrieve the supercell parameters.
        auto metadata_path =
            compound_dir / boost::filesystem::path("_metadata");

        if (!boost::filesystem::exists(metadata_path)) {
            std::cout << "ERROR: _metadata file for " << m << " is missing"
                      << std::endl;
            world.abort(1);
        }
        std::ifstream metadatareader;
        metadatareader.open(metadata_path.c_str());
        std::string linereader;
        // read first line (compound info) and skip
        std::getline(metadatareader, linereader);
        // obtain 2nd-order IFC supercell info
        std::getline(metadatareader, linereader);
        metadatareader.close();
        std::size_t index = linereader.find("=");
        std::stringstream metadataextractor(linereader.substr(index + 1));
        std::size_t tmp;
        metadataextractor >> tmp;
        scA_checker.insert(tmp);
        metadataextractor >> tmp;
        scB_checker.insert(tmp);
        metadataextractor >> tmp;
        scC_checker.insert(tmp);
    }

    if (polarity_checker.size() != 1) {
        std::cout << "ERROR: polarity mismatch in alloy components"
                  << std::endl;
        world.abort(1);
    }

    if (std::max(
            {scA_checker.size(), scB_checker.size(), scC_checker.size()}) !=
        1) {
        std::cout << "ERROR: supercell size mismatch in alloy components"
                  << std::endl;
        world.abort(1);
    }

    auto polar = *(polarity_checker.begin());
    auto scA = *(scA_checker.begin());
    auto scB = *(scB_checker.begin());
    auto scC = *(scC_checker.begin());

    // Import and store all data about each compound.
    std::vector<std::unique_ptr<alma::Crystal_structure>> poscar_ptrs;
    std::vector<std::unique_ptr<alma::Harmonic_ifcs>> IFC2_ptrs;
    std::vector<std::unique_ptr<alma::Dielectric_parameters>> born_ptrs;
    std::vector<std::unique_ptr<std::vector<alma::Thirdorder_ifcs>>> IFC3_ptrs;

    for (const auto& m : materialBase) {
        std::cout << "Loading crystal information for " << m << std::endl;
        auto compound_dir = boost::filesystem::path(m);
        auto poscar_path = compound_dir / boost::filesystem::path("POSCAR");

        if (!boost::filesystem::exists(poscar_path)) {
            std::cout << "ERROR: POSCAR file for " << m << " is missing"
                      << std::endl;
            world.abort(1);
        }
        poscar_ptrs.emplace_back(
            alma::load_POSCAR(poscar_path.string().c_str()));
        auto ifc2_path =
            compound_dir / boost::filesystem::path("FORCE_CONSTANTS");

        if (!boost::filesystem::exists(ifc2_path)) {
            std::cout << "ERROR: FORCE_CONSTANTS file for " << m
                      << " is missing" << std::endl;
            world.abort(1);
        }
        IFC2_ptrs.emplace_back(alma::load_FORCE_CONSTANTS(
            ifc2_path.string().c_str(), *(poscar_ptrs.back()), scA, scB, scC));

        if (do3ph) {
            auto ifc3_path =
                compound_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");

            if (!boost::filesystem::exists(ifc3_path)) {
                std::cout << "ERROR: FORCE_CONSTANTS_3RD file for " << m
                          << " is missing" << std::endl;
                world.abort(1);
            }
            IFC3_ptrs.emplace_back(alma::load_FORCE_CONSTANTS_3RD(
                ifc3_path.string().c_str(), *(poscar_ptrs.back())));
        }

        if (polar) {
            auto born_path = compound_dir / boost::filesystem::path("BORN");
            born_ptrs.emplace_back(alma::load_BORN(born_path.string().c_str()));
        }
    }

    // Build a Superlattice_structure object.
    auto nlayers = mixfractions.size();
    Eigen::ArrayXd profile{nlayers};
    std::copy(mixfractions.begin(), mixfractions.end(), profile.data());
    alma::Superlattice_structure superlattice(
        *(poscar_ptrs.front()), *(poscar_ptrs.back()), normal, profile);

    // Create a reproducible identifier for the profile with a small
    // probability of collision.
    auto strprofile = get_strprofile(mixfractions);
    auto uid = get_uid(mixfractions);
    std::cout << "Superlattice UID: " << uid << std::endl;

    // Compute the average concentration of each compound.
    double x2nd = superlattice.average;
    double x1st = 1. - superlattice.average;
    std::cout << "Average composition: " << std::endl;
    std::cout << (boost::format("%|1$|: %|2$2.2f|%%") % materialBase.front() %
                  (x1st * 100.))
                     .str()
              << std::endl;
    std::cout << (boost::format("%|1$|: %|2$2.2f|%%") % materialBase.back() %
                  (x2nd * 100.))
                     .str()
              << std::endl;

    // Create the output directory name automatically if nothing else
    // has been specified.
    boost::filesystem::current_path(launch_path);

    if (target_directory == "AUTO") {
        target_directory = (boost::format("superlattice_%|1$|_%|2$|") %
                            materialBase.front() % materialBase.back())
                               .str();
    }

    auto targetfolder = boost::filesystem::path(target_directory);

    // Create output directory if it does not exist yet.
    if (!(boost::filesystem::exists(targetfolder))) {
        boost::filesystem::create_directories(targetfolder);
    }

    if (target_filename == "AUTO") {
        // Create the target file name

        target_filename =
            (boost::format("superlattice_%|1$|%|3$0.4g|_%|2$|%|4$0.4g|") %
             materialBase.front() % materialBase.back() % (x1st) % (x2nd))
                .str();
        target_filename += "_" + uid + "_";
        target_filename += (boost::format("%d_%d_%d") % gridDensityA %
                            gridDensityB % gridDensityC)
                               .str() +
                           ".h5";
    }

    auto h5_target_file =
        targetfolder / boost::filesystem::path(target_filename);

    std::cout << "Target filename: " << h5_target_file.string() << std::endl;

    if (!overwrite) {
        if (boost::filesystem::exists(h5_target_file)) {
            std::cout << std::endl;
            std::cout
                << "INFO: Target HDF5 already exists, no computations needed."
                << std::endl;
            std::cout
                << "If you wish to force (re)creation of this superlattice,"
                << std::endl;
            std::cout << "change one of the following in the XML input file:"
                      << std::endl;
            std::cout << " (1) Turn on overwrite option using <overwrite/>"
                      << std::endl;
            std::cout << " (2) Specify alternate target directory using "
                         "<target directory=\"[target_dir]\"/>"
                      << std::endl;
            std::cout << std::endl;
            world.abort(1);
        }
    }

    // Build the harmonic part of the virtual crystal.
    std::cout << "Building average virtual crystal" << std::endl;
    auto vc_poscar = alma::vc_mix_structures(
        {*(poscar_ptrs.front()), *(poscar_ptrs.back())}, {x1st, x2nd});
    auto vc_ifcs = alma::vc_mix_harmonic_ifcs(
        {*(IFC2_ptrs.front()), *(IFC2_ptrs.back())}, {x1st, x2nd});
    std::unique_ptr<alma::Dielectric_parameters> vc_born;

    if (polar) {
        vc_born = alma::vc_mix_dielectric_parameters(
            {*(born_ptrs.front()), *(born_ptrs.back())}, {x1st, x2nd});
    }

    // Objects required to compute the spectrum.
    auto syms = alma::Symmetry_operations(*vc_poscar);
    auto factory = polar ? alma::make_unique<alma::Dynamical_matrix_builder>(
                               *vc_poscar, syms, *vc_ifcs, *vc_born)
                         : alma::make_unique<alma::Dynamical_matrix_builder>(
                               *vc_poscar, syms, *vc_ifcs);

    // Regular grid on which to compute w0.
    auto grid = polar ? alma::make_unique<alma::Gamma_grid>(*vc_poscar,
                                                            syms,
                                                            *vc_ifcs,
                                                            *vc_born,
                                                            gridDensityA,
                                                            gridDensityB,
                                                            gridDensityC)
                      : alma::make_unique<alma::Gamma_grid>(*vc_poscar,
                                                            syms,
                                                            *vc_ifcs,
                                                            gridDensityA,
                                                            gridDensityB,
                                                            gridDensityC);
    grid->enforce_asr();
    auto nqpoints = grid->nqpoints;
    std::size_t nequiv{grid->get_nequivalences()};
    std::cout << "Number of q points: " << nqpoints << std::endl;
    std::cout << "Number of inequivalent q points: " << nequiv << std::endl;

    // If requested, run the third-order calculation for the virtual crystal.
    std::unique_ptr<std::vector<alma::Threeph_process>> processes;

    if (do3ph) {
        std::cout << "Performing three-phonon calculations" << std::endl;
        auto vc_thirdorder = alma::vc_mix_thirdorder_ifcs(
            {*(IFC3_ptrs.front()), *(IFC3_ptrs.back())}, {x1st, x2nd});
        processes = alma::make_unique<std::vector<alma::Threeph_process>>(
            alma::find_allowed_threeph(*grid, world, 0.1));

        for (auto& p : *processes) {
            p.compute_gaussian();
            p.compute_vp2(*vc_poscar, *grid, *vc_thirdorder);
        }
    }
    else {
        std::cout
            << "Skipping the three-phonon calculations at the user's request"
            << std::endl;
        // Note that we still create a valid (but empty) vector of three-phonon
        // processes.
        processes = alma::make_unique<std::vector<alma::Threeph_process>>();
    }

    // Get the two sets of elastic scattering rates specific to superlattices.
    // 1 - Mass disorder in the effective medium.
    std::cout << "Computing scattering rates due to mass disorder" << std::endl;
    Eigen::ArrayXXd w0_disorder{superlattice.calc_w0_medium(*grid, world)};
    // 2 - Barriers.
    std::cout << "Computing scattering rates due to superlattice barriers"
              << std::endl;
    Eigen::ArrayXXd w0_barriers{
        superlattice.calc_w0_barriers(*grid, *factory, rng, world, nqline)};

    // Write out all data.
    std::cout << std::endl << "Writing output file" << std::endl;
    // First part: bulk data.
    alma::save_bulk_hdf5(h5_target_file.string().c_str(),
                         target_filename,
                         *vc_poscar,
                         syms,
                         *grid,
                         *processes,
                         world);
    // Second part: elastic scattering rates.
    alma::Scattering_subgroup disorder_group(
        (boost::format("superlattice_%|1$|_%|2$|_%|3$|_disorder") %
         materialBase.front() % materialBase.back() % uid)
            .str(),
        true,
        strprofile,
        w0_disorder);
    alma::write_scattering_subgroup(
        h5_target_file.string().c_str(), disorder_group, world);
    alma::Scattering_subgroup barriers_group(
        (boost::format("superlattice_%|1$|_%|2$|_%|3$|_barriers") %
         materialBase.front() % materialBase.back() % uid)
            .str(),
        false,
        strprofile,
        w0_barriers);
    alma::write_scattering_subgroup(
        h5_target_file.string().c_str(), barriers_group, world);
    std::cout << "\n[DONE.]" << std::endl;
}
