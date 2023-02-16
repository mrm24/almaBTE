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
/// Emulate ShengBTE using the facilities in the ALMA library.
///
/// The binary generated from this file will read a ShengBTE CONTROL
/// file, perform the same calculation as ShengBTE would, and generate
/// a ShengBTE-compatible output. A few limitations apply: in particular,
/// support for nanowire calculations using the method described in
/// Phys. Rev. B 85 (2012) 195436 is not implemented.

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <string>
#include <complex>
#include <functional>
#include <Eigen/Dense>
#include <boost/mpi.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/math/special_functions/pow.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/device/null.hpp>
#include <boost/math/distributions/normal.hpp>
#include <symmetry.hpp>
#include <utilities.hpp>
#include <structures.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <bulk_properties.hpp>
#include <dos.hpp>
#include <isotopic_scattering.hpp>
#include <processes.hpp>
#include <shengbte_iter.hpp>


extern "C" {
/// Data structure containing the information in an allocations
/// namelist from ShengBTE
struct sheng_allocations {
    /// Number of elements in the compound
    int nelements;
    /// Number of atoms in the compound
    int natoms;
    /// Number of divisions along each direction in reciprocal
    /// space
    int ngrid[3];
    /// Number of orientations for nanowires (not supported)
    int norientations;
};

/// Data structure containing the information in a parameters
/// namelist from ShengBTE
struct sheng_parameters {
    /// Temperature for single-T calculations
    double T;
    /// Broadening factor for the Gaussians
    double scalebroad;
    /// Minimum radius for nanowires (not supported)
    double rmin;
    /// Maximum radius for nanowires (not supported)
    double rmax;
    /// Radius step for nanowires (not supported)
    double dr;
    /// Maximum number of iterations of the iterative method
    int maxiter;
    /// Number of divisions for the DOS and cumulative kappa calculations.
    int nticks;
    /// Tolerance criterion for the iterative solver
    double eps;
    /// Minimum temperature for T-sweep calculations
    double T_min;
    /// Maximum temperature for T-sweep calculations
    double T_max;
    /// Temperature step for T-sweep calculations
    double T_step;
    /// Optional angular frequency cutoff for anharmonic calculations (not
    /// supported)
    double omega_max;
};

/// Data structure containing the information in a flags namelist
/// from ShengBTE.
struct sheng_flags {
    /// Include a nonanalytic correction?
    bool nonanalytic;
    /// Run the self-consistent solver or stay at the RTA level?
    bool convergence;
    /// Include isotopic scattering?
    bool isotopes;
    /// Fill in the masses and g factors automatically?
    bool autoisotopes;
    /// Compute the thermal conductivity of nanowires (not supported)
    bool nanowires;
    /// Compute only harmonic quantities
    bool onlyharmonic;
    /// Read the inputs in ESPRESSO format (not supported)
    bool espresso;
};

/// Parse the allocations namelist from a CONTROL file in the current directory.
extern sheng_allocations read_sheng_allocations(void);

/// Parse the flags namelist from a CONTROL file in the current directory.
extern sheng_parameters read_sheng_parameters(void);

/// Parse the flags namelist from a CONTROL file in the current directory.
extern sheng_flags read_sheng_flags(void);

/// Read the supercell size from a CONTROL file in the current directory.
extern void read_sheng_scell(struct sheng_allocations* allocs,
                             int* na,
                             int* nb,
                             int* nc);

/// Read the lattice vectors (including lfactor) from a CONTROL file
/// in the current directory.
extern void read_sheng_lattvec(struct sheng_allocations* allocs,
                               double lattvec[3][3]);

/// Read the dielectric tensor from a CONTROL file in the current directory.
extern void read_sheng_epsilon(struct sheng_allocations* allocs,
                               double epsilon[3][3]);

/// Read the list of atom types from a CONTROL file in the current directory.
extern void read_sheng_types(struct sheng_allocations* allocs, int* types);

/// Read the list of element masses from a CONTROL file in the current
/// directory.
extern void read_sheng_masses(struct sheng_allocations* allocs, double* types);

/// Read the list of mass disorder "g" factors from a CONTROL file in the
/// current directory.
extern void read_sheng_gfactors(struct sheng_allocations* allocs,
                                double* types);

/// Read the coordinates of an atom from a CONTROL file in the current
/// directory.
extern void read_sheng_position(struct sheng_allocations* allocs,
                                int index,
                                double position[3]);

/// Read the Born effective charge tensor of an atom from a CONTROL file
/// in the current directory.
extern void read_sheng_born(struct sheng_allocations* allocs,
                            int index,
                            double born[3][3]);

/// Read an element name from a CONTROL file in the current directory.
extern void read_sheng_element(struct sheng_allocations* allocs,
                               int index,
                               char element[4]);
}

// Small pieces of code used to make it more convenient to create files
// and write out information from process 0 only.
// Stream_buffer leading to a black hole.
boost::iostreams::stream_buffer<boost::iostreams::null_sink> null_buf{
    boost::iostreams::null_sink()};

// Open a file for writing from process 0, and create it if needed.
//
// @return a ostream for the open file on process 0, or the black hole
// stream anywhere else.
std::unique_ptr<std::ostream> open_from_master(const std::string& filename) {
    boost::mpi::communicator world;
    std::unique_ptr<std::ostream> nruter;
    auto my_id = world.rank();
    if (my_id == 0) {
        nruter = alma::make_unique<std::ofstream>(filename);
    }
    else {
        nruter = alma::make_unique<std::ostream>(&null_buf);
    }
    return std::move(nruter);
}

/// Create a directory name corresponding to a given temperature.
///
/// @param[in] T - temperature in K
/// @
std::string gen_subdir_name(double T) {
    auto intT = static_cast<int>(std::round(T));
    auto nruter = (boost::format("T%|d|K") % intT).str();
    return nruter;
}

/// Compute the 25th and 75th percentiles of log(sigma)
///
/// @param[in] sigma an Eigen array with all broadening parameters
/// @return an array containing the 25th and 75th percentiles
std::array<double, 2> calc_percentiles_log(
    const Eigen::Ref<const Eigen::ArrayXXd>& sigma) {
    auto rows = sigma.rows();
    auto cols = sigma.cols();

    std::vector<double> logsigma;
    logsigma.reserve(rows * cols);

    for (decltype(cols) c = 0; c < cols; ++c)
        for (decltype(rows) r = 0; r < rows; ++r)
            logsigma.emplace_back(sigma(r, c) == 0.
                                      ? std::numeric_limits<double>::lowest()
                                      : std::log(sigma(r, c)));
    std::sort(logsigma.begin(), logsigma.end());

    std::array<double, 2> nruter;
    nruter[0] = logsigma[25 * rows * cols / 100];
    nruter[1] = logsigma[75 * rows * cols / 100];
    return nruter;
}


/// Compute the Grüneisen parameter for each mode on a grid.
///
/// @param[in] cell - description of the unit cell
/// @param[in] grid - regular grid with phonon spectrum
/// @param[in] fc3 - set of third-order force constants
/// @return an array with the mode Grüneisen parameters for each q point
/// and each branch.
Eigen::ArrayXXd calc_mode_gruneisen(
    const alma::Crystal_structure& cell,
    const alma::Gamma_grid& grid,
    const std::vector<alma::Thirdorder_ifcs>& fc3) {
    // Factor to transform from nm*eV/(amu * A ** 3 * THz ** 2) to unitless
    double prefactor = alma::constants::e * 1e-3 / alma::constants::amu;
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    Eigen::ArrayXXd nruter(Eigen::ArrayXXd::Zero(nmodes, grid.nqpoints));

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        auto spectrum = grid.get_spectrum_at_q(iq);
        for (std::size_t im = 0; im < nmodes; ++im) {
            if (spectrum.omega(im) == 0.) {
                continue;
            }
            for (const auto& tri : fc3) {
                double massfactor =
                    std::sqrt(cell.get_mass(tri.i) * cell.get_mass(tri.j));
                double arg = grid.get_q(iq).dot(tri.rj);
                std::complex<double> factor1 =
                    (std::cos(arg) + alma::constants::imud * std::sin(arg)) /
                    massfactor;
                Eigen::Vector3d cart =
                    cell.lattvec * cell.positions.col(tri.k) + tri.rk;
                for (auto rr = 0; rr < 3; ++rr) {
                    std::complex<double> factor2 =
                        factor1 * std::conj(spectrum.wfs(3 * tri.i + rr, im));
                    for (auto ss = 0; ss < 3; ++ss) {
                        std::complex<double> factor3 =
                            factor2 * spectrum.wfs(3 * tri.j + ss, im);
                        for (auto tt = 0; tt < 3; ++tt) {
                            nruter(im, iq) += std::real(
                                factor3 * tri.ifc(rr, ss, tt) * cart(tt));
                        }
                    }
                }
            }
            nruter(im, iq) *=
                -prefactor / 6. / boost::math::pow<2>(spectrum.omega(im));
        }
    }
    return nruter;
}


/// Compute the total Grüneisen parameter as a weighted average of mode
/// Grüneisen parameters.
///
/// @param[in] grid - regular grid with phonon spectrum
/// @param[in] modegrun - Grüneisen parameters for each q point and each branch
/// @param[in] T - temperature in K
/// @return the total Grüneisen parameter at the given temperature
double calc_total_gruneisen(const alma::Gamma_grid& grid,
                            const Eigen::Ref<const Eigen::ArrayXXd>& modegrun,
                            double T) {
    double nruter = 0.;
    double denominator = 0.;
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();
    // The weight of each mode is proportional to its contribution to the heat
    // capacity.
    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        auto spectrum = grid.get_spectrum_at_q(iq);
        for (std::size_t im = 0; im < nmodes; ++im) {
            double weight = alma::bose_einstein_kernel(spectrum.omega(im), T);
            nruter += weight * modegrun(im, iq);
            denominator += weight;
        }
    }
    nruter /= denominator;
    return nruter;
}


int main(int argc, char** argv) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;
    auto my_id = world.rank();
    // Just like ShengBTE itself, this program takes no arguments and
    // reads all of its parameters from CONTROL.

    // Make sure that only one process can produce output.
    if (my_id != 0) {
        std::cout.rdbuf(&null_buf);
        std::cerr.rdbuf(&null_buf);
    }
    else {
        // And that the output is unbuffered.
        std::cout.setf(std::ios::unitbuf);
        std::cerr.setf(std::ios::unitbuf);
    }

    // Read in the basic information about array sizes.
    auto allocations = read_sheng_allocations();
    // Perform some preliminary sanity checks.
    if (allocations.norientations > 0) {
        std::cerr << "Error: nanowire calculations are not implemented"
                  << std::endl;
        std::cerr << "norientations must not be > 0" << std::endl;
        env.abort(1);
    }
    if (allocations.nelements < 1 ||
        allocations.natoms < allocations.nelements) {
        std::cerr << "Error: nelements, natoms must be > 0,"
                  << " natoms must be >= nelements" << std::endl;
        env.abort(1);
    }
    if (std::min({allocations.ngrid[0],
                  allocations.ngrid[1],
                  allocations.ngrid[2]}) < 1) {
        std::cerr << "Error: all components of ngrid must be must be > 0"
                  << std::endl;
        env.abort(1);
    }
    // Read in all the information about atoms and elements.
    int na;
    int nb;
    int nc;
    read_sheng_scell(&allocations, &na, &nb, &nc);
    if (std::min({na, nb, nc}) < 1) {
        std::cerr << "Error: all supercell sizes must be > 0" << std::endl;
        env.abort(1);
    }
    std::vector<int> types(allocations.natoms);
    read_sheng_types(&allocations, types.data());
    for (std::size_t i = 0; i < static_cast<std::size_t>(allocations.natoms);
         ++i) {
        if (types[i] < 0) {
            std::cerr << "Error: atom types must be initialized correctly"
                      << std::endl;
            env.abort(1);
        }
    }
    Eigen::Matrix3d lattvec;
    double tmp33[3][3];
    read_sheng_lattvec(&allocations, tmp33);
    for (std::size_t i = 0u; i < 3u; ++i) {
        for (std::size_t j = 0u; j < 3u; ++j) {
            lattvec(i, j) = tmp33[i][j];
        }
    }
    Eigen::Matrix3d epsilon;
    read_sheng_epsilon(&allocations, tmp33);
    for (std::size_t i = 0u; i < 3u; ++i) {
        for (std::size_t j = 0u; j < 3u; ++j) {
            epsilon(i, j) = tmp33[i][j];
        }
    }
    Eigen::MatrixXd positions(3, allocations.natoms);
    for (std::size_t i = 0; i < static_cast<std::size_t>(allocations.natoms);
         ++i) {
        read_sheng_position(&allocations, i, positions.col(i).data());
    }
    std::vector<Eigen::MatrixXd> born;
    for (std::size_t i = 0; i < static_cast<std::size_t>(allocations.natoms);
         ++i) {
        read_sheng_born(&allocations, i, tmp33);
        Eigen::MatrixXd thisborn(3, 3);
        for (std::size_t r = 0u; r < 3u; ++r) {
            for (std::size_t c = 0u; c < 3u; ++c) {
                thisborn(r, c) = tmp33[r][c];
            }
        }
        born.emplace_back(thisborn);
    }
    std::vector<std::string> elements;
    char tmpchars[4];
    for (std::size_t i = 0; i < static_cast<std::size_t>(allocations.nelements);
         ++i) {
        read_sheng_element(&allocations, i, tmpchars);
        elements.push_back(tmpchars);
    }
    for (auto& e : elements) {
        boost::trim(e);
    }
    std::vector<double> masses(allocations.nelements);
    read_sheng_masses(&allocations, masses.data());
    std::vector<double> gfactors(allocations.nelements);
    read_sheng_gfactors(&allocations, gfactors.data());
    // Check that the types vector is compatible with alma and
    // convert it to a numbers vector.
    bool types_valid = true;
    if (types[0] != 0) {
        types_valid = false;
    }
    for (std::size_t i = 1; i < static_cast<std::size_t>(allocations.natoms);
         ++i) {
        if (types[i] < types[i - 1] || types[i] > types[i - 1] + 1) {
            types_valid = false;
        }
    }
    if (!types_valid) {
        std::cerr << "Error: only contiguous and increasing 'types' '"
                  << "arrays are supported" << std::endl;
        env.abort(1);
    }
    std::vector<int> numbers(allocations.nelements);
    std::size_t j = 0;
    numbers[0] = 1;
    for (std::size_t i = 1; i < static_cast<std::size_t>(allocations.natoms);
         ++i) {
        if (types[i] != types[i - 1]) {
            ++j;
        }
        ++numbers[j];
    }
    // Read in the flag namelist and check for unsupported features.
    auto flags = read_sheng_flags();
    if (flags.nanowires) {
        std::cerr << "Error: nanowire calculations are not implemented"
                  << std::endl;
        env.abort(1);
    }
    if (flags.espresso) {
        std::cerr << "Error: Quantum ESPRESSO support is not implemented"
                  << std::endl;
        env.abort(1);
    }
    // Read in the parameters namelist and perform some last checks.
    auto parameters = read_sheng_parameters();
    if (parameters.omega_max >= 0) {
        std::cerr << "Error: the omega_max parameter is not supported"
                  << std::endl;
        env.abort(1);
    }
    if (parameters.T < 0 && parameters.T_min <= 0.) {
        std::cerr << "Error: T must be > 0 K" << std::endl;
        env.abort(1);
    }
    else if (parameters.T > 0.) {
        parameters.T_min = parameters.T;
        parameters.T_max = parameters.T;
        parameters.T_step = parameters.T;
    }

    // Create a Crystal_structure based on the information in CONTROL
    std::unique_ptr<alma::Crystal_structure> poscar;
    if (flags.autoisotopes) {
        poscar = alma::make_unique<alma::Crystal_structure>(
            lattvec, positions, elements, numbers);
    }
    else {
        poscar = alma::make_unique<alma::Crystal_structure>(
            lattvec, positions, elements, numbers, masses);
    }
    // And compute the symmetry information.
    auto syms = alma::Symmetry_operations(*poscar);

    std::cout << "Info: symmetry group " << syms.get_spacegroup_symbol()
              << " detected" << std::endl;
    std::cout << "Info: " << syms.get_nsym() << " symmetry operations"
              << std::endl;

    // Try to mimic all of ShengBTE's output files, although not down
    // to Fortran's spacing rules.
    if (my_id == 0) {
        std::ofstream f("BTE.ReciprocalLatticeVectors");
        for (std::size_t i = 0; i < 3; ++i) {
            f << poscar->rlattvec.col(i).transpose() << " # nm-1,b" << i + 1
              << std::endl;
        }
    }
    world.barrier();

    // Try to load the FORCE_CONSTANTS_2ND file
    auto ifcs =
        alma::load_FORCE_CONSTANTS("FORCE_CONSTANTS_2ND", *poscar, na, nb, nc);

    // And obtain the spectrum
    std::cout << "Info: about to obtain the spectrum" << std::endl;
    std::cout << "Info: expecting Phonopy 2nd-order format" << std::endl;
    std::unique_ptr<alma::Gamma_grid> grid;
    if (flags.nonanalytic) {
        auto dielectric =
            alma::make_unique<alma::Dielectric_parameters>(born, epsilon);
        grid = alma::make_unique<alma::Gamma_grid>(*poscar,
                                                   syms,
                                                   *ifcs,
                                                   *dielectric,
                                                   allocations.ngrid[0],
                                                   allocations.ngrid[1],
                                                   allocations.ngrid[2]);
    }

    else {
        grid = alma::make_unique<alma::Gamma_grid>(*poscar,
                                                   syms,
                                                   *ifcs,
                                                   allocations.ngrid[0],
                                                   allocations.ngrid[1],
                                                   allocations.ngrid[2]);
    }
    // Note that these lines are emitted after the spectrum calculation,
    // contrary to what happens in ShengBTE.
    std::cout << "Info: Ntot = " << grid->nqpoints << std::endl;
    std::cout << "Info: Nlist = " << grid->get_nequivalences() << std::endl;

    // The same thing goes for these two files, which ShengBTE creates
    // before the spectrum calculation.
    // Furthermore, the order of the q points is different.
    if (my_id == 0) {
        auto nequivs = grid->get_nequivalences();
        auto solver = poscar->rlattvec.colPivHouseholderQr();
        std::ofstream f("BTE.qpoints");
        for (std::size_t i = 0; i < nequivs; ++i) {
            f << std::setw(9) << i + 1 << " " << std::setw(9)
              << grid->get_representative(i) + 1 << " " << std::setw(9)
              << grid->get_cardinal(i);
            Eigen::VectorXd q = grid->get_q(grid->get_representative(i));
            q = solver.solve(q);
            f << std::scientific << std::setprecision(10);
            for (std::size_t ic = 0; ic < 3; ++ic) {
                if (std::abs(q(ic) < 1e-10)) {
                    q(ic) = 0.;
                }
                f << " " << std::setw(20) << q(ic);
            }
            f << std::endl;
        }
    }
    if (my_id == 0) {
        auto nequivs = grid->get_nequivalences();
        auto solver = poscar->rlattvec.colPivHouseholderQr();
        std::ofstream f("BTE.qpoints_full");
        for (std::size_t i = 0; i < grid->nqpoints; ++i) {
            Eigen::VectorXd q = grid->get_q(i);
            q = solver.solve(q);
            std::size_t found = 0;
            for (std::size_t ieq = 0; ieq < nequivs; ++ieq) {
                auto eq = grid->get_equivalence(ieq);
                for (const auto& m : eq) {
                    if (m == i) {
                        found = ieq + 1;
                        break;
                    }
                }
                if (found > 0) {
                    break;
                }
            }
            f << std::setw(9) << i + 1 << " " << std::setw(9) << found << " ";
            f << std::scientific << std::setprecision(10);
            for (std::size_t ic = 0; ic < 3; ++ic) {
                if (std::abs(q(ic) < 1e-10)) {
                    q(ic) = 0.;
                }
                f << " " << std::setw(20) << q(ic);
            }
            f << std::endl;
        }
    }
    world.barrier();

    // The artifactual nonzero acoustic frequencies will not match
    // the ones ShengBTE outputs for the same set of inputs due
    // to differences in the algorithms.
    std::cout << "Info: about to set the acoustic frequencies at Gamma to zero"
              << std::endl;
    std::cout << "Info: original values:" << std::endl;
    for (std::size_t i = 0; i < 3; ++i) {
        // Use Fortran-like 1-based indices for the output.
        std::cout << "Info: omega(1, " << i + 1
                  << ") = " << grid->get_spectrum_at_q(0).omega(i) << " rad/ps"
                  << std::endl;
    }
    grid->enforce_asr();
    std::cout << "Info: spectrum calculation finished" << std::endl;
    auto nmodes =
        static_cast<std::size_t>(grid->get_spectrum_at_q(0).omega.size());

    // Write out all the information about the spectrum we just found.
    if (my_id == 0) {
        auto nequivs = grid->get_nequivalences();
        std::ofstream f("BTE.omega");
        f << std::scientific << std::setprecision(10);
        for (std::size_t ieq = 0; ieq < nequivs; ++ieq) {
            auto spectrum =
                grid->get_spectrum_at_q(grid->get_representative(ieq));
            for (std::size_t im = 0;
                 im < static_cast<std::size_t>(spectrum.omega.size());
                 ++im) {
                f << std::setw(20) << spectrum.omega(im) << " ";
            }
            f << std::endl;
        }
    }
    if (my_id == 0) {
        auto nequivs = grid->get_nequivalences();
        std::ofstream f("BTE.v");
        std::ofstream ffull("BTE.v_full");
        f << std::scientific << std::setprecision(10);
        ffull << std::scientific << std::setprecision(10);
        auto spectrum = grid->get_spectrum_at_q(0);
        for (std::size_t im = 0; im < nmodes; ++im) {
            for (std::size_t ieq = 0; ieq < nequivs; ++ieq) {
                spectrum =
                    grid->get_spectrum_at_q(grid->get_representative(ieq));
                for (std::size_t ic = 0; ic < 3; ++ic) {
                    f << std::setw(20) << spectrum.vg(ic, im) << " ";
                }
                f << std::endl;
            }
            for (std::size_t iq = 0; iq < grid->nqpoints; ++iq) {
                spectrum = grid->get_spectrum_at_q(iq);
                for (std::size_t ic = 0; ic < 3; ++ic) {
                    ffull << std::setw(20) << spectrum.vg(ic, im) << " ";
                }
                ffull << std::endl;
            }
        }
    }
    world.barrier();

    // Compute locally adaptive estimates of the total and projected DOS
    // in a ShengBTE-like fashion.
    if (my_id == 0) {
        double omega_min = std::numeric_limits<double>::infinity();
        double omega_max = -std::numeric_limits<double>::infinity();
        for (std::size_t iq = 0; iq < grid->nqpoints; ++iq) {
            auto spectrum = grid->get_spectrum_at_q(iq);
            omega_max = std::max(omega_max, spectrum.omega.maxCoeff());
            omega_min = std::min(omega_min, spectrum.omega.minCoeff());
        }
        omega_max *= 1.1;
        Eigen::ArrayXd ticks(parameters.nticks);
        for (std::size_t i = 0; i < static_cast<std::size_t>(parameters.nticks);
             ++i) {
            ticks[i] = omega_min + (omega_max - omega_min) *
                                       static_cast<double>(i + 1) /
                                       parameters.nticks;
        }
        Eigen::ArrayXXd sigma(nmodes, grid->nqpoints);
        for (std::size_t iq = 0; iq < grid->nqpoints; ++iq) {
            auto spectrum = grid->get_spectrum_at_q(iq);
            for (std::size_t im = 0; im < nmodes; ++im) {
                sigma(im, iq) = grid->base_sigma(spectrum.vg.col(im));
            }
        }
        auto percent = calc_percentiles_log(sigma);
        double lbound = std::exp(percent[0] - 1.5 * (percent[1] - percent[0]));
        sigma = (sigma < lbound).select(lbound, sigma);
        Eigen::ArrayXd dos(Eigen::ArrayXd::Zero(parameters.nticks));
        Eigen::ArrayXXd pdos(
            Eigen::ArrayXXd::Zero(parameters.nticks, allocations.natoms));
        for (std::size_t iq = 0; iq < grid->nqpoints; ++iq) {
            auto spectrum = grid->get_spectrum_at_q(iq);
            for (std::size_t im = 0; im < nmodes; ++im) {
                auto dist = alma::make_unique<boost::math::normal>(
                    spectrum.omega(im), sigma(im, iq));
                for (std::size_t i = 0;
                     i < static_cast<std::size_t>(parameters.nticks);
                     ++i) {
                    double factor = boost::math::pdf(*dist, ticks[i]);
                    dos(i) += factor;
                    for (std::size_t ia = 0;
                         ia < static_cast<std::size_t>(poscar->get_natoms());
                         ++ia) {
                        pdos(i, ia) +=
                            factor *
                            std::abs(spectrum.wfs.block<3, 1>(3 * ia, im)
                                         .squaredNorm());
                    }
                }
            }
        }
        dos /= grid->nqpoints;
        pdos /= grid->nqpoints;
        // Write out the densities of states.
        std::ofstream fdos("BTE.dos");
        fdos << std::scientific << std::setprecision(5);
        std::ofstream fpdos("BTE.pdos");
        fpdos << std::scientific << std::setprecision(5);
        for (std::size_t i = 0; i < static_cast<std::size_t>(parameters.nticks);
             ++i) {
            fdos << std::setw(14) << ticks(i) << " " << std::setw(14) << dos(i)
                 << std::endl;
            fpdos << std::setw(14) << ticks(i) << " ";
            for (std::size_t ia = 0;
                 ia < static_cast<std::size_t>(poscar->get_natoms());
                 ++ia) {
                fpdos << std::setw(14) << pdos(i, ia) << " ";
            }
            fpdos << std::endl;
        }
    }
    world.barrier();

    // Look for allowed two-phonon processes.
    auto twoph_processes = alma::find_allowed_twoph(*grid, world);
    // Compute the associated elastic scattering rates, using either
    // the provided or the default values. If isotopes == false,
    // set the scattering rates to zero.
    Eigen::ArrayXXd w0_elastic(
        Eigen::ArrayXXd::Zero(3 * poscar->get_natoms(), grid->nqpoints));
    if (flags.isotopes) {
        if (flags.autoisotopes) {
            w0_elastic =
                alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world);
        }
        else {
            Eigen::VectorXd egfactors(gfactors.size());
            for (std::size_t i = 0; i < gfactors.size(); ++i) {
                egfactors(i) = gfactors[i];
            }
            w0_elastic = alma::calc_w0_twoph(
                *poscar, egfactors, *grid, twoph_processes, world);
        }
    }
    // Write out the two-phonon scattering rates, even if they are zero.
    if (my_id == 0) {
        std::ofstream f("BTE.w_isotopic");
        f << std::scientific << std::setprecision(10);
        auto nequivs = grid->get_nequivalences();
        for (std::size_t im = 0; im < nmodes; ++im) {
            for (std::size_t ieq = 0; ieq < nequivs; ++ieq) {
                auto spectrum =
                    grid->get_spectrum_at_q(grid->get_representative(ieq));
                f << std::setw(20) << spectrum.omega(im) << " " << std::setw(20)
                  << w0_elastic(im, grid->get_representative(ieq)) << std::endl;
            }
        }
    }
    world.barrier();

    // Compute some quantities which do not require any anharmonic information.
    std::cout << "Info: start calculating"
              << " specific heat and kappa in the small-grain limit"
              << std::endl;
    std::vector<double> Ts;
    for (std::size_t Tcounter = 0;
         Tcounter <=
         static_cast<std::size_t>(std::ceil(
             (parameters.T_max - parameters.T_min) / parameters.T_step));
         ++Tcounter) {
        double T = parameters.T_min + Tcounter * parameters.T_step;
        if (T > parameters.T_max && T < parameters.T_max + 1.) {
            break;
        }
        else if (T > parameters.T_max + 1.0) {
            T = parameters.T_max;
        }
        Ts.emplace_back(T);
    }
    if (my_id == 0) {
        std::ofstream f101("BTE.cvVsT");
        std::ofstream f102("BTE.KappaTensorVsT_sg");
        for (const auto T : Ts) {
            std::cout << "Info: Temperature= " << T << std::endl;
            // Since this is the first loop over temperatures,
            // try to create the subdirectory corresponding to
            // this temperature.
            std::string sdirname = gen_subdir_name(T);
            boost::filesystem::create_directory(sdirname);
            // And move to it.
            auto saved = boost::filesystem::current_path();
            boost::filesystem::current_path(sdirname);
            // Note the change to m ** 3 for the volume.
            auto cv = alma::calc_cv(*poscar, *grid, T) * 1e27;
            std::ofstream f("BTE.cv");
            f << cv << std::endl;
            f101 << std::fixed << std::setprecision(1);
            f101 << T << " ";
            f101 << std::scientific << std::setprecision(5);
            f101 << std::setw(14) << cv << std::endl;
            auto kappa_sg = alma::calc_kappa_sg(*poscar, *grid, T);
            kappa_sg = syms.symmetrize_m<double>(kappa_sg, true);
            std::ofstream f2("BTE.kappa_sg");
            f2 << std::scientific;
            f2 << std::setprecision(5);
            for (std::size_t ic = 0; ic < 9u; ++ic) {
                auto row = ic % 3u;
                auto col = ic / 3u;
                f2 << std::setw(14) << kappa_sg(row, col);
                if (ic < 8u) {
                    f2 << " ";
                }
            }
            f2 << std::endl;
            f102 << std::fixed << std::setprecision(1);
            f102 << T << " ";
            f102 << std::scientific << std::setprecision(5);
            for (std::size_t ic = 0; ic < 9u; ++ic) {
                auto row = ic % 3u;
                auto col = ic / 3u;
                f102 << std::setw(14) << kappa_sg(row, col);
                if (ic < 8u) {
                    f102 << " ";
                }
            }
            f102 << std::endl;
            // Move back to the parent directory after writing the output.
            boost::filesystem::current_path(saved);
        }
    }
    world.barrier();

    // Compute and write out the normalized boundary scattering rates
    // (as the norm of the velocities).
    if (my_id == 0) {
        auto nequivs = grid->get_nequivalences();
        std::ofstream f("BTE.w_boundary");
        f << std::scientific << std::setprecision(10);
        for (std::size_t im = 0; im < nmodes; ++im) {
            for (std::size_t ieq = 0; ieq < nequivs; ++ieq) {
                auto spectrum =
                    grid->get_spectrum_at_q(grid->get_representative(ieq));
                f << std::setw(20) << spectrum.omega(im) << " " << std::setw(20)
                  << spectrum.vg.col(im).matrix().norm() << std::endl;
            }
        }
    }
    world.barrier();

    // Look for allowed three-phonon processes.
    auto threeph_processes =
        alma::find_allowed_threeph(*grid, world, parameters.scalebroad);

    // Inform the user about how many processes have been found.
    std::size_t my_plus = 0;
    std::size_t my_minus = 0;
    for (const auto& p : threeph_processes) {
        if (p.type == alma::threeph_type::absorption) {
            ++my_plus;
        }
        else {
            ++my_minus;
        }
    }
    std::size_t ntotal_plus = 0;
    std::size_t ntotal_minus = 0;
    boost::mpi::all_reduce(
        world, my_plus, ntotal_plus, std::plus<std::size_t>());
    boost::mpi::all_reduce(
        world, my_minus, ntotal_minus, std::plus<std::size_t>());
    std::cout << "Info: Ntotal_plus = " << ntotal_plus << std::endl;
    std::cout << "Info: Ntotal_minus = " << ntotal_minus << std::endl;

    // Compute the Gaussian factors in the scattering amplitudes of each
    // process and use them to get the phase space volume for allowed
    // three-phonon processes, plus the weighted version of this quantity
    // for each temperature.
    std::cout << "Info: start calculating weighted phase space" << std::endl;
    auto P3_denominator = static_cast<double>(
        boost::math::pow<2>(grid->nqpoints) * boost::math::pow<3>(nmodes));
    double my_P3_plus = 0.;
    double my_P3_minus = 0.;
    Eigen::ArrayXXd my_P3_detailed_plus(
        Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
    Eigen::ArrayXXd my_P3_detailed_minus(
        Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
    std::vector<Eigen::ArrayXXd> my_WP3_detailed_plus;
    std::vector<Eigen::ArrayXXd> my_WP3_detailed_minus;
    for (std::size_t iT = 0; iT < Ts.size(); ++iT) {
        Eigen::ArrayXXd my_WP3_T_plus(
            Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
        my_WP3_detailed_plus.push_back(my_WP3_T_plus);
        Eigen::ArrayXXd my_WP3_T_minus(
            Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
        my_WP3_detailed_minus.push_back(my_WP3_T_minus);
    }
    for (auto& p : threeph_processes) {
        auto gaussian = p.compute_gaussian() / P3_denominator;
        if (p.type == alma::threeph_type::absorption) {
            my_P3_plus += gaussian * grid->get_cardinal(p.c);
            my_P3_detailed_plus(p.alpha[0], p.c) += gaussian;
            for (std::size_t iT = 0; iT < Ts.size(); ++iT) {
                gaussian =
                    p.compute_weighted_gaussian(*grid, Ts[iT]) / grid->nqpoints;
                my_WP3_detailed_plus[iT](p.alpha[0], p.c) += gaussian;
            }
        }
        else {
            my_P3_minus += gaussian * grid->get_cardinal(p.c);
            my_P3_detailed_minus(p.alpha[0], p.c) += gaussian;
            for (std::size_t iT = 0; iT < Ts.size(); ++iT) {
                gaussian =
                    p.compute_weighted_gaussian(*grid, Ts[iT]) / grid->nqpoints;
                my_WP3_detailed_minus[iT](p.alpha[0], p.c) += gaussian;
            }
        }
    }
    double P3_plus = 0.;
    double P3_minus = 0.;
    boost::mpi::all_reduce(world, my_P3_plus, P3_plus, std::plus<double>());
    boost::mpi::all_reduce(world, my_P3_minus, P3_minus, std::plus<double>());
    Eigen::ArrayXXd P3_detailed_plus(
        Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
    Eigen::ArrayXXd P3_detailed_minus(
        Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
    boost::mpi::all_reduce(world,
                           my_P3_detailed_plus.data(),
                           my_P3_detailed_plus.size(),
                           P3_detailed_plus.data(),
                           std::plus<double>());
    boost::mpi::all_reduce(world,
                           my_P3_detailed_minus.data(),
                           my_P3_detailed_minus.size(),
                           P3_detailed_minus.data(),
                           std::plus<double>());
    std::vector<Eigen::ArrayXXd> WP3_detailed_plus;
    std::vector<Eigen::ArrayXXd> WP3_detailed_minus;
    for (std::size_t iT = 0; iT < Ts.size(); ++iT) {
        Eigen::ArrayXXd WP3_T_plus(
            Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
        boost::mpi::all_reduce(world,
                               my_WP3_detailed_plus[iT].data(),
                               my_WP3_detailed_plus[iT].size(),
                               WP3_T_plus.data(),
                               std::plus<double>());
        WP3_detailed_plus.push_back(WP3_T_plus);
        Eigen::ArrayXXd WP3_T_minus(
            Eigen::ArrayXXd::Zero(nmodes, grid->get_nequivalences()));
        boost::mpi::all_reduce(world,
                               my_WP3_detailed_minus[iT].data(),
                               my_WP3_detailed_minus[iT].size(),
                               WP3_T_minus.data(),
                               std::plus<double>());
        WP3_detailed_minus.push_back(WP3_T_minus);
    }
    // Create the same extremely detailed set of files with phase
    // space information as ShengBTE.
    if (my_id == 0) {
        std::ofstream f("BTE.P3_plus");
        f << std::scientific << std::setprecision(10);
        for (std::size_t iq = 0; iq < grid->get_nequivalences(); ++iq) {
            for (std::size_t im = 0; im < nmodes; ++im) {
                f << std::setw(20) << P3_detailed_plus(im, iq) << " ";
            }
            f << std::endl;
        }
    }
    if (my_id == 0) {
        std::ofstream f("BTE.P3_minus");
        f << std::scientific << std::setprecision(10);
        for (std::size_t iq = 0; iq < grid->get_nequivalences(); ++iq) {
            for (std::size_t im = 0; im < nmodes; ++im) {
                f << std::setw(20) << P3_detailed_minus(im, iq) << " ";
            }
            f << std::endl;
        }
    }
    if (my_id == 0) {
        std::ofstream f("BTE.P3");
        f << std::scientific << std::setprecision(10);
        for (std::size_t iq = 0; iq < grid->get_nequivalences(); ++iq) {
            for (std::size_t im = 0; im < nmodes; ++im) {
                f << std::setw(20)
                  << (2. * P3_detailed_plus(im, iq) +
                      P3_detailed_minus(im, iq)) /
                         3.
                  << " ";
            }
            f << std::endl;
        }
    }
    if (my_id == 0) {
        std::ofstream f("BTE.P3_plus_total");
        f << std::scientific << std::setprecision(10);
        f << std::setw(20) << P3_plus << std::endl;
    }
    if (my_id == 0) {
        std::ofstream f("BTE.P3_minus_total");
        f << std::scientific << std::setprecision(10);
        f << std::setw(20) << P3_minus << std::endl;
    }
    if (my_id == 0) {
        std::ofstream f("BTE.P3_total");
        f << std::scientific << std::setprecision(10);
        f << std::setw(20) << (2. * P3_plus + P3_minus) / 3. << std::endl;
    }
    if (my_id == 0) {
        for (std::size_t iT = 0; iT < Ts.size(); ++iT) {
            auto T = Ts[iT];
            std::cout << "Info: Temperature= " << T << std::endl;
            std::string sdirname = gen_subdir_name(T);
            auto saved = boost::filesystem::current_path();
            boost::filesystem::current_path(sdirname);
            std::ofstream fplus("BTE.WP3_plus");
            std::ofstream fminus("BTE.WP3_minus");
            std::ofstream ftotal("BTE.WP3");
            fplus << std::scientific << std::setprecision(5);
            fminus << std::scientific << std::setprecision(5);
            ftotal << std::scientific << std::setprecision(5);
            for (std::size_t im = 0; im < nmodes; ++im) {
                for (std::size_t iq = 0; iq < grid->get_nequivalences(); ++iq) {
                    auto spectrum =
                        grid->get_spectrum_at_q(grid->get_representative(iq));
                    fplus << std::setw(14) << spectrum.omega(im) << " "
                          << std::setw(14) << WP3_detailed_plus[iT](im, iq)
                          << std::endl;
                    fminus << std::setw(14) << spectrum.omega(im) << " "
                           << std::setw(14) << WP3_detailed_minus[iT](im, iq)
                           << std::endl;
                    ftotal << std::setw(14) << spectrum.omega(im) << " "
                           << std::setw(14)
                           << WP3_detailed_plus[iT](im, iq) +
                                  WP3_detailed_minus[iT](im, iq)
                           << std::endl;
                }
            }
            boost::filesystem::current_path(saved);
        }
    }
    world.barrier();

    if (flags.onlyharmonic) {
        std::cout << "Info: onlyharmonic=.true., stopping here" << std::endl;
        std::exit(0);
    }

    // If onlyharmonic == false, continue and load the third-order force
    // constants.
    auto thirdorder =
        alma::load_FORCE_CONSTANTS_3RD("FORCE_CONSTANTS_3RD", *poscar);

    // Obtain the mode Grüneisen parameters.
    Eigen::ArrayXXd modegrun = calc_mode_gruneisen(*poscar, *grid, *thirdorder);
    // Write them out.
    if (my_id == 0) {
        std::ofstream f("BTE.gruneisen");
        f << std::scientific << std::setprecision(10);
        for (std::size_t iq = 0; iq < grid->get_nequivalences(); ++iq) {
            for (std::size_t im = 0; im < nmodes; ++im) {
                f << std::setw(20) << modegrun(im, grid->get_representative(iq))
                  << " ";
            }
            f << std::endl;
        }
    }
    // Obtain and write out the total Grüneisen parameter for each temperature.
    if (my_id == 0) {
        std::ofstream f("BTE.gruneisenVsT_total");
        for (const auto& T : Ts) {
            auto grun = calc_total_gruneisen(*grid, modegrun, T);
            f << std::fixed << std::setprecision(1) << std::setw(7) << T << " "
              << std::scientific << std::setprecision(5) << std::setw(14)
              << grun << std::endl;
            std::string sdirname = gen_subdir_name(T);
            auto saved = boost::filesystem::current_path();
            boost::filesystem::current_path(sdirname);
            std::ofstream f2("BTE.gruneisen_total");
            f2 << grun << std::endl;
            boost::filesystem::current_path(saved);
        }
    }
    world.barrier();

    std::cout << "Info: start calculating kappa" << std::endl;
    // Precompute all the matrix elements.
    for (auto& p : threeph_processes) {
        p.compute_vp2(*poscar, *grid, *thirdorder);
    }

    auto fp303 = open_from_master("BTE.KappaTensorVsT_RTA");
    auto fp403 = flags.convergence ? open_from_master("BTE.KappaTensorVsT_CONV")
                                   : alma::make_unique<std::ostream>(&null_buf);
    auto is_plus = [](const alma::Threeph_process& p) {
        return p.type == alma::threeph_type::absorption;
    };
    auto is_minus = [](const alma::Threeph_process& p) {
        return p.type == alma::threeph_type::emission;
    };
    // Compute the scattering rates and the conductivity at each temperature.
    for (const auto& T : Ts) {
        std::cout << "Info: Temperature= " << T << std::endl;
        std::string sdirname = gen_subdir_name(T);
        auto saved = boost::filesystem::current_path();
        boost::filesystem::current_path(sdirname);
        Eigen::ArrayXXd w3_plus(
            alma::calc_w0_threeph(*grid, threeph_processes, T, is_plus, world));
        Eigen::ArrayXXd w3_minus(alma::calc_w0_threeph(
            *grid, threeph_processes, T, is_minus, world));
        Eigen::ArrayXXd w3 = w3_plus + w3_minus;
        // Write out the RTA scattering rates in ShengBTE's detailed format.
        if (my_id == 0) {
            std::ofstream fplus("BTE.w_anharmonic_plus");
            std::ofstream fminus("BTE.w_anharmonic_minus");
            std::ofstream ftotal("BTE.w_anharmonic");
            fplus << std::scientific << std::setprecision(10);
            fminus << std::scientific << std::setprecision(10);
            ftotal << std::scientific << std::setprecision(10);
            for (std::size_t im = 0; im < nmodes; ++im) {
                for (std::size_t ieq = 0; ieq < grid->get_nequivalences();
                     ++ieq) {
                    auto spectrum =
                        grid->get_spectrum_at_q(grid->get_representative(ieq));
                    fplus << std::setw(20) << spectrum.omega(im) << " "
                          << std::setw(20)
                          << w3_plus(im, grid->get_representative(ieq))
                          << std::endl;
                    fminus << std::setw(20) << spectrum.omega(im) << " "
                           << std::setw(20)
                           << w3_minus(im, grid->get_representative(ieq))
                           << std::endl;
                    ftotal << std::setw(20) << spectrum.omega(im) << " "
                           << std::setw(20)
                           << w3(im, grid->get_representative(ieq))
                           << std::endl;
                }
            }
        }
        // Add in the elastic part and save the result.
        Eigen::ArrayXXd w0 = w3 + w0_elastic;
        if (my_id == 0) {
            std::ofstream f("BTE.w");
            f << std::scientific << std::setprecision(10);
            for (std::size_t im = 0; im < nmodes; ++im) {
                for (std::size_t ieq = 0; ieq < grid->get_nequivalences();
                     ++ieq) {
                    auto spectrum =
                        grid->get_spectrum_at_q(grid->get_representative(ieq));
                    f << std::setw(20) << spectrum.omega(im) << " "
                      << std::setw(20) << w0(im, grid->get_representative(ieq))
                      << std::endl;
                }
            }
        }
        world.barrier();

        auto fp2001 = open_from_master("BTE.kappa");
        auto fp2002 = open_from_master("BTE.kappa_tensor");
        auto fp2003 = open_from_master("BTE.kappa_scalar");
        // Compute and save the RTA thermal conductivity.
        auto iterator = alma::ShengBTE_iterator(
            *poscar, *grid, syms, threeph_processes, twoph_processes, T, world);
        Eigen::MatrixXd kappa_RTA = iterator.calc_current_kappa(T);
        kappa_RTA = syms.symmetrize_m<double>(kappa_RTA, true);
        *fp303 << std::fixed << std::setprecision(1) << std::setw(7) << T << " "
               << std::scientific << std::setprecision(5);
        *fp2002 << std::setw(9) << 0 << " " << std::scientific
                << std::setprecision(10);
        for (std::size_t c = 0; c < 3; ++c) {
            for (std::size_t r = 0; r < 3; ++r) {
                *fp303 << std::setw(14) << kappa_RTA(r, c) << " ";
                *fp2002 << std::setw(20) << kappa_RTA(r, c) << " ";
            }
        }
        *fp303 << std::endl;
        *fp2002 << std::endl;
        *fp2003 << std::setw(9) << 0 << " " << std::scientific
                << std::setprecision(10) << std::setw(20)
                << kappa_RTA.trace() / 3. << std::endl;
        *fp2001 << std::setw(9) << 0 << " " << std::scientific
                << std::setprecision(10);
        std::vector<Eigen::MatrixXd> kappa_RTA_branches;
        for (std::size_t im = 0; im < nmodes; ++im) {
            kappa_RTA_branches.push_back(syms.symmetrize_m<double>(
                iterator.calc_current_kappa_branch(T, im), true));
        }
        for (std::size_t i3 = 0; i3 < 3; ++i3) {
            for (std::size_t i2 = 0; i2 < 3; ++i2) {
                for (std::size_t im = 0; im < nmodes; ++im) {
                    *fp2001 << std::setw(20) << kappa_RTA_branches[im](i2, i3)
                            << " ";
                }
            }
        }
        *fp2001 << std::endl;
        Eigen::ArrayXXd w(w0);

        // Use the iterative solver if the user has requested it.
        // Note that this implementation includes elastic scattering
        // in the iteration, unlike ShengBTE itself.
        if (flags.convergence) {
            Eigen::MatrixXd kappa = kappa_RTA;
            // Iterate until convergence has been achieved or until
            // the maximum number of iterations is reached.
            int i = 0;
            for (i = 0; i < parameters.maxiter; ++i) {
                Eigen::MatrixXd kappa_old(kappa);
                iterator.next(*poscar,
                              *grid,
                              syms,
                              threeph_processes,
                              twoph_processes,
                              T);
                kappa = iterator.calc_current_kappa(T);
                kappa = syms.symmetrize_m<double>(kappa, true);
                double relchange =
                    (kappa - kappa_old).norm() / kappa_old.norm();
                // Save the information in the same detailed format as
                // for the RTA versions.
                *fp2002 << std::setw(9) << i + 1 << " " << std::scientific
                        << std::setprecision(10);
                for (std::size_t c = 0; c < 3; ++c) {
                    for (std::size_t r = 0; r < 3; ++r) {
                        *fp2002 << std::setw(20) << kappa(r, c) << " ";
                    }
                }
                *fp2002 << std::endl;
                *fp2003 << std::setw(9) << i + 1 << " " << std::scientific
                        << std::setprecision(10) << std::setw(20)
                        << kappa.trace() / 3. << std::endl;

                std::vector<Eigen::MatrixXd> kappa_branches;
                for (std::size_t im = 0; im < nmodes; ++im) {
                    kappa_branches.push_back(syms.symmetrize_m<double>(
                        iterator.calc_current_kappa_branch(T, im), true));
                }
                *fp2001 << std::setw(9) << i + 1 << " " << std::scientific
                        << std::setprecision(10);
                for (std::size_t i3 = 0; i3 < 3; ++i3) {
                    for (std::size_t i2 = 0; i2 < 3; ++i2) {
                        for (std::size_t im = 0; im < nmodes; ++im) {
                            *fp2001 << std::setw(20)
                                    << kappa_branches[im](i2, i3) << " ";
                        }
                    }
                }
                *fp2001 << std::endl;
                std::cout << "Info: Iteration " << i + 1 << std::endl;
                std::cout << "Info: Relative change = " << relchange
                          << std::endl;
                if (relchange < parameters.eps) {
                    break;
                }
            }
            *fp403 << std::fixed << std::setprecision(1) << std::setw(7) << T
                   << " " << std::scientific << std::setprecision(5);
            for (std::size_t c = 0; c < 3; ++c) {
                for (std::size_t r = 0; r < 3; ++r) {
                    *fp403 << std::setw(14) << kappa(r, c) << " ";
                }
            }
            *fp403 << std::setw(6) << i + 1 << std::endl;
            // Replace the RTA scattering rates with their converged
            // versions. There are no real relaxation times in the full
            // linearized BTE formalism, and thus these can be negative.
            w = iterator.calc_w();
        }
        // Save the final scattering rate (which may be RTA or not, depending on
        // the value of the convergence flag).
        if (my_id == 0) {
            std::ofstream f("BTE.w_final");
            f << std::scientific << std::setprecision(10);
            for (std::size_t im = 0; im < nmodes; ++im) {
                for (std::size_t ieq = 0; ieq < grid->get_nequivalences();
                     ++ieq) {
                    auto spectrum =
                        grid->get_spectrum_at_q(grid->get_representative(ieq));
                    f << std::setw(20) << spectrum.omega(im) << " "
                      << std::setw(20) << w(im, grid->get_representative(ieq))
                      << std::endl;
                }
            }
        }
        // Cumulative thermal conductivity as a function of MFP. To more
        // closely reproduce ShengBTE's behavior, we don't use the same
        // code as cumulativecurves.
        Eigen::ArrayXd ticks(parameters.nticks);
        for (int i = 0; i < parameters.nticks; ++i) {
            ticks(i) = std::pow(10, 8. * i / (parameters.nticks - 1.) - 2.);
        }
        auto cumulative = iterator.calc_cumulative_kappa_lambda(T, ticks);
        fp2002 = open_from_master("BTE.cumulative_kappa_tensor");
        fp2003 = open_from_master("BTE.cumulative_kappa_scalar");
        *fp2002 << std::scientific << std::setprecision(10);
        *fp2003 << std::scientific << std::setprecision(10);
        for (int i = 0; i < parameters.nticks; ++i) {
            *fp2002 << std::setw(20) << ticks(i) << " ";
            *fp2003 << std::setw(20) << ticks(i) << " ";
            cumulative[i] = syms.symmetrize_m<double>(cumulative[i], true);
            for (std::size_t c = 0; c < 3; ++c) {
                for (std::size_t r = 0; r < 3; ++r) {
                    *fp2002 << std::setw(20) << cumulative[i](r, c) << " ";
                }
            }
            *fp2002 << std::endl;
            *fp2003 << std::setw(20) << cumulative[i].trace() / 3. << std::endl;
        }
        // Cumulative thermal conductivity as a function of angular frequency.
        double omega_min = std::numeric_limits<double>::infinity();
        double omega_max = -std::numeric_limits<double>::infinity();
        for (std::size_t iq = 0; iq < grid->nqpoints; ++iq) {
            auto spectrum = grid->get_spectrum_at_q(iq);
            omega_max = std::max(omega_max, spectrum.omega.maxCoeff());
            omega_min = std::min(omega_min, spectrum.omega.minCoeff());
        }
        omega_max *= 1.1;
        for (int i = 0; i < parameters.nticks; ++i) {
            ticks(i) = omega_min +
                       (omega_max - omega_min) * (i + 1.0) / parameters.nticks;
        }
        cumulative = iterator.calc_cumulative_kappa_omega(T, ticks);
        if (my_id == 0) {
            std::ofstream f("BTE.cumulative_kappaVsOmega_tensor");
            f << std::scientific << std::setprecision(10);
            for (int i = 0; i < parameters.nticks; ++i) {
                f << std::setw(20) << ticks(i) << " ";
                cumulative[i] = syms.symmetrize_m<double>(cumulative[i], true);
                for (std::size_t c = 0; c < 3; ++c) {
                    for (std::size_t r = 0; r < 3; ++r) {
                        f << std::setw(20) << cumulative[i](r, c) << " ";
                    }
                }
                f << std::endl;
            }
        }
        boost::filesystem::current_path(saved);
    }
    std::cout << "Info: normal exit" << std::endl;
}