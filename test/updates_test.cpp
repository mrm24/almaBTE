// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <H5Cpp.h>
#include <boost/filesystem.hpp>
#include <fstream>
#include <bulk_hdf5.hpp>
#include <bulk_properties.hpp>
#include <gruneisen.hpp>
#include <interfaces.hpp>
#include <vasp_io.hpp>

namespace {
boost::mpi::communicator* world;
std::unique_ptr<alma::Crystal_structure> cell;
std::unique_ptr<alma::Symmetry_operations> syms;
std::unique_ptr<alma::Harmonic_ifcs> ifcs;
std::unique_ptr<alma::Gamma_grid> reference;
std::string temporary() {
    std::string path;
    if (!world->rank())
        path = (boost::filesystem::temp_directory_path() /
                boost::filesystem::unique_path("prealma-updates-%%%%-%%%%.h5"))
                   .string();
    boost::mpi::broadcast(*world, path, 0);
    return path;
}
void remove_file(const std::string& path) {
    world->barrier();
    if (!world->rank())
        boost::filesystem::remove(path);
}
alma::Threeph_processes empty_processes(const alma::Gamma_grid& grid) {
    const auto bounds = alma::my_jobs(
        grid.get_nequivalences() * 6, world->size(), world->rank());
    return alma::Threeph_processes(grid, bounds[0], bounds[1] - bounds[0]);
}
void save(const std::string& path, const alma::Gamma_grid& grid) {
    alma::save_bulk_hdf5(path.c_str(),
                         "Wigner regression",
                         *cell,
                         *syms,
                         grid,
                         empty_processes(grid),
                         *world);
}
TEST(Updates, WignerMatchesFiniteDifferencesAndFinalEigenbasis) {
    alma::Dynamical_matrix_builder factory(*cell, *syms, *ifcs);
    int degenerate_pairs = 0;
    for (const Eigen::Vector3d q : {Eigen::Vector3d(0.71, 1.13, 1.67),
                                    Eigen::Vector3d(0.71, 0., 0.),
                                    Eigen::Vector3d(0.71, 0.71, 0.71)}) {
        const auto sp = factory.get_spectrum(q);
        const auto n = sp->omega.size();
        for (Eigen::Index i = 1; i < n; ++i)
            degenerate_pairs +=
                alma::almost_equal(sp->omega(i), sp->omega(i - 1));
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d step = Eigen::Vector3d::Zero();
            step(axis) = 1e-5;
            const Eigen::MatrixXcd derivative =
                (factory.build(q + step)[0] - factory.build(q - step)[0]) /
                (2e-5);
            Eigen::VectorXd positions(n);
            for (int atom = 0; atom < n / 3; ++atom) {
                const Eigen::Vector3d tau =
                    cell->lattvec * cell->positions.col(atom);
                positions.segment<3>(3 * atom).setConstant(tau(axis));
            }
            const Eigen::MatrixXcd projected =
                sp->wfs.adjoint() * derivative * sp->wfs;
            const Eigen::MatrixXcd position =
                sp->wfs.adjoint() * positions.asDiagonal() * sp->wfs;
            for (Eigen::Index j = 0; j < n; ++j)
                for (Eigen::Index i = 0; i < n; ++i) {
                    const auto v = sp->wigner_v(axis, i + n * j);
                    EXPECT_NEAR(
                        0.,
                        std::abs(v - std::conj(sp->wigner_v(axis, j + n * i))),
                        1e-10);
                    const auto expected =
                        projected(i, j) / (sp->omega(i) + sp->omega(j)) -
                        alma::constants::imud * (sp->omega(j) - sp->omega(i)) *
                            position(i, j);
                    EXPECT_NEAR(0., std::abs(v - expected), 2e-7);
                    bool isolated = true;
                    for (Eigen::Index k = 0; k < n; ++k)
                        if (k != i &&
                            alma::almost_equal(sp->omega(k), sp->omega(i)))
                            isolated = false;
                    if (i == j && isolated) {
                        EXPECT_NEAR(sp->vg(axis, i), v.real(), 1e-9);
                    }
                }
        }
    }
    EXPECT_GT(degenerate_pairs, 0);
}
TEST(Updates, CoherenceTwoModeFormulaAndGaugeInvariance) {
    auto grid = *reference;
    for (std::size_t q = 0; q < grid.nqpoints; ++q) {
        auto& sp =
            const_cast<alma::Spectrum_at_point&>(grid.get_spectrum_at_q(q));
        sp.omega.setZero();
        sp.omega(0) = 2.;
        sp.omega(1) = 5.;
        sp.wigner_v.setZero();
        sp.wigner_v(0, 0 + 6 * 1) = {1., 2.};
        sp.wigner_v(0, 1 + 6 * 0) = {1., -2.};
    }
    Eigen::ArrayXXd rates = Eigen::ArrayXXd::Constant(6, grid.nqpoints, .25);
    const double T = 300., width = .5;
    auto capacity = [&](double frequency) {
        const double x = alma::constants::hbar * frequency * 1e12 /
                         (alma::constants::kB * T);
        const double occupation = 1. / std::expm1(x);
        return x * x * occupation * (occupation + 1.);
    };
    const double expected = .5e21 * alma::constants::kB / cell->V * 7. *
                            (capacity(2.) / 2. + capacity(5.) / 5.) *
                            (.5 * width / (9. + .25 * width * width)) * 5.;
    const auto actual =
        alma::calc_kappa_coherence(*cell, grid, *syms, rates, T);
    EXPECT_NEAR(expected, actual.trace(), 1e-11 * expected);
    EXPECT_LT((actual - Eigen::Matrix3d::Identity() * expected / 3.).norm(),
              1e-11 * expected);
    const auto phase = std::polar(1., .71);
    for (std::size_t q = 0; q < grid.nqpoints; ++q) {
        auto& sp =
            const_cast<alma::Spectrum_at_point&>(grid.get_spectrum_at_q(q));
        sp.wigner_v(0, 6) *= phase;
        sp.wigner_v(0, 1) = std::conj(sp.wigner_v(0, 6));
    }
    EXPECT_LT(
        (actual - alma::calc_kappa_coherence(*cell, grid, *syms, rates, T))
            .norm(),
        1e-11 * expected);
    rates.setZero();
    EXPECT_EQ(0.,
              alma::calc_kappa_coherence(*cell, grid, *syms, rates, T).norm());
    const_cast<alma::Spectrum_at_point&>(grid.get_spectrum_at_q(0))
        .wigner_v.resize(0, 0);
    EXPECT_THROW(alma::calc_kappa_coherence(*cell, grid, *syms, rates, T),
                 alma::value_error);
}
TEST(Updates, CoherenceInvariantUnderDegenerateSubspaceRotation) {
    auto grid = *reference;
    // Triclinic symmetry leaves cross-axis tensor entries visible.
    Eigen::Matrix3d lattice;
    lattice << .5, .13, .07, 0., .6, .11, 0., 0., .7;
    const alma::Crystal_structure triclinic(
        lattice, Eigen::MatrixXd::Zero(3, 1), {"Si"}, {1});
    const alma::Symmetry_operations low_syms(triclinic);
    Eigen::MatrixXcd rotation = Eigen::MatrixXcd::Identity(6, 6);
    rotation(0, 0) = rotation(1, 1) = std::cos(.43);
    rotation(0, 1) = rotation(1, 0) = std::complex<double>(0., std::sin(.43));
    for (std::size_t q = 0; q < grid.nqpoints; ++q) {
        auto& sp =
            const_cast<alma::Spectrum_at_point&>(grid.get_spectrum_at_q(q));
        sp.omega.setZero();
        sp.omega(0) = sp.omega(1) = 2.;
        sp.omega(2) = 5.;
        sp.wigner_v.setZero();
        sp.wigner_v(0, 0 + 6 * 2) = {1., 2.};
        sp.wigner_v(0, 1 + 6 * 2) = {3., 1.};
        sp.wigner_v(0, 2 + 6 * 0) = {1., -2.};
        sp.wigner_v(0, 2 + 6 * 1) = {3., -1.};
        sp.wigner_v(1, 0 + 6 * 2) = {2., -1.};
        sp.wigner_v(1, 1 + 6 * 2) = {.5, 1.5};
        sp.wigner_v(2, 0 + 6 * 2) = {-1., .25};
        sp.wigner_v(2, 1 + 6 * 2) = {.7, -.9};
        for (int axis = 1; axis < 3; ++axis)
            for (int i = 0; i < 2; ++i)
                sp.wigner_v(axis, 2 + 6 * i) =
                    std::conj(sp.wigner_v(axis, i + 6 * 2));
    }
    Eigen::ArrayXXd rates = Eigen::ArrayXXd::Constant(6, grid.nqpoints, .25);
    const auto initial =
        alma::calc_kappa_coherence(*cell, grid, low_syms, rates, 300.);
    EXPECT_GT(std::abs(initial(0, 1)), 1e-6 * initial.norm());
    for (std::size_t q = 0; q < grid.nqpoints; ++q) {
        auto& sp =
            const_cast<alma::Spectrum_at_point&>(grid.get_spectrum_at_q(q));
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::MatrixXcd op(6, 6);
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    op(i, j) = sp.wigner_v(axis, i + 6 * j);
            op = (rotation.adjoint() * op * rotation).eval();
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j)
                    sp.wigner_v(axis, i + 6 * j) = op(i, j);
        }
    }
    EXPECT_LT((initial -
               alma::calc_kappa_coherence(*cell, grid, low_syms, rates, 300.))
                  .norm(),
              1e-11 * initial.norm());
}
TEST(Updates, Wigner05RoundTripAnd04PopulationCompatibility) {
    const auto path = temporary();
    save(path, *reference);
    auto loaded = alma::load_bulk_hdf5(path.c_str(), *world);
    for (std::size_t q = 0; q < reference->nqpoints; ++q)
        EXPECT_EQ(0.,
                  (reference->get_spectrum_at_q(q).wigner_v -
                   std::get<3>(loaded)->get_spectrum_at_q(q).wigner_v)
                      .matrix()
                      .norm());
    if (!world->rank()) {
        H5::H5File file(path, H5F_ACC_RDWR);
        auto ds = file.openDataSet("/qpoint_grid/re_wigner_v");
        hsize_t shape[4];
        EXPECT_EQ(4, ds.getSpace().getSimpleExtentNdims());
        ds.getSpace().getSimpleExtentDims(shape);
        EXPECT_EQ(reference->nqpoints, shape[0]);
        EXPECT_EQ(3u, shape[1]);
        EXPECT_EQ(6u, shape[2]);
        EXPECT_EQ(6u, shape[3]);
        unsigned v = 4;
        file.openAttribute("minor_version")
            .write(H5::PredType::NATIVE_UINT, &v);
        H5Ldelete(file.getId(), "/qpoint_grid/re_wigner_v", H5P_DEFAULT);
        H5Ldelete(file.getId(), "/qpoint_grid/im_wigner_v", H5P_DEFAULT);
        auto group = file.openGroup("/qpoint_grid");
        H5Adelete(group.getId(), "wigner_available");
    }
    world->barrier();
    loaded = alma::load_bulk_hdf5(path.c_str(), *world);
    EXPECT_FALSE(
        std::get<3>(loaded)->get_spectrum_at_q(0).has_wigner_velocities());
    EXPECT_TRUE(std::get<4>(loaded)->empty());
    remove_file(path);
}
TEST(Updates, ForkLegacyWignerAndConverterMetadata) {
    const auto path = temporary();
    save(path, *reference);
    if (!world->rank()) {
        H5::H5File f(path, H5F_ACC_RDWR);
        unsigned version = 2;
        f.openAttribute("minor_version")
            .write(H5::PredType::NATIVE_UINT, &version);
        auto group = f.openGroup("/qpoint_grid");
        H5Adelete(group.getId(), "wigner_available");
        for (const auto* name : {"re_wigner_v", "im_wigner_v"}) {
            H5Ldelete(group.getId(), name, H5P_DEFAULT);
            const hsize_t dims[] = {reference->nqpoints, 3, 36};
            auto ds = group.createDataSet(
                name, H5::PredType::IEEE_F64LE, H5::DataSpace(3, dims));
            std::vector<double> data(reference->nqpoints * 3 * 36);
            for (std::size_t q = 0; q < reference->nqpoints; ++q)
                for (int axis = 0; axis < 3; ++axis)
                    for (int pair = 0; pair < 36; ++pair) {
                        const auto value =
                            reference->get_spectrum_at_q(q).wigner_v(axis,
                                                                     pair);
                        data[(q * 3 + axis) * 36 + pair] =
                            name[0] == 'r' ? value.real() : value.imag();
                    }
            ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        }
        H5Ldelete(f.getId(), "/threeph_processes", H5P_DEFAULT);
        f.createGroup("/threeph_processes");
        const hsize_t empty = 0, triples[] = {0, 3};
        for (const auto* name :
             {"class", "type", "domega", "sigma", "gaussian", "vp2"})
            f.createDataSet(
                std::string("/threeph_processes/") + name,
                name == std::string("class")  ? H5::PredType::STD_U64LE
                : name == std::string("type") ? H5::PredType::STD_U8LE
                                              : H5::PredType::IEEE_F64LE,
                H5::DataSpace(1, &empty));
        for (const auto* name : {"q", "alpha"})
            f.createDataSet(std::string("/threeph_processes/") + name,
                            H5::PredType::STD_U64LE,
                            H5::DataSpace(2, triples));
        const hsize_t size = 2;
        const double payload[] = {1.25, 3.5};
        group
            .createDataSet("user_payload",
                           H5::PredType::IEEE_F64LE,
                           H5::DataSpace(1, &size))
            .write(payload, H5::PredType::NATIVE_DOUBLE);
        H5::StrType text(H5::PredType::C_S1, 11);
        group.createAttribute("user_attribute", text, H5::DataSpace())
            .write(text, std::string("preserve me"));
        f.createGroup("/ifcs")
            .createDataSet(
                "payload", H5::PredType::IEEE_F64LE, H5::DataSpace(1, &size))
            .write(payload, H5::PredType::NATIVE_DOUBLE);
    }
    world->barrier();
    const auto loaded = alma::load_bulk_hdf5(path.c_str(), *world);
    for (std::size_t q = 0; q < reference->nqpoints; ++q)
        EXPECT_EQ(0.,
                  (reference->get_spectrum_at_q(q).wigner_v -
                   std::get<3>(loaded)->get_spectrum_at_q(q).wigner_v)
                      .matrix()
                      .norm());
    if (world->size() == 1) {
        const auto converted = temporary();
        const std::string command = std::string("\"") + ALMA_CONVERTER_PATH +
                                    "\" \"" + path + "\" \"" + converted + "\"";
        EXPECT_EQ(0, std::system(command.c_str()));
        auto data = alma::load_bulk_hdf5(converted.c_str(), *world);
        EXPECT_EQ(0.,
                  (reference->get_spectrum_at_q(1).wigner_v -
                   std::get<3>(data)->get_spectrum_at_q(1).wigner_v)
                      .matrix()
                      .norm());
        {
            H5::H5File f(converted, H5F_ACC_RDONLY);
            auto group = f.openGroup("/qpoint_grid");
            EXPECT_GT(H5Lexists(group.getId(), "user_payload", H5P_DEFAULT), 0);
            EXPECT_GT(H5Lexists(f.getId(), "/ifcs/payload", H5P_DEFAULT), 0);
            std::string attr;
            auto a = group.openAttribute("user_attribute");
            a.read(a.getStrType(), attr);
            EXPECT_EQ("preserve me", attr);
            unsigned version;
            f.openAttribute("minor_version")
                .read(H5::PredType::NATIVE_UINT, &version);
            EXPECT_EQ(5u, version);
        }
        remove_file(converted);
    }
    remove_file(path);
}

TEST(Updates, RejectsIncompleteAndMalformedWignerDataCollectively) {
    for (int scenario = 0; scenario < 5; ++scenario) {
        const auto path = temporary();
        save(path, *reference);
        if (!world->rank()) {
            H5::H5File file(path, H5F_ACC_RDWR);
            if (scenario == 0)
                H5Ldelete(
                    file.getId(), "/qpoint_grid/im_wigner_v", H5P_DEFAULT);
            if (scenario == 1) {
                auto g = file.openGroup("/qpoint_grid");
                H5Adelete(g.getId(), "wigner_available");
            }
            if (scenario == 2) {
                auto g = file.openGroup("/qpoint_grid");
                unsigned char zero = 0;
                g.openAttribute("wigner_available")
                    .write(H5::PredType::NATIVE_UCHAR, &zero);
            }
            if (scenario == 3) {
                H5Ldelete(
                    file.getId(), "/qpoint_grid/im_wigner_v", H5P_DEFAULT);
                const hsize_t dims[] = {reference->nqpoints, 3, 36};
                file.createDataSet("/qpoint_grid/im_wigner_v",
                                   H5::PredType::IEEE_F64LE,
                                   H5::DataSpace(3, dims));
            }
            if (scenario == 4) {
                auto ds = file.openDataSet("/qpoint_grid/re_wigner_v");
                auto space = ds.getSpace();
                const hsize_t point[] = {0, 0, 0, 0};
                space.selectElements(H5S_SELECT_SET, 1, point);
                double nan = std::numeric_limits<double>::quiet_NaN();
                ds.write(
                    &nan, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(), space);
            }
        }
        world->barrier();
        EXPECT_THROW(alma::load_bulk_hdf5(path.c_str(), *world),
                     alma::input_error);
        remove_file(path);
    }
}
TEST(Updates, IfcSubgroupsRoundTripAndCollectiveErrors) {
    const auto path = temporary();
    save(path, *reference);
    std::vector<alma::Thirdorder_ifcs> third;
    third.emplace_back(
        Eigen::Vector3d(1., 2., 3.), Eigen::Vector3d(4., 5., 6.), 0, 1, 0);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
                third[0].ifc(a, b, c) = 1. + a * 9 + b * 3 + c;
    alma::write_ifcs_subgroup(path.c_str(), *ifcs, third, *world);
    alma::Harmonic_ifcs harmonic({}, {}, 1, 1, 1);
    std::vector<alma::Thirdorder_ifcs> read;
    alma::load_ifcs_subgroup(path.c_str(), harmonic, read, *world);
    ASSERT_EQ(ifcs->ifcs.size(), harmonic.ifcs.size());
    ASSERT_EQ(1u, read.size());
    for (std::size_t k = 0; k < harmonic.ifcs.size(); ++k) {
        EXPECT_EQ(ifcs->pos[k], harmonic.pos[k]);
        EXPECT_EQ(0., (ifcs->ifcs[k] - harmonic.ifcs[k]).norm());
    }
    EXPECT_EQ(0., (third[0].rj - read[0].rj).norm());
    EXPECT_EQ(0., (third[0].rk - read[0].rk).norm());
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
                EXPECT_EQ(third[0].ifc(a, b, c), read[0].ifc(a, b, c));
    if (!world->rank()) {
        H5::H5File f(path, H5F_ACC_RDWR);
        H5Ldelete(f.getId(), "/ifcs/anharmonic/Rk", H5P_DEFAULT);
    }
    world->barrier();
    EXPECT_THROW(alma::load_ifcs_subgroup(path.c_str(), harmonic, read, *world),
                 alma::input_error);
    remove_file(path);
}
TEST(Updates, PhaseSpaceNormalizationAndEmptyAnharmonicity) {
    auto processes = empty_processes(*reference);
    if (processes.first_row() == 0 && processes.nrows()) {
        processes.push_back(alma::Threeph_process(
            0, {{0, 1, 2}}, {{3, 1, 2}}, alma::threeph_type::emission, .1, .2));
        processes.push_back(
            alma::Threeph_process(0,
                                  {{0, 1, 2}},
                                  {{3, 1, 2}},
                                  alma::threeph_type::absorption,
                                  .1,
                                  .2));
    }
    processes.finalize();
    Eigen::MatrixXd plus, minus, wp, wm;
    const auto phase = alma::calc_phase_space(
        *cell, *reference, 300., processes, plus, minus, wp, wm, *world);
    const double gaussian = std::exp(-.5 * .1 * .1 / (.2 * .2)) /
                            (std::sqrt(2. * alma::constants::pi) * .2);
    const double expected =
        gaussian / std::pow(static_cast<double>(reference->nqpoints), 2) / 216.;
    EXPECT_TRUE(std::isfinite(phase.second));
    EXPECT_NEAR(expected, phase.first, expected * 1e-12);
    EXPECT_NEAR(2. * minus.sum(), plus.sum(), 1e-14);
    auto empty = empty_processes(*reference);
    Eigen::MatrixXd mean;
    EXPECT_EQ(0.,
              alma::calc_anharmonicity(*cell, *reference, empty, mean, *world));
    EXPECT_EQ(0., mean.norm());
}
TEST(Updates, InterfaceBallisticLimitAndGruneisenMean) {
    const Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
    Eigen::ArrayXXd alpha = Eigen::ArrayXXd::Ones(6, reference->nqpoints);
    double sum = 0.;
    for (std::size_t q = 0; q < reference->nqpoints; ++q) {
        const auto& sp = reference->get_spectrum_at_q(q);
        for (int b = 0; b < 6; ++b)
            if (sp.omega(b) > 0. && sp.vg(0, b) > 0.)
                sum +=
                    sp.vg(0, b) * alma::bose_einstein_kernel(sp.omega(b), 300.);
    }
    const double expected =
        sum * 1e30 * alma::constants::kB / cell->V / reference->nqpoints;
    EXPECT_NEAR(expected,
                alma::interface_conductance(
                    *cell, *reference, axis, alpha, 300., *world),
                1e-10 * std::max(1., expected));
    alpha *= .4;
    EXPECT_NEAR(.4 * expected,
                alma::interface_conductance(
                    *cell, *reference, axis, alpha, 300., *world),
                1e-10 * std::max(1., expected));
    Eigen::ArrayXXd modes =
        Eigen::ArrayXXd::Constant(6, reference->nqpoints, 2.3);
    EXPECT_NEAR(
        2.3, alma::calc_total_gruneisen(*reference, modes, 300.), 1e-13);
    EXPECT_EQ(0.,
              alma::calc_mode_gruneisen(*cell, *reference, {}).matrix().norm());
}
TEST(Updates, PhonopyFormatsAndCartesianScaling) {
    alma::Crystal_structure unit(
        Eigen::Matrix3d::Identity(), Eigen::MatrixXd::Zero(3, 1), {"Si"}, {1});
    std::vector<std::vector<Eigen::MatrixXd>> results;
    for (int format = 0; format < 3; ++format) {
        const auto path = temporary();
        if (!world->rank()) {
            std::ofstream out(path);
            out << (format == 0 ? "2\n" : format == 1 ? "2 2\n" : "1 2\n");
            for (int a = 0; a < (format == 2 ? 1 : 2); ++a)
                for (int b = 0; b < 2; ++b) {
                    out << a + 1 << ' ' << b + 1 << '\n';
                    for (int x = 0; x < 3; ++x) {
                        for (int y = 0; y < 3; ++y)
                            out << 100 * a + 10 * b + 3 * x + y + 1 << ' ';
                        out << '\n';
                    }
                }
        }
        world->barrier();
        auto constants =
            alma::load_FORCE_CONSTANTS(path.c_str(), unit, 2, 1, 1);
        results.push_back(constants->ifcs);
        ASSERT_EQ(2u, constants->ifcs.size());
        remove_file(path);
    }
    for (std::size_t k = 0; k < 2; ++k) {
        EXPECT_EQ(0., (results[0][k] - results[1][k]).norm());
        EXPECT_EQ(0., (results[0][k] - results[2][k]).norm());
        EXPECT_EQ(1. + 10 * k, results[2][k](0, 0));
        EXPECT_EQ(9. + 10 * k, results[2][k](2, 2));
    }
    const auto path = temporary();
    if (!world->rank()) {
        std::ofstream out(path);
        out << "Si\n2\n4 0 0\n0 4 0\n0 0 4\nSi\n1\nCartesian\n1 2 3\n";
    }
    world->barrier();
    const auto loaded = alma::load_POSCAR(path.c_str());
    EXPECT_NEAR(.8, loaded->lattvec(0, 0), 1e-14);
    EXPECT_LT((loaded->positions.col(0) - Eigen::Vector3d(.25, .5, .75)).norm(),
              1e-14);
    remove_file(path);
}
TEST(Updates, PolarDerivativeAndCommensurateReconstruction) {
    const std::string folder = std::string(TEST_RESOURCE_DIR) + "/InAs/";
    const auto polar = alma::load_POSCAR((folder + "POSCAR").c_str());
    alma::Symmetry_operations symmetry(*polar);
    const auto constants = alma::load_FORCE_CONSTANTS(
        (folder + "FORCE_CONSTANTS").c_str(), *polar, 5, 5, 5);
    const auto born = alma::load_BORN((folder + "BORN").c_str());
    alma::Dynamical_matrix_builder uncorrected(*polar, symmetry, *constants);
    alma::Dynamical_matrix_builder wang(
        *polar, symmetry, *constants, *born, alma::nonanalytic_treatment::wang);
    alma::Dynamical_matrix_builder gonze(*polar,
                                         symmetry,
                                         *constants,
                                         *born,
                                         alma::nonanalytic_treatment::gonze);
    const Eigen::Vector3d q(.43, .77, 1.31);
    for (const auto* factory : {&wang, &gonze}) {
        const auto matrices = factory->build(q);
        EXPECT_LT((matrices[0] - matrices[0].adjoint()).norm(),
                  1e-7 * matrices[0].norm());
        const auto spectrum = factory->get_spectrum(q);
        EXPECT_LT(
            (matrices[0] * spectrum->wfs -
             spectrum->wfs * spectrum->omega.square().matrix().asDiagonal())
                .norm(),
            1e-7 * matrices[0].norm());
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d step = Eigen::Vector3d::Zero();
            step(axis) = 1e-5;
            const Eigen::MatrixXcd finite =
                (factory->build(q + step)[0] - factory->build(q - step)[0]) /
                (2e-5);
            EXPECT_LT((finite - matrices[axis + 1]).norm(),
                      2e-6 * std::max(1., finite.norm()));
        }
    }
    const Eigen::Vector3d commensurate =
        polar->rlattvec * Eigen::Vector3d(.2, 0., 0.);
    EXPECT_LT(
        (uncorrected.build(commensurate)[0] - gonze.build(commensurate)[0])
            .norm(),
        1e-7 * uncorrected.build(commensurate)[0].norm());
    Eigen::MatrixXcd translations = Eigen::MatrixXcd::Zero(6, 3);
    for (int atom = 0; atom < 2; ++atom)
        translations.block<3, 3>(3 * atom, 0) =
            std::sqrt(polar->get_mass(atom)) * Eigen::Matrix3d::Identity();
    const auto gamma = gonze.build(Eigen::Vector3d::Zero())[0];
    EXPECT_LT((gamma * translations).norm(),
              1e-6 * gamma.norm() * translations.norm());
}

} // namespace
int main(int argc, char** argv) {
    boost::mpi::environment environment(argc, argv);
    boost::mpi::communicator comm;
    world = &comm;
    H5::Exception::dontPrint();
    ::testing::InitGoogleTest(&argc, argv);
    const std::string folder = std::string(TEST_RESOURCE_DIR) + "/Si/";
    cell = alma::load_POSCAR((folder + "POSCAR").c_str());
    syms = std::make_unique<alma::Symmetry_operations>(*cell);
    ifcs = alma::load_FORCE_CONSTANTS(
        (folder + "FORCE_CONSTANTS").c_str(), *cell, 5, 5, 5);
    reference =
        std::make_unique<alma::Gamma_grid>(*cell, *syms, *ifcs, 2, 2, 2);
    reference->enforce_asr();
    return RUN_ALL_TESTS();
}
