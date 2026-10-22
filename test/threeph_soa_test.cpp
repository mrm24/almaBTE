// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <H5Cpp.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <bulk_hdf5.hpp>
#include <vasp_io.hpp>

namespace {
using Data = decltype(alma::load_bulk_hdf5(
    nullptr,
    std::declval<const boost::mpi::communicator&>()));
std::unique_ptr<Data> reference_data;
std::string native_file;
boost::mpi::communicator* communicator;
std::string temporary_file() {
    std::string filename;
    if (communicator->rank() == 0)
        filename =
            (boost::filesystem::temp_directory_path() /
             boost::filesystem::unique_path("prealma-soa-%%%%-%%%%-%%%%.h5"))
                .string();
    boost::mpi::broadcast(*communicator, filename, 0);
    return filename;
}
class Soa : public ::testing::Test {
protected:
    static void SetUpTestCase() {
        const auto legacy =
            std::string(TEST_RESOURCE_DIR) + "/Si/Si_threeph.h5";
        reference_data = std::make_unique<Data>(
            alma::load_bulk_hdf5(legacy.c_str(), *communicator));
        native_file = temporary_file();
        alma::save_bulk_hdf5(native_file.c_str(),
                             "SoA regression fixture",
                             *std::get<1>(*reference_data),
                             *std::get<2>(*reference_data),
                             *std::get<3>(*reference_data),
                             *std::get<4>(*reference_data),
                             *communicator);
    }
    static void TearDownTestCase() {
        communicator->barrier();
        if (communicator->rank() == 0)
            boost::filesystem::remove(native_file);
        reference_data.reset();
    }
    alma::Gamma_grid& grid() {
        return *std::get<3>(*reference_data);
    }
    alma::Threeph_processes& processes() {
        return *std::get<4>(*reference_data);
    }
    void malformed(const std::function<void(H5::H5File&)>& change) {
        const auto path = temporary_file();
        if (communicator->rank() == 0) {
            boost::filesystem::copy_file(native_file, path);
            H5::H5File file(path, H5F_ACC_RDWR);
            change(file);
        }
        communicator->barrier();
        EXPECT_THROW(alma::load_bulk_hdf5(path.c_str(), *communicator),
                     alma::input_error);
        communicator->barrier();
        if (communicator->rank() == 0)
            boost::filesystem::remove(path);
    }
};
TEST_F(Soa, LegacyCountsAndScalarRates) {
    std::size_t total;
    boost::mpi::all_reduce(
        *communicator, processes().size(), total, std::plus<std::size_t>());
    EXPECT_EQ(175718u, total);
    alma::Threeph_scattering prepared(processes(), grid());
    for (double T : {200., 300., 650.}) {
        const auto rates = prepared.rates(T);
        Eigen::ArrayXXd local =
            Eigen::ArrayXXd::Zero(processes().nbranches(), grid().nqpoints);
        for (auto sign :
             {alma::threeph_type::emission, alma::threeph_type::absorption}) {
            const int which = sign == alma::threeph_type::absorption ? 1 : 0;
            const auto& table = processes().table(sign);
            for (std::size_t row = 0; row < processes().nrows(); ++row)
                for (std::size_t i = table.offsets[row];
                     i < table.offsets[row + 1];
                     ++i) {
                    auto value = processes().view(sign, row, i).to_value();
                    const auto rate = value.compute_gamma(grid(), T);
                    EXPECT_NEAR(rate,
                                rates[which][i],
                                1e-13 * std::max(1., std::abs(rate)));
                    local(value.alpha[0], value.q[0]) +=
                        (which ? 1. : 0.5) * rate;
                }
        }
        for (std::size_t c = 0; c < grid().get_nequivalences(); ++c) {
            const auto equivalent = grid().get_equivalence(c);
            for (std::size_t i = 1; i < equivalent.size(); ++i)
                local.col(equivalent[i]) = local.col(equivalent[0]);
        }
        Eigen::ArrayXXd expected(local.rows(), local.cols());
        boost::mpi::all_reduce(*communicator,
                               local.data(),
                               local.size(),
                               expected.data(),
                               std::plus<double>());
        const auto actual =
            alma::calc_w0_threeph(grid(), processes(), T, *communicator);
        EXPECT_LT((actual - expected).matrix().norm(),
                  1e-11 * std::max(1., expected.matrix().norm()));
        auto plus = alma::calc_w0_threeph(
            grid(),
            processes(),
            T,
            [](const alma::Const_threeph_process_view& p) {
                return p.type == alma::threeph_type::absorption;
            },
            *communicator);
        auto minus = alma::calc_w0_threeph(
            grid(),
            processes(),
            T,
            [](const alma::Const_threeph_process_view& p) {
                return p.type == alma::threeph_type::emission;
            },
            *communicator);
        EXPECT_LT((actual - plus - minus).matrix().norm(),
                  1e-11 * std::max(1., actual.matrix().norm()));
    }
}
TEST_F(Soa, LegacyVersion01RoundTrip) {
    const auto path = std::string(TEST_RESOURCE_DIR) + "/Ge/Ge12.h5";
    auto data = alma::load_bulk_hdf5(path.c_str(), *communicator);
    auto& legacy_processes = *std::get<4>(data);
    auto& legacy_grid = *std::get<3>(data);
    std::size_t total;
    boost::mpi::all_reduce(*communicator,
                           legacy_processes.size(),
                           total,
                           std::plus<std::size_t>());
    EXPECT_EQ(176100u, total);
    const auto expected = alma::calc_w0_threeph(
        legacy_grid, legacy_processes, 300., *communicator);
    // Reference from the original AoS kernels with the same spglib version.
    EXPECT_NEAR(1779.0423281715462, expected.sum(), 1e-9);
    const auto output = temporary_file();
    alma::save_bulk_hdf5(output.c_str(),
                         std::get<0>(data),
                         *std::get<1>(data),
                         *std::get<2>(data),
                         legacy_grid,
                         legacy_processes,
                         *communicator);
    auto loaded = alma::load_bulk_hdf5(output.c_str(), *communicator);
    const auto actual = alma::calc_w0_threeph(
        *std::get<3>(loaded), *std::get<4>(loaded), 300., *communicator);
    EXPECT_LT((actual - expected).matrix().norm(), 1e-11);
    communicator->barrier();
    if (communicator->rank() == 0)
        boost::filesystem::remove(output);
}
TEST_F(Soa, ViewsAndOwningSnapshots) {
    static_assert(
        std::is_same<
            decltype(*std::declval<const alma::Threeph_processes&>().begin()),
            alma::Const_threeph_process_view>::value,
        "const iteration must be read-only");
    EXPECT_THROW(processes().at(processes().size()), std::out_of_range);
    if (processes().empty())
        return;
    const auto p = processes()[0];
    const auto row = p.c * processes().nbranches() + p.alpha[0];
    alma::Threeph_processes owner(grid(), row, 1);
    owner.push_back(alma::Threeph_process(p.c, p.q, p.alpha, p.type, 0.1, 0.2));
    owner.finalize();
    const auto& read_only = owner;
    const auto gaussian = read_only[0].compute_gaussian();
    EXPECT_EQ(0u, owner.table(p.type).flags[0]);
    auto first = owner[0];
    auto second = first;
    EXPECT_DOUBLE_EQ(gaussian, second.compute_gaussian());
    EXPECT_EQ(alma::Threeph_table::gaussian_ready,
              owner.table(p.type).flags[0]);
    EXPECT_THROW(first.get_vp2(), alma::exception);
    auto snapshot = first.to_value();
    EXPECT_EQ(p.q, snapshot.q);
    EXPECT_EQ(p.alpha, snapshot.alpha);
    EXPECT_DOUBLE_EQ(gaussian, snapshot.compute_gaussian());
    auto thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        (std::string(TEST_RESOURCE_DIR) + "/Si/FORCE_CONSTANTS_3RD").c_str(),
        *std::get<1>(*reference_data));
    const auto expected = snapshot.compute_vp2(
        *std::get<1>(*reference_data), grid(), *thirdorder);
    EXPECT_DOUBLE_EQ(
        expected,
        first.compute_vp2(*std::get<1>(*reference_data), grid(), *thirdorder));
    EXPECT_DOUBLE_EQ(expected, second.get_vp2());
}
TEST_F(Soa, PackedIndexBounds) {
    alma::Threeph_indices narrow, wide(true);
    const std::uint64_t large =
        std::uint64_t(std::numeric_limits<std::uint32_t>::max()) + 7;
    narrow.push_back(17);
    EXPECT_THROW(narrow.push_back(large), alma::value_error);
    EXPECT_THROW(narrow.set(0, large), alma::value_error);
    wide.push_back(large);
    EXPECT_EQ(large, wide[0]);
    wide.validate();
}
TEST_F(Soa, SymmetryLookupOwnership) {
    using Images = std::vector<std::size_t>;
    static_assert(std::is_same<decltype(std::declval<const alma::Gamma_grid&>()
                                            .equivalent_qpoints(0)),
                               const Images&>::value,
                  "lvalue lookup should borrow immutable storage");
    static_assert(std::is_same<decltype(std::declval<alma::Gamma_grid&&>()
                                            .equivalent_qpoints(0)),
                               Images>::value,
                  "temporary lookup must own its result");
    static_assert(std::is_same<decltype(std::declval<const alma::Gamma_grid&&>()
                                            .equivalent_qpoints(0)),
                               Images>::value,
                  "const temporary lookup must own its result");
    const auto& first = grid().equivalent_qpoints(0);
    const auto& second = grid().equivalent_qpoints(0);
    EXPECT_EQ(&first, &second);
    auto owned = alma::Gamma_grid(*std::get<1>(*reference_data),
                                  *std::get<2>(*reference_data),
                                  4,
                                  4,
                                  4)
                     .equivalent_qpoints(0);
    EXPECT_FALSE(owned.empty());
    for (auto image : owned)
        EXPECT_EQ(0u, image);
}
TEST_F(Soa, CollisionMatchesScalarProcessTraversal) {
    Eigen::MatrixXd F(3 * processes().nbranches(), grid().nqpoints);
    for (Eigen::Index i = 0; i < F.size(); ++i)
        F.data()[i] = std::sin(0.013 * i);
    for (double T : {200., 500.}) {
        Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(F.rows(), F.cols());
        // Independent reference: the old per-process symmetry traversal.
        for (const auto p : processes()) {
            const auto gamma = p.to_value().compute_gamma(grid(), T);
            const auto s = static_cast<int>(p.type);
            std::map<std::size_t, bool> seen;
            const auto& images1 = grid().equivalent_qpoints(p.q[0]);
            const auto& images2 = grid().equivalent_qpoints(p.q[1]);
            const auto& images3 = grid().equivalent_qpoints(p.q[2]);
            for (std::size_t op = 0; op < images1.size(); ++op) {
                const auto q1 = images1[op], q2 = images2[op], q3 = images3[op];
                if (seen[q1] || q1 == grid().nqpoints ||
                    q2 == grid().nqpoints || q3 == grid().nqpoints)
                    continue;
                expected.block<3, 1>(3 * p.alpha[0], q1) +=
                    2. / (3. - s) * gamma *
                    (F.block<3, 1>(3 * p.alpha[2], q3) -
                     s * F.block<3, 1>(3 * p.alpha[1], q2));
                seen[q1] = true;
            }
        }
        const auto actual =
            alma::Threeph_collision(processes(), grid(), T).apply(F);
        EXPECT_LT((actual - expected).norm(),
                  1e-11 * std::max(1., expected.norm()));
    }
}
TEST_F(Soa, NativeAndMpiRoundTrips) {
    auto loaded = alma::load_bulk_hdf5(native_file.c_str(), *communicator);
    auto& copy = *std::get<4>(loaded);
    EXPECT_EQ(processes().first_row(), copy.first_row());
    EXPECT_EQ(processes().nrows(), copy.nrows());
    EXPECT_EQ(processes().size(), copy.size());
    for (auto sign :
         {alma::threeph_type::emission, alma::threeph_type::absorption}) {
        const auto& a = processes().table(sign);
        const auto& b = copy.table(sign);
        EXPECT_EQ(a.offsets, b.offsets);
        EXPECT_EQ(a.second.narrow(), b.second.narrow());
        EXPECT_EQ(a.third.narrow(), b.third.narrow());
        EXPECT_EQ(a.gaussian, b.gaussian);
        EXPECT_EQ(a.vp2, b.vp2);
        EXPECT_EQ(a.domega, b.domega);
        EXPECT_EQ(a.sigma, b.sigma);
        EXPECT_EQ(a.flags, b.flags);
    }
    std::stringstream buffer;
    {
        boost::archive::binary_oarchive archive(buffer);
        archive << processes();
    }
    alma::Threeph_processes restored;
    {
        boost::archive::binary_iarchive archive(buffer);
        archive >> restored;
    }
    EXPECT_EQ(processes().size(), restored.size());
    const auto reference =
        alma::calc_w0_threeph(grid(), processes(), 300., *communicator);
    const auto actual =
        alma::calc_w0_threeph(grid(), restored, 300., *communicator);
    EXPECT_LT((actual - reference).matrix().norm(), 1e-12);
}
TEST_F(Soa, EmptyRanksAndPartialPreparation) {
    const auto rows = grid().get_nequivalences() * processes().nbranches();
    alma::Threeph_processes local(grid(),
                                  communicator->rank() == 0 ? 0 : rows,
                                  communicator->rank() == 0 ? rows : 0);
    if (communicator->rank() == 0 && !processes().empty()) {
        const auto p = processes()[0];
        local.push_back(
            alma::Threeph_process(p.c, p.q, p.alpha, p.type, 0.1, 0.2));
        local.push_back(p.to_value());
        local.finalize();
    }
    const auto path = temporary_file();
    alma::save_bulk_hdf5(path.c_str(),
                         "partial",
                         *std::get<1>(*reference_data),
                         *std::get<2>(*reference_data),
                         grid(),
                         local,
                         *communicator);
    auto loaded = alma::load_bulk_hdf5(path.c_str(), *communicator);
    auto& copy = *std::get<4>(loaded);
    std::size_t total;
    boost::mpi::all_reduce(
        *communicator, copy.size(), total, std::plus<std::size_t>());
    EXPECT_EQ(2u, total);
    unsigned partial = 0, complete = 0;
    for (auto p : copy) {
        if (p.is_vp2_computed())
            ++complete;
        else
            ++partial;
    }
    unsigned n_partial, n_complete;
    boost::mpi::all_reduce(
        *communicator, partial, n_partial, std::plus<unsigned>());
    boost::mpi::all_reduce(
        *communicator, complete, n_complete, std::plus<unsigned>());
    EXPECT_EQ(1u, n_partial);
    EXPECT_EQ(1u, n_complete);
    EXPECT_THROW(alma::calc_w0_threeph(grid(), copy, 300., *communicator),
                 alma::exception);
    communicator->barrier();
    if (communicator->rank() == 0)
        boost::filesystem::remove(path);
    local.clear();
    const auto empty_path = temporary_file();
    alma::save_bulk_hdf5(empty_path.c_str(),
                         "empty",
                         *std::get<1>(*reference_data),
                         *std::get<2>(*reference_data),
                         grid(),
                         local,
                         *communicator);
    auto empty = alma::load_bulk_hdf5(empty_path.c_str(), *communicator);
    EXPECT_TRUE(std::get<4>(empty)->empty());
    EXPECT_DOUBLE_EQ(
        0.,
        alma::calc_w0_threeph(grid(), *std::get<4>(empty), 300., *communicator)
            .sum());
    communicator->barrier();
    if (communicator->rank() == 0)
        boost::filesystem::remove(empty_path);
}
TEST_F(Soa, RejectsUnsupportedVersionOnAllRanks) {
    malformed([](H5::H5File& file) {
        unsigned version = 6;
        file.openAttribute("minor_version")
            .write(H5::PredType::NATIVE_UINT, &version);
    });
}
TEST_F(Soa, RejectsMalformedOffsetsOnAllRanks) {
    malformed([](H5::H5File& file) {
        auto dataset = file.openDataSet("/threeph_processes/emission/offsets");
        auto space = dataset.getSpace();
        const hsize_t pos = 0;
        space.selectElements(H5S_SELECT_SET, 1, &pos);
        std::uint64_t bad = 1;
        dataset.write(
            &bad, H5::PredType::NATIVE_UINT64, H5::DataSpace(), space);
    });
}
TEST_F(Soa, RejectsInvalidPartnerOnAllRanks) {
    malformed([](H5::H5File& file) {
        auto dataset = file.openDataSet("/threeph_processes/emission/second");
        auto space = dataset.getSpace();
        hsize_t count;
        space.getSimpleExtentDims(&count);
        const hsize_t pos = count - 1;
        space.selectElements(H5S_SELECT_SET, 1, &pos);
        std::uint64_t bad = std::numeric_limits<std::uint32_t>::max();
        dataset.write(
            &bad, H5::PredType::NATIVE_UINT64, H5::DataSpace(), space);
    });
}
TEST_F(Soa, RejectsColumnLengthMismatchOnAllRanks) {
    malformed([](H5::H5File& file) {
        H5Ldelete(
            file.getId(), "/threeph_processes/emission/flags", H5P_DEFAULT);
        const hsize_t bad = 1;
        file.createDataSet("/threeph_processes/emission/flags",
                           H5::PredType::STD_U8LE,
                           H5::DataSpace(1, &bad));
    });
}
} // namespace
int main(int argc, char** argv) {
    boost::mpi::environment environment(argc, argv);
    boost::mpi::communicator world;
    communicator = &world;
    H5::Exception::dontPrint();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
