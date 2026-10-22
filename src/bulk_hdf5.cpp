// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2015-2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <bitset>
#include <cstdint>
#include <numeric>
#include <H5Cpp.h>
#include <bulk_hdf5.hpp>
#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

namespace alma {
namespace {
constexpr std::size_t io_batch = 65536;
// Every rank participates, including ranks with no rows. Failures are reported
// before the next point-to-point transfer, so a bad file cannot strand a peer.
template <class F>
void root_action(const boost::mpi::communicator& comm, F action) {
    std::string error;
    if (comm.rank() == 0) {
        try {
            action();
        }
        catch (const H5::Exception& e) {
            error = e.getDetailMsg();
        }
        catch (const std::exception& e) {
            error = e.what();
        }
    }
    boost::mpi::broadcast(comm, error, 0);
    if (!error.empty())
        throw input_error("bulk HDF5: " + error);
}
std::size_t checked_size(const std::vector<hsize_t>& dims) {
    std::size_t n = 1;
    for (auto d : dims) {
        if (d && n > std::numeric_limits<std::size_t>::max() / d)
            throw input_error("dataset dimensions overflow size_t");
        n *= d;
    }
    return n;
}
std::vector<hsize_t> dimensions(const DataSet& dataset) {
    const auto space = dataset.getSpace();
    std::vector<hsize_t> dims(space.getSimpleExtentNdims());
    if (!dims.empty())
        space.getSimpleExtentDims(dims.data());
    return dims;
}
void require_dimensions(H5File& file,
                        const std::string& name,
                        const std::vector<hsize_t>& dims) {
    if (dimensions(file.openDataSet(name)) != dims)
        throw input_error("invalid dimensions for " + name);
    checked_size(dims);
}
void require_unsigned(H5File& file,
                      const std::string& name,
                      bool index = false) {
    const auto type = file.openDataSet(name).getDataType();
    const auto bytes = type.getSize();
    if (type.getClass() != H5T_INTEGER ||
        H5Tget_sign(type.getId()) != H5T_SGN_NONE ||
        (index && bytes != 4 && bytes != 8))
        throw input_error("expected unsigned integer dataset: " + name);
}
void transfer_slice(DataSet& dataset,
                    bool write,
                    const DataType& type,
                    void* data,
                    const std::vector<hsize_t>& start,
                    const std::vector<hsize_t>& count) {
    if (!checked_size(count))
        return;
    auto space = dataset.getSpace();
    space.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
    DataSpace memory(count.size(), count.data());
    if (write)
        dataset.write(data, type, memory, space);
    else
        dataset.read(data, type, memory, space);
}
template <class T>
std::vector<T> read_column(H5File& file,
                           const std::string& name,
                           std::uint64_t first,
                           std::size_t n,
                           const DataType& type) {
    std::vector<T> result(n);
    auto dataset = file.openDataSet(name);
    for (std::size_t i = 0; i < n; i += io_batch) {
        const auto count = std::min(io_batch, n - i);
        transfer_slice(
            dataset, false, type, result.data() + i, {first + i}, {count});
    }
    return result;
}
template <class T>
void write_column(H5File& file,
                  const std::string& name,
                  std::uint64_t first,
                  const std::vector<T>& values,
                  const DataType& type) {
    auto dataset = file.openDataSet(name);
    for (std::size_t i = 0; i < values.size(); i += io_batch) {
        const auto count = std::min(io_batch, values.size() - i);
        transfer_slice(dataset,
                       true,
                       type,
                       const_cast<T*>(values.data() + i),
                       {first + i},
                       {count});
    }
}
void create_column(H5File& file,
                   const std::string& name,
                   std::uint64_t n,
                   const DataType& type) {
    const hsize_t length = n;
    file.createDataSet(name, type, DataSpace(1, &length));
}
std::string sign_path(int which) {
    return which ? "/threeph_processes/absorption/"
                 : "/threeph_processes/emission/";
}
threeph_type sign_value(int which) {
    return which ? threeph_type::absorption : threeph_type::emission;
}
} // namespace

void write_string_attribute(H5Object& object,
                            const std::string& name,
                            const std::string& value) {
    StrType type(PredType::C_S1, std::max<std::size_t>(1, value.size()));
    type.setCset(H5T_CSET_UTF8);
    auto attribute = object.createAttribute(name, type, DataSpace());
    attribute.write(type, value);
}
template <class H, class T>
void create_eigen_dataset(H& loc,
                          const Eigen::DenseBase<T>& data,
                          const std::string& name,
                          const std::string& units) {
    const hsize_t dims[] = {static_cast<hsize_t>(data.rows()),
                            static_cast<hsize_t>(data.cols())};
    auto dataset =
        loc.createDataSet(name, PredType::IEEE_F64LE, DataSpace(2, dims));
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        contiguous = data;
    dataset.write(contiguous.data(), PredType::NATIVE_DOUBLE);
    write_string_attribute(dataset, "Units", units);
}
void write_file_attributes(H5File& file, const std::string& description) {
    auto root = file.openGroup("/");
    write_string_attribute(root, "timestamp", get_timestamp());
    write_string_attribute(root, "description", description);
    unsigned version = BULK_HDF5_MAJOR_VERSION;
    auto attribute =
        file.createAttribute("major_version", PredType::STD_U16LE, DataSpace());
    attribute.write(PredType::NATIVE_UINT, &version);
    version = BULK_HDF5_MINOR_VERSION;
    attribute =
        file.createAttribute("minor_version", PredType::STD_U16LE, DataSpace());
    attribute.write(PredType::NATIVE_UINT, &version);
}
void create_groups(H5File& file) {
    for (const auto name : {"/crystal_structure",
                            "/symmetry_operations",
                            "/qpoint_grid",
                            "/threeph_processes",
                            "/scattering"})
        file.createGroup(name);
}
std::string read_file_attributes(H5File& file) {
    unsigned major, minor;
    file.openAttribute("major_version").read(PredType::NATIVE_UINT, &major);
    file.openAttribute("minor_version").read(PredType::NATIVE_UINT, &minor);
    if (major != 0 || (minor != 1 && minor != 2 && minor != 4 && minor != 5))
        throw input_error("unsupported bulk HDF5 version " +
                          std::to_string(major) + "." + std::to_string(minor));
    auto attribute = file.openAttribute("description");
    std::string result;
    attribute.read(attribute.getStrType(), result);
    return result;
}

namespace {
void write_metadata(H5File& file,
                    const std::string& description,
                    const Crystal_structure& cell,
                    const Symmetry_operations& symmetries,
                    const Gamma_grid& grid) {
    write_file_attributes(file, description);
    create_groups(file);
    create_eigen_dataset(
        file, cell.lattvec, "/crystal_structure/lattvec", "nm");
    create_eigen_dataset(file,
                         cell.positions,
                         "/crystal_structure/positions",
                         "lattice coordinates");
    std::size_t longest = 1;
    for (const auto& element : cell.elements)
        longest = std::max(longest, element.size());
    StrType type(PredType::C_S1, longest);
    type.setCset(H5T_CSET_UTF8);
    create_column(
        file, "/crystal_structure/elements", cell.elements.size(), type);
    auto elements = file.openDataSet("/crystal_structure/elements");
    for (std::size_t i = 0; i < cell.elements.size(); ++i) {
        auto space = elements.getSpace();
        const hsize_t pos = i;
        space.selectElements(H5S_SELECT_SET, 1, &pos);
        elements.write(cell.elements[i], type, DataSpace(), space);
    }
    create_column(file,
                  "/crystal_structure/numbers",
                  cell.numbers.size(),
                  PredType::STD_I32LE);
    write_column(file,
                 "/crystal_structure/numbers",
                 0,
                 cell.numbers,
                 PredType::NATIVE_INT);
    auto symprec = file.createDataSet(
        "/symmetry_operations/symprec", PredType::IEEE_F64LE, DataSpace());
    symprec.write(&symmetries.symprec, PredType::NATIVE_DOUBLE);
    create_column(file, "/qpoint_grid/nq", 3, PredType::STD_I32LE);
    write_column(file,
                 "/qpoint_grid/nq",
                 0,
                 std::vector<int>{grid.na, grid.nb, grid.nc},
                 PredType::NATIVE_INT);
    const hsize_t bands = grid.get_spectrum_at_q(0).omega.size();
    const hsize_t wdims[] = {grid.nqpoints, bands},
                  fdims[] = {grid.nqpoints, bands, bands};
    const hsize_t vdims[] = {grid.nqpoints, 3, bands};
    auto omega = file.createDataSet(
        "/qpoint_grid/omega", PredType::IEEE_F64LE, DataSpace(2, wdims));
    auto real = file.createDataSet(
        "/qpoint_grid/re_wf", PredType::IEEE_F64LE, DataSpace(3, fdims));
    auto imag = file.createDataSet(
        "/qpoint_grid/im_wf", PredType::IEEE_F64LE, DataSpace(3, fdims));
    auto velocity = file.createDataSet(
        "/qpoint_grid/vg", PredType::IEEE_F64LE, DataSpace(3, vdims));
    write_string_attribute(omega, "Units", "rad / ps");
    write_string_attribute(velocity, "Units", "nm / ps");
    const bool has_wigner = grid.get_spectrum_at_q(0).has_wigner_velocities();
    for (std::size_t q = 0; q < grid.nqpoints; ++q) {
        const auto& sp = grid.get_spectrum_at_q(q);
        if (sp.has_wigner_velocities() != has_wigner ||
            (has_wigner &&
             (sp.wigner_v.rows() != 3 ||
              sp.wigner_v.cols() != static_cast<Eigen::Index>(bands * bands) ||
              !sp.wigner_v.isFinite().all())))
            throw input_error("inconsistent or nonfinite Wigner velocity data");
    }
    auto qgroup = file.openGroup("/qpoint_grid");
    auto available = qgroup.createAttribute(
        "wigner_available", PredType::STD_U8LE, DataSpace());
    const unsigned char present = has_wigner ? 1 : 0;
    available.write(PredType::NATIVE_UCHAR, &present);
    DataSet wigner_real, wigner_imag;
    if (has_wigner) {
        const hsize_t dims[] = {grid.nqpoints, 3, bands, bands};
        wigner_real = file.createDataSet("/qpoint_grid/re_wigner_v",
                                         PredType::IEEE_F64LE,
                                         DataSpace(4, dims));
        wigner_imag = file.createDataSet("/qpoint_grid/im_wigner_v",
                                         PredType::IEEE_F64LE,
                                         DataSpace(4, dims));
        for (auto* dataset : {&wigner_real, &wigner_imag}) {
            write_string_attribute(*dataset, "Units", "nm / ps");
            write_string_attribute(
                *dataset, "axes", "qpoint,cartesian,bra_band,ket_band");
            write_string_attribute(*dataset,
                                   "phase_convention",
                                   "smooth; atomic positions included");
        }
    }
    for (hsize_t q = 0; q < grid.nqpoints; ++q) {
        const auto& spectrum = grid.get_spectrum_at_q(q);
        transfer_slice(omega,
                       true,
                       PredType::NATIVE_DOUBLE,
                       const_cast<double*>(spectrum.omega.data()),
                       {q, 0},
                       {1, bands});
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            buffer = spectrum.wfs.real();
        transfer_slice(real,
                       true,
                       PredType::NATIVE_DOUBLE,
                       buffer.data(),
                       {q, 0, 0},
                       {1, bands, bands});
        buffer = spectrum.wfs.imag();
        transfer_slice(imag,
                       true,
                       PredType::NATIVE_DOUBLE,
                       buffer.data(),
                       {q, 0, 0},
                       {1, bands, bands});
        buffer = spectrum.vg;
        transfer_slice(velocity,
                       true,
                       PredType::NATIVE_DOUBLE,
                       buffer.data(),
                       {q, 0, 0},
                       {1, 3, bands});
        if (has_wigner) {
            // Band-pair order is explicit on disk; Eigen's flattened in-memory
            // convention is bra + bands * ket.
            std::vector<double> re(3 * bands * bands), im(re.size());
            for (hsize_t axis = 0; axis < 3; ++axis)
                for (hsize_t i = 0; i < bands; ++i)
                    for (hsize_t j = 0; j < bands; ++j) {
                        const auto value =
                            spectrum.wigner_v(axis, i + bands * j);
                        const auto offset = (axis * bands + i) * bands + j;
                        re[offset] = value.real();
                        im[offset] = value.imag();
                    }
            transfer_slice(wigner_real,
                           true,
                           PredType::NATIVE_DOUBLE,
                           re.data(),
                           {q, 0, 0, 0},
                           {1, 3, bands, bands});
            transfer_slice(wigner_imag,
                           true,
                           PredType::NATIVE_DOUBLE,
                           im.data(),
                           {q, 0, 0, 0},
                           {1, 3, bands, bands});
        }
    }
    auto processes = file.openGroup("/threeph_processes");
    write_string_attribute(
        processes, "mode_encoding", "q*nbranches+branch; zero-based");
    write_string_attribute(
        processes, "row_encoding", "class*nbranches+branch; zero-based");
    std::vector<std::uint64_t> representatives;
    for (std::size_t c = 0; c < grid.get_nequivalences(); ++c)
        representatives.push_back(grid.get_representative(c));
    create_column(file,
                  "/threeph_processes/representatives",
                  representatives.size(),
                  PredType::STD_U64LE);
    write_column(file,
                 "/threeph_processes/representatives",
                 0,
                 representatives,
                 PredType::NATIVE_UINT64);
}
struct Metadata {
    std::string description;
    unsigned minor = 0;
    std::vector<double> lattice, positions;
    std::vector<std::string> elements;
    std::vector<int> numbers, nq;
    double symprec = 0.;
    std::vector<Spectrum_at_point> spectrum;
    template <class Archive> void serialize(Archive& ar, unsigned int) {
        ar & description & minor & lattice & positions & elements & numbers &
            nq & symprec & spectrum;
    }
};
Metadata read_metadata(H5File& file) {
    Metadata result;
    result.description = read_file_attributes(file);
    file.openAttribute("minor_version")
        .read(PredType::NATIVE_UINT, &result.minor);
    require_dimensions(file, "/crystal_structure/lattvec", {3, 3});
    result.lattice.resize(9);
    file.openDataSet("/crystal_structure/lattvec")
        .read(result.lattice.data(), PredType::NATIVE_DOUBLE);
    const auto positions_dims =
        dimensions(file.openDataSet("/crystal_structure/positions"));
    if (positions_dims.size() != 2 || positions_dims[0] != 3 ||
        positions_dims[1] == 0)
        throw input_error("invalid crystal positions dimensions");
    result.positions.resize(checked_size(positions_dims));
    file.openDataSet("/crystal_structure/positions")
        .read(result.positions.data(), PredType::NATIVE_DOUBLE);
    auto elements = file.openDataSet("/crystal_structure/elements");
    const auto element_dims = dimensions(elements);
    if (element_dims.size() != 1 || element_dims[0] == 0)
        throw input_error("invalid element dimensions");
    for (hsize_t i = 0; i < element_dims[0]; ++i) {
        std::string element;
        auto space = elements.getSpace();
        space.selectElements(H5S_SELECT_SET, 1, &i);
        elements.read(element, elements.getStrType(), DataSpace(), space);
        result.elements.push_back(element);
    }
    require_dimensions(file, "/crystal_structure/numbers", element_dims);
    result.numbers = read_column<int>(file,
                                      "/crystal_structure/numbers",
                                      0,
                                      element_dims[0],
                                      PredType::NATIVE_INT);
    std::uint64_t atoms = 0;
    for (auto n : result.numbers) {
        if (n <= 0)
            throw input_error("nonpositive atom count");
        atoms += n;
    }
    if (atoms != positions_dims[1])
        throw input_error("atom counts and positions disagree");
    require_dimensions(file, "/symmetry_operations/symprec", {});
    file.openDataSet("/symmetry_operations/symprec")
        .read(&result.symprec, PredType::NATIVE_DOUBLE);
    if (!std::isfinite(result.symprec) || result.symprec <= 0.)
        throw input_error("invalid symmetry tolerance");
    require_dimensions(file, "/qpoint_grid/nq", {3});
    result.nq =
        read_column<int>(file, "/qpoint_grid/nq", 0, 3, PredType::NATIVE_INT);
    for (auto n : result.nq)
        if (n <= 0)
            throw input_error("nonpositive q-grid dimension");
    const auto nq = checked_size({static_cast<hsize_t>(result.nq[0]),
                                  static_cast<hsize_t>(result.nq[1]),
                                  static_cast<hsize_t>(result.nq[2])});
    if (nq > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw input_error(
            "q-grid exceeds the supported int q-point indexing range");
    const auto bands = checked_size({3, atoms});
    require_dimensions(file, "/qpoint_grid/omega", {nq, bands});
    require_dimensions(file, "/qpoint_grid/re_wf", {nq, bands, bands});
    require_dimensions(file, "/qpoint_grid/im_wf", {nq, bands, bands});
    require_dimensions(file, "/qpoint_grid/vg", {nq, 3, bands});
    auto omega = file.openDataSet("/qpoint_grid/omega"),
         real = file.openDataSet("/qpoint_grid/re_wf");
    auto imag = file.openDataSet("/qpoint_grid/im_wf"),
         velocity = file.openDataSet("/qpoint_grid/vg");
    const bool has_real =
        H5Lexists(file.getId(), "/qpoint_grid/re_wigner_v", H5P_DEFAULT) > 0;
    const bool has_imag =
        H5Lexists(file.getId(), "/qpoint_grid/im_wigner_v", H5P_DEFAULT) > 0;
    if (has_real != has_imag)
        throw input_error("incomplete Wigner real/imaginary dataset pair");
    auto qgroup = file.openGroup("/qpoint_grid");
    const bool has_marker = H5Aexists(qgroup.getId(), "wigner_available") > 0;
    if (result.minor == 5 && !has_marker)
        throw input_error("missing Wigner availability marker");
    if (has_marker) {
        unsigned char present = 0;
        auto marker = qgroup.openAttribute("wigner_available");
        if (marker.getSpace().getSimpleExtentType() != H5S_SCALAR ||
            marker.getTypeClass() != H5T_INTEGER ||
            marker.getDataType().getSize() != 1 ||
            H5Tget_sign(marker.getDataType().getId()) != H5T_SGN_NONE)
            throw input_error("invalid Wigner availability marker type");
        marker.read(PredType::NATIVE_UCHAR, &present);
        if (present > 1 || (present != 0) != has_real)
            throw input_error("inconsistent Wigner availability marker");
    }
    DataSet wigner_real, wigner_imag;
    bool flattened = false;
    if (has_real) {
        wigner_real = file.openDataSet("/qpoint_grid/re_wigner_v");
        wigner_imag = file.openDataSet("/qpoint_grid/im_wigner_v");
        const auto dims = dimensions(wigner_real);
        flattened =
            dims == std::vector<hsize_t>{nq, 3, checked_size({bands, bands})};
        if (dims != dimensions(wigner_imag) ||
            (dims != std::vector<hsize_t>{nq, 3, bands, bands} &&
             !(result.minor < 5 && flattened)))
            throw input_error("invalid Wigner velocity dimensions");
        for (const auto* dataset : {&wigner_real, &wigner_imag})
            if (dataset->getTypeClass() != H5T_FLOAT ||
                dataset->getDataType().getSize() != 8)
                throw input_error("Wigner velocities must be float64");
    }
    result.spectrum.reserve(nq);
    for (std::size_t q = 0; q < nq; ++q) {
        Eigen::ArrayXd frequencies(bands);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            re(bands, bands), im(bands, bands), vg(3, bands);
        transfer_slice(omega,
                       false,
                       PredType::NATIVE_DOUBLE,
                       frequencies.data(),
                       {q, 0},
                       {1, bands});
        transfer_slice(real,
                       false,
                       PredType::NATIVE_DOUBLE,
                       re.data(),
                       {q, 0, 0},
                       {1, bands, bands});
        transfer_slice(imag,
                       false,
                       PredType::NATIVE_DOUBLE,
                       im.data(),
                       {q, 0, 0},
                       {1, bands, bands});
        transfer_slice(velocity,
                       false,
                       PredType::NATIVE_DOUBLE,
                       vg.data(),
                       {q, 0, 0},
                       {1, 3, bands});
        Eigen::MatrixXcd wfs =
            re.cast<std::complex<double>>() +
            constants::imud * im.cast<std::complex<double>>();
        Eigen::ArrayXXcd wigner;
        if (has_real) {
            std::vector<double> re(3 * bands * bands), im(re.size());
            const std::vector<hsize_t> start =
                flattened ? std::vector<hsize_t>{q, 0, 0}
                          : std::vector<hsize_t>{q, 0, 0, 0};
            const std::vector<hsize_t> count =
                flattened ? std::vector<hsize_t>{1, 3, bands * bands}
                          : std::vector<hsize_t>{1, 3, bands, bands};
            transfer_slice(wigner_real,
                           false,
                           PredType::NATIVE_DOUBLE,
                           re.data(),
                           start,
                           count);
            transfer_slice(wigner_imag,
                           false,
                           PredType::NATIVE_DOUBLE,
                           im.data(),
                           start,
                           count);
            wigner.resize(3, bands * bands);
            for (std::size_t axis = 0; axis < 3; ++axis)
                for (std::size_t i = 0; i < bands; ++i)
                    for (std::size_t j = 0; j < bands; ++j) {
                        const auto offset =
                            flattened ? axis * bands * bands + i + bands * j
                                      : (axis * bands + i) * bands + j;
                        if (!std::isfinite(re[offset]) ||
                            !std::isfinite(im[offset]))
                            throw input_error("nonfinite Wigner velocity");
                        wigner(axis, i + bands * j) = {re[offset], im[offset]};
                    }
        }
        result.spectrum.emplace_back(frequencies, wfs, vg, wigner);
    }
    return result;
}
struct Partition {
    std::size_t first = 0, rows = 0;
    std::array<std::vector<std::uint64_t>, 2> counts;
    template <class Archive> void serialize(Archive& ar, unsigned int) {
        ar & first & rows & counts;
    }
};
using Offsets = std::array<std::vector<std::uint64_t>, 2>;
void validate_offsets(const std::vector<std::uint64_t>& offsets,
                      std::size_t rows) {
    if (offsets.size() != rows + 1 || offsets.front() != 0 ||
        !std::is_sorted(offsets.begin(), offsets.end()) ||
        offsets.back() > std::numeric_limits<std::size_t>::max())
        throw input_error("invalid three-phonon offsets");
}
std::vector<std::size_t> partition_rows(const Offsets& offsets,
                                        std::size_t rows,
                                        int ranks) {
    std::vector<std::uint64_t> work(rows + 1);
    for (std::size_t r = 0; r <= rows; ++r) {
        if (offsets[0][r] >
            std::numeric_limits<std::uint64_t>::max() - offsets[1][r])
            throw input_error("three-phonon process count overflows uint64");
        work[r] = offsets[0][r] + offsets[1][r];
    }
    std::vector<std::size_t> boundaries(ranks + 1);
    boundaries.back() = rows;
    for (int rank = 1; rank < ranks; ++rank) {
        if (work.back() == 0)
            boundaries[rank] = my_jobs(rows, ranks, rank)[0];
        else {
            const auto target = work.back() / ranks * rank +
                                (work.back() % ranks) * rank / ranks;
            boundaries[rank] =
                std::lower_bound(work.begin(), work.end(), target) -
                work.begin();
        }
    }
    return boundaries;
}
void allocate_table(Threeph_table& table,
                    const std::vector<std::uint64_t>& global,
                    std::size_t first,
                    std::size_t rows) {
    table.offsets.resize(rows + 1);
    for (std::size_t r = 0; r <= rows; ++r)
        table.offsets[r] = global[first + r] - global[first];
    const auto n = table.offsets.back();
    table.second.resize(n);
    table.third.resize(n);
    table.domega.resize(n);
    table.sigma.resize(n);
    table.gaussian.resize(n);
    table.vp2.resize(n);
    table.flags.resize(n);
}
void write_table(H5File& file,
                 const Threeph_processes& processes,
                 const Offsets& global) {
    for (int which = 0; which < 2; ++which) {
        const auto path = sign_path(which);
        const auto& table = processes.table(sign_value(which));
        const auto first = global[which][processes.first_row()];
        if (table.second.wide()) {
            write_column(file,
                         path + "second",
                         first,
                         table.second.wide_values(),
                         PredType::NATIVE_UINT64);
            write_column(file,
                         path + "third",
                         first,
                         table.third.wide_values(),
                         PredType::NATIVE_UINT64);
        }
        else {
            write_column(file,
                         path + "second",
                         first,
                         table.second.narrow(),
                         PredType::NATIVE_UINT32);
            write_column(file,
                         path + "third",
                         first,
                         table.third.narrow(),
                         PredType::NATIVE_UINT32);
        }
        write_column(file,
                     path + "domega",
                     first,
                     table.domega,
                     PredType::NATIVE_DOUBLE);
        write_column(
            file, path + "sigma", first, table.sigma, PredType::NATIVE_DOUBLE);
        write_column(file,
                     path + "gaussian",
                     first,
                     table.gaussian,
                     PredType::NATIVE_DOUBLE);
        write_column(
            file, path + "vp2", first, table.vp2, PredType::NATIVE_DOUBLE);
        write_column(
            file, path + "flags", first, table.flags, PredType::NATIVE_UINT8);
    }
}
} // namespace

void save_bulk_hdf5(const char* filename,
                    const std::string& description,
                    const Crystal_structure& cell,
                    const Symmetry_operations& symmetries,
                    const Gamma_grid& grid,
                    const Threeph_processes& input,
                    const boost::mpi::communicator& comm) {
    const auto bands =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());
    const auto rows = checked_size({grid.get_nequivalences(), bands});
    const auto limits = my_jobs(rows, comm.size(), comm.rank());
    Threeph_processes empty(grid, limits[0], limits[1] - limits[0]);
    const auto& processes =
        input.nbranches() == 0 && input.empty() ? empty : input;
    std::string local_error;
    try {
        processes.validate();
        if (processes.nbranches() != bands ||
            processes.nqpoints() != grid.nqpoints)
            throw input_error("processes do not match the q grid");
    }
    catch (const std::exception& e) {
        local_error = e.what();
    }
    std::vector<std::string> errors;
    boost::mpi::all_gather(comm, local_error, errors);
    for (const auto& error : errors)
        if (!error.empty())
            throw input_error(error);
    Partition local;
    local.first = processes.first_row();
    local.rows = processes.nrows();
    for (int which = 0; which < 2; ++which) {
        const auto& offsets = processes.table(sign_value(which)).offsets;
        for (std::size_t row = 0; row < local.rows; ++row)
            local.counts[which].push_back(offsets[row + 1] - offsets[row]);
    }
    std::vector<Partition> partitions;
    boost::mpi::gather(comm, local, partitions, 0);
    Offsets global;
    std::unique_ptr<H5File> file;
    root_action(comm, [&] {
        std::size_t next = 0;
        for (const auto& p : partitions) {
            if (p.first != next || p.rows > rows - next)
                throw input_error("MPI process rows must form consecutive, "
                                  "disjoint partitions");
            next += p.rows;
        }
        if (next != rows)
            throw input_error("MPI process rows do not cover the grid");
        file = std::make_unique<H5File>(filename, H5F_ACC_TRUNC);
        write_metadata(*file, description, cell, symmetries, grid);
        for (int which = 0; which < 2; ++which) {
            global[which].assign(rows + 1, 0);
            for (const auto& p : partitions)
                for (std::size_t r = 0; r < p.rows; ++r)
                    global[which][p.first + r + 1] = p.counts[which][r];
            for (std::size_t r = 0; r < rows; ++r) {
                if (global[which][r + 1] >
                    std::numeric_limits<std::uint64_t>::max() -
                        global[which][r])
                    throw input_error("three-phonon offsets overflow uint64");
                global[which][r + 1] += global[which][r];
            }
            const auto path = sign_path(which);
            file->createGroup(path.substr(0, path.size() - 1));
            create_column(
                *file, path + "offsets", rows + 1, PredType::STD_U64LE);
            write_column(*file,
                         path + "offsets",
                         0,
                         global[which],
                         PredType::NATIVE_UINT64);
            const auto total = global[which].back();
            const auto modes = checked_size({grid.nqpoints, bands});
            const auto& index_type =
                modes && modes - 1 > std::numeric_limits<std::uint32_t>::max()
                    ? PredType::STD_U64LE
                    : PredType::STD_U32LE;
            create_column(*file, path + "second", total, index_type);
            create_column(*file, path + "third", total, index_type);
            for (auto name : {"domega", "sigma", "gaussian", "vp2"})
                create_column(*file, path + name, total, PredType::IEEE_F64LE);
            create_column(*file, path + "flags", total, PredType::STD_U8LE);
            for (auto name : {"domega", "sigma", "gaussian", "vp2"}) {
                auto dataset = file->openDataSet(path + name);
                write_string_attribute(
                    dataset,
                    "Units",
                    std::string(name) == "gaussian" ? "ps / rad"
                    : std::string(name) == "vp2"    ? "THz**4 / (kg nm**2)"
                                                    : "rad / ps");
            }
        }
    });
    // Transfer one rank's columns at a time. I/O uses bounded hyperslabs and
    // root never builds a second global process table.
    for (int rank = 0; rank < comm.size(); ++rank) {
        Threeph_processes other;
        if (rank != 0) {
            if (comm.rank() == rank)
                comm.send(0, 710, processes);
            if (comm.rank() == 0)
                comm.recv(rank, 710, other);
        }
        root_action(comm, [&] {
            write_table(*file, rank == 0 ? processes : other, global);
        });
    }
    root_action(comm, [&] {
        file->flush(H5F_SCOPE_GLOBAL);
        file.reset();
    });
}

namespace {
Offsets read_offsets(H5File& file, const Gamma_grid& grid) {
    const auto rows = checked_size(
        {grid.get_nequivalences(),
         static_cast<hsize_t>(grid.get_spectrum_at_q(0).omega.size())});
    auto group = file.openGroup("/threeph_processes");
    const std::string representatives_path =
        "/threeph_processes/representatives";
    require_dimensions(file, representatives_path, {grid.get_nequivalences()});
    require_unsigned(file, representatives_path);
    const auto representatives =
        read_column<std::uint64_t>(file,
                                   representatives_path,
                                   0,
                                   grid.get_nequivalences(),
                                   PredType::NATIVE_UINT64);
    for (std::size_t c = 0; c < representatives.size(); ++c)
        if (representatives[c] != grid.get_representative(c))
            throw input_error("stored irreducible representatives do not match "
                              "the symmetry grid");
    for (const auto& attribute :
         std::vector<std::pair<std::string, std::string>>{
             {"mode_encoding", "q*nbranches+branch; zero-based"},
             {"row_encoding", "class*nbranches+branch; zero-based"}}) {
        auto a = group.openAttribute(attribute.first);
        std::string value;
        a.read(a.getStrType(), value);
        if (value != attribute.second)
            throw input_error("unsupported " + attribute.first);
    }
    Offsets offsets;
    for (int which = 0; which < 2; ++which) {
        const auto path = sign_path(which);
        require_dimensions(file, path + "offsets", {rows + 1});
        require_unsigned(file, path + "offsets");
        offsets[which] = read_column<std::uint64_t>(
            file, path + "offsets", 0, rows + 1, PredType::NATIVE_UINT64);
        validate_offsets(offsets[which], rows);
        const auto n = offsets[which].back();
        for (auto name :
             {"second", "third", "domega", "sigma", "gaussian", "vp2", "flags"})
            require_dimensions(file, path + name, {n});
        require_unsigned(file, path + "second", true);
        require_unsigned(file, path + "third", true);
        require_unsigned(file, path + "flags");
        if (file.openDataSet(path + "flags").getDataType().getSize() != 1)
            throw input_error("process flags must be uint8");
        for (auto name : {"domega", "sigma", "gaussian", "vp2"}) {
            const auto type = file.openDataSet(path + name).getDataType();
            if (type.getClass() != H5T_FLOAT || type.getSize() != 8)
                throw input_error("process numeric columns must be float64");
        }
    }
    return offsets;
}
Threeph_table read_table(H5File& file,
                         const Gamma_grid& grid,
                         const Offsets& offsets,
                         int which,
                         std::size_t first_row,
                         std::size_t rows) {
    const auto bands =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());
    const auto modes = checked_size({grid.nqpoints, bands});
    Threeph_table table(modes &&
                        modes - 1 > std::numeric_limits<std::uint32_t>::max());
    allocate_table(table, offsets[which], first_row, rows);
    const auto first = offsets[which][first_row], n = table.size();
    const auto path = sign_path(which);
    // Even a 64-bit file can be read into a narrow in-memory table after bounds
    // checks; the file's integer width is independent of the host size_t.
    for (const auto partner : {"second", "third"}) {
        auto dataset = file.openDataSet(path + partner);
        auto& column =
            std::string(partner) == "second" ? table.second : table.third;
        std::vector<std::uint64_t> buffer(std::min(io_batch, n));
        for (std::size_t i = 0; i < n; i += io_batch) {
            const auto count = std::min(io_batch, n - i);
            transfer_slice(dataset,
                           false,
                           PredType::NATIVE_UINT64,
                           buffer.data(),
                           {first + i},
                           {count});
            for (std::size_t j = 0; j < count; ++j)
                column.set(i + j, buffer[j]);
        }
    }
    table.domega = read_column<double>(
        file, path + "domega", first, n, PredType::NATIVE_DOUBLE);
    table.sigma = read_column<double>(
        file, path + "sigma", first, n, PredType::NATIVE_DOUBLE);
    table.gaussian = read_column<double>(
        file, path + "gaussian", first, n, PredType::NATIVE_DOUBLE);
    table.vp2 = read_column<double>(
        file, path + "vp2", first, n, PredType::NATIVE_DOUBLE);
    table.flags = read_column<std::uint8_t>(
        file, path + "flags", first, n, PredType::NATIVE_UINT8);
    return table;
}
struct Legacy_chunk {
    std::vector<std::uint64_t> c, q, alpha;
    std::vector<std::uint8_t> type;
    std::vector<double> domega, sigma, gaussian, vp2;
};
Legacy_chunk legacy_chunk(H5File& file,
                          std::size_t first,
                          std::size_t n,
                          bool full) {
    const std::string path = "/threeph_processes/";
    Legacy_chunk result;
    result.c = read_column<std::uint64_t>(
        file, path + "class", first, n, PredType::NATIVE_UINT64);
    result.type = read_column<std::uint8_t>(
        file, path + "type", first, n, PredType::NATIVE_UINT8);
    result.alpha.resize(3 * n);
    auto alpha = file.openDataSet(path + "alpha");
    transfer_slice(alpha,
                   false,
                   PredType::NATIVE_UINT64,
                   result.alpha.data(),
                   {first, 0},
                   {n, 3});
    if (full) {
        result.q.resize(3 * n);
        auto q = file.openDataSet(path + "q");
        transfer_slice(q,
                       false,
                       PredType::NATIVE_UINT64,
                       result.q.data(),
                       {first, 0},
                       {n, 3});
        result.domega = read_column<double>(
            file, path + "domega", first, n, PredType::NATIVE_DOUBLE);
        result.sigma = read_column<double>(
            file, path + "sigma", first, n, PredType::NATIVE_DOUBLE);
        result.gaussian = read_column<double>(
            file, path + "gaussian", first, n, PredType::NATIVE_DOUBLE);
        result.vp2 = read_column<double>(
            file, path + "vp2", first, n, PredType::NATIVE_DOUBLE);
    }
    return result;
}
Offsets legacy_offsets(H5File& file, const Gamma_grid& grid, std::size_t& n) {
    const std::string path = "/threeph_processes/";
    const auto bands =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());
    const auto rows = checked_size({grid.get_nequivalences(), bands});
    const auto dims = dimensions(file.openDataSet(path + "class"));
    if (dims.size() != 1)
        throw input_error("invalid legacy process count");
    n = checked_size(dims);
    for (auto name : {"class", "type", "domega", "sigma", "gaussian", "vp2"})
        require_dimensions(file, path + name, {n});
    for (auto name : {"q", "alpha"})
        require_dimensions(file, path + name, {n, 3});
    for (auto name : {"class", "q", "alpha", "type"})
        require_unsigned(file, path + name);
    Offsets offsets;
    for (auto& column : offsets)
        column.assign(rows + 1, 0);
    for (std::size_t first = 0; first < n; first += io_batch) {
        const auto chunk =
            legacy_chunk(file, first, std::min(io_batch, n - first), false);
        for (std::size_t i = 0; i < chunk.c.size(); ++i) {
            if (chunk.c[i] >= grid.get_nequivalences() || (chunk.type[i] & ~7u))
                throw input_error("invalid legacy process class or flags");
            for (int j = 0; j < 3; ++j)
                if (chunk.alpha[3 * i + j] >= bands)
                    throw input_error("invalid legacy branch index");
            const auto row = chunk.c[i] * bands + chunk.alpha[3 * i];
            ++offsets[chunk.type[i] & 1][row + 1];
        }
    }
    for (auto& column : offsets)
        std::partial_sum(column.begin(), column.end(), column.begin());
    return offsets;
}
struct Legacy_payload {
    std::vector<std::uint64_t> row, second, third;
    std::vector<double> domega, sigma, gaussian, vp2;
    std::vector<std::uint8_t> mask;
    template <class Archive> void serialize(Archive& ar, unsigned int) {
        ar & row & second & third & domega & sigma & gaussian & vp2 & mask;
    }
};
std::unique_ptr<Threeph_processes> read_legacy(
    H5File* file,
    const Gamma_grid& grid,
    const Offsets& offsets,
    const std::vector<std::size_t>& boundaries,
    std::size_t n,
    const boost::mpi::communicator& comm) {
    const auto first_row = boundaries[comm.rank()],
               rows = boundaries[comm.rank() + 1] - first_row;
    auto result = std::make_unique<Threeph_processes>(grid, first_row, rows);
    const auto bands = result->nbranches();
    std::array<Threeph_table, 2> tables{
        {Threeph_table(result->table(threeph_type::emission).second.wide()),
         Threeph_table(result->table(threeph_type::absorption).second.wide())}};
    std::array<std::vector<std::uint64_t>, 2> cursor;
    for (int which = 0; which < 2; ++which) {
        allocate_table(tables[which], offsets[which], first_row, rows);
        cursor[which] = tables[which].offsets;
    }
    // Two bounded passes: first count rows, then route each chunk to its owning
    // rank and fill preallocated columns. No global AoS or global SoA copy.
    for (std::size_t first = 0; first < n; first += io_batch) {
        std::vector<Legacy_payload> payloads(comm.rank() == 0 ? comm.size()
                                                              : 0);
        root_action(comm, [&] {
            const auto chunk =
                legacy_chunk(*file, first, std::min(io_batch, n - first), true);
            for (std::size_t i = 0; i < chunk.c.size(); ++i) {
                for (int j = 0; j < 3; ++j)
                    if (chunk.q[3 * i + j] >= grid.nqpoints)
                        throw input_error("invalid legacy q-point index");
                if (chunk.q[3 * i] != grid.get_representative(chunk.c[i]))
                    throw input_error("legacy incoming q point is not a class "
                                      "representative");
                const auto row = chunk.c[i] * bands + chunk.alpha[3 * i];
                const auto owner = std::upper_bound(boundaries.begin(),
                                                    boundaries.end(),
                                                    row) -
                                   boundaries.begin() - 1;
                auto& payload = payloads[owner];
                payload.row.push_back(row);
                payload.second.push_back(chunk.q[3 * i + 1] * bands +
                                         chunk.alpha[3 * i + 1]);
                payload.third.push_back(chunk.q[3 * i + 2] * bands +
                                        chunk.alpha[3 * i + 2]);
                payload.domega.push_back(chunk.domega[i]);
                payload.sigma.push_back(chunk.sigma[i]);
                payload.gaussian.push_back(chunk.type[i] & 2 ? chunk.gaussian[i]
                                                             : 0.);
                payload.vp2.push_back(chunk.type[i] & 4 ? chunk.vp2[i] : 0.);
                payload.mask.push_back(chunk.type[i]);
            }
        });
        Legacy_payload payload;
        if (comm.rank() == 0) {
            for (int rank = 1; rank < comm.size(); ++rank)
                comm.send(rank, 712, payloads[rank]);
            payload = std::move(payloads[0]);
        }
        else
            comm.recv(0, 712, payload);
        for (std::size_t i = 0; i < payload.row.size(); ++i) {
            const auto which = payload.mask[i] & 1;
            const auto row = payload.row[i] - first_row;
            const auto dest = cursor[which][row]++;
            auto& table = tables[which];
            table.second.set(dest, payload.second[i]);
            table.third.set(dest, payload.third[i]);
            table.domega[dest] = payload.domega[i];
            table.sigma[dest] = payload.sigma[i];
            table.gaussian[dest] = payload.gaussian[i];
            table.vp2[dest] = payload.vp2[i];
            table.flags[dest] = (payload.mask[i] >> 1) & 3;
        }
    }
    for (int which = 0; which < 2; ++which)
        result->set_table(sign_value(which), std::move(tables[which]));
    return result;
}
} // namespace

std::tuple<std::string,
           std::unique_ptr<Crystal_structure>,
           std::unique_ptr<Symmetry_operations>,
           std::unique_ptr<Gamma_grid>,
           std::unique_ptr<Threeph_processes>>
load_bulk_hdf5(const char* filename, const boost::mpi::communicator& comm) {
    std::unique_ptr<H5File> file;
    Metadata metadata;
    root_action(comm, [&] {
        file = std::make_unique<H5File>(filename, H5F_ACC_RDONLY);
        metadata = read_metadata(*file);
    });
    boost::mpi::broadcast(comm, metadata, 0);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> lattice(
        metadata.lattice.data());
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        positions(metadata.positions.data(), 3, metadata.positions.size() / 3);
    auto cell = std::make_unique<Crystal_structure>(
        lattice, positions, metadata.elements, metadata.numbers);
    auto symmetries =
        std::make_unique<Symmetry_operations>(*cell, metadata.symprec);
    auto grid = std::make_unique<Gamma_grid>(
        *cell, *symmetries, metadata.nq[0], metadata.nq[1], metadata.nq[2]);
    grid->spectrum.swap(metadata.spectrum);
    const auto rows = checked_size(
        {grid->get_nequivalences(),
         static_cast<hsize_t>(grid->get_spectrum_at_q(0).omega.size())});
    Offsets offsets;
    std::size_t legacy_count = 0;
    root_action(comm, [&] {
        offsets = metadata.minor >= 4
                      ? read_offsets(*file, *grid)
                      : legacy_offsets(*file, *grid, legacy_count);
    });
    boost::mpi::broadcast(comm, offsets, 0);
    boost::mpi::broadcast(comm, legacy_count, 0);
    const auto boundaries = partition_rows(offsets, rows, comm.size());
    std::unique_ptr<Threeph_processes> processes;
    if (metadata.minor < 4)
        processes = read_legacy(
            file.get(), *grid, offsets, boundaries, legacy_count, comm);
    else {
        for (int rank = 0; rank < comm.size(); ++rank) {
            std::unique_ptr<Threeph_processes> local;
            root_action(comm, [&] {
                const auto first = boundaries[rank],
                           count = boundaries[rank + 1] - first;
                local =
                    std::make_unique<Threeph_processes>(*grid, first, count);
                for (int which = 0; which < 2; ++which)
                    local->set_table(
                        sign_value(which),
                        read_table(*file, *grid, offsets, which, first, count));
            });
            if (rank == 0) {
                if (comm.rank() == 0)
                    processes = std::move(local);
            }
            else {
                if (comm.rank() == 0)
                    comm.send(rank, 713, *local);
                if (comm.rank() == rank) {
                    processes = std::make_unique<Threeph_processes>();
                    comm.recv(0, 713, *processes);
                }
            }
        }
    }
    return std::make_tuple(metadata.description,
                           std::move(cell),
                           std::move(symmetries),
                           std::move(grid),
                           std::move(processes));
}


std::vector<std::string> list_scattering_subgroups(
    const char* filename,
    const boost::mpi::communicator& comm) {
    auto my_id = comm.rank();

    std::unique_ptr<H5File> h5f;
    std::vector<std::string> nruter;

    // The file is only read from the process with id 0.
    if (my_id == 0) {
        h5f = std::make_unique<H5File>(filename, H5F_ACC_RDONLY);
        // Try to open the "scattering" group in the file,
        // and stop if it does not exist.
        bool found = true;
        Group scattering;
        try {
            scattering = h5f->openGroup("/scattering");
        }
        catch (GroupIException e) {
            found = false;
        }

        // If the group exists, get the number of objects therein.
        if (found) {
            std::size_t nobjects{scattering.getNumObjs()};

            // Iterate through them to find those that are groups.
            for (std::size_t i = 0; i < nobjects; ++i) {
                H5G_obj_t obj_type = scattering.getObjTypeByIdx(i);

                // And add them to the returned vector.
                if (obj_type == H5G_GROUP)
                    nruter.push_back(scattering.getObjnameByIdx(i));
            }
        }
    }
    // Broadcast the data to all processes.
    boost::mpi::broadcast(comm, nruter, 0);
    // And return the list of subgroups.
    return nruter;
}


Scattering_subgroup load_scattering_subgroup(
    const char* filename,
    const std::string& groupname,
    const boost::mpi::communicator& comm) {
    auto my_id = comm.rank();

    std::unique_ptr<H5File> h5f;
    std::string name;
    bool preserves_symmetry;
    std::string description;
    Eigen::ArrayXXd w0;
    std::vector<std::string> attributes;
    std::vector<std::string> datasets;
    std::vector<std::string> groups;

    if (my_id == 0) {
        h5f = std::make_unique<H5File>(filename, H5F_ACC_RDONLY);
        std::string fullname{std::string("/scattering/") + groupname};
        // Open the group (the operation can fail if it does not
        // exist).
        Group group = h5f->openGroup(fullname);
        // Read the two mandatory attributes.
        Attribute symm_attr{group.openAttribute("preserves_symmetry")};
        // Since HDF5 does not have a boolean datatype,
        // an unsigned int is used.
        unsigned int tmp;
        symm_attr.read(PredType::NATIVE_UINT, &tmp);
        preserves_symmetry = static_cast<bool>(tmp);
        Attribute description_attr{group.openAttribute("description")};
        StrType description_type = description_attr.getStrType();
        description_attr.read(description_type, description);
        // Read the mandatory dataset.
        DataSet dset = group.openDataSet("w0");
        DataSpace dspace = dset.getSpace();
        hsize_t dims[2];
        dspace.getSimpleExtentDims(dims, nullptr);
        w0.resize(dims[0], dims[1]);
        DataSpace scalar;
        hsize_t pos[] = {1, 1};

        for (hsize_t i = 0; i < dims[0]; ++i) {
            pos[0] = i;

            for (hsize_t j = 0; j < dims[1]; ++j) {
                pos[1] = j;
                dspace.selectElements(H5S_SELECT_SET, 1, pos);
                double value;
                dset.read(&value, PredType::NATIVE_DOUBLE, scalar, dspace);
                w0(i, j) = value;
            }
        }
        // List all non-mandatory elements in the subgroup.
        std::size_t nattrs = group.getNumAttrs();

        for (std::size_t i = 0; i < nattrs; ++i) {
            Attribute attr{group.openAttribute(i)};
            std::string name{attr.getName()};

            if ((name != "preserves_symmetry") && (name != "description"))
                attributes.emplace_back(name);
        }
        std::size_t nobjects = group.getNumObjs();

        for (std::size_t i = 0; i < nobjects; ++i) {
            H5G_obj_t obj_type = group.getObjTypeByIdx(i);
            std::string obj_name = group.getObjnameByIdx(i);

            if (obj_name == "w0")
                continue;

            // Objects other than groups or datasets are
            // silently ignored.
            if (obj_type == H5G_GROUP) {
                groups.emplace_back(name);
            }
            else if (obj_type == H5G_DATASET) {
                datasets.emplace_back(name);
            }
        }
    }
    // Broadcast the result to the remaining processes.
    boost::mpi::broadcast(comm, preserves_symmetry, 0);
    boost::mpi::broadcast(comm, description, 0);
    std::size_t rows;
    std::size_t cols;

    if (my_id == 0) {
        rows = w0.rows();
        cols = w0.cols();
    }
    boost::mpi::broadcast(comm, rows, 0);
    boost::mpi::broadcast(comm, cols, 0);

    if (my_id != 0)
        w0.resize(rows, cols);
    boost::mpi::broadcast(comm, w0.data(), w0.size(), 0);
    boost::mpi::broadcast(comm, attributes, 0);
    boost::mpi::broadcast(comm, datasets, 0);
    boost::mpi::broadcast(comm, groups, 0);
    // And construct the final object.
    return Scattering_subgroup(groupname,
                               preserves_symmetry,
                               description,
                               w0,
                               attributes,
                               datasets,
                               groups);
}


void write_scattering_subgroup(const char* filename,
                               const Scattering_subgroup& group,
                               const boost::mpi::communicator& comm) {
    auto my_id = comm.rank();

    // Processes other than ID 0 do not do anything other than wait.
    if (my_id == 0) {
        std::unique_ptr<H5File> h5f;
        h5f = std::make_unique<H5File>(filename, H5F_ACC_RDWR);
        // Check if the shape of the scattering rates is compatible
        // with the crystal structure and q-point grid stored in
        // the file.
        DataSet pos_dset = h5f->openDataSet("/crystal_structure/positions");
        DataSpace pos_dspace = pos_dset.getSpace();
        hsize_t pos_size[2];
        pos_dspace.getSimpleExtentDims(pos_size, nullptr);
        std::size_t ndof_file = 3 * pos_size[1];
        int nq[3];
        DataSet nq_dset = h5f->openDataSet("/qpoint_grid/nq");
        DataSpace nq_dspace = nq_dset.getSpace();
        nq_dspace.selectAll();
        nq_dset.read(nq, PredType::NATIVE_INT, nq_dspace, nq_dspace);
        std::size_t nqpoints_file = nq[0] * nq[1] * nq[2];

        if ((ndof_file != static_cast<std::size_t>(group.w0.rows())) ||
            (nqpoints_file != static_cast<std::size_t>(group.w0.cols())))
            throw value_error("the scattering rate array has the"
                              " wrong shape");
        // Check if the "/scattering" group exists,
        // create it otherwise.
        Group scattering;
        try {
            scattering = h5f->openGroup("/scattering");
        }
        catch (GroupIException e) {
            scattering = h5f->createGroup("/scattering");
        }
        // Write the two mandatory attributes.
        Group subgroup = scattering.createGroup(group.name);
        write_string_attribute(subgroup, "description", group.description);
        unsigned int tmp = group.preserves_symmetry ? 1 : 0;
        DataSpace dspace = DataSpace();
        Attribute attr{subgroup.createAttribute(
            "preserves_symmetry", PredType::STD_U16LE, dspace)};
        attr.write(PredType::NATIVE_UINT, &tmp);
        // Write the mandatory dataset.
        create_eigen_dataset(subgroup, group.w0, "w0", "1 / ps");
    }
    comm.barrier();
}


void write_ifcs_subgroup(const char* filename,
                         const Harmonic_ifcs& harmonic,
                         const std::vector<Thirdorder_ifcs>& anharmonic,
                         const boost::mpi::communicator& comm) {
    root_action(comm, [&] {
        if (harmonic.na <= 0 || harmonic.nb <= 0 || harmonic.nc <= 0 ||
            harmonic.ifcs.empty() ||
            harmonic.pos.size() != harmonic.ifcs.size())
            throw value_error("invalid harmonic IFCs");
        H5File file(filename, H5F_ACC_RDWR);
        if (H5Lexists(file.getId(), "/ifcs", H5P_DEFAULT) <= 0)
            file.createGroup("/ifcs");
        if (H5Lexists(file.getId(), "/ifcs/harmonic", H5P_DEFAULT) > 0 ||
            H5Lexists(file.getId(), "/ifcs/anharmonic", H5P_DEFAULT) > 0)
            throw value_error("IFC subgroups already exist");
        const auto dofs = harmonic.ifcs.front().rows();
        if (dofs == 0 || dofs % 3)
            throw value_error("invalid harmonic IFC matrix dimensions");
        for (const auto& block : harmonic.ifcs)
            if (block.rows() != dofs || block.cols() != dofs ||
                !block.allFinite())
                throw value_error(
                    "inconsistent harmonic IFC matrix dimensions or values");
        file.createGroup("/ifcs/harmonic");
        file.createGroup("/ifcs/anharmonic");
        create_column(file, "/ifcs/harmonic/supercell", 3, PredType::STD_I32LE);
        write_column(file,
                     "/ifcs/harmonic/supercell",
                     0,
                     std::vector<int>{harmonic.na, harmonic.nb, harmonic.nc},
                     PredType::NATIVE_INT);
        const hsize_t n = harmonic.ifcs.size(), d = static_cast<hsize_t>(dofs);
        const hsize_t pdims[] = {n, 3}, hdims[] = {n, d * d};
        auto positions = file.createDataSet(
            "/ifcs/harmonic/Rj", PredType::STD_I32LE, DataSpace(2, pdims));
        auto constants = file.createDataSet("/ifcs/harmonic/force_constants",
                                            PredType::IEEE_F64LE,
                                            DataSpace(2, hdims));
        write_string_attribute(
            constants, "matrix_order", "column-major; row+ndof*column");
        for (hsize_t k = 0; k < n; ++k) {
            transfer_slice(positions,
                           true,
                           PredType::NATIVE_INT,
                           const_cast<int*>(harmonic.pos[k].data()),
                           {k, 0},
                           {1, 3});
            transfer_slice(constants,
                           true,
                           PredType::NATIVE_DOUBLE,
                           const_cast<double*>(harmonic.ifcs[k].data()),
                           {k, 0},
                           {1, d * d});
        }
        const hsize_t na = anharmonic.size();
        const hsize_t rdims[] = {na, 3}, adims[] = {na, 27};
        auto rj = file.createDataSet(
            "/ifcs/anharmonic/Rj", PredType::IEEE_F64LE, DataSpace(2, rdims));
        auto rk = file.createDataSet(
            "/ifcs/anharmonic/Rk", PredType::IEEE_F64LE, DataSpace(2, rdims));
        auto atoms = file.createDataSet(
            "/ifcs/anharmonic/atoms", PredType::STD_U64LE, DataSpace(2, rdims));
        auto values = file.createDataSet("/ifcs/anharmonic/force_constants",
                                         PredType::IEEE_F64LE,
                                         DataSpace(2, adims));
        for (hsize_t k = 0; k < na; ++k) {
            const auto& block = anharmonic[k];
            if (block.rj.size() != 3 || block.rk.size() != 3 ||
                !block.rj.allFinite() || !block.rk.allFinite() ||
                std::max({block.i, block.j, block.k}) >=
                    static_cast<std::size_t>(dofs / 3))
                throw value_error("invalid third-order IFC metadata");
            std::uint64_t indices[] = {block.i, block.j, block.k};
            std::array<double, 27> data;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    for (int c = 0; c < 3; ++c)
                        data[c + 3 * (b + 3 * a)] = block.ifc(a, b, c);
            if (!std::all_of(data.begin(), data.end(), [](double x) {
                    return std::isfinite(x);
                }))
                throw value_error("nonfinite third-order IFC");
            transfer_slice(rj,
                           true,
                           PredType::NATIVE_DOUBLE,
                           const_cast<double*>(block.rj.data()),
                           {k, 0},
                           {1, 3});
            transfer_slice(rk,
                           true,
                           PredType::NATIVE_DOUBLE,
                           const_cast<double*>(block.rk.data()),
                           {k, 0},
                           {1, 3});
            transfer_slice(
                atoms, true, PredType::NATIVE_UINT64, indices, {k, 0}, {1, 3});
            transfer_slice(values,
                           true,
                           PredType::NATIVE_DOUBLE,
                           data.data(),
                           {k, 0},
                           {1, 27});
        }
    });
}

void load_ifcs_subgroup(const char* filename,
                        Harmonic_ifcs& harmonic,
                        std::vector<Thirdorder_ifcs>& anharmonic,
                        const boost::mpi::communicator& comm) {
    root_action(comm, [&] {
        H5File file(filename, H5F_ACC_RDONLY);
        require_dimensions(file, "/ifcs/harmonic/supercell", {3});
        const auto supercell = read_column<int>(
            file, "/ifcs/harmonic/supercell", 0, 3, PredType::NATIVE_INT);
        if (std::min({supercell[0], supercell[1], supercell[2]}) <= 0)
            throw input_error("invalid IFC supercell");
        auto values = file.openDataSet("/ifcs/harmonic/force_constants");
        auto shape = dimensions(values);
        if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0)
            throw input_error("invalid harmonic IFC dimensions");
        checked_size(shape);
        const hsize_t d =
            static_cast<hsize_t>(std::sqrt(static_cast<double>(shape[1])));
        if (d * d != shape[1] || d % 3)
            throw input_error(
                "harmonic IFC matrix must be square with 3N rows");
        require_dimensions(file, "/ifcs/harmonic/Rj", {shape[0], 3});
        auto positions = file.openDataSet("/ifcs/harmonic/Rj");
        harmonic.na = supercell[0];
        harmonic.nb = supercell[1];
        harmonic.nc = supercell[2];
        harmonic.pos.resize(shape[0]);
        harmonic.ifcs.resize(shape[0]);
        for (hsize_t k = 0; k < shape[0]; ++k) {
            harmonic.ifcs[k].resize(d, d);
            transfer_slice(positions,
                           false,
                           PredType::NATIVE_INT,
                           harmonic.pos[k].data(),
                           {k, 0},
                           {1, 3});
            transfer_slice(values,
                           false,
                           PredType::NATIVE_DOUBLE,
                           harmonic.ifcs[k].data(),
                           {k, 0},
                           {1, d * d});
            if (!harmonic.ifcs[k].allFinite())
                throw input_error("nonfinite harmonic IFC");
        }
        auto thirds = file.openDataSet("/ifcs/anharmonic/force_constants");
        shape = dimensions(thirds);
        if (shape.size() != 2 || shape[1] != 27)
            throw input_error("invalid anharmonic IFC dimensions");
        checked_size(shape);
        for (const auto name : {"/ifcs/anharmonic/Rj",
                                "/ifcs/anharmonic/Rk",
                                "/ifcs/anharmonic/atoms"})
            require_dimensions(file, name, {shape[0], 3});
        auto rj = file.openDataSet("/ifcs/anharmonic/Rj"),
             rk = file.openDataSet("/ifcs/anharmonic/Rk"),
             atoms = file.openDataSet("/ifcs/anharmonic/atoms");
        anharmonic.clear();
        anharmonic.reserve(shape[0]);
        for (hsize_t k = 0; k < shape[0]; ++k) {
            Eigen::Vector3d j, l;
            std::uint64_t indices[3];
            std::array<double, 27> data;
            transfer_slice(
                rj, false, PredType::NATIVE_DOUBLE, j.data(), {k, 0}, {1, 3});
            transfer_slice(
                rk, false, PredType::NATIVE_DOUBLE, l.data(), {k, 0}, {1, 3});
            transfer_slice(
                atoms, false, PredType::NATIVE_UINT64, indices, {k, 0}, {1, 3});
            transfer_slice(thirds,
                           false,
                           PredType::NATIVE_DOUBLE,
                           data.data(),
                           {k, 0},
                           {1, 27});
            if (!j.allFinite() || !l.allFinite() ||
                std::max({indices[0], indices[1], indices[2]}) >= d / 3 ||
                !std::all_of(data.begin(), data.end(), [](double x) {
                    return std::isfinite(x);
                }))
                throw input_error("invalid third-order IFC data");
            anharmonic.emplace_back(j, l, indices[0], indices[1], indices[2]);
            anharmonic.back().swap_block(data);
        }
    });
    boost::mpi::broadcast(comm, harmonic, 0);
    boost::mpi::broadcast(comm, anharmonic, 0);
}
} // namespace alma
