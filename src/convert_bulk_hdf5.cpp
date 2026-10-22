// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <bulk_hdf5.hpp>
#include <boost/filesystem.hpp>
#include <H5Cpp.h>
#include <iostream>

namespace {
// Merge extra attributes/objects recursively, retaining the freshly generated
// core schema and version attributes. In particular, preserve user metadata
// nested under qpoint_grid without replacing the migrated Wigner datasets.
void preserve_attributes(const H5::H5Object& source, H5::H5Object& target) {
    for (int i = 0; i < source.getNumAttrs(); ++i) {
        auto attribute = source.openAttribute(i);
        const auto name = attribute.getName();
        if (H5Aexists(target.getId(), name.c_str()) > 0)
            continue;
        const auto type = attribute.getDataType();
        const auto space = attribute.getSpace();
        const auto count =
            static_cast<std::size_t>(space.getSimpleExtentNpoints());
        if (count > std::numeric_limits<std::size_t>::max() / type.getSize())
            throw alma::input_error("attribute size overflows size_t: " + name);
        std::vector<unsigned char> data(
            std::max<std::size_t>(1, count * type.getSize()));
        attribute.read(type, data.data());
        auto copied = target.createAttribute(name, type, space);
        copied.write(type, data.data());
        if (H5Tis_variable_str(type.getId()) > 0 ||
            H5Tdetect_class(type.getId(), H5T_VLEN) > 0)
            H5Dvlen_reclaim(
                type.getId(), space.getId(), H5P_DEFAULT, data.data());
    }
}
void preserve_group(H5::Group source,
                    H5::Group target,
                    const std::string& path) {
    preserve_attributes(source, target);
    for (hsize_t i = 0; i < source.getNumObjs(); ++i) {
        const auto name = source.getObjnameByIdx(i);
        // The entire old process group is replaced by validated SoA tables.
        if (path == "/" && name == "threeph_processes")
            continue;
        if (H5Lexists(target.getId(), name.c_str(), H5P_DEFAULT) <= 0) {
            if (H5Ocopy(source.getId(),
                        name.c_str(),
                        target.getId(),
                        name.c_str(),
                        H5P_DEFAULT,
                        H5P_DEFAULT) < 0)
                throw alma::input_error("could not preserve object " + path +
                                        name);
        }
        else if (source.getObjTypeByIdx(i) == H5G_GROUP) {
            preserve_group(source.openGroup(name),
                           target.openGroup(name),
                           path + name + "/");
        }
        else if (source.getObjTypeByIdx(i) == H5G_DATASET) {
            auto from = source.openDataSet(name), to = target.openDataSet(name);
            preserve_attributes(from, to);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    boost::mpi::environment environment(argc, argv);
    boost::mpi::communicator comm;
    if (argc != 3) {
        if (!comm.rank())
            std::cerr << "usage: convert_bulk_hdf5 INPUT OUTPUT\n";
        return 1;
    }
    std::string error;
    if (!comm.rank()) {
        try {
            if (boost::filesystem::exists(argv[2]))
                error = "output already exists; choose a new filename";
        }
        catch (const std::exception& e) {
            error = e.what();
        }
    }
    boost::mpi::broadcast(comm, error, 0);
    if (!error.empty()) {
        if (!comm.rank())
            std::cerr << error << '\n';
        return 1;
    }
    try {
        auto data = alma::load_bulk_hdf5(argv[1], comm);
        alma::save_bulk_hdf5(argv[2],
                             std::get<0>(data),
                             *std::get<1>(data),
                             *std::get<2>(data),
                             *std::get<3>(data),
                             *std::get<4>(data),
                             comm);
        if (!comm.rank()) {
            try {
                H5::H5File source(argv[1], H5F_ACC_RDONLY),
                    target(argv[2], H5F_ACC_RDWR);
                preserve_group(
                    source.openGroup("/"), target.openGroup("/"), "/");
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
            throw alma::input_error(error);
    }
    catch (const std::exception& e) {
        if (!comm.rank())
            std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
