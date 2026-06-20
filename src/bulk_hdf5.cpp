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
/// Definitions corresponding to bulk_hdf5.hpp.

#include <bitset>
#include <cstdint>
#include <H5Cpp.h>
#include <bulk_hdf5.hpp>

// There seems to be no way to avoid this "using" without
// causing a lot of compilation problems.
#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif // ifndef H5_NO_NAMESPACE

/// Constants used to build the "/threeph_processes/type"
/// and "/fourph_processes/type"
/// dataset. See below for details.
constexpr std::size_t ABSORPTION_BIT     = 0;
constexpr std::size_t GAUSSIAN_BIT       = 1;
constexpr std::size_t VP2_BIT            = 2;
constexpr std::size_t RECOMBINATION_BIT  = 3;
constexpr std::size_t REDISTRIBUTION_BIT = 4;
constexpr std::size_t SPLITTING_BIT      = 5;

namespace alma {
/// Add a string-valued attribute to an HDF5 file, group,
/// or dataset
///
/// Note that UTF-8 encoding is assumed.
/// @param[in, out] loc - HDF5 group, or dataset
/// @param[in] name - attribute identifier
/// @param[in] value - attribute value
void write_string_attribute(H5Object& obj,
                            const std::string& name,
                            const std::string& value) {
    DataSpace attr_space = DataSpace();
    StrType attr_type(PredType::C_S1, value.size());

    attr_type.setCset(H5T_CSET_UTF8);
    Attribute attr(obj.createAttribute(name, attr_type, attr_space));
    attr.write(attr_type, value.data());
}


/// Create a dataset from an Eigen object.
///
/// All objects are assumed to be 2D, and their contents
/// are assumed to be double-precission floating point numbers.
/// @param[in,out] loc - the HDF5 file or group
/// @param[in] data - the Eigen object
/// @param[in] name - the name of the dataset
/// @param[in] units - the units of the data
template <typename H, typename T>
void create_eigen_dataset(H& loc,
                          const Eigen::DenseBase<T>& data,
                          const std::string& name,
                          const std::string& units) {
    hsize_t rows = static_cast<hsize_t>(data.rows());
    hsize_t cols = static_cast<hsize_t>(data.cols());
    hsize_t dims[] = {rows, cols};
    DataSpace dspace(2, dims);
    DataSpace scalar;
    DataSet dset = loc.createDataSet(name, PredType::IEEE_F64LE, dspace);
    hsize_t pos[] = {1, 1};

    // Write the elements one by one to avoid storage order issues.
    for (decltype(rows) i = 0; i < rows; ++i) {
        pos[0] = i;

        for (decltype(cols) j = 0; j < cols; ++j) {
            pos[1] = j;
            dspace.selectElements(H5S_SELECT_SET, 1, pos);
            double value = data(i, j);
            dset.write(&value, PredType::NATIVE_DOUBLE, scalar, dspace);
        }
    }
    write_string_attribute(dset, "Units", units);
}


/// Write some metadata to the HDF5 file as attributes.
///
/// @param[in,out] file - the HDF5 file
/// @param[in] description - a free-format description
void write_file_attributes(H5File& file, const std::string& description) {
    Group root = file.openGroup("/");
    // Creation timestamp.
    write_string_attribute(root, "timestamp", get_timestamp());
    // File version.
    unsigned int version = BULK_HDF5_MAJOR_VERSION;
    DataSpace version_space = DataSpace();
    Attribute version_attr(file.createAttribute(
        "major_version", PredType::STD_U16LE, version_space));
    version_attr.write(PredType::NATIVE_UINT, &version);
    version = BULK_HDF5_MINOR_VERSION;
    version_attr = file.createAttribute(
        "minor_version", PredType::STD_U16LE, version_space);
    version_attr.write(PredType::NATIVE_UINT, &version);
    // Free-format description.
    write_string_attribute(root, "description", description);
}


/// Create some predefined groups in the HDF5 file.
///
/// @param[in,out] file - the hdf5 file
void create_groups(H5File& file) {
    file.createGroup("/crystal_structure");
    file.createGroup("/symmetry_operations");
    file.createGroup("/qpoint_grid");
    file.createGroup("/threeph_processes");
    file.createGroup("/fourph_processes");
    file.createGroup("/scattering");
    file.createGroup("/ifcs");
}


void save_bulk_hdf5(const char* filename,
                    const std::string& description,
                    const Crystal_structure& cell,
                    const Symmetry_operations& symmetries,
                    const Gamma_grid& grid,
                    const std::vector<Threeph_process>& processes_3ph,
                    const std::vector<Fourph_process>&  processes_4ph,
                    const boost::mpi::communicator& comm) {
    auto my_id = comm.rank();
    auto my_nprocs_3ph = processes_3ph.size();
    auto my_nprocs_4ph = processes_4ph.size();

    decltype(my_nprocs_3ph) nprocs_3ph, nprocs_4ph;
    boost::mpi::reduce(
        comm, my_nprocs_3ph, nprocs_3ph, std::plus<decltype(my_nprocs_3ph)>(), 0);
    boost::mpi::all_reduce(
        comm, my_nprocs_4ph, nprocs_4ph, std::plus<decltype(my_nprocs_4ph)>());

    // Only process 0 will handle the actual output.
    if (my_id == 0) {
        // This part of the function is rather long, but also very
        // repetitive. Each DataSet requires specifying its data space,
        // data type and so on and so forth.
        // Create or overwrite the HDF5 file.
        H5File h5f(filename, H5F_ACC_TRUNC);
        // Global attributes.
        write_file_attributes(h5f, description);
        // Create some groups in the file.
        create_groups(h5f);
        // Lattice vectors.
        create_eigen_dataset(
            h5f, cell.lattvec, "/crystal_structure/lattvec", "nm");
        // Atomic positions.
        create_eigen_dataset(h5f,
                             cell.positions,
                             "/crystal_structure/positions",
                             "lattice coordinates");
        // Elements.
        auto nelements = cell.elements.size();
        hsize_t elements_dims[] = {nelements};
        DataSpace elements_dspace(1, elements_dims);
        DataSpace scalar;
        auto longest_el =
            std::max_element(cell.elements.begin(),
                             cell.elements.end(),
                             [](const std::string& s1, const std::string& s2) {
                                 return s1.size() < s2.size();
                             });
        StrType elements_type(PredType::C_S1, longest_el->size());
        elements_type.setCset(H5T_CSET_UTF8);
        DataSet elements_dset = h5f.createDataSet(
            "/crystal_structure/elements", elements_type, elements_dspace);

        for (hsize_t i = 0; i < nelements; ++i) {
            elements_dspace.selectElements(H5S_SELECT_SET, 1, &i);
            elements_dset.write(
                cell.elements[i], elements_type, scalar, elements_dspace);
        }
        // Number of atoms belonging to each element.
        DataSet numbers_dset = h5f.createDataSet(
            "/crystal_structure/numbers", PredType::STD_I32LE, elements_dspace);
        elements_dspace.selectAll();
        numbers_dset.write(cell.numbers.data(),
                           PredType::NATIVE_INT,
                           elements_dspace,
                           elements_dspace);
        // Tolerance for symmetry search.
        DataSet symprec_dset = h5f.createDataSet(
            "/symmetry_operations/symprec", PredType::IEEE_F64LE, scalar);
        symprec_dset.write(
            &(symmetries.symprec), PredType::NATIVE_DOUBLE, scalar, scalar);
        // Dimensions of the q-point grid.
        int nq[] = {grid.na, grid.nb, grid.nc};
        hsize_t nq_dims[] = {3};
        DataSpace nq_dspace(1, nq_dims);
        DataSet nq_dset = h5f.createDataSet(
            "/qpoint_grid/nq", PredType::STD_I32LE, nq_dspace);
        nq_dspace.selectAll();
        nq_dset.write(nq, PredType::NATIVE_INT, nq_dspace, nq_dspace);
        // Harmonic properties at each q point.
        // Frequencies.
        hsize_t nmodes =
            static_cast<hsize_t>(grid.get_spectrum_at_q(0).omega.size());
        hsize_t omega_dims[] = {grid.nqpoints, nmodes};
        DataSpace omega_dspace(2, omega_dims);
        DataSet omega_dset = h5f.createDataSet(
            "/qpoint_grid/omega", PredType::IEEE_F64LE, omega_dspace);
        hsize_t omega_dims_m[] = {nmodes};
        DataSpace omega_dspace_m(1, omega_dims_m);
        hsize_t omega_start[] = {0, 0};
        hsize_t omega_count[] = {1, nmodes};
        omega_dspace_m.selectAll();

        for (hsize_t i = 0; i < grid.nqpoints; ++i) {
            omega_start[0] = i;
            omega_dspace.selectHyperslab(
                H5S_SELECT_SET, omega_count, omega_start);
            omega_dset.write(grid.get_spectrum_at_q(i).omega.data(),
                             PredType::NATIVE_DOUBLE,
                             omega_dspace_m,
                             omega_dspace);
        }
        write_string_attribute(omega_dset, "Units", "rad / ps");
        // Real and imaginary parts of the wave functions.
        hsize_t wf_dims[] = {grid.nqpoints, nmodes, nmodes};
        DataSpace wf_dspace(3, wf_dims);
        DataSet wf_re_dset = h5f.createDataSet(
            "/qpoint_grid/re_wf", PredType::IEEE_F64LE, wf_dspace);
        DataSet wf_im_dset = h5f.createDataSet(
            "/qpoint_grid/im_wf", PredType::IEEE_F64LE, wf_dspace);
        hsize_t wf_pos[] = {0, 0, 0};

        for (hsize_t i = 0; i < grid.nqpoints; ++i) {
            wf_pos[0] = i;

            for (hsize_t j1 = 0; j1 < nmodes; ++j1) {
                wf_pos[1] = j1;

                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {
                    wf_pos[2] = j2;
                    wf_dspace.selectElements(H5S_SELECT_SET, 1, wf_pos);
                    auto value = grid.get_spectrum_at_q(i).wfs(j1, j2);
                    double re = value.real();
                    double im = value.imag();
                    wf_re_dset.write(
                        &re, PredType::NATIVE_DOUBLE, scalar, wf_dspace);
                    wf_im_dset.write(
                        &im, PredType::NATIVE_DOUBLE, scalar, wf_dspace);
                }
            }
        }
        // Cartesian components of the group velocities.
        hsize_t vg_dims[] = {grid.nqpoints, 3, nmodes};
        DataSpace vg_dspace(3, vg_dims);
        DataSet vg_dset = h5f.createDataSet(
            "/qpoint_grid/vg", PredType::IEEE_F64LE, vg_dspace);
        hsize_t vg_pos[] = {0, 0, 0};

        for (hsize_t i = 0; i < grid.nqpoints; ++i) {
            vg_pos[0] = i;

            for (hsize_t j1 = 0; j1 < 3; ++j1) {
                vg_pos[1] = j1;

                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {
                    vg_pos[2] = j2;
                    vg_dspace.selectElements(H5S_SELECT_SET, 1, vg_pos);
                    double value = grid.get_spectrum_at_q(i).vg(j1, j2);
                    vg_dset.write(
                        &value, PredType::NATIVE_DOUBLE, scalar, vg_dspace);
                }
            }
        }
        write_string_attribute(vg_dset, "Units", "nm / ps");

        // Cartesian components of the Wigner matrix velocities.
        hsize_t wigner_v_dims[] = {grid.nqpoints, 3, nmodes*nmodes};
        DataSpace wigner_v_dspace(3, wigner_v_dims);
        DataSet re_wigner_v_dset = h5f.createDataSet(
            "/qpoint_grid/re_wigner_v", PredType::IEEE_F64LE, wigner_v_dspace);
        DataSet im_wigner_v_dset = h5f.createDataSet(
            "/qpoint_grid/im_wigner_v", PredType::IEEE_F64LE, wigner_v_dspace);
        hsize_t wigner_v_pos[] = {0, 0, 0};

        for (hsize_t i = 0; i < grid.nqpoints; ++i) {
            wigner_v_pos[0] = i;

            for (hsize_t j1 = 0; j1 < 3; ++j1) {
                wigner_v_pos[1] = j1;

                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {

                        for (hsize_t j3 = 0; j3 < nmodes; ++j3) {

                            wigner_v_pos[2] = j2+nmodes*j3;

                            wigner_v_dspace.selectElements(H5S_SELECT_SET, 1, wigner_v_pos);
                            auto value =
                                grid.get_spectrum_at_q(i).wigner_v(j1, j2+nmodes*j3);
                            double re = value.real();
                            double im = value.imag();
                            re_wigner_v_dset.write(
                                &re, PredType::NATIVE_DOUBLE, scalar, wigner_v_dspace);
                            im_wigner_v_dset.write(
                                &im, PredType::NATIVE_DOUBLE, scalar, wigner_v_dspace);

                    }
                }
            }
        }
        write_string_attribute(re_wigner_v_dset, "Units", "nm / ps");
        write_string_attribute(im_wigner_v_dset, "Units", "nm / ps");
        // Allowed three-phonon processes. See processes.hpp for
        // details.
        hsize_t c_dims[] = {nprocs_3ph};
        DataSpace c_dspace(1, c_dims);
        DataSet c_dset = h5f.createDataSet(
            "/threeph_processes/class", PredType::STD_U64LE, c_dspace);
        hsize_t q_dims[] = {nprocs_3ph, 3};
        DataSpace q_dspace(2, q_dims);
        DataSet q_dset = h5f.createDataSet(
            "/threeph_processes/q", PredType::STD_U64LE, q_dspace);
        DataSet alpha_dset = h5f.createDataSet(
            "/threeph_processes/alpha", PredType::STD_U64LE, q_dspace);
        // This dataset summarized the data in the "type",
        // "gaussian_computed" and "vp2_computed" attributes.
        // It is built as a bitmask with bits 0, 1 and 2 selecting
        // between the two possible values of these attributes,
        // respectively.
        DataSet type_dset = h5f.createDataSet(
            "/threeph_processes/type", PredType::STD_U8LE, c_dspace);
        DataSet domega_dset = h5f.createDataSet(
            "/threeph_processes/domega", PredType::IEEE_F64LE, c_dspace);
        DataSet sigma_dset = h5f.createDataSet(
            "/threeph_processes/sigma", PredType::IEEE_F64LE, c_dspace);
        DataSet gaussian_dset = h5f.createDataSet(
            "/threeph_processes/gaussian", PredType::IEEE_F64LE, c_dspace);
        DataSet vp2_dset = h5f.createDataSet(
            "/threeph_processes/vp2", PredType::IEEE_F64LE, c_dspace);
        write_string_attribute(domega_dset, "Units", "rad / ps");
        write_string_attribute(sigma_dset, "Units", "rad / ps");
        write_string_attribute(gaussian_dset, "Units", "ps / rad");
        write_string_attribute(vp2_dset, "Units", "THz**4 / (kg nm**2)");
        hsize_t q_pos[] = {0, 0};
        hsize_t q_count[] = {1, 3};
        hsize_t q_dims_m[] = {3};
        DataSpace q_dspace_m(1, q_dims_m);
        q_dspace_m.selectAll();

        // Begin by three-phonon processes identified by this
        // MPI process.
        for (hsize_t i = 0; i < my_nprocs_3ph; ++i) {
            c_dspace.selectElements(H5S_SELECT_SET, 1, &i);
            q_pos[0] = i;
            q_dspace.selectHyperslab(H5S_SELECT_SET, q_count, q_pos);
            c_dset.write(
                &(processes_3ph[i].c), PredType::NATIVE_SIZE_T, scalar, c_dspace);
            q_dset.write(processes_3ph[i].q.data(),
                         PredType::NATIVE_SIZE_T,
                         q_dspace_m,
                         q_dspace);
            alpha_dset.write(processes_3ph[i].alpha.data(),
                             PredType::NATIVE_SIZE_T,
                             q_dspace_m,
                             q_dspace);
            std::bitset<8> mask(0u);

            if (processes_3ph[i].type == threeph_type::absorption)
                mask.set(ABSORPTION_BIT);

            if (processes_3ph[i].gaussian_computed)
                mask.set(GAUSSIAN_BIT);

            if (processes_3ph[i].vp2_computed)
                mask.set(VP2_BIT);
            uint8_t short_mask = static_cast<uint8_t>(mask.to_ulong());
            type_dset.write(
                &short_mask, PredType::NATIVE_UINT8, scalar, c_dspace);
            domega_dset.write(&(processes_3ph[i].domega),
                              PredType::NATIVE_DOUBLE,
                              scalar,
                              c_dspace);
            sigma_dset.write(&(processes_3ph[i].sigma),
                             PredType::NATIVE_DOUBLE,
                             scalar,
                             c_dspace);
            gaussian_dset.write(&(processes_3ph[i].gaussian),
                                PredType::NATIVE_DOUBLE,
                                scalar,
                                c_dspace);
            vp2_dset.write(
                &(processes_3ph[i].vp2), PredType::NATIVE_DOUBLE, scalar, c_dspace);
        }
        // And proceed with the remaining MPI processes.
        std::vector<Threeph_process> other;

        for (auto ip = 1; ip < comm.size(); ++ip) {
            other.clear();
            comm.recv(ip, 0, other);
            auto other_nproc = other.size();

            for (hsize_t i = 0; i < other_nproc; ++i) {
                ++q_pos[0];
                c_dspace.selectElements(H5S_SELECT_SET, 1, &(q_pos[0]));
                q_dspace.selectHyperslab(H5S_SELECT_SET, q_count, q_pos);
                c_dset.write(
                    &(other[i].c), PredType::NATIVE_SIZE_T, scalar, c_dspace);
                q_dset.write(other[i].q.data(),
                             PredType::NATIVE_SIZE_T,
                             q_dspace_m,
                             q_dspace);
                alpha_dset.write(other[i].alpha.data(),
                                 PredType::NATIVE_SIZE_T,
                                 q_dspace_m,
                                 q_dspace);
                std::bitset<8> mask(0u);

                if (other[i].type == threeph_type::absorption)
                    mask.set(ABSORPTION_BIT);

                if (other[i].gaussian_computed)
                    mask.set(GAUSSIAN_BIT);

                if (other[i].vp2_computed)
                    mask.set(VP2_BIT);
                uint8_t short_mask = static_cast<uint8_t>(mask.to_ulong());
                type_dset.write(
                    &short_mask, PredType::NATIVE_UINT8, scalar, c_dspace);
                domega_dset.write(&(other[i].domega),
                                  PredType::NATIVE_DOUBLE,
                                  scalar,
                                  c_dspace);
                sigma_dset.write(&(other[i].sigma),
                                 PredType::NATIVE_DOUBLE,
                                 scalar,
                                 c_dspace);
                gaussian_dset.write(&(other[i].gaussian),
                                    PredType::NATIVE_DOUBLE,
                                    scalar,
                                    c_dspace);
                vp2_dset.write(
                    &(other[i].vp2), PredType::NATIVE_DOUBLE, scalar, c_dspace);
            }
        }


        // Allowed four-phonon processes. See processes.hpp for
        // details.
        {
            hsize_t c_dims[] = {nprocs_4ph};
            DataSpace c_dspace(1, c_dims);
            DataSet c_dset = h5f.createDataSet(
                "/fourph_processes/class", PredType::STD_U64LE, c_dspace);
            hsize_t q_dims[] = {nprocs_4ph, 4};
            DataSpace q_dspace(2, q_dims);
            DataSet q_dset = h5f.createDataSet(
                "/fourph_processes/q", PredType::STD_U64LE, q_dspace);
            DataSet alpha_dset = h5f.createDataSet(
                "/fourph_processes/alpha", PredType::STD_U64LE, q_dspace);
            // This dataset summarized the data in the "type",
            // "gaussian_computed" and "vp2_computed" attributes.
            // It is built as a bitmask with bits 0, 1 and 2 selecting
            // between the two possible values of these attributes,
            // respectively.
            DataSet type_dset = h5f.createDataSet(
                "/fourph_processes/type", PredType::STD_U8LE, c_dspace);
            DataSet domega_dset = h5f.createDataSet(
                "/fourph_processes/domega", PredType::IEEE_F64LE, c_dspace);
            DataSet sigma_dset = h5f.createDataSet(
                "/fourph_processes/sigma", PredType::IEEE_F64LE, c_dspace);
            DataSet gaussian_dset = h5f.createDataSet(
                "/fourph_processes/gaussian", PredType::IEEE_F64LE, c_dspace);
            DataSet vp2_dset = h5f.createDataSet(
                "/fourph_processes/vp2", PredType::IEEE_F64LE, c_dspace);
            write_string_attribute(domega_dset, "Units", "rad / ps");
            write_string_attribute(sigma_dset, "Units", "rad / ps");
            write_string_attribute(gaussian_dset, "Units", "ps / rad");

            write_string_attribute(vp2_dset, "Units", "THz**4 / (kg**4 nm**4)");
            hsize_t q_pos[] = {0, 0};
            hsize_t q_count[] = {1, 4};
            hsize_t q_dims_m[] = {4};
            DataSpace q_dspace_m(1, q_dims_m);
            q_dspace_m.selectAll();

            // Begin by three-phonon processes identified by this
            // MPI process.
            for (hsize_t i = 0; i < my_nprocs_4ph; ++i) {
                c_dspace.selectElements(H5S_SELECT_SET, 1, &i);
                q_pos[0] = i;
                q_dspace.selectHyperslab(H5S_SELECT_SET, q_count, q_pos);
                c_dset.write(
                    &(processes_4ph[i].c), PredType::NATIVE_SIZE_T, scalar, c_dspace);
                q_dset.write(processes_4ph[i].q.data(),
                            PredType::NATIVE_SIZE_T,
                            q_dspace_m,
                            q_dspace);
                alpha_dset.write(processes_4ph[i].alpha.data(),
                                PredType::NATIVE_SIZE_T,
                                q_dspace_m,
                                q_dspace);
                                
                std::bitset<8> mask(0u);

                if (processes_4ph[i].type == fourph_type::recombination) {
                    mask.set(RECOMBINATION_BIT);
                } else if (processes_4ph[i].type == fourph_type::redistribution) {
                    mask.set(REDISTRIBUTION_BIT);
                } else {
                    mask.set(SPLITTING_BIT);
                }

                if (processes_4ph[i].gaussian_computed)
                    mask.set(GAUSSIAN_BIT);

                if (processes_4ph[i].vp2_computed)
                    mask.set(VP2_BIT);

                uint8_t short_mask = static_cast<uint8_t>(mask.to_ulong());
                type_dset.write(
                    &short_mask, PredType::NATIVE_UINT8, scalar, c_dspace);
                domega_dset.write(&(processes_4ph[i].domega),
                                PredType::NATIVE_DOUBLE,
                                scalar,
                                c_dspace);
                sigma_dset.write(&(processes_4ph[i].sigma),
                                PredType::NATIVE_DOUBLE,
                                scalar,
                                c_dspace);
                gaussian_dset.write(&(processes_4ph[i].gaussian),
                                    PredType::NATIVE_DOUBLE,
                                    scalar,
                                    c_dspace);
                vp2_dset.write(
                    &(processes_4ph[i].vp2), PredType::NATIVE_DOUBLE, scalar, c_dspace);
            }
            // And proceed with the remaining MPI processes.
            std::vector<Fourph_process> other;

            for (auto ip = 1; ip < comm.size(); ++ip) {
                other.clear();
                comm.recv(ip, 0, other);
                auto other_nproc = other.size();

                for (hsize_t i = 0; i < other_nproc; ++i) {
                    ++q_pos[0];
                    c_dspace.selectElements(H5S_SELECT_SET, 1, &(q_pos[0]));
                    q_dspace.selectHyperslab(H5S_SELECT_SET, q_count, q_pos);
                    c_dset.write(
                        &(other[i].c), PredType::NATIVE_SIZE_T, scalar, c_dspace);
                    q_dset.write(other[i].q.data(),
                                PredType::NATIVE_SIZE_T,
                                q_dspace_m,
                                q_dspace);
                    alpha_dset.write(other[i].alpha.data(),
                                    PredType::NATIVE_SIZE_T,
                                    q_dspace_m,
                                    q_dspace);

                    std::bitset<8> mask(0u);

                    if (processes_4ph[i].type == fourph_type::recombination) {
                        mask.set(RECOMBINATION_BIT);
                    } else if (processes_4ph[i].type == fourph_type::redistribution) {
                        mask.set(REDISTRIBUTION_BIT);
                    } else {
                        mask.set(SPLITTING_BIT);
                    }


                    if (other[i].gaussian_computed)
                        mask.set(GAUSSIAN_BIT);

                    if (other[i].vp2_computed)
                        mask.set(VP2_BIT);
                    uint8_t short_mask = static_cast<uint8_t>(mask.to_ulong());
                    type_dset.write(
                        &short_mask, PredType::NATIVE_UINT8, scalar, c_dspace);
                    domega_dset.write(&(other[i].domega),
                                    PredType::NATIVE_DOUBLE,
                                    scalar,
                                    c_dspace);
                    sigma_dset.write(&(other[i].sigma),
                                    PredType::NATIVE_DOUBLE,
                                    scalar,
                                    c_dspace);
                    gaussian_dset.write(&(other[i].gaussian),
                                        PredType::NATIVE_DOUBLE,
                                        scalar,
                                        c_dspace);
                    vp2_dset.write(
                        &(other[i].vp2), PredType::NATIVE_DOUBLE, scalar, c_dspace);
                }
            }
        }
    }
    else {
        // The remaining MPI processes just need to send their
        // data to process 0.
        comm.send(0, 0, processes_3ph);

        // For four processes do something in case one has 4ph
        // vector populated
        if (nprocs_4ph > 0) comm.send(0, 0, processes_4ph);
    }   
}


/// Read an HDF5 file's metadata and check if the
/// versions are compatible.
///
/// @param[in] file - the HDF5 file
/// @return the description stored in the file
std::string read_file_attributes(H5File& file) {
    unsigned int version;
    Attribute version_attr = file.openAttribute("major_version");

    version_attr.read(PredType::NATIVE_UINT, &version);

    if (version != BULK_HDF5_MAJOR_VERSION)
        throw input_error("wrong major version read from HDF5 file");
    Attribute description_attr = file.openAttribute("description");
    StrType description_type = description_attr.getStrType();
    std::string nruter;
    description_attr.read(description_type, nruter);
    return nruter;
}


std::tuple<std::string,
           std::unique_ptr<Crystal_structure>,
           std::unique_ptr<Symmetry_operations>,
           std::unique_ptr<Gamma_grid>,
           std::unique_ptr<std::vector<Threeph_process>>,
           std::unique_ptr<std::vector<Fourph_process>>>
load_bulk_hdf5(const char* filename, const boost::mpi::communicator& comm) {
    auto my_id = comm.rank();

    std::unique_ptr<H5File> h5f;
    std::string description;

    if (my_id == 0) {
        h5f = alma::make_unique<H5File>(filename, H5F_ACC_RDONLY);
        description = read_file_attributes(*h5f);
    }
    // Each piece of data read is broadcast to all
    // processes in this communicator.
    DataSpace scalar;
    // Lattice vectors.
    Eigen::MatrixXd lattvec(3, 3);

    if (my_id == 0) {
        DataSet lattvec_dset = h5f->openDataSet("/crystal_structure/lattvec");
        DataSpace lattvec_dspace = lattvec_dset.getSpace();
        hsize_t pos[2];
        double value;

        for (hsize_t i = 0; i < 3; ++i) {
            pos[0] = i;

            for (hsize_t j = 0; j < 3; ++j) {
                pos[1] = j;
                lattvec_dspace.selectElements(H5S_SELECT_SET, 1, pos);
                lattvec_dset.read(
                    &value, PredType::NATIVE_DOUBLE, scalar, lattvec_dspace);
                lattvec(i, j) = value;
            }
        }
    }
    boost::mpi::broadcast(comm, lattvec.data(), 9, 0);
    boost::mpi::broadcast(comm, description, 0);
    // Atomic positions.
    hsize_t natoms;
    Eigen::MatrixXd positions;

    if (my_id == 0) {
        DataSet pos_dset = h5f->openDataSet("/crystal_structure/positions");
        DataSpace pos_dspace = pos_dset.getSpace();
        hsize_t pos_size[2];
        pos_dspace.getSimpleExtentDims(pos_size, nullptr);
        natoms = pos_size[1];
        positions.resize(pos_size[0], pos_size[1]);
        hsize_t pos_pos[2];
        double value;

        for (hsize_t i = 0; i < 3; ++i) {
            pos_pos[0] = i;

            for (hsize_t j = 0; j < natoms; ++j) {
                pos_pos[1] = j;
                pos_dspace.selectElements(H5S_SELECT_SET, 1, pos_pos);
                pos_dset.read(
                    &value, PredType::NATIVE_DOUBLE, scalar, pos_dspace);
                positions(i, j) = value;
            }
        }
    }
    boost::mpi::broadcast(comm, natoms, 0);

    if (my_id != 0)
        positions.resize(3, natoms);
    boost::mpi::broadcast(comm, positions.data(), 3 * natoms, 0);
    // Elements.
    std::vector<std::string> elements;

    if (my_id == 0) {
        DataSet elements_dset = h5f->openDataSet("/crystal_structure/elements");
        StrType elements_type = elements_dset.getStrType();
        DataSpace elements_dspace = elements_dset.getSpace();
        hsize_t nelements;
        elements_dspace.getSimpleExtentDims(&nelements, nullptr);
        std::string value;

        for (hsize_t i = 0; i < nelements; ++i) {
            elements_dspace.selectElements(H5S_SELECT_SET, 1, &i);
            elements_dset.read(value, elements_type, scalar, elements_dspace);
	    value.erase(std::remove_if(value.begin(), value.end(), [](char c) {
                       return  (c == '\t') or (c == '\n') or (c == '\r') or
		               (c == '\0') or (c < 32 || c > 126);
                     }), value.end());
            elements.push_back(value);
        }
    }
    boost::mpi::broadcast(comm, elements, 0);
    // Number of atoms belonging to each element.
    std::vector<int> numbers(elements.size());

    if (my_id == 0) {
        DataSet numbers_dset = h5f->openDataSet("/crystal_structure/numbers");
        DataSpace numbers_dspace = numbers_dset.getSpace();
        numbers_dset.read(numbers.data(),
                          PredType::NATIVE_INT,
                          numbers_dspace,
                          numbers_dspace);
    }
    boost::mpi::broadcast(comm, numbers, 0);
    // Build a Crystal_structure object.
    auto nruter1 =
        alma::make_unique<Crystal_structure>(lattvec, positions, elements, numbers);
    // Tolerance for symmetry search.
    double symprec;

    if (my_id == 0) {
        DataSet symprec_dset = h5f->openDataSet("/symmetry_operations/symprec");
        symprec_dset.read(&symprec, PredType::NATIVE_DOUBLE, scalar, scalar);
    }
    boost::mpi::broadcast(comm, symprec, 0);
    // Build a Symmetry_operations object.
    auto nruter2 = alma::make_unique<Symmetry_operations>(*nruter1, symprec);
    // Dimensions of the q-point grid.
    int nq[3];

    if (my_id == 0) {
        DataSet nq_dset = h5f->openDataSet("/qpoint_grid/nq");
        DataSpace nq_dspace = nq_dset.getSpace();
        nq_dspace.selectAll();
        nq_dset.read(nq, PredType::NATIVE_INT, nq_dspace, nq_dspace);
    }
    boost::mpi::broadcast(comm, nq, 3, 0);
    // Build a stub Gamma_grid object, to be filled in the
    // next section
    auto nruter3 =
        alma::make_unique<Gamma_grid>(*nruter1, *nruter2, nq[0], nq[1], nq[2]);
    // Harmonic properties at each q point.
    hsize_t nqpoints;
    hsize_t nmodes;

    if (my_id == 0) {
        DataSet omega_dset = h5f->openDataSet("/qpoint_grid/omega");
        DataSpace omega_dspace = omega_dset.getSpace();
        hsize_t omega_dims[2];
        omega_dspace.getSimpleExtentDims(omega_dims, nullptr);
        nqpoints = omega_dims[0];
        nmodes = omega_dims[1];
        boost::mpi::broadcast(comm, nqpoints, 0);
        boost::mpi::broadcast(comm, nmodes, 0);
        DataSet vg_dset = h5f->openDataSet("/qpoint_grid/vg");
        DataSpace vg_dspace = vg_dset.getSpace();
        DataSet wf_re_dset = h5f->openDataSet("/qpoint_grid/re_wf");
        DataSpace wf_dspace = wf_re_dset.getSpace();
        DataSet wf_im_dset = h5f->openDataSet("/qpoint_grid/im_wf");
        DataSet re_wigner_v_dset = h5f->openDataSet("/qpoint_grid/re_wigner_v");
        DataSpace wigner_v_dspace = re_wigner_v_dset.getSpace();
        DataSet im_wigner_v_dset = h5f->openDataSet("/qpoint_grid/im_wigner_v");
        Eigen::ArrayXd omega(nmodes);
        Eigen::MatrixXcd wfs(nmodes, nmodes);
        Eigen::ArrayXXd vg(3, nmodes);
        Eigen::ArrayXXcd wigner_v(3, nmodes*nmodes);
        hsize_t omega_start[2];
        hsize_t omega_count[2];
        hsize_t vg_pos[3];
        hsize_t wf_pos[3];
        hsize_t wigner_v_pos[3];
        double value;
        double re;
        double im;
        omega_start[1] = 0;
        omega_count[0] = 1;
        omega_count[1] = nmodes;
        hsize_t omega_dims_m[] = {nmodes};
        DataSpace omega_dspace_m(1, omega_dims_m);
        omega_dspace_m.selectAll();

        for (hsize_t i = 0; i < nqpoints; ++i) {
            omega_start[0] = i;
            vg_pos[0] = i;
            wigner_v_pos[0] = i;
            wf_pos[0] = i;
            omega_dspace.selectHyperslab(
                H5S_SELECT_SET, omega_count, omega_start);
            omega_dset.read(omega.data(),
                            PredType::NATIVE_DOUBLE,
                            omega_dspace_m,
                            omega_dspace);
            boost::mpi::broadcast(comm, omega.data(), omega.size(), 0);

            for (hsize_t j1 = 0; j1 < nmodes; ++j1) {
                wf_pos[1] = j1;

                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {
                    wf_pos[2] = j2;
                    wf_dspace.selectElements(H5S_SELECT_SET, 1, wf_pos);
                    wf_re_dset.read(
                        &re, PredType::NATIVE_DOUBLE, scalar, wf_dspace);
                    wf_im_dset.read(
                        &im, PredType::NATIVE_DOUBLE, scalar, wf_dspace);
                    boost::mpi::broadcast(comm, re, 0);
                    boost::mpi::broadcast(comm, im, 0);
                    wfs(j1, j2) = std::complex<double>(re, im);
                }
            }

            for (hsize_t j1 = 0; j1 < 3; ++j1) {
                vg_pos[1] = j1;
                wigner_v_pos[1] = j1;

                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {
                    vg_pos[2] = j2;
                    vg_dspace.selectElements(H5S_SELECT_SET, 1, vg_pos);
                    vg_dset.read(
                        &value, PredType::NATIVE_DOUBLE, scalar, vg_dspace);
                    boost::mpi::broadcast(comm, value, 0);
                    vg(j1, j2) = value;

                    for (hsize_t j3 = 0; j3 < nmodes; ++j3) {
                        wigner_v_pos[2] = j2+nmodes*j3;
                        wigner_v_dspace.selectElements(H5S_SELECT_SET, 1, wigner_v_pos);
                        re_wigner_v_dset.read(
                            &re, PredType::NATIVE_DOUBLE, scalar, wigner_v_dspace);
                        im_wigner_v_dset.read(
                            &im, PredType::NATIVE_DOUBLE, scalar, wigner_v_dspace);
                        boost::mpi::broadcast(comm, re, 0);
                        boost::mpi::broadcast(comm, im, 0);
                        wigner_v(j1, j2+nmodes*j3) = std::complex<double>(re, im);
                    }
                }
            }
            nruter3->spectrum.emplace_back(omega, wfs, vg, wigner_v);
        }
    }
    else {
        boost::mpi::broadcast(comm, nqpoints, 0);
        boost::mpi::broadcast(comm, nmodes, 0);
        Eigen::ArrayXd omega(nmodes);
        Eigen::MatrixXcd wfs(nmodes, nmodes);
        Eigen::ArrayXXd vg(3, nmodes);
        Eigen::ArrayXXcd wigner_v(3, nmodes*nmodes);
        double value;
        double re;
        double im;

        for (hsize_t i = 0; i < nqpoints; ++i) {
            boost::mpi::broadcast(comm, omega.data(), omega.size(), 0);

            for (hsize_t j1 = 0; j1 < nmodes; ++j1) {
                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {
                    boost::mpi::broadcast(comm, re, 0);
                    boost::mpi::broadcast(comm, im, 0);
                    wfs(j1, j2) = std::complex<double>(re, im);
                }
            }

            for (hsize_t j1 = 0; j1 < 3; ++j1) {
                for (hsize_t j2 = 0; j2 < nmodes; ++j2) {
                    boost::mpi::broadcast(comm, value, 0);
                    vg(j1, j2) = value;
                    for (hsize_t j3 = 0; j3 < nmodes; ++j3) {
                        boost::mpi::broadcast(comm, re, 0);
                        boost::mpi::broadcast(comm, im, 0);
                        wigner_v(j1, j2+nmodes*j3) = std::complex<double>(re, im);
                    }
                }
            }
            nruter3->spectrum.emplace_back(omega, wfs, vg, wigner_v);
        }
    }
    // Create an empty vector of Threeph_process objects.
    auto nruter4 = alma::make_unique<std::vector<Threeph_process>>();

    if (my_id == 0) {
        DataSet c_dset = h5f->openDataSet("/threeph_processes/class");
        DataSet q_dset = h5f->openDataSet("/threeph_processes/q");
        DataSet alpha_dset = h5f->openDataSet("/threeph_processes/alpha");
        DataSet type_dset = h5f->openDataSet("/threeph_processes/type");
        DataSet domega_dset = h5f->openDataSet("/threeph_processes/domega");
        DataSet sigma_dset = h5f->openDataSet("/threeph_processes/sigma");
        DataSet gaussian_dset = h5f->openDataSet("/threeph_processes/gaussian");
        DataSet vp2_dset = h5f->openDataSet("/threeph_processes/vp2");
        DataSpace c_dspace = c_dset.getSpace();
        DataSpace q_dspace = q_dset.getSpace();
        // Read the total number of three-phonon processes
        // stored in the file.
        hsize_t nprocs;
        c_dspace.getSimpleExtentDims(&nprocs, nullptr);
        auto nmpi = comm.size();
        // Read the first few for process 0 and send the rest
        // to the other MPI processes.
        hsize_t q_pos[] = {0, 0};
        hsize_t q_count[] = {1, 3};
        hsize_t q_dims_m[] = {3};
        std::size_t c_value;
        std::array<std::size_t, 3> q_value;
        std::array<std::size_t, 3> alpha_value;
        uint8_t short_mask;
        double domega_value;
        double sigma_value;
        DataSpace q_dspace_m(1, q_dims_m);

        for (decltype(nmpi) impi = 0; impi < nmpi; ++impi) {
            std::vector<Threeph_process> other;
            auto limits = my_jobs(nprocs, nmpi, impi);

            if (impi != 0) {
                other.reserve(limits[1] - limits[0]);
                comm.send(impi, 0, limits[1] - limits[0]);
            }

            for (hsize_t i = limits[0]; i < limits[1]; ++i) {
                c_dspace.selectElements(H5S_SELECT_SET, 1, &i);
                q_pos[0] = i;
                q_dspace.selectHyperslab(H5S_SELECT_SET, q_count, q_pos);
                c_dset.read(
                    &c_value, PredType::NATIVE_SIZE_T, scalar, c_dspace);
                q_dset.read(q_value.data(),
                            PredType::NATIVE_SIZE_T,
                            q_dspace_m,
                            q_dspace);
                alpha_dset.read(alpha_value.data(),
                                PredType::NATIVE_SIZE_T,
                                q_dspace_m,
                                q_dspace);
                type_dset.read(
                    &short_mask, PredType::NATIVE_UINT8, scalar, c_dspace);
                std::bitset<8> mask(static_cast<unsigned long>(short_mask));
                domega_dset.read(
                    &domega_value, PredType::NATIVE_DOUBLE, scalar, c_dspace);
                sigma_dset.read(
                    &sigma_value, PredType::NATIVE_DOUBLE, scalar, c_dspace);
                threeph_type type_value =
                    (mask.test(ABSORPTION_BIT) ? threeph_type::absorption
                                               : threeph_type::emission);
                Threeph_process p(c_value,
                                  q_value,
                                  alpha_value,
                                  type_value,
                                  domega_value,
                                  sigma_value);

                if (mask.test(GAUSSIAN_BIT)) {
                    p.gaussian_computed = true;
                    gaussian_dset.read(&(p.gaussian),
                                       PredType::NATIVE_DOUBLE,
                                       scalar,
                                       c_dspace);
                }

                if (mask.test(VP2_BIT)) {
                    p.vp2_computed = true;
                    vp2_dset.read(
                        &(p.vp2), PredType::NATIVE_DOUBLE, scalar, c_dspace);
                }

                if (impi == 0)
                    nruter4->emplace_back(p);
                else
                    other.emplace_back(p);
            }

            if (impi != 0)
                comm.send(impi, 0, other);
        }
    }
    else {
        std::size_t toreserve;
        comm.recv(0, 0, toreserve);
        nruter4->reserve(toreserve);
        comm.recv(0, 0, *nruter4);
    }

    // Create an empty vector of Fourph_process objects.
    auto nruter5 = alma::make_unique<std::vector<Fourph_process>>();

    if (my_id == 0) {
        DataSet c_dset = h5f->openDataSet("/fourph_processes/class");
        DataSet q_dset = h5f->openDataSet("/fourph_processes/q");
        DataSet alpha_dset = h5f->openDataSet("/fourph_processes/alpha");
        DataSet type_dset = h5f->openDataSet("/fourph_processes/type");
        DataSet domega_dset = h5f->openDataSet("/fourph_processes/domega");
        DataSet sigma_dset = h5f->openDataSet("/fourph_processes/sigma");
        DataSet gaussian_dset = h5f->openDataSet("/fourph_processes/gaussian");
        DataSet vp2_dset = h5f->openDataSet("/fourph_processes/vp2");
        DataSpace c_dspace = c_dset.getSpace();
        DataSpace q_dspace = q_dset.getSpace();
        // Read the total number of three-phonon processes
        // stored in the file.
        hsize_t nprocs;
        c_dspace.getSimpleExtentDims(&nprocs, nullptr);
        auto nmpi = comm.size();
        // Read the first few for process 0 and send the rest
        // to the other MPI processes.
        hsize_t q_pos[] = {0, 0};
        hsize_t q_count[] = {1, 4};
        hsize_t q_dims_m[] = {4};
        std::size_t c_value;
        std::array<std::size_t, 4> q_value;
        std::array<std::size_t, 4> alpha_value;
        uint8_t short_mask;
        double domega_value;
        double sigma_value;
        DataSpace q_dspace_m(1, q_dims_m);

        for (decltype(nmpi) impi = 0; impi < nmpi; ++impi) {
            std::vector<Fourph_process> other;
            auto limits = my_jobs(nprocs, nmpi, impi);

            if (impi != 0) {
                other.reserve(limits[1] - limits[0]);
                comm.send(impi, 0, limits[1] - limits[0]);
            }

            for (hsize_t i = limits[0]; i < limits[1]; ++i) {
                c_dspace.selectElements(H5S_SELECT_SET, 1, &i);
                q_pos[0] = i;
                q_dspace.selectHyperslab(H5S_SELECT_SET, q_count, q_pos);
                c_dset.read(
                    &c_value, PredType::NATIVE_SIZE_T, scalar, c_dspace);
                q_dset.read(q_value.data(),
                            PredType::NATIVE_SIZE_T,
                            q_dspace_m,
                            q_dspace);
                alpha_dset.read(alpha_value.data(),
                                PredType::NATIVE_SIZE_T,
                                q_dspace_m,
                                q_dspace);
                type_dset.read(
                    &short_mask, PredType::NATIVE_UINT8, scalar, c_dspace);
                std::bitset<8> mask(static_cast<unsigned long>(short_mask));
                domega_dset.read(
                    &domega_value, PredType::NATIVE_DOUBLE, scalar, c_dspace);
                sigma_dset.read(
                    &sigma_value, PredType::NATIVE_DOUBLE, scalar, c_dspace);
                fourph_type type_value;
                
                if (mask.test(RECOMBINATION_BIT)) {
                    type_value = fourph_type::recombination;
                } else if (mask.test(REDISTRIBUTION_BIT)) {
                    type_value = fourph_type::redistribution;
                } else {
                    type_value = fourph_type::splitting;
                }


                
                Fourph_process p(c_value,
                                 q_value,
                                 alpha_value,
                                 type_value,
                                 domega_value,
                                 sigma_value);

                if (mask.test(GAUSSIAN_BIT)) {
                    p.gaussian_computed = true;
                    gaussian_dset.read(&(p.gaussian),
                                       PredType::NATIVE_DOUBLE,
                                       scalar,
                                       c_dspace);
                }

                if (mask.test(VP2_BIT)) {
                    p.vp2_computed = true;
                    vp2_dset.read(
                        &(p.vp2), PredType::NATIVE_DOUBLE, scalar, c_dspace);
                }

                if (impi == 0)
                    nruter5->emplace_back(p);
                else
                    other.emplace_back(p);
            }

            if (impi != 0)
                comm.send(impi, 0, other);
        }
    }
    else {
        std::size_t toreserve;
        comm.recv(0, 0, toreserve);
        nruter4->reserve(toreserve);
        comm.recv(0, 0, *nruter5);
    }


    // Return a tuple with all objects.
    return std::make_tuple(description,
                           std::move(nruter1),
                           std::move(nruter2),
                           std::move(nruter3),
                           std::move(nruter4),
                           std::move(nruter5));
}


std::vector<std::string> list_scattering_subgroups(
    const char* filename,
    const boost::mpi::communicator& comm) {
    auto my_id = comm.rank();

    std::unique_ptr<H5File> h5f;
    std::vector<std::string> nruter;

    // The file is only read from the process with id 0.
    if (my_id == 0) {
        h5f = alma::make_unique<H5File>(filename, H5F_ACC_RDONLY);
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
        h5f = alma::make_unique<H5File>(filename, H5F_ACC_RDONLY);
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
        h5f = alma::make_unique<H5File>(filename, H5F_ACC_RDWR);
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
                         Harmonic_ifcs& harmonic_ifcs,
                         const std::vector<Thirdorder_ifcs>& anharmonic_ifcs,
                         const boost::mpi::communicator& comm) {
    
    auto my_id = comm.rank();

    // Processes other than ID 0 do not do anything other than wait.
    if (my_id == 0) {
        std::unique_ptr<H5File> h5f;
        h5f = alma::make_unique<H5File>(filename, H5F_ACC_RDWR);
        // Check if the "/ifcs" group exists,
        // create it otherwise.
        Group ifcs;
        try {
            ifcs = h5f->openGroup("/ifcs");
        }
        catch (GroupIException e) {
            ifcs = h5f->createGroup("/ifcs");
        }

        /// Obtain the number of harmonic IFCs
        std::size_t n_harmonic_ifcs = harmonic_ifcs.ifcs.size();
        std::size_t ndof = harmonic_ifcs.ifcs[0].rows();

        /// Write the supercell used to compute the harmonic IFCs
        int n_supercell[] = {harmonic_ifcs.na, harmonic_ifcs.nb, harmonic_ifcs.nc};
        hsize_t nsupercell_dims[] = {3};
        DataSpace nsupercell_dspace(1, nsupercell_dims);
        DataSet nsupercell_dset = h5f->createDataSet(
            "/ifcs/harmonic/supercell", PredType::STD_I32LE, nsupercell_dspace);
        nsupercell_dspace.selectAll();
        nsupercell_dset.write(n_supercell, PredType::NATIVE_INT, nsupercell_dspace, nsupercell_dspace);

        hsize_t positions_dims[] = {n_harmonic_ifcs, 3};
        DataSpace positions_dspace(2, positions_dims);
        DataSet positions_dset = h5f->createDataSet(
            "/ifcs/harmonic/Rj", PredType::NATIVE_INT, positions_dspace);

        hsize_t hifcs_dims[] = {n_harmonic_ifcs, ndof*ndof};
        DataSpace hifcs_dspace(2, hifcs_dims);
        DataSet hifcs_dset = h5f->createDataSet(
            "/ifcs/harmonic/force_constants", PredType::IEEE_F64LE, hifcs_dspace);
        
        hsize_t positions_pos[] = {0, 0};
        hsize_t positions_count[] = {1, 3};
        hsize_t positions_dims_m[] = {3};
        DataSpace positions_dspace_m(1, positions_dims_m);

        hsize_t hifcs_pos[] = {0, 0};
        hsize_t hifcs_count[] = {1, ndof*ndof};
        hsize_t hifcs_dims_m[] = {ndof*ndof};
        DataSpace hifcs_dspace_m(1, hifcs_dims_m);

        // Write the IFCs
        for (std::size_t i = 0; i < n_harmonic_ifcs; i++) {
            ++positions_pos[0];
            positions_dspace.selectHyperslab(H5S_SELECT_SET, positions_count, positions_pos);
            positions_dset.write(harmonic_ifcs.pos[i].data(),
                                 PredType::NATIVE_INT,
                                 positions_dspace_m,
                                 positions_dspace);
            ++hifcs_pos[0];
            hifcs_dspace.selectHyperslab(H5S_SELECT_SET, hifcs_count, hifcs_pos);
            hifcs_dset.write(harmonic_ifcs.ifcs[i].data(),
                             PredType::IEEE_F64LE,
                             hifcs_dspace_m,
                             hifcs_dspace);
        }

        /// Obtain the number of harmonic IFCs
        std::size_t n_anharmonic_ifcs = anharmonic_ifcs.size();

        hsize_t Rk_dims[] = {n_anharmonic_ifcs, 3};
        DataSpace Rk_dspace(2, Rk_dims);
        DataSet Rk_dset = h5f->createDataSet(
            "/ifcs/anharmonic/Rk", PredType::IEEE_F64LE, Rk_dspace);
        hsize_t Rj_dims[] = {n_anharmonic_ifcs, 3};
        DataSpace Rj_dspace(2, Rj_dims);
        DataSet Rj_dset = h5f->createDataSet(
            "/ifcs/anharmonic/Rj", PredType::IEEE_F64LE, Rj_dspace);
        hsize_t atoms_dims[] = {n_anharmonic_ifcs, 3};
        DataSpace atoms_dspace(2, atoms_dims);
        DataSet atoms_dset = h5f->createDataSet(
            "/ifcs/anharmonic/atoms", PredType::NATIVE_SIZE_T, atoms_dspace);

        hsize_t aifcs_dims[] = {n_anharmonic_ifcs, 27};
        DataSpace aifcs_dspace(2, aifcs_dims);
        DataSet aifcs_dset = h5f->createDataSet(
            "/ifcs/anharmonic/force_constants", PredType::IEEE_F64LE, aifcs_dspace);


        hsize_t Rk_pos[] = {0, 0};
        hsize_t Rk_count[] = {1, 3};
        hsize_t Rk_dims_m[] = {3};
        DataSpace Rk_dspace_m(1, Rk_dims_m);
        hsize_t Rj_pos[] = {0, 0};
        hsize_t Rj_count[] = {1, 3};
        hsize_t Rj_dims_m[] = {3};
        DataSpace Rj_dspace_m(1, Rj_dims_m);
        hsize_t atoms_pos[] = {0, 0};
        hsize_t atoms_count[] = {1, 3};
        hsize_t atoms_dims_m[] = {3};
        DataSpace atoms_dspace_m(1, atoms_dims_m);

        hsize_t aifcs_pos[] = {0, 0};
        hsize_t aifcs_count[] = {1, 27};
        hsize_t aifcs_dims_m[] = {27};
        DataSpace aifcs_dspace_m(1, aifcs_dims_m);


        for (std::size_t i = 0; i < n_anharmonic_ifcs; i++) {
            ++Rk_pos[0];
            Rk_dspace.selectHyperslab(H5S_SELECT_SET, Rk_count, Rk_pos);
            Rk_dset.write(anharmonic_ifcs[i].rk.data(),
                          PredType::IEEE_F64LE,
                          Rk_dspace_m,
                          Rk_dspace);
            ++Rj_pos[0];
            Rj_dspace.selectHyperslab(H5S_SELECT_SET, Rj_count, Rj_pos);
            Rj_dset.write(anharmonic_ifcs[i].rj.data(),
                          PredType::IEEE_F64LE,
                          Rj_dspace_m,
                          Rj_dspace);

            ++atoms_pos[0];
            atoms_dspace.selectHyperslab(H5S_SELECT_SET, atoms_count, atoms_pos);
            std::array<std::size_t,3> atoms_ = {anharmonic_ifcs[i].i, anharmonic_ifcs[i].j, anharmonic_ifcs[i].k};
            atoms_dset.write(atoms_.data(),
                             PredType::NATIVE_SIZE_T,
                             atoms_dspace_m,
                             atoms_dspace);

            ++aifcs_pos[0];
            aifcs_dspace.selectHyperslab(H5S_SELECT_SET, aifcs_count, aifcs_pos);
            aifcs_dset.write(&(anharmonic_ifcs[i].ifc(0,0,0)),
                             PredType::IEEE_F64LE,
                             aifcs_dspace_m,
                             aifcs_dspace);
        }

    }
    comm.barrier();

}

void load_ifcs_subgroup(const char* filename,
                         Harmonic_ifcs& harmonic_ifcs,
                         std::vector<Thirdorder_ifcs>& anharmonic_ifcs,
                         const boost::mpi::communicator& comm) {
    
    auto my_id = comm.rank();

    // Processes other than ID 0 do not do anything other than wait.
    if (my_id == 0) {
        std::unique_ptr<H5File> h5f;
        h5f = alma::make_unique<H5File>(filename, H5F_ACC_RDWR);

        
        DataSet nsupercell_dset = h5f->openDataSet("/ifcs/harmonic/supercell");
        DataSpace nsupercell_dspace = nsupercell_dset.getSpace();
        DataSet positions_dset = h5f->openDataSet("/ifcs/harmonic/Rj");
        DataSpace positions_dspace = positions_dset.getSpace();
        DataSet hifcs_dset = h5f->openDataSet("/ifcs/harmonic/force_constants");
        DataSpace hifcs_dspace = hifcs_dset.getSpace();

        /// Get the number of harmonic components
        hsize_t hsizes[2];
        hifcs_dspace.getSimpleExtentDims(hsizes, nullptr);
        std::size_t nharm = hsizes[0];
        std::size_t ndof  = std::sqrt(hsizes[1]);
        
        /// Get the number of supercells
        int n_supercell[3];
        nsupercell_dspace.selectAll();
        nsupercell_dset.read(n_supercell, PredType::NATIVE_INT, nsupercell_dspace, nsupercell_dspace);


        std::vector<Eigen::MatrixXd> ifcs_(nharm, Eigen::MatrixXd::Zero(ndof,ndof));
        std::vector<Triple_int> pos_(nharm);

        hsize_t positions_pos[] = {0, 0};
        hsize_t positions_count[] = {1, 3};
        hsize_t positions_dims_m[] = {3};
        DataSpace positions_dspace_m(1, positions_dims_m);

        hsize_t hifcs_pos[] = {0, 0};
        hsize_t hifcs_count[] = {1, ndof*ndof};
        hsize_t hifcs_dims_m[] = {ndof*ndof};
        DataSpace hifcs_dspace_m(1, hifcs_dims_m);

        // Read the IFCs
        for (std::size_t i = 0; i < nharm; i++) {
            ++positions_pos[0];
            positions_dspace.selectHyperslab(H5S_SELECT_SET, positions_count, positions_pos);
            positions_dset.read(pos_[i].data(),
                                 PredType::NATIVE_INT,
                                 positions_dspace_m,
                                 positions_dspace);
            ++hifcs_pos[0];
            hifcs_dspace.selectHyperslab(H5S_SELECT_SET, hifcs_count, hifcs_pos);
            hifcs_dset.read(ifcs_[i].data(),
                             PredType::IEEE_F64LE,
                             hifcs_dspace_m,
                             hifcs_dspace);
        }

        harmonic_ifcs.na = n_supercell[0];
        harmonic_ifcs.nb = n_supercell[1];
        harmonic_ifcs.nc = n_supercell[2];
        harmonic_ifcs.pos.swap(pos_);
        harmonic_ifcs.ifcs.swap(ifcs_);

        /// Anharmonic ifcs
        DataSet Rj_dset = h5f->openDataSet("/ifcs/anharmonic/Rj");
        DataSpace Rj_dspace = Rj_dset.getSpace();
        DataSet Rk_dset = h5f->openDataSet("/ifcs/anharmonic/Rk");
        DataSpace Rk_dspace = Rk_dset.getSpace();
        DataSet atoms_dset = h5f->openDataSet("/ifcs/anharmonic/atoms");
        DataSpace atoms_dspace = atoms_dset.getSpace();
        DataSet aifcs_dset = h5f->openDataSet("/ifcs/harmonic/force_constants");
        DataSpace aifcs_dspace = aifcs_dset.getSpace();

        /// Get the number of anharmonic components
        hsize_t asizes[2];
        Rj_dspace.getSimpleExtentDims(asizes, nullptr);
        std::size_t nanharm = hsizes[0];

        anharmonic_ifcs.clear();
        anharmonic_ifcs.reserve(nanharm);

        hsize_t Rk_pos[] = {0, 0};
        hsize_t Rk_count[] = {1, 3};
        hsize_t Rk_dims_m[] = {3};
        DataSpace Rk_dspace_m(1, Rk_dims_m);
        hsize_t Rj_pos[] = {0, 0};
        hsize_t Rj_count[] = {1, 3};
        hsize_t Rj_dims_m[] = {3};
        DataSpace Rj_dspace_m(1, Rj_dims_m);
        hsize_t atoms_pos[] = {0, 0};
        hsize_t atoms_count[] = {1, 3};
        hsize_t atoms_dims_m[] = {3};
        DataSpace atoms_dspace_m(1, atoms_dims_m);

        hsize_t aifcs_pos[] = {0, 0};
        hsize_t aifcs_count[] = {1, 27};
        hsize_t aifcs_dims_m[] = {27};
        DataSpace aifcs_dspace_m(1, aifcs_dims_m);



        for (std::size_t i = 0; i < nanharm; i++) {

            Eigen::Vector3d Rk, Rj;

            ++Rk_pos[0];
            Rk_dspace.selectHyperslab(H5S_SELECT_SET, Rk_count, Rk_pos);
            Rk_dset.read(Rk.data(),
                          PredType::IEEE_F64LE,
                          Rk_dspace_m,
                          Rk_dspace);
            ++Rj_pos[0];
            Rj_dspace.selectHyperslab(H5S_SELECT_SET, Rj_count, Rj_pos);
            Rj_dset.read(Rj.data(),
                          PredType::IEEE_F64LE,
                          Rj_dspace_m,
                          Rj_dspace);

            ++atoms_pos[0];
            atoms_dspace.selectHyperslab(H5S_SELECT_SET, atoms_count, atoms_pos);
            std::array<std::size_t,3> atoms_;
            atoms_dset.read(atoms_.data(),
                             PredType::NATIVE_SIZE_T,
                             atoms_dspace_m,
                             atoms_dspace);

            ++aifcs_pos[0];
            std::array<double,27> ifcs_;
            aifcs_dspace.selectHyperslab(H5S_SELECT_SET, aifcs_count, aifcs_pos);
            aifcs_dset.read(ifcs_.data(),
                            PredType::IEEE_F64LE,
                            aifcs_dspace_m,
                            aifcs_dspace);

            anharmonic_ifcs.emplace_back(Rj,Rk,atoms_[0],atoms_[1],atoms_[2]);
            anharmonic_ifcs[i].swap_block(ifcs_);
        }

    }
    
    boost::mpi::broadcast(comm, harmonic_ifcs, 0);
    boost::mpi::broadcast(comm, anharmonic_ifcs, 0);

    comm.barrier();
}

} // namespace alma

