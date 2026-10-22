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

#pragma once

/// @file
///
/// MPI-parallel saving/loading of N-phonon processes (3ph, 4ph) to/from
/// MessagePack files. File layout: one uint64 with the total number of
/// processes, followed by that many packed processes, in MPI-rank order.

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/mpi.hpp>
#include <msgpack.hpp>
#include <processes.hpp>
#include <utilities.hpp>

namespace alma {

/// Build the name of a MessagePack processes file from the HDF5 file name.
///
/// @param[in] h5filename - name of the .h5 file
/// @param[in] tag - "3ph" or "4ph"
/// @return h5filename with ".h5" replaced by ".<tag>.msgpack"
inline std::string processes_msgpack_filename(const std::string& h5filename,
                                              const std::string& tag) {
    std::string nruter(h5filename);
    auto pos = nruter.rfind(".h5");
    if (pos == std::string::npos)
        // No .h5 extension (e.g. temporary files): just append the suffix.
        nruter += "." + tag + ".msgpack";
    else
        nruter.replace(pos, 3, "." + tag + ".msgpack");
    return nruter;
}


/// Save the processes held by every MPI rank into a single MessagePack
/// file (a header-only file if there are no processes at all, so that
/// stale files are overwritten). Must be called by all ranks.
///
/// @param[in] filename - msgpack file to (over)write
/// @param[in] processes - processes owned by this rank
/// @param[in] comm - MPI communicator
template <class process_t>
void save_processes_msgpack(const std::string& filename,
                            const std::vector<process_t>& processes,
                            const boost::mpi::communicator& comm) {
    std::uint64_t my_n = processes.size(), total = 0;
    boost::mpi::all_reduce(comm, my_n, total, std::plus<std::uint64_t>());

    comm.barrier();
    if (comm.rank() == 0) {
        std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
        if (!ofs) throw std::runtime_error("Failed to open file for writting: " + filename);
        msgpack::pack(ofs, total);
        for (const auto& elem : processes) msgpack::pack(ofs, elem);
        ofs.flush();
    }
    comm.barrier();

    // The other ranks append, one at a time.
    for (int iproc = 1; iproc < comm.size(); ++iproc) {
        if (iproc == comm.rank()) {
            std::ofstream ofs(filename, std::ios::binary | std::ios::app);
            if (!ofs) throw std::runtime_error("Failed to open file for writting: " + filename);
            for (const auto& elem : processes) msgpack::pack(ofs, elem);
            ofs.flush();
        }
        comm.barrier();
    }
}


/// Load processes from a MessagePack file. Each rank reads only its own
/// chunk of the file (as given by my_jobs). Must be called by all ranks.
///
/// @param[in] filename - msgpack file to read
/// @param[in] comm - MPI communicator
/// @param[in] blank - a placeholder process, overwritten by every record read
/// @return the processes belonging to this rank
template <class process_t>
std::vector<process_t> load_processes_msgpack(const std::string& filename,
                                              const boost::mpi::communicator& comm,
                                              process_t blank) {
    std::vector<process_t> nruter;
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Failed to open file for reading: " + filename);

    msgpack::unpacker pac;
    auto read_more = [&](std::size_t want) {
        pac.reserve_buffer(want);
        ifs.read(pac.buffer(), want);
        std::streamsize got = ifs.gcount();
        if (got <= 0) return std::size_t(0);
        pac.buffer_consumed(static_cast<std::size_t>(got));
        return static_cast<std::size_t>(got);
    };
    constexpr std::size_t chunk_size = 5 << 20; // 5 MB at a time

    msgpack::object_handle oh;
    auto next = [&](const std::string& what) {
        while (!pac.next(oh)) {
            if (read_more(chunk_size) == 0)
                throw std::runtime_error("Unexpected EOF " + what + " in " + filename);
        }
    };

    next("reading the header");
    std::size_t ntot = oh.get().as<std::uint64_t>();

    auto limits = alma::my_jobs(ntot, comm.size(), comm.rank());
    nruter.reserve(limits[1] - limits[0]);

    for (std::size_t i = 0; i < limits[0]; ++i) next("while skipping entries");

    for (std::size_t i = limits[0]; i < limits[1]; ++i) {
        next("while reading rank " + std::to_string(comm.rank()) + " entries");
        oh.get().convert(blank);
        nruter.emplace_back(blank);
    }
    return nruter;
}

} // namespace alma
