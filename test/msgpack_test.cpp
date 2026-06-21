#include <boost/mpi.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>
#include <msgpack.hpp>
#include <processes.hpp>

int main(int argc, char** argv) {

    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    const auto rank = world.rank();
    const auto nproc = world.size();
    const auto root = 0;
    const std::string filename_4ph = "4ph.testing.msgpack";
    const std::size_t n4ph_ref = 5000000;
    std::size_t n4ph_check;

    std::vector<alma::Fourph_process> reference, check;

    /// Filling
    {

        auto limits = alma::my_jobs(n4ph_ref, world.size(), world.rank());
        reference.reserve(limits[1] - limits[0]);

        for (auto i = limits[0]; i < limits[1]; i++) {
            alma::fourph_type type;
            if (i % 3 == 0) {
                type = alma::fourph_type::splitting;
            } else if (i % 2 == 0) {
                type = alma::fourph_type::recombination;
            } else {
                type = alma::fourph_type::redistribution;
            }
            // if ( world.rank() == nproc -1) std::cout << world.rank() << '\t' << i << '\t' << static_cast<int>(type) << std::endl;
            alma::Fourph_process p(i, {i,i+1,i+2,i+3},{i+4,i+5,i+6,i+7}, type, 2.*i + 0.5, 4.*i + 0.5);
            p.compute_gaussian();
            p.set_vp2(6.*i);
            reference.emplace_back(p);
        }

    }

    /// Dump into a file
    {
        /// The master
        world.barrier();
        if (world.rank() == 0) {
            std::ofstream ofs(filename_4ph, std::ios::binary | std::ios::trunc);
            if (!ofs) throw std::runtime_error("Failed to open file for writting: " + filename_4ph);
            msgpack::pack(ofs, static_cast<std::uint64_t>(n4ph_ref));
            for (const auto& elem : reference) msgpack::pack(ofs, elem);
            ofs.flush();
            ofs.close();
        }
        world.barrier();

        /// The other processes append
        for (auto iproc = 1; iproc < world.size(); iproc++) {
            if (iproc == world.rank()) {
                std::ofstream ofs(filename_4ph, std::ios::binary | std::ios::app);
                if (!ofs) throw std::runtime_error("Failed to open file for writting: " + filename_4ph);
                for (const auto& elem : reference) msgpack::pack(ofs, elem);
                ofs.flush();
                ofs.close();
            }
            world.barrier();
        }
    }

    /// Reading
    {
        std::ifstream ifs(filename_4ph, std::ios::binary);
        if (!ifs) throw std::runtime_error("Failed to open file for reading: " + filename_4ph);

        msgpack::unpacker pac;
        auto read_more = [&](std::size_t want) {
            // Ensure unpacker has room, read from file into its internal buffer
            pac.reserve_buffer(want);
            ifs.read(pac.buffer(), want);
            std::streamsize got = ifs.gcount();
            if (got <= 0) return std::size_t(0);
            pac.buffer_consumed(static_cast<std::size_t>(got));
            return static_cast<std::size_t>(got);
        };

        constexpr std::size_t chunk_size = 5 << 20; // 5 MB at a time, it is more than enough to hold several 4ph objects

        /// Read the size
        msgpack::object_handle oh;
        while (!pac.next(oh)) {
            if (read_more(chunk_size) == 0)
                throw std::runtime_error("Unexpected EOF reading from " + filename_4ph);
        }
        std::size_t n4ph = oh.get().as<std::uint64_t>();

        auto limits = alma::my_jobs(n4ph, world.size(), world.rank());
        check.reserve(limits[1] - limits[0]);

        // Skip records before limits[0]
        for (std::size_t i = 0; i < limits[0]; ++i) {
            while (!pac.next(oh)) {
                if (read_more(chunk_size) == 0)
                    throw std::runtime_error("Unexpected EOF while skipping entries in " + filename_4ph);
            }
        }

        // Read this rank's chunk [limits[0], limits[1]) ---
        for (std::size_t i = limits[0]; i < limits[1]; ++i) {
            while (!pac.next(oh)) {
                if (read_more(chunk_size) == 0)
                    throw std::runtime_error("Unexpected EOF while reading rank's (" +
                        std::to_string(world.rank()) + ") entries in " + filename_4ph);
            }
            alma::Fourph_process p(0, {0,0,0,0},{0,0,0,0}, alma::fourph_type::splitting, 0., 0.);
            oh.get().convert(p);
            check.emplace_back(p);
            // if ( world.rank() == nproc -1) std::cout << world.rank() << '\t' << i << '\t' << static_cast<int>(p.type) << std::endl;
        }
        
        n4ph_check = n4ph;

        ifs.close();
    }

    /// Testing
     {
        if (n4ph_check != n4ph_ref) {
            std::cerr << "[rank " << rank << "] total size mismatch \n";
            world.abort(1);
        }
        if (check.size() != reference.size()) {
            std::cerr << "[rank " << rank << "] size mismatch: check="
                      << check.size() << " reference=" << reference.size() << "\n";
            world.abort(1);
        }
        for (std::size_t i = 0; i < reference.size(); ++i) {

            assert(check[i] == reference[i] && "mismatch between read-back and reference record");
        }
        world.barrier();
        if (rank == root) std::cout << "All ranks: read-back matches reference.\n";
    }
    return 0;
}
