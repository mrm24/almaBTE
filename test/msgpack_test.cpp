#include <boost/mpi.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <stdexcept>
#include <processes_msgpack.hpp>

namespace {

/// Build the reference processes owned by this rank.
template <std::size_t N, class process_type>
std::vector<alma::ph_process<N, process_type>> make_reference(
    std::size_t ntot, const boost::mpi::communicator& world,
    const std::vector<process_type>& types) {
    using proc_t = alma::ph_process<N, process_type>;
    std::vector<proc_t> reference;
    auto limits = alma::my_jobs(ntot, world.size(), world.rank());
    reference.reserve(limits[1] - limits[0]);
    for (auto i = limits[0]; i < limits[1]; i++) {
        std::array<std::size_t, N> q, alpha;
        for (std::size_t j = 0; j < N; ++j) {
            q[j] = i + j;
            alpha[j] = i + N + j;
        }
        proc_t p(i, q, alpha, types[i % types.size()], 2. * i + 0.5, 4. * i + 0.5);
        p.compute_gaussian();
        p.set_vp2(6. * i);
        reference.emplace_back(p);
    }
    return reference;
}

/// Write, read back and compare.
template <std::size_t N, class process_type>
void roundtrip(const std::string& filename, std::size_t ntot,
               const std::vector<process_type>& types,
               const alma::ph_process<N, process_type>& blank,
               const boost::mpi::communicator& world) {
    auto reference = make_reference<N>(ntot, world, types);
    alma::save_processes_msgpack(filename, reference, world);
    world.barrier();
    auto check = alma::load_processes_msgpack(filename, world, blank);
    if (check.size() != reference.size()) {
        std::cerr << "[rank " << world.rank() << "] size mismatch: check=" << check.size()
                  << " reference=" << reference.size() << "\n";
        world.abort(1);
    }
    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (!(check[i] == reference[i])) {
            std::cerr << "[rank " << world.rank() << "] mismatch at record " << i << "\n";
            world.abort(1);
        }
    }
    world.barrier();
}

} // namespace

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;
    const std::size_t nref = 500000;

    roundtrip<4, alma::fourph_type>(
        "4ph.testing.msgpack", nref,
        {alma::fourph_type::splitting, alma::fourph_type::recombination,
         alma::fourph_type::redistribution},
        alma::Fourph_process(0, {0, 0, 0, 0}, {0, 0, 0, 0}, alma::fourph_type::splitting, 0., 0.),
        world);

    roundtrip<3, alma::threeph_type>(
        "3ph.testing.msgpack", nref,
        {alma::threeph_type::emission, alma::threeph_type::absorption},
        alma::Threeph_process(0, {0, 0, 0}, {0, 0, 0}, alma::threeph_type::emission, 0., 0.),
        world);

    // Empty set of processes: header-only file.
    roundtrip<3, alma::threeph_type>(
        "3ph.empty.testing.msgpack", 0, {alma::threeph_type::emission},
        alma::Threeph_process(0, {0, 0, 0}, {0, 0, 0}, alma::threeph_type::emission, 0., 0.),
        world);

    if (world.rank() == 0) std::cout << "All ranks: read-back matches reference (3ph and 4ph).\n";
    return 0;
}
