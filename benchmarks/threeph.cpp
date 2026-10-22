// Benchmark the real three-phonon RTA and iterative paths, including legacy
// I/O.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <bulk_hdf5.hpp>
#include <shengbte_iter.hpp>

int main(int argc, char** argv) {
    boost::mpi::environment environment(argc, argv);
    boost::mpi::communicator world;
    if (argc < 2) {
        if (world.rank() == 0)
            std::cerr
                << "usage: benchmark_threeph FILE [REPETITIONS] [REFERENCE]\n";
        return 1;
    }
    const int repeats = argc > 2 ? std::stoi(argv[2]) : 3;
    if (repeats < 1)
        return 1;
    using clock = std::chrono::steady_clock;
    auto start = clock::now();
    auto data = alma::load_bulk_hdf5(argv[1], world);
    const auto load =
        std::chrono::duration<double>(clock::now() - start).count();
    auto& cell = *std::get<1>(data);
    auto& syms = *std::get<2>(data);
    auto& grid = *std::get<3>(data);
    auto& processes = *std::get<4>(data);
    const double temperature = 300.;
    Eigen::ArrayXXd rates;
    start = clock::now();
    for (int i = 0; i < repeats; ++i)
        rates = alma::calc_w0_threeph(grid, processes, temperature, world);
    const auto rta =
        std::chrono::duration<double>(clock::now() - start).count() / repeats;
    std::vector<alma::Twoph_process> elastic;
    start = clock::now();
    alma::ShengBTE_iterator iterator(
        cell, grid, syms, processes, elastic, temperature, world);
    const auto prepare =
        std::chrono::duration<double>(clock::now() - start).count();
    start = clock::now();
    for (int i = 0; i < repeats; ++i)
        iterator.next(cell, grid, syms, processes, elastic, temperature);
    const auto iteration =
        std::chrono::duration<double>(clock::now() - start).count() / repeats;
    const auto kappa = iterator.calc_current_kappa(temperature);
    std::size_t total;
    boost::mpi::reduce(
        world, processes.size(), total, std::plus<std::size_t>(), 0);
    if (world.rank() == 0) {
        std::cout << std::setprecision(17) << "processes=" << total
                  << " ranks=" << world.size() << " load_s=" << load
                  << " rta_s=" << rta << " prepare_s=" << prepare
                  << " iteration_s=" << iteration << " rate_sum=" << rates.sum()
                  << " kappa_trace=" << kappa.trace() << '\n';
        if (argc > 3) {
            std::ofstream reference(argv[3]);
            reference << std::setprecision(17);
            for (Eigen::Index i = 0; i < rates.size(); ++i)
                reference << rates.data()[i] << '\n';
            for (Eigen::Index i = 0; i < kappa.size(); ++i)
                reference << kappa.data()[i] << '\n';
        }
    }
}
