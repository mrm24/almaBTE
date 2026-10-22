// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
extern "C" {
struct sheng_allocations {
    int nelements, natoms, ngrid[3], norientations;
};
void read_sheng_gfactors(sheng_allocations*, double*);
void read_sheng_element(sheng_allocations*, int, char*);
}
int main() {
    namespace fs = std::filesystem;
    const auto original = fs::current_path();
    const auto temporary =
        fs::temp_directory_path() /
        ("prealma-control-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(temporary);
    fs::current_path(temporary);
    {
        std::ofstream output("CONTROL");
        output << "&crystal\n elements='Ga','N', gfactors=0.1,0.2\n/\n";
    }
    sheng_allocations allocations{2, 4, {1, 1, 1}, 0};
    // A per-atom write would overwrite the sentinels beyond the two elements.
    double factors[] = {7., 7., 123., 456.};
    read_sheng_gfactors(&allocations, factors);
    char element[4];
    read_sheng_element(&allocations, 0, element);
    const bool valid = factors[0] == .1 && factors[1] == .2 &&
                       factors[2] == 123. && factors[3] == 456. &&
                       std::string(element) == "Ga ";
    fs::current_path(original);
    fs::remove_all(temporary);
    if (!valid)
        std::cerr
            << "ShengBTE reader confused per-element and per-atom arrays\n";
    return valid ? 0 : 1;
}
