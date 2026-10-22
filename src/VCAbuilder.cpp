// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2015-2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#include <material_builder.hpp>
#include <exception>
#include <iostream>
int main(int argc, char** argv) {
    try {
        return alma::run_material_builder(
            argc, argv, alma::Builder_task::material);
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
