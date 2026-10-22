// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2015-2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace alma {
enum class Builder_task { material, gruneisen };
/// Shared XML/material loading for VCAbuilder and the Gruneisen executable.
int run_material_builder(int argc, char** argv, Builder_task task);
} // namespace alma
