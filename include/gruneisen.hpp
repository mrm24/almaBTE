// Copyright 2023-2026 Martí Raya-Moreno <marti.raya.moreno@physik.hu-berlin.de>
// Copyright 2015-2026 The ALMA Project Developers
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <qpoint_grid.hpp>
namespace alma {
/// Mode Gruneisen parameters from third-order IFCs (dimensionless).
Eigen::ArrayXXd calc_mode_gruneisen(const Crystal_structure&,
                                    const Gamma_grid&,
                                    const std::vector<Thirdorder_ifcs>&);
/// Heat-capacity-weighted mean at a positive temperature.
double calc_total_gruneisen(const Gamma_grid&,
                            const Eigen::Ref<const Eigen::ArrayXXd>&,
                            double);
} // namespace alma
