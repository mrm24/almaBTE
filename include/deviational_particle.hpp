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
/// Code used to describe deviational particles.

#include <algorithm>
#include <Eigen/Dense>

namespace alma {
/// Simple enum representing a sign.
enum class particle_sign { minus = -1, plus = 1 };

/// Convenience function for obtaining a particle_sign.
///
/// @param[in] value - anything that can be compared to 0
/// @return a particle_sign. 0 is considered positive.
template <typename T> inline particle_sign get_particle_sign(const T& value) {
    if (value >= 0)
        return particle_sign::plus;
    else
        return particle_sign::minus;
}


/// Each object of this class represents a deviational particle
/// in the simulation.
class D_particle {
public:
    /// Cartesian coordinates of the particle [nm].
    Eigen::VectorXd pos;
    /// q point index in some regular grid.
    std::size_t q;
    /// Mode index.
    std::size_t alpha;
    /// Sign of the particle.
    particle_sign sign;
    /// Current time.
    double t;
    /// Basic constructor.
    D_particle(const Eigen::VectorXd& _pos,
               std::size_t _q,
               std::size_t _alpha,
               particle_sign _sign,
               double _t)
        : pos(std::move(_pos)), q(_q), alpha(_alpha), sign(_sign), t(_t) {
    }


    /// Copy constructor.
    D_particle(const D_particle& original)
        : pos(original.pos), q(original.q), alpha(original.alpha),
          sign(original.sign), t(original.t) {
    }


    /// Swap the data from two objects.
    ///
    /// @param[in,out] a - first object
    /// @param[in,out] b - second object
    friend void swap(D_particle& a, D_particle& b) {
        a.pos.swap(b.pos);
        std::swap(a.q, b.q);
        std::swap(a.alpha, b.alpha);
        std::swap(a.sign, b.sign);
        std::swap(a.t, b.t);
    }


    /// Assignment operator (copy and swap).
    D_particle& operator=(D_particle original) {
        swap(*this, original);
        return *this;
    }
};
} // namespace alma
