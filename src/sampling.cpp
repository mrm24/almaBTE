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
/// Definitions corresponding to sampling.cpp

#include <cmath>
#include <vector>
#include <map>
#include <utilities.hpp>
#include <exceptions.hpp>
#include <sampling.hpp>

namespace alma {
void Grid_distribution::fill_cumulative(const std::vector<double>& p) {
    // Check that the vector size if correct.
    std::size_t n = p.size();

    if (n != this->nmodes * this->nqpoints)
        throw value_error("the argument size is incompatible with the grid");

    // Check that there are no negative elements.
    if (*std::min_element(p.begin(), p.end()) < 0.)
        throw value_error("probabilities must be >= 0.");
    this->cumulative.reserve(n);
    // Compute the unnormalized distribution function.
    std::partial_sum(p.begin(), p.end(), std::back_inserter(this->cumulative));
    // Normalize it.
    double last = this->cumulative[n - 1];

    for (auto& it : this->cumulative)
        it /= last;
}


BE_derivative_distribution::BE_derivative_distribution(
    const Gamma_grid& grid,
    const Eigen::Ref<const Eigen::ArrayXXd>& w,
    double T,
    randutils::mt19937_rng& _rng)
    : Grid_distribution(grid, _rng) {
    // Compute the unnormalized probabilities.
    std::vector<double> pmf(this->nmodes * grid.nqpoints);

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < this->nmodes; ++im) {
            if (spectrum.omega(im) <= 0.)
                pmf[this->nmodes * iq + im] = 0.;
            else
                pmf[this->nmodes * iq + im] =
                    bose_einstein_kernel(spectrum.omega(im), T) * w(im, iq);
        }
    }
    this->fill_cumulative(pmf);
}


Nabla_T_distribution::Nabla_T_distribution(
    const Gamma_grid& grid,
    const Eigen::Ref<const Eigen::Vector3d>& nablaT,
    double T,
    randutils::mt19937_rng& _rng)
    : Grid_distribution(grid, _rng) {
    if (nablaT.norm() == 0.)
        throw value_error("invalid vector");
    std::vector<double> pmf(this->nmodes * grid.nqpoints);

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < this->nmodes; ++im)
            if (spectrum.omega(im) == 0.)
                pmf[this->nmodes * iq + im] = 0.;
            else
                pmf[this->nmodes * iq + im] =
                    std::fabs(nablaT.dot(spectrum.vg.col(im).matrix()) *
                              bose_einstein_kernel(spectrum.omega(im), T));
    }
    this->fill_cumulative(pmf);
}

Isothermal_wall_distribution::Isothermal_wall_distribution(
    const Gamma_grid& grid,
    double Twall,
    double Teq,
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    randutils::mt19937_rng& _rng)
    : Grid_distribution(grid, _rng) {
    if (normal.norm() == 0.)
        throw value_error("invalid normal vector");
    // Compute the unnormalized probabilities.
    // They are zero for all states going into the wall.
    std::vector<double> pmf(this->nmodes * grid.nqpoints);

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < this->nmodes; ++im) {
            auto pos = this->nmodes * iq + im;
            double vn = normal.dot(spectrum.vg.col(im).matrix());

            if ((vn <= 0.) || almost_equal(0., spectrum.omega(im)))
                pmf[pos] = 0.;
            else
                pmf[pos] = vn * spectrum.omega(im) *
                           std::fabs(bose_einstein(spectrum.omega(im), Twall) -
                                     bose_einstein(spectrum.omega(im), Teq));
        }
    }
    this->fill_cumulative(pmf);
}

planar_source_distribution::planar_source_distribution(
    const Gamma_grid& grid,
    double Tref,
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    randutils::mt19937_rng& _rng)
    : Grid_distribution(grid, _rng) {
    if (normal.norm() == 0.)
        throw value_error("invalid normal vector");
    // Compute the unnormalized probabilities.
    // They are zero for all states going into the wall.
    std::vector<double> pmf(this->nmodes * grid.nqpoints);

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < this->nmodes; ++im) {
            auto pos = this->nmodes * iq + im;

            double vn = normal.dot(spectrum.vg.col(im).matrix());
            double C = alma::bose_einstein_kernel(spectrum.omega(im), Tref);

            if ((vn <= 0.) || almost_equal(0., spectrum.omega(im))) {
                pmf[pos] = 0.;
            }
            else {
                pmf[pos] = C * vn;
            }
        }
    }
    this->fill_cumulative(pmf);
}

std::vector<Diffuse_mismatch_distribution::dos_tuple>
Diffuse_mismatch_distribution::get_modes(
    const Gamma_grid& gridA,
    const Gamma_grid& gridB,
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    double scalebroad) {
    if (normal.norm() == 0.) {
        throw value_error("invalid normal vector");
    }

    std::vector<dos_tuple> result;

    Eigen::Vector3d u = normal / normal.norm();

    std::size_t NqA = gridA.nqpoints;
    std::size_t NbranchesA = gridA.get_spectrum_at_q(0).omega.size();

    std::size_t NqB = gridB.nqpoints;
    std::size_t NbranchesB = gridB.get_spectrum_at_q(0).omega.size();

    result.reserve(NqA * NbranchesA + NqB * NbranchesB);

    // Process all modes in the A grid
    for (std::size_t nq = 0; nq < NqA; nq++) {
        auto& spectrum = gridA.get_spectrum_at_q(nq);

        for (std::size_t nbranch = 0; nbranch < NbranchesA; nbranch++) {
            double vn = u.dot(spectrum.vg.col(nbranch).matrix());
            double C =
                alma::bose_einstein_kernel(spectrum.omega(nbranch),
                                           this->Tref) /
                (alma::constants::kB * gridA.nqpoints * this->volume['A']);
            result.emplace_back(std::make_tuple(
                'A',
                nbranch,
                nq,
                vn,
                C,
                Gaussian_for_DOS(gridA, nq, nbranch, scalebroad)));
        }
    }

    // Process all modes in the B grid
    for (std::size_t nq = 0; nq < NqB; nq++) {
        auto& spectrum = gridB.get_spectrum_at_q(nq);

        for (std::size_t nbranch = 0; nbranch < NbranchesB; nbranch++) {
            double vn = u.dot(spectrum.vg.col(nbranch).matrix());
            double C =
                alma::bose_einstein_kernel(spectrum.omega(nbranch),
                                           this->Tref) /
                (alma::constants::kB * gridB.nqpoints * this->volume['B']);
            result.emplace_back(std::make_tuple(
                'B',
                nbranch,
                nq,
                vn,
                C,
                Gaussian_for_DOS(gridB, nq, nbranch, scalebroad)));
        }
    }

    return result;
}

Diffuse_mismatch_distribution::Diffuse_mismatch_distribution(
    const Gamma_grid& gridA,
    const Crystal_structure& poscarA,
    const Gamma_grid& gridB,
    const Crystal_structure& poscarB,
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    double scalebroad,
    randutils::mt19937_rng& _rng,
    double _Tref)
    :

      rng(_rng) {
    this->Tref = _Tref;

    if (normal.norm() == 0.) {
        throw value_error("invalid normal vector");
    }

    this->Nbranches['A'] = gridA.get_spectrum_at_q(0).omega.size();
    // this->Nq['A'] = gridA.nqpoints;
    this->Ntot['A'] = gridA.get_spectrum_at_q(0).omega.size() * gridA.nqpoints;

    this->Nbranches['B'] = gridB.get_spectrum_at_q(0).omega.size();
    // this->Nq['B'] = gridB.nqpoints;
    this->Ntot['B'] = gridB.get_spectrum_at_q(0).omega.size() * gridB.nqpoints;

    this->volume['A'] = poscarA.V;
    this->volume['B'] = poscarB.V;

    std::vector<Diffuse_mismatch_distribution::dos_tuple> allmodes =
        this->get_modes(gridA, gridB, normal, scalebroad);

    // INITIALISE LOOKUP TABLES
    lookup_incidentA.reserve(this->Ntot['A']);
    for (std::size_t idxA = 0; idxA < this->Ntot['A']; idxA++) {
        std::vector<lookup_entry> empty;
        this->lookup_incidentA.emplace_back(empty);
    }

    lookup_incidentB.reserve(this->Ntot['B']);
    for (std::size_t idxB = 0; idxB < this->Ntot['B']; idxB++) {
        std::vector<lookup_entry> empty;
        this->lookup_incidentB.emplace_back(empty);
    }

    // DETERMINE ALL ALLOWED TRANSITIONS

    // variables used for normalisation of cumulative distributions
    double totalsumA = 0.0;
    double totalsumB = 0.0;

    for (std::size_t listidx = 0; listidx < allmodes.size(); listidx++) {
        Diffuse_mismatch_distribution::dos_tuple& mode_in =
            allmodes.at(listidx);
        char side_in = std::get<0>(mode_in);
        double vproj_in = std::get<3>(mode_in);

        // only process modes incident on the interface

        if ((side_in == 'A' && vproj_in > 0.0) ||
            (side_in == 'B' && vproj_in < 0.0)) {
            // construct overall index from branch and q
            std::size_t idx_in =
                std::get<1>(mode_in) +
                std::get<2>(mode_in) * this->Nbranches[side_in];

            // obtain target frequency
            double omega_target = std::get<5>(mode_in).mu;

            // search for compatible transitions

            for (std::size_t searchidx = 0; searchidx < allmodes.size();
                 searchidx++) {
                Diffuse_mismatch_distribution::dos_tuple& candidate =
                    allmodes.at(searchidx);
                char side_out = std::get<0>(candidate);
                double vproj_out = std::get<3>(candidate);

                // check if the candidate points AWAY from the interface
                bool emission = ((side_out == 'A' && vproj_out < 0.0) ||
                                 (side_out == 'B' && vproj_out > 0.0));

                // check if the candidate is energetically compatible
                bool is_omega_compatible =
                    (omega_target > std::get<5>(candidate).lbound) &&
                    (omega_target < std::get<5>(candidate).ubound);

                // only process valid candidates
                if (emission && is_omega_compatible) {
                    // compute single index
                    std::size_t idx_out =
                        std::get<1>(candidate) +
                        std::get<2>(candidate) * this->Nbranches[side_out];

                    // compute unnormalised contribution to cumulative
                    // probability = C*vproj*Gaussian
                    double raw_probability =
                        std::get<4>(candidate) * std::abs(vproj_out) *
                        std::get<5>(candidate).get_contribution(omega_target);

                    // register this mode in the list of valid transistions
                    lookup_entry entry(
                        std::make_tuple(side_out, idx_out, raw_probability));

                    if (side_in == 'A') {
                        this->lookup_incidentA.at(idx_in).emplace_back(entry);
                        totalsumA += raw_probability;
                    }
                    else {
                        this->lookup_incidentB.at(idx_in).emplace_back(entry);
                        totalsumB += raw_probability;
                    }

                } // end compatible candidate
            }     // done scanning over candidates
        }         // end incident mode
    }             // done scanning over modes

    // fix incident modes that currently have no valid output mode
    // by looking for the closest compatible reflection

    for (std::size_t listidx = 0; listidx < allmodes.size(); listidx++) {
        Diffuse_mismatch_distribution::dos_tuple& mode_in =
            allmodes.at(listidx);
        char side_in = std::get<0>(mode_in);
        double vproj_in = std::get<3>(mode_in);
        std::size_t idx_in = std::get<1>(mode_in) +
                             std::get<2>(mode_in) * this->Nbranches[side_in];

        // only process incident modes that need fixing

        bool fixA = false;
        if (side_in == 'A') {
            fixA =
                vproj_in > 0.0 && this->lookup_incidentA.at(idx_in).size() == 0;
        }

        bool fixB = false;
        if (side_in == 'B') {
            fixB =
                vproj_in < 0.0 && this->lookup_incidentB.at(idx_in).size() == 0;
        }

        if (fixA || fixB) {
            // obtain target frequency
            double omega_target = std::get<5>(mode_in).mu;

            // scan over possible reflections

            int idx_out = -1;
            char side_out;
            double deltamin = 1e300;
            double vproj_out;
            double C_out;

            for (std::size_t searchidx = 0; searchidx < allmodes.size();
                 searchidx++) {
                Diffuse_mismatch_distribution::dos_tuple& candidate =
                    allmodes.at(searchidx);
                side_out = std::get<0>(candidate);
                double vproj_candidate = std::get<3>(candidate);

                // only consider reflections pointing AWAY from the interface
                bool reflection = (side_in == side_out);
                bool emission = (side_out == 'A' && vproj_candidate < 0.0) ||
                                (side_out == 'B' && vproj_candidate > 0.0);

                if (reflection && emission) {
                    // compute single index
                    std::size_t idx_candidate =
                        std::get<1>(candidate) +
                        std::get<2>(candidate) * this->Nbranches[side_out];

                    // look for the smallest energy mismatch
                    double omega = std::get<5>(candidate).mu;
                    double delta_omega = std::abs(omega - omega_target);
                    if (delta_omega < deltamin) {
                        idx_out = idx_candidate;
                        vproj_out = vproj_candidate;
                        C_out = std::get<4>(candidate);
                        deltamin = delta_omega;
                    }
                }
            } // done scanning over all candidates

            if (idx_out == -1) {
                throw value_error(
                    "no matching mode was found for some incident mode");
            }
            else {
                // compute unnormalised contribution to cumulative probability

                double raw_probability = C_out * std::abs(vproj_out);
                lookup_entry entry(
                    std::make_tuple(side_out, idx_out, raw_probability));
                if (side_in == 'A') {
                    this->lookup_incidentA.at(idx_in).emplace_back(entry);
                    totalsumA += raw_probability;
                }
                else {
                    this->lookup_incidentB.at(idx_in).emplace_back(entry);
                    totalsumB += raw_probability;
                }
            }
        } // end mode needs fixing
    }     // end scanning over all modes

    // CONSTRUCT THE NORMALISED CUMULATIVE PROBABILITY DISTRIBUTIONS

    double pcumulA = 0.0;

    for (std::size_t idxA = 0; idxA < this->Ntot['A']; idxA++) {
        for (std::size_t sub_idxA = 0;
             sub_idxA < this->lookup_incidentA.at(idxA).size();
             sub_idxA++) {
            lookup_entry entry = this->lookup_incidentA.at(idxA).at(sub_idxA);
            pcumulA += std::get<2>(entry) / totalsumA;
            lookup_entry new_entry = lookup_entry(std::make_tuple(
                std::get<0>(entry), std::get<1>(entry), pcumulA));
            this->lookup_incidentA.at(idxA).at(sub_idxA) = new_entry;
        }
    }

    double pcumulB = 0.0;

    for (std::size_t idxB = 0; idxB < this->Ntot['B']; idxB++) {
        for (std::size_t sub_idxB = 0;
             sub_idxB < this->lookup_incidentB.at(idxB).size();
             sub_idxB++) {
            lookup_entry entry = this->lookup_incidentB.at(idxB).at(sub_idxB);
            pcumulB += std::get<2>(entry) / totalsumB;
            lookup_entry new_entry = lookup_entry(std::make_tuple(
                std::get<0>(entry), std::get<1>(entry), pcumulB));
            this->lookup_incidentB.at(idxB).at(sub_idxB) = new_entry;
        }
    }
}

char Diffuse_mismatch_distribution::reemit(char side_in, D_particle& particle) {
    std::size_t idx_in = particle.alpha + particle.q * this->Nbranches[side_in];

    char side_out = '0';
    int idx_out = -1;

    double p_lower;
    double p_upper;
    double p;

    if (side_in == 'A') {
        if (this->lookup_incidentA.at(idx_in).size() == 0) {
            std::cout << "WARNING: \"not a valid incident mode\" exception "
                         "occurred at interface."
                      << std::endl;
            std::cout << "Particle will be discarded." << std::endl;
            return 'X';
        }

        // obtain range of cumulative probabilities pertaining to incident mode
        p_lower = std::get<2>(this->lookup_incidentA.at(idx_in).front());
        p_upper = std::get<2>(this->lookup_incidentA.at(idx_in).back());
        p = rng.uniform(p_lower, p_upper);

        for (std::size_t subidx = 0;
             subidx < this->lookup_incidentA.at(idx_in).size();
             subidx++) {
            if (std::get<2>(this->lookup_incidentA.at(idx_in).at(subidx)) >=
                p) {
                side_out =
                    std::get<0>(this->lookup_incidentA.at(idx_in).at(subidx));
                idx_out =
                    std::get<1>(this->lookup_incidentA.at(idx_in).at(subidx));
                break;
            }
        }
    }

    else { // side_in=='B'

        if (this->lookup_incidentB.at(idx_in).size() == 0) {
            std::cout << "WARNING: \"not a valid incident mode\" exception "
                         "occurred at interface."
                      << std::endl;
            std::cout << "Particle will be discarded." << std::endl;
            return 'X';
        }

        // obtain range of cumulative probabilities pertaining to incident mode
        p_lower = std::get<2>(this->lookup_incidentB.at(idx_in).front());
        p_upper = std::get<2>(this->lookup_incidentB.at(idx_in).back());
        p = rng.uniform(p_lower, p_upper);

        for (std::size_t subidx = 0;
             subidx < this->lookup_incidentB.at(idx_in).size();
             subidx++) {
            if (std::get<2>(this->lookup_incidentB.at(idx_in).at(subidx)) >=
                p) {
                side_out =
                    std::get<0>(this->lookup_incidentB.at(idx_in).at(subidx));
                idx_out =
                    std::get<1>(this->lookup_incidentB.at(idx_in).at(subidx));
                break;
            }
        }
    }

    // update the particle with the selected mode
    particle.q = idx_out / this->Nbranches[side_out];
    particle.alpha = idx_out % this->Nbranches[side_out];

    return side_out;
}


std::vector<Elastic_interface_distribution::dos_tuple>
Elastic_interface_distribution::get_all_modes(const Gamma_grid& gridA,
                                              const Gamma_grid& gridB,
                                              double scalebroad) const {
    auto ntotA = this->nqA * this->nmodesA;
    auto ntotB = this->nqB * this->nmodesB;

    std::vector<dos_tuple> nruter;
    nruter.reserve(ntotA + ntotB);

    // Process all modes from A.
    for (std::size_t iq = 0; iq < gridA.nqpoints; ++iq)
        for (std::size_t im = 0; im < this->nmodesA; ++im)
            nruter.emplace_back(std::make_tuple(
                'A', im, iq, Gaussian_for_DOS(gridA, iq, im, scalebroad)));

    // And then all modes from B.
    for (std::size_t iq = 0; iq < gridB.nqpoints; ++iq)
        for (std::size_t im = 0; im < this->nmodesB; ++im)
            nruter.emplace_back(std::make_tuple(
                'B', im, iq, Gaussian_for_DOS(gridB, iq, im, scalebroad)));
    return nruter;
}


Elastic_interface_distribution::Elastic_interface_distribution(
    const Gamma_grid& gridA,
    const Gamma_grid& gridB,
    double scalebroad,
    randutils::mt19937_rng& _rng)
    : nqA(gridA.nqpoints), nqB(gridB.nqpoints),
      nmodesA(
          static_cast<std::size_t>(gridA.get_spectrum_at_q(0).omega.size())),
      nmodesB(
          static_cast<std::size_t>(gridB.get_spectrum_at_q(0).omega.size())),
      VA(2. * constants::pi / std::abs(gridA.rlattvec.determinant())),
      VB(2. * constants::pi / std::abs(gridB.rlattvec.determinant())),
      directions(3, this->nqA * this->nmodesA + this->nqB * this->nmodesB),
      rng(_rng) {
    auto allmodes = this->get_all_modes(gridA, gridB, scalebroad);
    auto ntotA = this->nqA * this->nmodesA;
    auto ntotB = this->nqB * this->nmodesB;
    auto ntot = ntotA + ntotB;

    // Iterate through all pairs of modes looking for
    // transitions allowed by the conservation of energy.
    this->allowed.resize(ntot);
    this->cumulative.resize(ntot);

    for (std::size_t i = 0; i < ntot; ++i) {
        auto& details = allmodes[i];
        auto im = std::get<1>(details);
        auto iq = std::get<2>(details);
        std::size_t index;

        if (std::get<0>(details) == 'A') {
            index = im + iq * this->nmodesA;
            this->directions.col(index) =
                gridA.get_spectrum_at_q(iq).vg.col(im);
        }
        else {
            index = im + iq * this->nmodesB + ntotA;
            this->directions.col(index) =
                gridB.get_spectrum_at_q(iq).vg.col(im);
        }
        this->directions.col(index) /= this->directions.col(index).norm();
        auto omega = std::get<3>(details).mu;

        for (std::size_t ip = 0; ip < ntot; ++ip) {
            auto& detailsp = allmodes[ip];

            // The second mode must be energetically compatible with
            // the first one.
            if ((std::get<3>(detailsp).lbound > omega) ||
                (std::get<3>(detailsp).ubound < omega))
                continue;
            auto imp = std::get<1>(detailsp);
            auto iqp = std::get<2>(detailsp);
            std::size_t indexp;
            // Note that the contribution to the DOS is weighted by
            // the number density of each compound and the inverse
            // of the number of q points.
            double weight;

            if (std::get<0>(detailsp) == 'A') {
                indexp = imp + iqp * this->nmodesA;
                weight =
                    static_cast<double>(this->nmodesA) / this->nqA / this->VA;
            }
            else {
                indexp = imp + iqp * this->nmodesB + ntotA;
                weight =
                    static_cast<double>(this->nmodesB) / this->nqB / this->VB;
            }
            this->allowed[index].emplace_back(indexp);
            auto p = weight * std::get<3>(detailsp).get_contribution(omega);
            this->cumulative[index].push_back(
                (this->cumulative[index].size() == 0
                     ? 0.
                     : this->cumulative[index].back()) +
                p);
        }
    }

    // Normalize all cumulative distributions.
    for (auto& c : this->cumulative)
        if (c.size() != 0) {
            double last = c.back();

            for (auto& e : c)
                e /= last;
        }
}


char Elastic_interface_distribution::reemit(
    char incidence,
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    D_particle& particle) {
    auto ntotA = this->nmodesA * this->nqA;
    // Map the arguments to a single index.
    auto index = incidence == 'A'
                     ? particle.alpha + particle.q * this->nmodesA
                     : particle.alpha + particle.q * this->nmodesB + ntotA;
    double s = (incidence == 'A' ? 1. : -1.);

    if (s * normal.dot(directions.col(index)) <= 0.)
        throw value_error("not a valid incident mode");
    bool found = false;

    for (auto& d : this->allowed[index]) {
        double sp = (d >= ntotA ? 1. : -1.);

        if (sp * normal.dot(this->directions.col(d)) > 0.) {
            found = true;
            break;
        }
    }

    if (!found)
        throw value_error("no valid corresponding outgoing modes");
    // Invert the cumulative distribution function for that
    // incident mode so as to obtain a new random deviate.
    std::size_t deviate;
    double sp;

    do {
        auto r = std::lower_bound(this->cumulative[index].begin(),
                                  this->cumulative[index].end(),
                                  this->rng.uniform(0., 1.)) -
                 this->cumulative[index].begin();
        deviate = this->allowed[index][r];
        sp = (deviate >= ntotA ? 1. : -1.);
    } while (sp * normal.dot(this->directions.col(deviate)) <= 0.);

    // Update the particle data with this information.
    if (deviate >= ntotA) {
        deviate -= ntotA;
        particle.q = deviate / this->nmodesB;
        particle.alpha = deviate % this->nmodesB;
        return 'B';
    }
    else {
        particle.q = deviate / this->nmodesA;
        particle.alpha = deviate % this->nmodesA;
        return 'A';
    }
}


std::vector<Elastic_distribution::dos_tuple>
Elastic_distribution::get_all_modes(const Gamma_grid& grid,
                                    double scalebroad) const {
    std::vector<dos_tuple> nruter;

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq)
        for (std::size_t im = 0; im < this->nmodes; ++im) {
            nruter.emplace_back(std::make_tuple(
                im, iq, Gaussian_for_DOS(grid, iq, im, scalebroad)));
        }
    return nruter;
}


Elastic_distribution::Elastic_distribution(const Gamma_grid& grid,
                                           double scalebroad,
                                           randutils::mt19937_rng& _rng)
    : nq(grid.nqpoints),
      nmodes(static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size())),
      directions(3, nq * nmodes), rng(_rng) {
    auto allmodes = this->get_all_modes(grid, scalebroad);
    auto ntot = this->nq * this->nmodes;

    // Iterate through all pairs of modes looking for
    // transitions allowed by the conservation of energy.
    this->allowed.resize(ntot);
    this->cumulative.resize(ntot);

    for (std::size_t i = 0; i < ntot; ++i) {
        auto& details = allmodes[i];
        auto im = std::get<0>(details);
        auto iq = std::get<1>(details);
        // In the process, fill the directions array.
        directions.col(i) = grid.get_spectrum_at_q(iq).vg.col(im);
        directions.col(i) /= directions.col(i).norm();
        std::size_t index = im + iq * this->nmodes;
        auto omega = std::get<2>(details).mu;

        for (std::size_t ip = 0; ip < ntot; ++ip) {
            auto& detailsp = allmodes[ip];

            if ((std::get<2>(detailsp).lbound > omega) ||
                (std::get<2>(detailsp).ubound < omega))
                continue;
            auto imp = std::get<0>(detailsp);
            auto iqp = std::get<1>(detailsp);
            std::size_t indexp;
            indexp = imp + iqp * this->nmodes;
            this->allowed[index].emplace_back(indexp);
            auto p = std::get<2>(detailsp).get_contribution(omega);
            this->cumulative[index].push_back(
                (this->cumulative[index].size() == 0
                     ? 0.
                     : this->cumulative[index].back()) +
                p);
        }
    }

    // Normalize all cumulative distributions.
    for (auto& c : this->cumulative)
        if (c.size() != 0) {
            double last = c.back();

            for (auto& e : c)
                e /= last;
        }
}


void Elastic_distribution::scatter(D_particle& particle) {
    // Map the arguments to a single index.
    auto index = particle.alpha + particle.q * this->nmodes;
    // Invert the cumulative distribution function for that
    // incident mode so as to obtain a new random deviate.
    auto r = std::lower_bound(this->cumulative[index].begin(),
                              this->cumulative[index].end(),
                              this->rng.uniform(0., 1.)) -
             this->cumulative[index].begin();
    auto deviate = this->allowed[index][r];

    particle.q = deviate / this->nmodes;
    particle.alpha = deviate % this->nmodes;
}


void Elastic_distribution::scatter(
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    const int refsign,
    D_particle& particle) {
    if (normal.norm() == 0.)
        throw value_error("invalid vector");

    if (refsign == 0)
        throw value_error("refsign cannot be zero");
    auto index = particle.alpha + particle.q * this->nmodes;
    auto sign = signum(refsign);
    // Perform the same operation as in the one-argument version of
    // scatter() until we get an acceptable output.
    std::size_t deviate;

    do {
        auto r = std::lower_bound(this->cumulative[index].begin(),
                                  this->cumulative[index].end(),
                                  this->rng.uniform(0., 1.)) -
                 this->cumulative[index].begin();
        deviate = this->allowed[index][r];
    } while (signum(normal.dot(this->directions.col(deviate))) != sign);
    particle.q = deviate / this->nmodes;
    particle.alpha = deviate % this->nmodes;
}
} // namespace alma
