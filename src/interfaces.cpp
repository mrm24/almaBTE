// Copyright 2022-2024 Martí Raya Moreno
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

/// This file contains the definitions of interfaces.hpp

#include <interfaces.hpp>

namespace alma {


interface::interface(Crystal_structure& poscar_A,
                     Gamma_grid& grid_A,
                     Symmetry_operations& syms_A,
                     Eigen::Vector3d axis_A,
                     Crystal_structure& poscar_B,
                     Gamma_grid& grid_B,
                     Symmetry_operations& syms_B,
                     Eigen::Vector3d axis_B) : 
                     
                     poscar{ {std::make_shared<Crystal_structure>(poscar_A),std::make_shared<Crystal_structure>(poscar_B)} },
                     grid{ {std::make_shared<Gamma_grid>(grid_A),std::make_shared<Gamma_grid>(grid_B)} },
                     syms{ {std::make_shared<Symmetry_operations>(syms_A),std::make_shared<Symmetry_operations>(syms_B)} },
                     orientation{ {axis_A,axis_B} } {};


constant_interface::constant_interface(
                       Crystal_structure& poscar_A,
                       Gamma_grid& grid_A,
                       Symmetry_operations& syms_A,
                       Eigen::Vector3d axis_A,
                       Crystal_structure& poscar_B,
                       Gamma_grid& grid_B,
                       Symmetry_operations& syms_B,
                       Eigen::Vector3d axis_B,
                       double constant_alpha_) : 
                       interface(poscar_A,grid_A,syms_A,axis_A,
                           poscar_B,grid_B,syms_B,axis_B) , 
                 constant_alpha(constant_alpha_) {};

double constant_interface::get_transmission(
                            std::size_t iq, std::size_t ib) const {
    return constant_alpha;
}


std::vector<DMM_interface::dos_tuple> DMM_interface::get_modes(
        const Gamma_grid& grid_A,
        const Gamma_grid& grid_B,
        const Eigen::Ref<const Eigen::Vector3d>& axis_A,
        const Eigen::Ref<const Eigen::Vector3d>& axis_B,
        double scalebroad) {

    if (axis_A.norm() == 0. or axis_B.norm() == 0.) {
        std::cout << axis_A << std::endl << std::endl;
        std::cout << axis_B << std::endl;
        throw value_error("Invalid normal vectors");
    }

    std::vector<dos_tuple> result;

    Eigen::Vector3d u_A = axis_A / axis_A.norm();
    Eigen::Vector3d u_B   = axis_B / axis_B.norm();

    std::size_t Nq_A = grid_A.nqpoints;
    std::size_t Nbranches_A = grid_A.get_spectrum_at_q(0).omega.size();

    std::size_t Nq_B = grid_B.nqpoints;
    std::size_t Nbranches_B = grid_B.get_spectrum_at_q(0).omega.size();

    result.reserve(Nq_A * Nbranches_A + Nq_B * Nbranches_B);

    // Process all modes in the semi grid
    for (std::size_t nq = 0; nq < Nq_A; nq++) {
        auto& spectrum = grid_A.get_spectrum_at_q(nq);

        for (std::size_t nbranch = 0; nbranch < Nbranches_A; nbranch++) {
            double vn = u_A.dot(spectrum.vg.col(nbranch).matrix());
            double C =
                alma::bose_einstein_kernel(spectrum.omega(nbranch),
                                           this->Tref) /
                (alma::constants::kB * grid_A.nqpoints * this->Vuc_A);
            result.emplace_back(std::make_tuple(
                "A",
                nbranch,
                nq,
                vn,
                C,
                Gaussian_for_DOS(grid_A, nq, nbranch, scalebroad)));
        }
    }

    // Process all modes in the B grid
    for (std::size_t nq = 0; nq < Nq_B; nq++) {
        auto& spectrum = grid_B.get_spectrum_at_q(nq);

        for (std::size_t nbranch = 0; nbranch < Nbranches_B; nbranch++) {
            double vn = u_B.dot(spectrum.vg.col(nbranch).matrix());
            double C =
                alma::bose_einstein_kernel(spectrum.omega(nbranch),
                                           this->Tref) /
                (alma::constants::kB * grid_B.nqpoints * this->Vuc_B);
            result.emplace_back(std::make_tuple(
                "B",
                nbranch,
                nq,
                vn,
                C,
                Gaussian_for_DOS(grid_B, nq, nbranch, scalebroad)));
        }
    }

    return result;
}

DMM_interface::DMM_interface(Crystal_structure& poscar_A,
                  Gamma_grid& grid_A,
                  Symmetry_operations& syms_A,
                  Eigen::Vector3d& axis_A,
                  Crystal_structure& poscar_B,
                  Gamma_grid& grid_B,
                  Symmetry_operations& syms_B,
                  Eigen::Vector3d& axis_B,
                  double Tref_) :
                  interface(poscar_A,grid_A,syms_A,axis_A,
                           poscar_B,grid_B,syms_B,axis_B), Tref(Tref_) {

    this->Nbranches_A = grid_A.get_spectrum_at_q(0).omega.size();
    this->Ntot_A = grid_A.get_spectrum_at_q(0).omega.size() * grid_A.nqpoints;

    this->Nbranches_B = grid_B.get_spectrum_at_q(0).omega.size();
    this->Ntot_B = grid_B.get_spectrum_at_q(0).omega.size() * grid_B.nqpoints;

    this->Vuc_A = poscar_A.V;
    this->Vuc_B = poscar_B.V;
    
    this->Nbranches["A"] = this->Nbranches_A;
    this->Nbranches["B"] = this->Nbranches_B;
    
    
    std::vector<DMM_interface::dos_tuple> allmodes =
        this->get_modes(grid_A, grid_B, axis_A, axis_B, 0.1);
    
    
    // INITIALISE LOOKUP TABLES
    lookup_incident_A.reserve(this->Ntot_A);
    for (std::size_t idxA = 0; idxA < this->Ntot_A; idxA++) {
        std::vector<lookup_entry> empty;
        this->lookup_incident_A.emplace_back(empty);
    }

    lookup_incident_B.reserve(this->Ntot_B);
    for (std::size_t idxB = 0; idxB < this->Ntot_B; idxB++) {
        std::vector<lookup_entry> empty;
        this->lookup_incident_B.emplace_back(empty);
    }

    // DETERMINE ALL ALLOWED TRANSITIONS

    for (std::size_t listidx = 0; listidx < allmodes.size(); listidx++) {
        DMM_interface::dos_tuple& mode_in =
            allmodes.at(listidx);
        std::string side_in = std::get<0>(mode_in);
        double vproj_in = std::get<3>(mode_in);

        // only process modes incident on the interface

        if ((side_in == "A"  && vproj_in > 0.0) ||
            (side_in == "B"  && vproj_in < 0.0)) {
            // construct overall index from branch and q
            std::size_t idx_in =
                std::get<1>(mode_in) +
                std::get<2>(mode_in) * this->Nbranches[side_in];

            // obtain target frequency
            double omega_target = std::get<5>(mode_in).mu;

            // search for compatible transitions

            for (std::size_t searchidx = 0; searchidx < allmodes.size();
                 searchidx++) {
                DMM_interface::dos_tuple& candidate =
                    allmodes.at(searchidx);
                std::string side_out = std::get<0>(candidate);
                double vproj_out = std::get<3>(candidate);

                // check if the candidate points AWAY from the interface
                bool emission = ((side_out == "A"  && vproj_out < 0.0) ||
                                 (side_out == "B"  && vproj_out > 0.0));

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

                    if (side_in == "A" ) {
                        this->lookup_incident_A.at(idx_in).emplace_back(entry);
                        //totalsumA += raw_probability;
                    }
                    else {
                        this->lookup_incident_B.at(idx_in).emplace_back(entry);
                        //totalsumB += raw_probability;
                    }

                } // end compatible candidate
            }     // done scanning over candidates
        }         // end incident mode
    }             // done scanning over modes
    
    // fix incident modes that currently have no valid output mode
    // by looking for the closest compatible reflection

    for (std::size_t listidx = 0; listidx < allmodes.size(); listidx++) {
        DMM_interface::dos_tuple& mode_in =
            allmodes.at(listidx);
        auto side_in = std::get<0>(mode_in);
        double vproj_in = std::get<3>(mode_in);
        std::size_t idx_in = std::get<1>(mode_in) +
                             std::get<2>(mode_in) * this->Nbranches[side_in];

        // only process incident modes that need fixing

        bool fixA = false;
        if (side_in == "A" ) {
            fixA =
                vproj_in > 0.0 && this->lookup_incident_A.at(idx_in).size() == 0;
        }

        bool fixB = false;
        if (side_in == "B" ) {
            fixB =
                vproj_in < 0.0 && this->lookup_incident_B.at(idx_in).size() == 0;
        }

        if (fixA || fixB) {
            // obtain target frequency
            double omega_target = std::get<5>(mode_in).mu;

            // scan over possible reflections

            int idx_out = -1;
            std::string side_out;
            double deltamin = 1e300;
            double vproj_out;
            double C_out;

            for (std::size_t searchidx = 0; searchidx < allmodes.size();
                 searchidx++) {
                DMM_interface::dos_tuple& candidate =
                    allmodes.at(searchidx);
                side_out = std::get<0>(candidate);
                double vproj_candidate = std::get<3>(candidate);

                // only consider reflections pointing AWAY from the interface
                bool reflection = (side_in == side_out);
                bool emission = (side_out == "A"  && vproj_candidate < 0.0) ||
                                (side_out == "B"  && vproj_candidate > 0.0);

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

                double raw_probability =
                    C_out * std::abs(vproj_out);
                lookup_entry entry(
                    std::make_tuple(side_out, idx_out, raw_probability));
                if (side_in == "A" ) {
                    this->lookup_incident_A.at(idx_in).emplace_back(entry);
                }
                else {
                    this->lookup_incident_B.at(idx_in).emplace_back(entry);
                }
            }
        } // end mode needs fixing
    }     // end scanning over all modes
}


double DMM_interface::get_transmission(std::size_t iq, std::size_t ib) const {
    
    auto idx_in = ib + iq * (this->Nbranches).at("A");
    
    double all = 0.;
    double trans = 0.;
    
    if (this->lookup_incident_A.at(idx_in).size() == 0) {
        throw value_error("Not outgoing mode has found");
    }
    
    
    for (std::size_t sub_idx_in = 0;
        sub_idx_in < this->lookup_incident_A.at(idx_in).size();
        sub_idx_in++) {
        lookup_entry entry = this->lookup_incident_A.at(idx_in).at(sub_idx_in);
        all += std::get<2>(entry);
        if (std::get<0>(entry)=="B")
            trans += std::get<2>(entry);
    }
    
    return trans/all;
    
}

AMM_interface::rotated_qmesh::rotated_qmesh(const alma::Gamma_grid& grid,    
        const alma::Crystal_structure& poscar,
        const Eigen::Vector3d&  axis) :
            rlat(poscar.rlattvec), 
            invrlat(poscar.rlattvec.inverse()){
            
        /// We compute the matrices to go from [qx,qy,qz] to [qparx,qpary,qparz,qnorm]

        Eigen::Vector3d naxis = axis / axis.norm();

        double ux = naxis(0);
        double uy = naxis(1);
        double uz = naxis(2);

        this->rot.resize(4,3);

        this->rot << 1 - ux*ux , -uy*ux , -uz*ux,
                      -ux*uy   , 1-uy*uy, -uz*uy,
                      -ux*uz   , -uy*uz ,1-uz*uz,
                      ux       ,  uy    , uz    ;

        /// Compute the Moore-Penrose inverse
        Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cqr(this->rot);
        this->invrot = cqr.pseudoInverse();

        /// Get rotated qpoints and min and max in transport direction
        int na = grid.na;
        int nb = grid.nb;
        int nc = grid.nc;


        this->cq.resize((na+1)*(nb+1)*(nc+1),4);

        int counter = 0;
        for (int i = 0; i <= na ; i++) for (int j = 0; j <= nb ; j++) for (int k=0; k <= nc; k++) {
            Eigen::Vector3d q;
            q << static_cast<double>(i)/na , static_cast<double>(j)/nb, static_cast<double>(k)/nc;
            q = poscar.rlattvec * q;
            cq.row(counter) = this->rot * q;
            counter++;
        }

        this->qmin = this->cq.col(3).minCoeff();
        this->qmax = this->cq.col(3).maxCoeff();

}
        
        
std::vector<Eigen::Vector3d> 
    AMM_interface::rotated_qmesh::get_qlist(const Eigen::Vector3d q){
        /// Get rotated vector
        Eigen::Vector4d qrot = this->rot * this->rlat * q;

        double q1 = qrot(0);
        double q2 = qrot(1);
        double q3 = qrot(2);

        int Np = 501;

        auto dq = (this->qmax - this->qmin) / (Np - 1);
        Eigen::VectorXd list_q4 = Eigen::VectorXd::LinSpaced(Np,
            this->qmin,this->qmax + dq);

        std::vector<Eigen::Vector3d> qlist;
        qlist.reserve(Np);

        for (int i = 0; i < Np; i++) {
            Eigen::Vector4d qcandidate;
            Eigen::Vector3d qcart, qred;
            qcandidate << q1 , q2 , q3, list_q4(i);
            /// Unrotate the candidate
            qcart = this->invrot * qcandidate;
            qlist.push_back(qcart);
        }

        return qlist;
}

std::vector<Eigen::Vector3d> AMM_interface::rotated_qmesh::get_qlist(const Eigen::Vector3d q,
            const AMM_interface::rotated_qmesh& omesh){
    
        /// Get rotated vector
        Eigen::Vector4d qrot = this->rot * this->rlat * q;

        double q1 = qrot(0);
        double q2 = qrot(1);
        double q3 = qrot(2);

        int Np = 501;

        auto dq = (this->qmax - this->qmin) / (Np - 1);
        Eigen::VectorXd list_q4 = Eigen::VectorXd::LinSpaced(Np,
            this->qmin,this->qmax + dq);

        std::vector<Eigen::Vector3d> qlist;
        qlist.reserve(Np);

        for (int i = 0; i < Np; i++) {
            Eigen::Vector4d qcandidate;
            Eigen::Vector3d qcart;
            qcandidate << q1 , q2 , q3, list_q4(i);
            /// Unrotate the candidate to new lattice
            qcart = omesh.invrot * qcandidate;

            qlist.push_back(qcart);
        }

        return qlist;
}


std::vector<double> 
AMM_interface::get_conserve_energy(std::vector<Eigen::Vector3d>& qlist,
                                   Dynamical_matrix_builder& dyn,
                                   double omega_target,
                                   Eigen::Vector3d u_axis,
                                   AMM_interface::interface_process what) {
    
    
    std::vector<double> vouts;
    
    /// Ignore the upper bound

    Eigen::Vector3d n_axis = u_axis / u_axis.norm();

    for (std::size_t i = 0; i < qlist.size() - 1; i++) {
        
        auto q0 = qlist[i];
        auto q1 = qlist[i+1];

        auto data0 = dyn.get_spectrum(qlist[i]);
        auto data1 = dyn.get_spectrum(qlist[i+1]);

        auto nb = data0->omega.size();


        for (int iband = 0; iband < nb; iband++) {

            double w0 = data0->omega(iband);
            double w1 = data1->omega(iband);

            ///Linear interpolate only in this case
            if ((omega_target >= w0 and omega_target <= w1) or
                (omega_target >= w1 and omega_target <= w0)) {

                double dq    = (q1 - q0).norm();
                double slope = (w1 - w0) / dq;

                /// Compute the qpoint conserving the energy by linear interpolation
                double qdisp =  (omega_target - w0) / slope;

                Eigen::Vector3d q_target = q0 + (q1 - q0)/dq * qdisp;

                auto data_target = dyn.get_spectrum(q_target);

                Eigen::Vector3d vout = data_target->vg.col(iband);

                double vproj = vout.dot(n_axis);

                if (what == interface_process::bounce) {
                    if (vproj < 0.) {
                        vouts.emplace_back(vproj);
                    }
                }

                if (what == interface_process::transmit) {
                    if (vproj > 0.) {
                        vouts.emplace_back(vproj);
                    }
                }
            }
        }
    }
    
    return vouts;
}

AMM_interface::AMM_interface(Crystal_structure& poscar_A,
                  Gamma_grid& grid_A,
                  Symmetry_operations& syms_A,
                  Eigen::Vector3d& axis_A,
                  Dynamical_matrix_builder& dyn_A,
                  Crystal_structure& poscar_B,
                  Gamma_grid& grid_B,
                  Symmetry_operations& syms_B,
                  Eigen::Vector3d& axis_B,
                  Dynamical_matrix_builder& dyn_B,
                  boost::mpi::communicator& world) :
                  interface(poscar_A,grid_A,syms_A,axis_A,
                           poscar_B,grid_B,syms_B,axis_B) {
                      
    /// Get sizes
    int nbands_A = grid_A.get_spectrum_at_q(0).omega.size();
    int nq_A     = grid_A.nqpoints;
    
    /// Make room for transmission coefficients
    this->transmission.resize(nbands_A,nq_A);
    this->transmission.setZero();
    
    /// Get rotated qmeshes
    auto rot_mesh_A = rotated_qmesh(grid_A,poscar_A,axis_A.normalized());
    auto rot_mesh_B = rotated_qmesh(grid_B,poscar_B,axis_B.normalized());
    
    /// Only do it partially (each proc takes care of a part)
    auto limits = alma::my_jobs(nq_A, world.size(), world.rank());


    for (std::size_t iq = limits[0]; iq < limits[1]; iq++){
        
        /// Get the reduced coordinates of the qpoint
        auto indexes = grid_A.one_to_three(iq);
        Eigen::Vector3d qred;         
        
        qred <<  static_cast<double>(indexes[0])/grid_A.na,
                 static_cast<double>(indexes[1])/grid_A.nb,
                 static_cast<double>(indexes[2])/grid_A.nc;
                                      
        
        auto sp = grid_A.get_spectrum_at_q(iq);
        
        /// Get the list of qvectors complying with parallel
        /// momentum conservation
        auto qs_bounce = rot_mesh_A.get_qlist(qred); 
        auto qs_trans  = rot_mesh_A.get_qlist(qred,rot_mesh_B);

        
        for (int ib = 0; ib < nbands_A; ib++) {
            /// Get velocity
            Eigen::Vector3d vg = sp.vg.col(ib); 
            double v_i = vg.dot(axis_A.normalized());
            
            /// Ignore states not going into the interface
            if (v_i <= 0. or alma::almost_equal(v_i,0.))
                continue;
            
            double omega = sp.omega(ib);
            
            /// Get incident velocities
            auto vs_incident = get_conserve_energy(qs_bounce,
                                                   dyn_A,
                                                   omega,
                                                   axis_A.normalized(),
                                                   interface_process::transmit);


            /// Get reflected velocities
            auto vs_bounce = get_conserve_energy(qs_bounce,
                                                 dyn_A,
                                                 omega,
                                                 axis_A.normalized(),
                                                 interface_process::bounce);
                                                 
            /// Get transmitted velocities
            auto vs_trans =  get_conserve_energy(qs_trans,
                                                 dyn_B,
                                                 omega,
                                                 axis_B.normalized(),
                                                 interface_process::transmit);
            
            int N_i = vs_incident.size();
            int N_t = vs_trans.size();
            int N_r = vs_bounce.size();

            v_i = (N_i == 0) ? 0. :
                1./(N_i*N_i) *
              std::accumulate(vs_incident.begin(),vs_incident.end(),0.);

            double v_r = (N_r == 0) ? 0. :
              1./(N_r*N_r) * std::abs(
              std::accumulate(vs_bounce.begin(),vs_bounce.end(),0.));
              
            double v_t = (N_t == 0) ? 0. :
              1./(N_t*N_t) *
              std::accumulate(vs_trans.begin(),vs_trans.end(),0.); 

            double sqrt_in = v_r * v_i + v_i * v_t - v_t * v_r;

            double tcoeff = (v_t / v_i) * boost::math::pow<2>(
                (v_r + std::sqrt(sqrt_in))/( v_t + v_r ));

            /// That means total internal reflexion other is for numerical problems
            if (std::isnan(tcoeff) or std::isinf(tcoeff))
                continue;
            this->transmission(ib,iq) = tcoeff;
        }
    }

    /// Share among procs_
    Eigen::ArrayXXd transmission_(this->transmission);
    boost::mpi::all_reduce(world,transmission_.data(),
                           transmission_.size(),
                           this->transmission.data(),
                           std::plus<double>());

}

double AMM_interface::get_transmission(std::size_t iq, std::size_t ib) const {
    return this->transmission(ib,iq);
}

} // namespace alma


