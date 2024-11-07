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

#pragma once

/// @file
/// Classes and functions used to compute the
/// transmission coefficient between 
/// two materials



#include <array>
#include <Eigen/Dense>
#include <structures.hpp>
#include <qpoint_grid.hpp>
#include <dos.hpp>
#include <processes.hpp>
#include <dynamical_matrix.hpp>

namespace alma {


/// Computes the conductance of an interface
/// @param[in] poscar - the cell informaton of the material in the A side
/// @param[in] grid   - the phonon information of the material in the A side
/// @param[in] axis   - the transport axis along the material in the A side
/// @param[in] alpha  - the transmission coefficients from A to B. Shape: (qpts,nbands) 
/// @param[in] Tref   - the reference temperature of the interface (i.e. the equilibrium one)
/// @param[in] world  - the mpi communicator
double  interface_conductance(const Crystal_structure& poscar,
                              const Gamma_grid& grid,
                              const Eigen::Ref<const Eigen::Vector3d> axis,
                              const Eigen::Ref<const Eigen::ArrayXd> alpha,
			      const double Tref,
			      boost::mpi::communicator& world);

/// Base class to model interface transmission
/// coefficients from A to B
class interface {
    
protected:
    /// description of the unit cell
    std::array<std::shared_ptr<Crystal_structure>,2> poscar;
    /// phonon spectrum on a regular q-point grid
    std::array<std::shared_ptr<Gamma_grid>,2> grid;
    /// symmetry operations object
    std::array<std::shared_ptr<Symmetry_operations>,2> syms;
    ///Surfces orientation
    std::array<Eigen::Vector3d,2> orientation;
    
public:
    
    /// Empty constructor
    interface() = default;
    
    /// Basic constructor
    /// @param[in] poscar_A - the cell informaton of the materia in the A side
    /// @param[in] grid_A   - the phonon information of the materia in the A side
    /// @param[in] syms_A   - the symmetry information of the materia in the A side
    /// @param[in] axis_A   - the transport axis along the materia in the A side (pointing into the materia in the B side)
    /// @param[in] poscar_B   - the cell informaton of the materia in the B side
    /// @param[in] grid_B     - the phonon information of the materia in the B side
    /// @param[in] syms_B     - the symmetry information of the materia in the B side
    /// @param[in] axis_B     - the transport axis along the materia in the B side (pointing into the materia in the B side from the materia in the A side)
    interface(Crystal_structure& poscar_A,
              Gamma_grid& grid_A,
              Symmetry_operations& syms_A,
              Eigen::Vector3d axis_A,
              Crystal_structure& poscar_B,
              Gamma_grid& grid_B,
              Symmetry_operations& syms_B,
              Eigen::Vector3d axis_B);
              
    
    ///Virtual trivial destructor
    virtual ~interface(){};
    

    /// Virtual function to obtain the transmission from one side
    /// to the other
    /// @param[in] iq - q-point index
    /// @param[in] ib - phonon band index
    /// @return transmission probability (double)
    virtual double get_transmission(std::size_t iq, std::size_t ib) const {
        std::cerr << "ERROR: This should not be used" << std::endl;
        exit(EXIT_FAILURE);
        return 0.;
    };
    
    
};


/// Interface model in which all phonons have equal
/// transmission coefficient. Thus, no harmonic
/// information is used to obtain such a coefficient
/// Therefore, it is a rather poor model, which 
/// should only be used for testing and debuging

class constant_interface : public interface {

private:
    
    /// Transmission probability from the materia in the A side to the materia in the B side
    /// Within constant class, that is deemed equal for all modes, without
    /// taking into account the harmonic properties
    const double constant_alpha;

public:
    
    
    /// @param[in] poscar_A - the cell informaton of the materia in the A side
    /// @param[in] grid_A   - the phonon information of the materia in the A side
    /// @param[in] syms_A   - the symmetry information of the materia in the A side
    /// @param[in] axis_A   - the transport axis along the materia in the A side (pointing into the materia in the B side)
    /// @param[in] poscar_B   - the cell informaton of the materia in the B side
    /// @param[in] grid_B     - the phonon information of the materia in the B side
    /// @param[in] syms_B     - the symmetry information of the materia in the B side
    /// @param[in] axis_B     - the transport axis along the materia in the B side (pointing into the materia in the B side from the materia in the A side)
    /// @param[in] constant_alpha_ - constant transmission probability for all phonon modes
    constant_interface(Crystal_structure& poscar_A,
                       Gamma_grid& grid_A,
                       Symmetry_operations& syms_A,
                       Eigen::Vector3d axis_A,
                       Crystal_structure& poscar_B,
                       Gamma_grid& grid_B,
                       Symmetry_operations& syms_B,
                       Eigen::Vector3d axis_B,
                       double constant_alpha_);
    
    /// Override of get_transmission of base class, see it for more information
    /// @param[in] iq - q-point index
    /// @param[in] ib - phonon band index
    /// @return transmission probability (double)
    double get_transmission(std::size_t iq, std::size_t ib) const;
    
};

/// Interface model using DMM it has most part of code rewritten from
/// Diffuse_mismatch_distribution in sampling.hpp

class DMM_interface : public interface {

private:

    /// Number of branches
    std::size_t Nbranches_A , Nbranches_B;
    std::map<std::string,std::size_t> Nbranches;
    /// Number of phonon modes
    std::size_t Ntot_A,  Ntot_B;
    /// Unit cell volumes
    double Vuc_A , Vuc_B;
    /// Tuple describing the origin of each available mode in A or B and its
    /// contribution to the DOS.
    /// The first element is the material side
    /// The second element is a branch index
    /// The third element is a q-point index
    /// The fourth element is the projection of its velocity on the normal to
    /// the surface. The fifth element is the heat capacity The sixth element is
    /// a contribution to the DOS
    using dos_tuple = std::
        tuple<std::string, std::size_t, std::size_t, double, double, Gaussian_for_DOS>;
    /// Short notation for tuple specifying the material side, mode index, and
    /// cumulative probability
    typedef std::tuple<std::string, std::size_t, double> lookup_entry;
    /// Lookup table for incident materia in the A side/materia in the B side modes
    std::vector<std::vector<lookup_entry>> lookup_incident_A;
    std::vector<std::vector<lookup_entry>> lookup_incident_B;
    /// Reference temperature (for computing heat capacities)
    double Tref;
    ///
    /// Gather information about all available phonon modes.
    ///
    /// @param[in] grid_A - phonon spectrum of material A
    /// @param[in] grid_B - phonon spectrum of material B
    /// @param[in] normal - a normal vector pointing from A to B.
    /// @param[in] scalebroad - factor modulating all the broadenings
    /// @return a vector of dos_tuples describing all modes
    std::vector<dos_tuple> get_modes(
        const Gamma_grid& grid_A,
        const Gamma_grid& grid_B,
        const Eigen::Ref<const Eigen::Vector3d>& axis_A,
        const Eigen::Ref<const Eigen::Vector3d>& axis_B,
        double scalebroad);


public:


    /// @param[in] poscar_A - the cell informaton of the materia in the A side
    /// @param[in] grid_A   - the phonon information of the materia in the A side
    /// @param[in] syms_A   - the symmetry information of the materia in the A side
    /// @param[in] axis_A   - the transport axis along the materia in the A side (pointing into the materia in the B side)
    /// @param[in] poscar_B   - the cell information of the materia in the B side
    /// @param[in] grid_B     - the phonon information of the materia in the B side
    /// @param[in] syms_B     - the symmetry information of the materia in the B side
    /// @param[in] axis_B     - the transport axis along the materia in the B side (pointing into the materia in the B side from the materia in the A side)
    /// @param[in] Tref_       - the reference temperature (computing heat capacities)
    DMM_interface(Crystal_structure& poscar_A,
                  Gamma_grid& grid_A,
                  Symmetry_operations& syms_A,
                  Eigen::Vector3d& axis_A,
                  Crystal_structure& poscar_B,
                  Gamma_grid& grid_B,
                  Symmetry_operations& syms_B,
                  Eigen::Vector3d& axis_B,
                  double Tref_ = 300.);

    /// Override of get_transmission of base class, see it for more information
    /// @param[in] iq - q-point index
    /// @param[in] ib - phonon band index
    /// @return transmission probability (double)
    double get_transmission(std::size_t iq, std::size_t ib) const;
};


/// Interface model using pseudo-AMM, it uses an
/// interpolator to satify the momentum and energy constraints

class AMM_interface : public interface {

private:

    /// Class to contain oriented mesh
    /// the qpoint is oriented with z-axis pointing to 
    /// transport axis
    class rotated_qmesh {

    private:
        /// Limiting values of the 1st BZ
        double qmin , qmax;
        /// List of qpoints in cartesian coordinates
        /// given in [qparx,qpary,qparz,qnorm] 
        Eigen::MatrixXd cq;
        /// Rotation matrix and inverse
        Eigen::MatrixXd rot, invrot;
        /// Reciprocal-lattice and its inverse
        Eigen::Matrix3d rlat, invrlat;
        
    public:

        /// Basic constructor
        /// @param[in] grid - object containing qgrid properties
        /// @param[in] poscar - crystal structure data
        /// @param[in] axis - projection axis
        rotated_qmesh(const alma::Gamma_grid& grid,
                    const alma::Crystal_structure& poscar,
                    const Eigen::Vector3d&  axis);
        
        /// Given unrotated q provide unique set of
        /// cartesian qpoints conserving the q
        /// in the same material
        /// @param[in] q - unrotated qred
        std::vector<Eigen::Vector3d> get_qlist(const Eigen::Vector3d q);
        
        /// Given unrotated q provide unique set of
        /// cartesian qpoints conserving the q
        /// in a different material
        /// @param[in] q - unrotated qred
        /// @param[in] omesh - rotated output mesh
        std::vector<Eigen::Vector3d> get_qlist(const Eigen::Vector3d q,
                    const rotated_qmesh& omesh);
    };
    
    /// The possible outcomes of phonon arriving the interface
    enum class interface_process {bounce = -1, transmit = 1 };
    
    /// Returns the velocites complying with all the constraints
    /// @param[in] qlist - list of all q-vectors complying with parallel momentum conservation
    /// @param[in] interpolator - the intorpolation object
    /// @param[in] omega_target - the target energy
    /// @param[in] u_axis       - the transport axis
    /// @param[in] what         - the process
    /// @return vector containing the projected velocity for the pseudo-AMM transmission probability computation
    std::vector<double> 
    get_conserve_energy(std::vector<Eigen::Vector3d>& qlist,
                         Dynamical_matrix_builder& dyn,
                         double omega_target,
                         Eigen::Vector3d u_axis,
                         interface_process what);
    
    /// Array containing the probability
    Eigen::ArrayXXd transmission;
    
public:

    /// @param[in] poscar_A - the cell informaton of the materia in the A side
    /// @param[in] grid_A   - the phonon information of the materia in the A side
    /// @param[in] syms_A   - the symmetry information of the materia in the A side
    /// @param[in] axis_A   - the transport axis along the materia in the A side (pointing into the materia in the B side)
    /// @param[in] poscar_B   - the cell information of the materia in the B side
    /// @param[in] grid_B     - the phonon information of the materia in the B side
    /// @param[in] syms_B     - the symmetry information of the materia in the B side
    /// @param[in] axis_B     - the transport axis along the materia in the B side (pointing into the materia in the B side from the materia in the A side)
    AMM_interface(Crystal_structure& poscar_A,
                  Gamma_grid& grid_A,
                  Symmetry_operations& syms_A,
                  Eigen::Vector3d& axis_A,
                  Dynamical_matrix_builder& dyn_A,
                  Crystal_structure& poscar_B,
                  Gamma_grid& grid_B,
                  Symmetry_operations& syms_B,
                  Eigen::Vector3d& axis_B,
                  Dynamical_matrix_builder& dyn_B,
                  boost::mpi::communicator& world);
    
    /// Override of get_transmission of base class, see it for more information
    /// @param[in] iq - q-point index
    /// @param[in] ib - phonon band index
    /// @return transmission probability (double)
    double get_transmission(std::size_t iq, std::size_t ib) const;
    
};

} // namespace alma
