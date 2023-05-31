// Copyright 2021 Martí Raya Moreno
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
/// Definition of gpu_solver.hpp
//  ADD HERE ANY future implementation


#include <gpu_solver.hpp>
#include <cmakevars.hpp>
/// Symmetry preserving scaling algorithm
#include <unsupported/Eigen/src/IterativeSolvers/Scaling.h>

#if GPU_BUILD
#include <magma_v2.h>
#include <magmasparse.h>
#endif


namespace alma {
double linear_iterative_solver_gpu(Eigen::SparseMatrix<double,Eigen::RowMajor>& A_,
                  Eigen::VectorXd& b_,
                  Eigen::VectorXd& x_){

    double rel_err = 1.0e+25;
#if GPU_BUILD

    std::cout << "Starting the GPU solver" << std::endl;

    /// SCALING
    Eigen::IterScaling<Eigen::SparseMatrix<double,Eigen::RowMajor>> scal;
    scal.computeRef(A_);
    std::cout << "## Sparsity: " << A_.nonZeros()/(A_.rows()*A_.cols()) << std::endl;
    b_ = scal.LeftScaling().cwiseProduct(b_);
    x_ = scal.RightScaling().cwiseInverse().cwiseProduct(x_);

    std::cout << "##Init rel error " << (A_*x_ - b_).norm()/b_.norm() << std::endl;


    ///INIT MAGMA
    magma_init();
    magma_dopts dopts;
    magma_queue_t queue;
    magma_queue_create( 0, &queue );
    magma_d_matrix A, A_d, x, x_d, b, b_d;

    static const double one = MAGMA_D_MAKE(1.0, 0.0);
    static const double zero = MAGMA_D_MAKE(0.0, 0.0);

    // Here we make the interface between Eigen and MAGMA
    // Essentially we put sparse matrices in a way that MAGMA understands it
    magma_dcsrset( A_.rows(), A_.cols(), A_.outerIndexPtr(), A_.innerIndexPtr(), A_.valuePtr() ,&A, queue );
    magma_dvset( b_.rows(), b_.cols(), b_.data(), &b, queue );
    magma_dvset( x_.rows(), x_.cols(), x_.data(), &x, queue );

    // Copy the linear system to the device (i.e. GPU)
    magma_d_vtransfer( b, &b_d, Magma_CPU, Magma_DEV, queue );
    magma_queue_sync( queue );
    magma_d_mtransfer( A, &A_d, Magma_CPU, Magma_DEV, queue );
    magma_queue_sync( queue );
    magma_d_mtransfer( x, &x_d, Magma_CPU, Magma_DEV, queue );
    magma_queue_sync( queue );

    // Configure solver (TODO: provide user control of it)
    dopts.solver_par.solver = Magma_GMRES;
    dopts.solver_par.rtol = 1.0e-16;
    dopts.solver_par.atol = 1.0e-19;
    dopts.solver_par.maxiter = 5000;
    dopts.solver_par.restart = 70;

    // Use JACOBI preconditioner (not the best but it is efficient from memory and computational point of view)
    dopts.precond_par.solver  = Magma_JACOBI;

    magma_dsolverinfo_init( &dopts.solver_par, &dopts.precond_par, queue );
    // solve the linear system
    magma_d_precondsetup(A,b, &dopts.solver_par, &dopts.precond_par, queue );
    magma_d_solver( A_d, b_d, &x_d, &dopts, queue );

    // Sync CPU and GPU
    magma_queue_sync( queue );
    std::cout << "###############GPU SOLVER INFO##################\n";
    std::cout << "#Residuals of linear system:" << std::endl;
    std::cout << "  -Initial residual " <<     dopts.solver_par.init_res << std::endl;
    std::cout << "  -Final residual " <<       dopts.solver_par.final_res << std::endl;
    std::cout << "  -Iteratively residual "   << dopts.solver_par.iter_res << std::endl;
    std::cout << "  -info   "                 << dopts.solver_par.info  << std::endl;
    std::cout << "  -niter  "                 << dopts.solver_par.numiter  << std::endl;
    std::cout << "################################################\n"; 

    // Copy the solution vector back to the host and pass it back to the application
    magma_d_mtransfer( x_d, &x, Magma_DEV, Magma_CPU, queue );
    magma_queue_sync( queue );

    ///Return to Eigen
    for (int i = 0; i<x_.rows(); i++)
        x_(i) = (x.val)[i];

    rel_err = (A_*x_ - b_).norm()/b_.norm();

    x_ = scal.RightScaling().cwiseProduct(x_);

    magma_queue_sync( queue );

    // clean up the memory
    magma_dsolverinfo_free( &dopts.solver_par, &dopts.precond_par, queue );
    magma_d_mfree(&A_d, queue );
    magma_d_mfree(&A, queue );
    magma_d_mfree(&x_d, queue );
    magma_d_mfree(&x, queue);
    magma_d_mfree(&b_d, queue );
    magma_d_mfree(&b, queue );
    magma_queue_sync( queue );

    // finalize MAGMA
    magma_queue_destroy( queue );
    magma_finalize();
#endif
    return rel_error;
}
}

