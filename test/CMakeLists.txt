# Build all the necessary executables.
add_executable(threeph_soa_test threeph_soa_test.cpp)
target_link_libraries(threeph_soa_test alma_framework gtest)
add_test(NAME threeph_soa_test COMMAND threeph_soa_test)
add_test(NAME threeph_soa_mpi_test COMMAND ${MPIEXEC_EXECUTABLE}
  ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:threeph_soa_test>
  ${MPIEXEC_POSTFLAGS})
set_tests_properties(threeph_soa_test threeph_soa_mpi_test PROPERTIES
  TIMEOUT 120 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")

add_executable(metatest metatest.cpp)
target_link_libraries(metatest gtest gtest_main)

add_executable(poscar_test poscar_test.cpp)
target_link_libraries(poscar_test alma_framework gtest gtest_main)

add_executable(force_constants_test force_constants_test.cpp)
target_link_libraries(force_constants_test alma_framework gtest gtest_main)

add_executable(born_test born_test.cpp)
target_link_libraries(born_test alma_framework gtest gtest_main)

add_executable(mass_test mass_test.cpp)
target_link_libraries(mass_test alma_framework gtest gtest_main)

add_executable(vc_test vc_test.cpp)
target_link_libraries(vc_test alma_framework gtest gtest_main)

add_executable(dynamical_matrix_test dynamical_matrix_test.cpp)
target_link_libraries(dynamical_matrix_test alma_framework gtest gtest_main)

add_executable(symmetry_test symmetry_test.cpp)
target_link_libraries(symmetry_test alma_framework gtest gtest_main)

add_executable(mpi_test mpi_test.cpp)
target_link_libraries(mpi_test alma_framework gtest gtest_main)

add_executable(qpoint_grid_test qpoint_grid_test.cpp)
target_link_libraries(qpoint_grid_test alma_framework gtest gtest_main)

add_executable(cv_test cv_test.cpp)
target_link_libraries(cv_test alma_framework gtest gtest_main)

add_executable(kappa_sg_test kappa_sg_test.cpp)
target_link_libraries(kappa_sg_test alma_framework gtest gtest_main)

add_executable(processes_test processes_test.cpp)
target_link_libraries(processes_test alma_framework gtest gtest_main)

add_executable(p3_test p3_test.cpp)
target_link_libraries(p3_test alma_framework gtest gtest_main)

add_executable(force_constants_3rd_test force_constants_3rd_test.cpp)
target_link_libraries(force_constants_3rd_test alma_framework gtest gtest_main)

add_executable(conductivity_test conductivity_test.cpp)
target_link_libraries(conductivity_test alma_framework gtest gtest_main)

add_executable(beyondRTA_test beyondRTA_test.cpp)
target_link_libraries(beyondRTA_test alma_framework gtest gtest_main)

add_executable(proc_serialization_test proc_serialization_test.cpp)
target_link_libraries(proc_serialization_test alma_framework gtest gtest_main)

add_executable(hdf5_write_test hdf5_write_test.cpp)
target_link_libraries(hdf5_write_test alma_framework gtest gtest_main)

add_executable(hdf5_scattering_test hdf5_scattering_test.cpp)
target_link_libraries(hdf5_scattering_test alma_framework gtest gtest_main)

add_executable(hdf5_read_test hdf5_read_test.cpp)
target_link_libraries(hdf5_read_test alma_framework gtest gtest_main)

add_executable(isotopic_scattering_test isotopic_scattering_test.cpp)
target_link_libraries(isotopic_scattering_test alma_framework gtest)

add_executable(green1d_test green1d_test.cpp)
target_link_libraries(green1d_test alma_framework gtest gtest_main)

add_executable(analytic1D_basicprop_test analytic1D_basicprop_test.cpp)
target_link_libraries(analytic1D_basicprop_test alma_framework gtest)

add_executable(analytic1D_psi_test analytic1D_psi_test.cpp)
target_link_libraries(analytic1D_psi_test alma_framework gtest)

add_executable(analytic1D_SPR-FL_test analytic1D_SPR-FL_test.cpp)
target_link_libraries(analytic1D_SPR-FL_test alma_framework gtest)

add_executable(analytic1D_SPR_test analytic1D_SPR_test.cpp)
target_link_libraries(analytic1D_SPR_test alma_framework gtest)

add_executable(analytic1D_MSD_test analytic1D_MSD_test.cpp)
target_link_libraries(analytic1D_MSD_test alma_framework gtest)

add_executable(superlattice_builder_test superlattice_builder_test.cpp)
target_link_libraries(superlattice_builder_test alma_framework gtest)
add_dependencies(superlattice_builder_test superlattice_builder)

add_executable(kappa_Tsweep_test kappa_Tsweep_test.cpp)
target_link_libraries(kappa_Tsweep_test alma_framework gtest)
add_dependencies(kappa_Tsweep_test kappa_Tsweep)

add_executable(cumulativecurves_test cumulativecurves_test.cpp)
target_link_libraries(cumulativecurves_test alma_framework gtest)
add_dependencies(cumulativecurves_test cumulativecurves)

add_executable(kappa_crossplanefilms_test kappa_crossplanefilms_test.cpp)
target_link_libraries(kappa_crossplanefilms_test alma_framework gtest)
add_dependencies(kappa_crossplanefilms_test kappa_crossplanefilms)

add_executable(kappa_inplanefilms_test kappa_inplanefilms_test.cpp)
target_link_libraries(kappa_inplanefilms_test alma_framework gtest)
add_dependencies(kappa_inplanefilms_test kappa_inplanefilms)

add_executable(steady_montecarlo1d_test steady_montecarlo1d_test.cpp)
target_link_libraries(steady_montecarlo1d_test alma_framework gtest)
add_dependencies(steady_montecarlo1d_test steady_montecarlo1d)

add_executable(transient_analytic1d_test transient_analytic1d_test.cpp)
target_link_libraries(transient_analytic1d_test alma_framework gtest)
add_dependencies(transient_analytic1d_test transient_analytic1d)

# Define all tests.
add_test(NAME metatest COMMAND metatest)
add_test(NAME poscar_test COMMAND poscar_test)
add_test(NAME force_constants_test COMMAND force_constants_test)
add_test(NAME born_test COMMAND born_test)
add_test(NAME mass_test COMMAND mass_test)
add_test(NAME vc_test COMMAND vc_test)
add_test(NAME dynamical_matrix_test COMMAND dynamical_matrix_test)
add_test(NAME symmetry_test COMMAND symmetry_test)
add_test(NAME mpi_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:symmetry_test>)
add_test(NAME qpoint_grid_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:qpoint_grid_test>)
add_test(NAME cv_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:cv_test>)
add_test(NAME kappa_sg_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:kappa_sg_test>)
add_test(NAME processes_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:processes_test>)
add_test(NAME p3_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:p3_test>)
add_test(NAME force_constants_3rd_test COMMAND force_constants_3rd_test)
add_test(NAME conductivity_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:conductivity_test>)
add_test(NAME beyondRTA_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:beyondRTA_test>)
add_test(NAME proc_serialization_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:proc_serialization_test>)
add_test(NAME hdf5_write_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:hdf5_write_test>)
add_test(NAME hdf5_scattering_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:hdf5_scattering_test>)
add_test(NAME hdf5_read_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:hdf5_read_test>)
add_test(NAME isotopic_scattering_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:isotopic_scattering_test>)
add_test(NAME green1d_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:green1d_test>)
add_test(NAME analytic1D_basicprop_test COMMAND analytic1D_basicprop_test)
add_test(NAME analytic1D_psi_test COMMAND analytic1D_psi_test)
add_test(NAME analytic1D_SPR-FL_test COMMAND analytic1D_SPR-FL_test)
add_test(NAME analytic1D_SPR_test COMMAND analytic1D_SPR_test)
add_test(NAME analytic1D_MSD_test COMMAND analytic1D_MSD_test)
add_test(NAME superlattice_builder_test COMMAND superlattice_builder_test)
add_test(NAME kappa_Tsweep_test COMMAND kappa_Tsweep_test)
add_test(NAME cumulativecurves_test COMMAND cumulativecurves_test)
add_test(NAME kappa_crossplanefilms_test COMMAND kappa_crossplanefilms_test)
add_test(NAME kappa_inplanefilms_test COMMAND kappa_inplanefilms_test)
add_test(NAME transient_analytic1d_test COMMAND transient_analytic1d_test)
add_test(NAME steady_montecarlo1d_test COMMAND steady_montecarlo1d_test)

add_executable(updates_test updates_test.cpp)
target_link_libraries(updates_test alma_framework gtest)
add_test(NAME updates_test COMMAND updates_test)
add_test(NAME updates_mpi_test COMMAND ${MPIEXEC_EXECUTABLE}
  ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:updates_test> ${MPIEXEC_POSTFLAGS})
set_tests_properties(updates_test updates_mpi_test PROPERTIES
  TIMEOUT 180 ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")

add_dependencies(updates_test convert_bulk_hdf5)
target_compile_definitions(updates_test PRIVATE ALMA_CONVERTER_PATH="$<TARGET_FILE:convert_bulk_hdf5>")

if(BUILD_SHENGBTEPP)
  add_executable(shengcontrol_test shengcontrol_test.cpp ../src/read_shengcontrol.f90)
  target_compile_features(shengcontrol_test PRIVATE cxx_std_17)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shengcontrol_mod)
  set_target_properties(shengcontrol_test PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shengcontrol_mod)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    target_compile_options(shengcontrol_test PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-fcheck=bounds>)
  endif()
  add_test(NAME shengcontrol_test COMMAND shengcontrol_test)
endif()
