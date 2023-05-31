# Library that will be used in the executable and in the unit tests.
add_library(alma_framework STATIC structures.cpp
  vasp_io.cpp
  periodic_table.cpp
  dynamical_matrix.cpp
  symmetry.cpp
  qpoint_grid.cpp
  isotopic_scattering.cpp
  processes.cpp
  bulk_hdf5.cpp
  sampling.cpp
  bulk_properties.cpp
  green1d.cpp
  vc.cpp
  superlattices.cpp
  analytic1d.cpp
  beyondRTA.cpp
  shengbte_iter.cpp
  gpu_solver.cpp
)

# Define our executables.

add_executable(VCAbuilder VCAbuilder.cpp)
target_link_libraries(VCAbuilder alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(superlattice_builder superlattice_builder.cpp)
target_link_libraries(superlattice_builder alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(phononinfo phononinfo.cpp)
target_link_libraries(phononinfo alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(kappa_Tsweep kappa_Tsweep.cpp)
target_link_libraries(kappa_Tsweep alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(kappa_crossplanefilms kappa_crossplanefilms.cpp)
target_link_libraries(kappa_crossplanefilms alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(kappa_inplanefilms kappa_inplanefilms.cpp)
target_link_libraries(kappa_inplanefilms alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(cumulativecurves cumulativecurves.cpp)
target_link_libraries(cumulativecurves alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(transient_analytic1d transient_analytic1d.cpp)
target_link_libraries(transient_analytic1d alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(steady_montecarlo1d steady_montecarlo1d.cpp)
target_link_libraries(steady_montecarlo1d alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

add_executable(steady_montecarlo1d_powersource steady_montecarlo1d_powersource.cpp)
target_link_libraries(steady_montecarlo1d_powersource alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})

# shengbtepp is a separate target that will not be built by "make all"
if(BUILD_SHENGBTEPP)
  add_executable(shengbtepp EXCLUDE_FROM_ALL read_shengcontrol.f90 shengbtepp.cpp)
  target_link_libraries(shengbtepp alma_framework
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg ${HDF5_LIBS} Threads::Threads ${GPU_LIBS})
endif()

# These custom commands provide the tests with the right path to some of the executables in ALMA
add_custom_command(TARGET superlattice_builder POST_BUILD COMMAND echo const std::string superlattice_builder = \"$<TARGET_FILE:superlattice_builder>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_superlattice_builder.hpp VERBATIM)
add_custom_command(TARGET kappa_Tsweep POST_BUILD COMMAND echo const std::string kappa_Tsweep = \"$<TARGET_FILE:kappa_Tsweep>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_Tsweep.hpp VERBATIM)
add_custom_command(TARGET cumulativecurves POST_BUILD COMMAND echo const std::string cumulativecurves = \"$<TARGET_FILE:cumulativecurves>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_cumulativecurves.hpp VERBATIM)
add_custom_command(TARGET kappa_crossplanefilms POST_BUILD COMMAND echo const std::string kappa_crossplanefilms = \"$<TARGET_FILE:kappa_crossplanefilms>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_crossplanefilms.hpp VERBATIM)
add_custom_command(TARGET kappa_inplanefilms POST_BUILD COMMAND echo const std::string kappa_inplanefilms = \"$<TARGET_FILE:kappa_inplanefilms>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_inplanefilms.hpp VERBATIM)
add_custom_command(TARGET steady_montecarlo1d POST_BUILD COMMAND echo const std::string steady_montecarlo1d = \"$<TARGET_FILE:steady_montecarlo1d>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_steady_montecarlo1d.hpp VERBATIM)
add_custom_command(TARGET transient_analytic1d POST_BUILD COMMAND echo const std::string transient_analytic1d = \"$<TARGET_FILE:transient_analytic1d>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_transient_analytic1d.hpp VERBATIM)
