# Library that will be used in the executable and in the unit tests.
add_library(alma_framework STATIC structures.cpp
  vasp_io.cpp
  periodic_table.cpp
  dynamical_matrix.cpp
  symmetry.cpp
  qpoint_grid.cpp
  isotopic_scattering.cpp
  processes.cpp
  threeph_processes.cpp
  bulk_hdf5.cpp
  sampling.cpp
  bulk_properties.cpp
  green1d.cpp
  vc.cpp
  superlattices.cpp
  analytic1d.cpp
  beyondRTA.cpp
  shengbte_iter.cpp
  interfaces.cpp
  gruneisen_properties.cpp
  material_builder.cpp
)

# Propagate the language requirement and MPI usage requirements to consumers.
target_compile_features(alma_framework PUBLIC cxx_std_17)
target_compile_definitions(alma_framework PUBLIC BOOST_LOG_DYN_LINK)
target_link_libraries(alma_framework PUBLIC MPI::MPI_CXX ${ALMA_SPGLIB_TARGET}
  Boost::filesystem Boost::mpi Boost::serialization Boost::log HDF5::HDF5 Threads::Threads)
if(ALMA_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(alma_framework PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(benchmark_threeph EXCLUDE_FROM_ALL ../benchmarks/threeph.cpp)
target_link_libraries(benchmark_threeph alma_framework)

add_executable(convert_bulk_hdf5 convert_bulk_hdf5.cpp)
target_link_libraries(convert_bulk_hdf5 alma_framework)

# Define our executables.

add_executable(VCAbuilder VCAbuilder.cpp)
target_link_libraries(VCAbuilder alma_framework)

add_executable(superlattice_builder superlattice_builder.cpp)
target_link_libraries(superlattice_builder alma_framework)

add_executable(phononinfo phononinfo.cpp)
target_link_libraries(phononinfo alma_framework)

add_executable(kappa_Tsweep kappa_Tsweep.cpp)
target_link_libraries(kappa_Tsweep alma_framework)

add_executable(kappa_crossplanefilms kappa_crossplanefilms.cpp)
target_link_libraries(kappa_crossplanefilms alma_framework)

add_executable(kappa_inplanefilms kappa_inplanefilms.cpp)
target_link_libraries(kappa_inplanefilms alma_framework)

add_executable(cumulativecurves cumulativecurves.cpp)
target_link_libraries(cumulativecurves alma_framework)

add_executable(transient_analytic1d transient_analytic1d.cpp)
target_link_libraries(transient_analytic1d alma_framework)

add_executable(steady_montecarlo1d steady_montecarlo1d.cpp)
target_link_libraries(steady_montecarlo1d alma_framework)

add_executable(steady_montecarlo1d_powersource steady_montecarlo1d_powersource.cpp)
target_link_libraries(steady_montecarlo1d_powersource alma_framework)

add_executable(interfaces interface_conductance.cpp)
target_link_libraries(interfaces alma_framework)

add_executable(gruneisen gruneisen.cpp)
target_link_libraries(gruneisen alma_framework)

# shengbtepp is a separate target that will not be built by "make all"
if(BUILD_SHENGBTEPP)
  add_executable(shengbtepp EXCLUDE_FROM_ALL read_shengcontrol.f90 shengbtepp.cpp)
  target_link_libraries(shengbtepp alma_framework)
endif()

# These custom commands provide the tests with the right path to some of the executables in ALMA
add_custom_command(TARGET superlattice_builder POST_BUILD COMMAND echo const std::string superlattice_builder = \"$<TARGET_FILE:superlattice_builder>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_superlattice_builder.hpp VERBATIM)
add_custom_command(TARGET kappa_Tsweep POST_BUILD COMMAND echo const std::string kappa_Tsweep = \"$<TARGET_FILE:kappa_Tsweep>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_Tsweep.hpp VERBATIM)
add_custom_command(TARGET cumulativecurves POST_BUILD COMMAND echo const std::string cumulativecurves = \"$<TARGET_FILE:cumulativecurves>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_cumulativecurves.hpp VERBATIM)
add_custom_command(TARGET kappa_crossplanefilms POST_BUILD COMMAND echo const std::string kappa_crossplanefilms = \"$<TARGET_FILE:kappa_crossplanefilms>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_crossplanefilms.hpp VERBATIM)
add_custom_command(TARGET kappa_inplanefilms POST_BUILD COMMAND echo const std::string kappa_inplanefilms = \"$<TARGET_FILE:kappa_inplanefilms>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_inplanefilms.hpp VERBATIM)
add_custom_command(TARGET steady_montecarlo1d POST_BUILD COMMAND echo const std::string steady_montecarlo1d = \"$<TARGET_FILE:steady_montecarlo1d>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_steady_montecarlo1d.hpp VERBATIM)
add_custom_command(TARGET transient_analytic1d POST_BUILD COMMAND echo const std::string transient_analytic1d = \"$<TARGET_FILE:transient_analytic1d>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_transient_analytic1d.hpp VERBATIM)
