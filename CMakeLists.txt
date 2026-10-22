# Match the minimum required by the optional spglib 2.7 download.
cmake_minimum_required(VERSION 3.25)
project(ALMA CXX C)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Use C++17 for the integrated code. Allow toolchains and callers to
# request a newer standard without silently overriding their choice.
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard (at least 17)")
if(CMAKE_CXX_STANDARD LESS 17)
  message(FATAL_ERROR "ALMA and its Boost dependencies require C++17 or newer")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep optimized builds with debug symbols as the default, while respecting
# explicit build types and the flags supplied by the user or toolchain.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# Try to find a suitable Fortran compiler to build the ShengBTE emulator.
# If it is not found, the target is skipped.
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
  include(FortranCInterface)
  FortranCInterface_VERIFY(QUIET)
  FortranCInterface_VERIFY(CXX QUIET)
  if(FortranCInterface_VERIFIED_C AND FortranCInterface_VERIFIED_CXX)
    set(BUILD_SHENGBTEPP TRUE)
  else()
    message("Fortran compiler not compatible with C/C++ compilers, the shengbtepp target will not be available")
    set(BUILD_SHENGBTEPP FALSE)
  endif()
else()
  message("Fortran compiler not found, the shengbtepp target will not be available")
  set(BUILD_SHENGBTEPP FALSE)
endif()

# Try to find a threading library. Superfluous in most systems,
# but lack of this seems to lead to errors in some cases.
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
option(ALMA_OPENMP "Enable OpenMP threading" ON)
if(ALMA_OPENMP)
  find_package(OpenMP QUIET COMPONENTS CXX)
endif()

# Do not use Eigen's parallel algorithms.
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# Detect and set up MPI.
find_package(MPI REQUIRED COMPONENTS CXX)

# Compiler flags (currently only for g++ or clang++).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-local-typedef>")
  endif()
endif()

# Set up some variables so that they can be used in the code.
set(ALMA_VERSION_MAJOR 1)
set(ALMA_VERSION_MINOR 3)
set(BULK_HDF5_MAJOR_VERSION 0)
set(BULK_HDF5_MINOR_VERSION 5)
set(TEST_RESOURCE_DIR "${PROJECT_SOURCE_DIR}/test_resources")


# Tell the compiler where to find the headers.
include_directories(BEFORE "${PROJECT_SOURCE_DIR}/external/eigen")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/external/randutils")
include_directories("${PROJECT_SOURCE_DIR}/external/base-n/include")

# Boost >= 1.75 supplies its CMake package and header-only Boost.System.
find_package(Boost 1.75.0 CONFIG REQUIRED COMPONENTS filesystem mpi serialization log)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Discover an installed spglib, or download a pinned release when requested.
include(PrealmaSpglib)

# Add the main source directory.
add_subdirectory(src)

# Set up Google Test and Google Mock.
# Isolate the policy compatibility override needed by the bundled Google Mock
# and Google Test when configuring with CMake 4 or newer.
function(alma_add_gmock)
  if(NOT DEFINED CMAKE_POLICY_VERSION_MINIMUM OR
      CMAKE_POLICY_VERSION_MINIMUM VERSION_LESS "3.10")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.10)
  endif()
  add_subdirectory(external/gmock-1.7.0)
endfunction()
alma_add_gmock()
enable_testing()
include_directories("${PROJECT_SOURCE_DIR}/external/gmock-1.7.0/include")
include_directories("${PROJECT_SOURCE_DIR}/external/gmock-1.7.0/gtest/include")

# Add the directory containing the tests.
add_subdirectory(test)

# Generate the configuration with Doxygen if possible.
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
  file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/doc/logo_for_doxygen.png
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/doc)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
    COMMENT "Generating documentation with Doxygen" VERBATIM
    )
else()
  message(WARNING "Doxygen not found, documentation cannot be generated")
endif()

# Write out a header file with all information needed at compile time
configure_file (
  "${PROJECT_SOURCE_DIR}/include/cmakevars.hpp.in"
  "${PROJECT_BINARY_DIR}/include/cmakevars.hpp"
  )
