# Installed spglib first; opt in to downloading a pinned release if absent.
# Discovery order follows ShengBTE trunk, with a C header check for bare libs.
option(ALMA_FETCH_SPGLIB "Download and build spglib if it cannot be found" OFF)

find_package(Spglib CONFIG QUIET)
if(TARGET Spglib::symspg)
  set(ALMA_SPGLIB_TARGET Spglib::symspg)
  message(STATUS "Found spglib ${Spglib_VERSION} (CMake package)")
endif()

if(NOT ALMA_SPGLIB_TARGET)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(ALMA_SPGLIB QUIET IMPORTED_TARGET spglib)
    if(TARGET PkgConfig::ALMA_SPGLIB)
      set(ALMA_SPGLIB_TARGET PkgConfig::ALMA_SPGLIB)
      message(STATUS "Found spglib ${ALMA_SPGLIB_VERSION} (pkg-config)")
    endif()
  endif()
endif()

if(NOT ALMA_SPGLIB_TARGET)
  find_path(ALMA_SPGLIB_INCLUDE_DIR NAMES spglib.h)
  find_library(ALMA_SPGLIB_LIBRARY NAMES symspg)
  if(ALMA_SPGLIB_INCLUDE_DIR AND ALMA_SPGLIB_LIBRARY)
    add_library(alma_spglib UNKNOWN IMPORTED)
    set_target_properties(alma_spglib PROPERTIES
      IMPORTED_LOCATION "${ALMA_SPGLIB_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ALMA_SPGLIB_INCLUDE_DIR}")
    set(ALMA_SPGLIB_TARGET alma_spglib)
    message(STATUS "Found spglib: ${ALMA_SPGLIB_LIBRARY}")
  endif()
endif()

if(NOT ALMA_SPGLIB_TARGET)
  if(NOT ALMA_FETCH_SPGLIB)
    message(FATAL_ERROR "spglib was not found. Install its development package, "
      "set CMAKE_PREFIX_PATH to its installation, or enable -DALMA_FETCH_SPGLIB=ON.")
  endif()
  message(STATUS "spglib not found; downloading v2.7.0")
  include(FetchContent)
  set(SPGLIB_WITH_Fortran OFF CACHE BOOL "" FORCE)
  set(SPGLIB_WITH_Python OFF CACHE BOOL "" FORCE)
  set(SPGLIB_WITH_TESTS OFF CACHE BOOL "" FORCE)
  set(SPGLIB_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(spglib
    GIT_REPOSITORY https://github.com/spglib/spglib.git
    GIT_TAG v2.7.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(spglib)
  set(ALMA_SPGLIB_TARGET Spglib::symspg)
endif()
