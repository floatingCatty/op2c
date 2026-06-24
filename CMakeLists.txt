cmake_minimum_required(VERSION 3.15...4.0)

# project settings
project(
  int2c 
  LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(GNUInstallDirs)

# Ensure all static libraries are built with -fPIC so they can be linked into the Python extension
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_TESTING "Build tests" OFF)
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  if(BUILD_TESTING)
    find_package(GTest)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
          googletest
          URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        include(GoogleTest)
    endif()
  endif()
endif()
# Find MPI for BOTH the standalone (_op2c) and the add_subdirectory (rescu++ _core) builds.
# The old top-level-only guard skipped this in the subdirectory build, leaving nao_int2c with
# the stub `typedef int MPI_Comm` while _core uses real <mpi.h> — an ABI mismatch (Op2c ctor
# undefined symbol / garbage handle). Running it unconditionally makes nao_int2c's MPI_Comm
# match _core's so the C++ operator construction can call Op2c. See doc/design/38.
find_package(MPI) # Try finding MPI, but don't error out yet

if(NOT TARGET MPI::MPI_CXX)
  # If MPI not found, define dummy library for serial build
  add_library(MPI::MPI_CXX INTERFACE IMPORTED)
  # Note: __MPI definition should only be added if MPI IS found.
else()
  # Only add __MPI for global if we found it and want to use it? 
  # Actually, base relies on __MPI definition. 
  # If we want serial build, we must NOT define __MPI.
endif()

find_path(MATIO_INCLUDE_DIR
    NAMES matio.h
    HINTS $ENV{CONDA_PREFIX}/include
)
find_library(MATIO_LIBRARY
    NAMES matio
    HINTS $ENV{CONDA_PREFIX}/lib
)
if(NOT MATIO_INCLUDE_DIR OR NOT MATIO_LIBRARY)
    message(FATAL_ERROR "libmatio is required for rescumat *_AtomicData.mat support. "
                        "Install conda-forge::libmatio in the rescupp conda environment.")
endif()
add_library(MATIO::MATIO UNKNOWN IMPORTED)
set_target_properties(MATIO::MATIO PROPERTIES
    IMPORTED_LOCATION "${MATIO_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MATIO_INCLUDE_DIR}"
)
message(STATUS "Using libmatio: ${MATIO_LIBRARY}")
# # include_directories(${MPI_CXX_INCLUDE_PATH})
# add_compile_definitions(__MPI)
# add_compile_definitions(__LCAO)

# set source path
set(SOURCE_DIR "${PROJECT_SOURCE_DIR}/")
message(STATUS "set source_dir as ${SOURCE_DIR}")
set(BASE_PATH "${SOURCE_DIR}/source_base")
message(STATUS "set base_path as ${BASE_PATH}")
set(NAO_PATH "${SOURCE_DIR}/source_basis/module_nao")
message(STATUS "set nao_path as ${NAO_PATH}")

# set(MKL_INTERFACE lp64)
# find_package(MKL CONFIG REQUIRED)

# add math_libs
#
# Run the BLAS/LAPACK/FFTW/OpenMP detection (which creates the BLAS::BLAS / LAPACK::LAPACK /
# FFTW::FFTW IMPORTED targets the nao_* libs and the _op2c binding link against) when building
# standalone OR when a parent project requests the Python bindings as a subproject
# (``-DOP2C_BUILD_PYTHON=ON``, set by rescu++'s CMake for the single-wheel build). Without this the
# subproject build leaves those targets undefined and _op2c fails to link. find_package(BLAS/LAPACK)
# is idempotent and the FFTW import is guarded by ``if(NOT TARGET FFTW::FFTW)``, so this is safe even
# when the parent already found the same libraries.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR OR OP2C_BUILD_PYTHON)
    # Options
    option(USE_OPENMP "Use OpenMP" ON)
    set(BLAS_BACKEND "Auto" CACHE STRING "BLAS Backend (Auto, MKL, OpenBLAS, Generic)")
    set_property(CACHE BLAS_BACKEND PROPERTY STRINGS "Auto" "MKL" "OpenBLAS" "Generic")

    # OpenMP
    if(USE_OPENMP)
        find_package(OpenMP REQUIRED)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        link_libraries(OpenMP::OpenMP_CXX)
    endif()

    # --- Auto-detection: resolve "Auto" to a concrete backend ---
    if(BLAS_BACKEND STREQUAL "Auto")
        find_library(MKL_RT_LIB NAMES mkl_rt mkl_rt.2 mkl_rt.1
                     HINTS $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64 $ENV{CONDA_PREFIX}/lib
                     NO_DEFAULT_PATH)
        find_library(MKL_RT_LIB NAMES mkl_rt mkl_rt.2 mkl_rt.1)
        if(MKL_RT_LIB)
            message(STATUS "Auto-detected MKL SDL: ${MKL_RT_LIB}")
            set(BLAS_BACKEND "MKL")
        else()
            message(STATUS "MKL not found, falling back to Generic BLAS")
            set(BLAS_BACKEND "Generic")
        endif()
    endif()

    # BLAS/LAPACK Selection
    if(BLAS_BACKEND STREQUAL "MKL")
        set(INT2C_USE_MKL ON)
        add_compile_definitions(__MKL RSATB_USE_MKL)
        if(DEFINED ENV{MKLROOT})
            include_directories($ENV{MKLROOT}/include)
        endif()
        if(NOT MKL_RT_LIB)
            find_library(MKL_RT_LIB NAMES mkl_rt mkl_rt.2 mkl_rt.1
                         HINTS $ENV{MKLROOT}/lib $ENV{MKLROOT}/lib/intel64 $ENV{CONDA_PREFIX}/lib)
        endif()
        if(NOT MKL_RT_LIB)
            message(FATAL_ERROR "BLAS_BACKEND=MKL but libmkl_rt.so not found. "
                    "Set BLAS_BACKEND=Auto or Generic, or install MKL.")
        endif()
        message(STATUS "Using MKL SDL: ${MKL_RT_LIB}")
        add_library(BLAS::BLAS INTERFACE IMPORTED)
        set_target_properties(BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES "${MKL_RT_LIB}")
        add_library(LAPACK::LAPACK INTERFACE IMPORTED)
        set_target_properties(LAPACK::LAPACK PROPERTIES INTERFACE_LINK_LIBRARIES "${MKL_RT_LIB}")
    elseif(BLAS_BACKEND STREQUAL "OpenBLAS")
        set(BLA_VENDOR "OpenBLAS")
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        add_compile_definitions(__OPENBLAS RSATB_USE_OPENBLAS)
    else()
        # Generic
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif()

    # FFTW
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3)
    set(INT2C_USE_FFTW ON)
    
    if(NOT TARGET FFTW::FFTW)
      add_library(FFTW::FFTW INTERFACE IMPORTED)
      set_target_properties(FFTW::FFTW PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}" 
        INTERFACE_LINK_LIBRARIES     "${FFTW3_LIBRARIES}")
    endif()
    message(STATUS "FFTW_INCLUDE_DIRS: ${FFTW3_INCLUDE_DIRS}")
    list(APPEND math_libs FFTW::FFTW LAPACK::LAPACK BLAS::BLAS)
endif()

# add include directories

# include_directories(
#     ${BASE_PATH} 
#     ${SOURCE_DIR} 
#     ${SOURCE_DIR}/source_base/module_container 
#     ${SOURCE_DIR}/source_base/module_device 
#     )

# Add new source directory
add_subdirectory(src)

# Tests (needs update for new targets)
# Tests (needs update for new targets)
if(BUILD_TESTING)
    # TODO: Update tests to link against nao_core, nao_int2c etc.
    # add_executable(test_nao test/test.cpp)
    # target_link_libraries(test_nao PRIVATE nao_core MPI::MPI_CXX GTest::gtest)
    add_subdirectory(test)
endif()

# --- Target-based definitions ---
# --- Target-based definitions ---
# Now we should attach these to the new interface targets in src/CMakeLists.txt or here.
# For now, let's just make sure they are available globally or attached to nao_utils
target_compile_definitions(nao_utils PUBLIC __LCAO)
if(TARGET MPI::MPI_CXX AND MPI_FOUND)
    target_compile_definitions(nao_utils PUBLIC __MPI)
endif()
if(INT2C_USE_MKL)
    target_compile_definitions(nao_utils PUBLIC __MKL)
endif()
if(INT2C_USE_FFTW)
    # target_compile_definitions(nao_utils PUBLIC __FFTW) # Check if needed
endif()


# --- Installation ---

# Bindings are now in src/bindings, added via add_subdirectory(src)
