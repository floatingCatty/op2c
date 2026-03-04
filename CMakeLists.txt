cmake_minimum_required(VERSION 3.15...4.0)

# project settings
project(
  int2c 
  LANGUAGES C CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake")
include(GNUInstallDirs)

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
  find_package(MPI) # Try finding MPI, but don't error out yet
endif()

if(NOT TARGET MPI::MPI_CXX)
  # If MPI not found, define dummy library for serial build
  add_library(MPI::MPI_CXX INTERFACE IMPORTED)
  # Note: __MPI definition should only be added if MPI IS found.
else()
  # Only add __MPI for global if we found it and want to use it? 
  # Actually, base relies on __MPI definition. 
  # If we want serial build, we must NOT define __MPI.
endif()
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

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Options
    option(USE_OPENMP "Use OpenMP" ON)
    set(BLAS_BACKEND "Generic" CACHE STRING "BLAS Backend (MKL, OpenBLAS, Generic)")
    set_property(CACHE BLAS_BACKEND PROPERTY STRINGS "MKL" "OpenBLAS" "Generic")

    # OpenMP
    if(USE_OPENMP)
        find_package(OpenMP REQUIRED)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_link_options(${OpenMP_CXX_LIBRARIES})
    endif()

    # BLAS/LAPACK Selection
    if(BLAS_BACKEND STREQUAL "MKL")
        set(BLA_VENDOR "Intel10_64_dyn")
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
        set(INT2C_USE_MKL ON)
        add_compile_definitions(__MKL RSATB_USE_MKL)
        if(DEFINED ENV{MKLROOT})
            include_directories($ENV{MKLROOT}/include)
        endif()
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
