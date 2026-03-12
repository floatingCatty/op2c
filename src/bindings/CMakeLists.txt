# src/bindings/CMakeLists.txt

include(FetchContent)

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, fetching from GitHub...")
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()
pybind11_add_module(_op2c py_nao.cpp)

# On Linux/ELF, prevent numpy's implicitly linked MKL component libraries from
# leaking into our shared object (causes PARDISO thread collisions).
# This flag is not supported on macOS (Mach-O linker).
if(UNIX AND NOT APPLE)
    target_link_options(_op2c PRIVATE
        "-Wl,--exclude-libs,libmkl_intel_thread.so"
        "-Wl,--exclude-libs,libiomp5.so")
endif()

# Link against our libraries; add BLAS::BLAS only if the target was created
target_link_libraries(_op2c PRIVATE nao_int2c nao_pseudo nao_core nao_math nao_utils)
if(TARGET BLAS::BLAS)
    target_link_libraries(_op2c PRIVATE BLAS::BLAS)
endif()

# Ensure includes are found
target_include_directories(_op2c PRIVATE ${CMAKE_SOURCE_DIR}/src)

install(TARGETS _op2c DESTINATION .)
