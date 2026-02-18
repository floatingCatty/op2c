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

pybind11_add_module(_estate py_nao.cpp)

# Link against our new libraries
target_link_libraries(_estate PRIVATE nao_int2c nao_pseudo nao_core nao_math nao_utils)

# Ensure includes are found
target_include_directories(_estate PRIVATE ${CMAKE_SOURCE_DIR}/src)

install(TARGETS _estate DESTINATION .)
