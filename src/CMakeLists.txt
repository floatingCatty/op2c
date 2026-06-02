# src/CMakeLists.txt

# --- Math & Utils ---

add_library(nao_utils
    utils/parallel_common.cpp
    utils/memory.cpp
    utils/mcd.c
    utils/container/base/core/refcount.cpp
    utils/container/base/core/cpu_allocator.cpp
    # utils/container/base/core/gpu_allocator.cpp # Requires CUDA/ROCm
    utils/container/ATen/core/tensor.cpp
    utils/container/ATen/core/tensor_map.cpp
    utils/container/ATen/core/tensor_types.cpp
    utils/container/ATen/core/tensor_buffer.cpp
    utils/container/ATen/core/tensor_shape.cpp
    utils/container/ATen/kernels/memory_impl.cpp
    utils/container/ATen/kernels/linalg.cpp
    utils/container/ATen/kernels/lapack.cpp
    utils/container/ATen/kernels/blas.cpp
    
    # Exclude kernels/ops unless needed, but adding all for completeness except tests
    utils/container/ATen/ops/einsum_op.cpp
    utils/container/ATen/ops/linalg_op.cpp
)
target_include_directories(nao_utils PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/container
)
if(TARGET MPI::MPI_CXX)
    target_link_libraries(nao_utils PUBLIC MPI::MPI_CXX)
endif()


add_library(nao_math
    math/spherical_bessel_transformer.cpp
    math/ylm.cpp
    math/math_sphbes.cpp
    math/math_integral.cpp
    math/linalg/complexmatrix.cpp
    math/linalg/complexarray.cpp
    math/linalg/matrix.cpp
    math/linalg/realarray.cpp
    math/interpolation/cubic_spline.cpp
    math/projgen.cpp
)
target_link_libraries(nao_math PUBLIC nao_utils)
target_include_directories(nao_math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/math)

if(TARGET LAPACK::LAPACK)
    target_link_libraries(nao_math PUBLIC LAPACK::LAPACK)
endif()
if(TARGET BLAS::BLAS)
    target_link_libraries(nao_math PUBLIC BLAS::BLAS)
endif()

if(INT2C_USE_FFTW)
   target_link_libraries(nao_math PUBLIC FFTW::FFTW)
endif()
if(__MKL)
    # math libs linked via interface in root
endif()


# --- NAO Core ---

add_library(nao_core
    io/rescumat_mat.cpp
    nao/radial_function.cpp
    nao/orbital_shell.cpp
    nao/atomic_basis.cpp
    nao/atomic_radials.cpp # Check if needed or legacy
)
target_link_libraries(nao_core PUBLIC nao_math MATIO::MATIO)

# --- Pseudopotential ---

add_library(nao_pseudo
    pseudopotential/pseudo.cpp
    pseudopotential/pseudo_atom.cpp
    pseudopotential/beta_projectors.cpp
    pseudopotential/soc.cpp
    pseudopotential/io/read_pp.cpp
    pseudopotential/io/read_pp_blps.cpp
    pseudopotential/io/read_pp_complete.cpp
    pseudopotential/io/read_pp_upf100.cpp
    pseudopotential/io/read_pp_upf201.cpp
    pseudopotential/io/read_pp_vwr.cpp
    pseudopotential/io/read_pseudo.cpp
)
target_link_libraries(nao_pseudo PUBLIC nao_core)

# --- Int2C ---

add_library(nao_int2c
    int2c/table.cpp
    int2c/integrator.cpp
    int2c/bundle.cpp
    int2c/gaunt_table.cpp
    int2c/op2c.cpp
)
target_link_libraries(nao_int2c PUBLIC nao_pseudo) # Depends on pseudo for beta integrals

# --- Bindings ---
add_subdirectory(bindings)
