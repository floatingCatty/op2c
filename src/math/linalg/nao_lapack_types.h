#ifndef NAO_LAPACK_TYPES_H
#define NAO_LAPACK_TYPES_H

#if defined(__MKL)
    #if defined(MKL_ILP64)
        typedef long long nao_lapack_int;
    #else
        typedef int nao_lapack_int;
    #endif
#else
    // For Generic BLAS/LAPACK (OpenBLAS, Netlib), the standard is 'int' (32-bit).
    typedef int nao_lapack_int;
#endif

#endif // NAO_LAPACK_TYPES_H
