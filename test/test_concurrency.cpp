#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <omp.h>

#include "int2c/op2c.hpp"
#include "math/linalg/vector3.h"

// Test Op2c Concurrency functionality
TEST(Op2cTest, ThreadSafety)
{
#ifdef TEST_DATA_DIR
    std::string data_dir = TEST_DATA_DIR;
#else
    std::string data_dir = "./test/pporb/";
#endif
    std::string orb_dir = data_dir;
    std::string psd_dir = data_dir;
    std::vector<std::string> orb_name = {"C_gga_7au_100Ry_2s2p1d.orb"};
    std::vector<std::string> psd_name = {"C_ONCV_PBE-1.0.upf"};
    std::string log_file = "test_concurrency_cpp.log";

    int ntype = 1;
    int nspin = 1;
    bool lspinorb = false;

#ifdef __MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    int comm = 0;
#endif

    // 1. Initialize Op2c (Shared Resource)
    Op2c op2c(ntype, nspin, lspinorb, orb_dir, orb_name, psd_dir, psd_name, comm, log_file);

    int max_threads = omp_get_max_threads();
    std::cout << "Running ThreadSafety test with " << max_threads << " OpenMP threads." << std::endl;

    int n_tasks = 1000;
    std::vector<double> results(n_tasks, 0.0);
    bool failed = false;

    // 2. Concurrent Execution
    #pragma omp parallel for
    for (int i = 0; i < n_tasks; ++i)
    {
        try {
            size_t itype = 0;
            size_t jtype = 0;
            // Shift Rj slightly for each task to avoid identical calculations (though caching might handle it)
            ModuleBase::Vector3<double> Rij(2.0 + i * 0.001, 0.0, 0.0);
            bool is_transpose = false;
            std::vector<double> v;
            
            // Thread-safe call?
            op2c.overlap(itype, jtype, Rij, is_transpose, v, nullptr, nullptr, nullptr);
            
            if (!v.empty()) {
                results[i] = v[0]; // Just store first element
            }
        } catch (...) {
            failed = true;
        }
    }

    EXPECT_FALSE(failed) << "One or more threads threw an exception.";
    
    // Check minimal correctness (results shouldn't be all zero if overlap is significant)
    int non_zero_count = 0;
    for(double val : results) {
        if(std::abs(val) > 1e-10) non_zero_count++;
    }
    EXPECT_GT(non_zero_count, 0) << "All overlap calculations returned near-zero/empty results.";
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
#endif
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
