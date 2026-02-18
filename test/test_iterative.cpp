#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

#include "int2c/op2c.hpp"
#include "math/linalg/matrix.h"
#include "math/linalg/vector3.h"

TEST(Op2cTest, IterativeStability)
{
    std::string data_dir = "./test/pporb/";
    std::string orb_dir = data_dir;
    std::string psd_dir = data_dir;
    std::vector<std::string> orb_name = {"C_gga_7au_100Ry_2s2p1d.orb"};
    std::vector<std::string> psd_name = {"C_ONCV_PBE-1.0.upf"};
    std::string log_file = "test_iterative.log";

    int ntype = 1;
    int nspin = 1;
    bool lspinorb = false;

#ifdef __MPI
    MPI_Comm comm = MPI_COMM_WORLD;
#else
    int comm = 0;
#endif

    Op2c op2c(ntype, nspin, lspinorb, orb_dir, orb_name, psd_dir, psd_name, comm, log_file);

    size_t itype = 0;
    size_t jtype = 0;
    ModuleBase::Vector3<double> Rij(2.0, 0.0, 0.0);
    bool is_transpose = false;
    
    std::vector<double> norms;
    auto start = std::chrono::high_resolution_clock::now();
    
    for(int i=0; i<100; ++i) {
        std::vector<double> v;
        op2c.overlap(itype, jtype, Rij, is_transpose, v, nullptr, nullptr, nullptr);
        
        double sum_sq = 0.0;
        for(double val : v) sum_sq += val * val;
        norms.push_back(std::sqrt(sum_sq));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "100 iterations took " << elapsed.count() << "s" << std::endl;

    double sum = 0.0;
    for(double n : norms) sum += n;
    double avg = sum / norms.size();
    
    double max_diff = 0.0;
    for(double n : norms) {
        max_diff = std::max(max_diff, std::abs(n - avg));
    }
    
    std::cout << "Max deviation: " << max_diff << std::endl;
    EXPECT_LT(max_diff, 1e-14);
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
