#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "int2c/op2c.hpp"
#include "math/linalg/matrix.h"
#include "math/linalg/vector3.h"

TEST(Op2cTest, DifferentialTest)
{
    std::string data_dir = "./test/pporb/";
    std::string orb_dir = data_dir;
    std::vector<std::string> orb_name = {"C_gga_7au_100Ry_2s2p1d.orb"};
    // Note: No pseudos, matching test_optional.py
    std::string psd_dir = "";
    std::vector<std::string> psd_name = {};
    std::string log_file = "test_diff.log";

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
    std::vector<double> v;

    op2c.overlap(itype, jtype, Rij, is_transpose, v, nullptr, nullptr, nullptr);
    
    double sum_sq = 0.0;
    for(double val : v) sum_sq += val * val;
    double norm = std::sqrt(sum_sq);
    
    double expected_norm = 2.208489944340288;
    
    std::cout << "Calculated: " << norm << ", Expected: " << expected_norm << std::endl;
    EXPECT_NEAR(norm, expected_norm, 1e-15);
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
