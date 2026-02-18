import unittest
import sys
import os
import numpy as np
import time

# Ensure we can import the build module
build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build"))
if build_path not in sys.path:
    sys.path.append(build_path)

try:
    from _estate import Op2c
except ImportError:
    sys.path.append(build_path)
    from _estate import Op2c

class TestOp2CIterative(unittest.TestCase):
    def setUp(self):
        self.orb_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/"
        self.orb_name = ["C_gga_7au_100Ry_2s2p1d.orb"]
        self.psd_dir = self.orb_dir
        self.psd_name = ["C_ONCV_PBE-1.0.upf"]
        self.ntype = 1
        self.nspin = 1
        self.lspinorb = False
        self.op2c = Op2c(self.ntype, self.nspin, self.lspinorb, 
                         self.orb_dir, self.orb_name, 
                         self.psd_dir, self.psd_name, 
                         "test_iterative.log", 0)

    def test_overlap_stability(self):
        itype = 0
        jtype = 0
        Rij = [2.0, 0.0, 0.0]
        
        norms = []
        start_time = time.time()
        for i in range(100):
            v, _, _, _ = self.op2c.overlap_deriv(itype, jtype, Rij, False)
            norms.append(np.linalg.norm(v))
        end_time = time.time()
        
        print(f"100 iterations took {end_time - start_time:.4f}s")
        
        # Check that all norms are identical
        avg_norm = np.mean(norms)
        max_diff = np.max(np.abs(np.array(norms) - avg_norm))
        print(f"Max deviation from mean: {max_diff}")
        self.assertLess(max_diff, 1e-14, "Results not consistent across iterations")

if __name__ == '__main__':
    unittest.main()
