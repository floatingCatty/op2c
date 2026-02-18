import unittest
import sys
import os
import numpy as np

from _op2c import Op2c


class TestOp2CDifferentiate(unittest.TestCase):
    def setUp(self):
        self.orb_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/"
        self.orb_name = ["C_gga_7au_100Ry_2s2p1d.orb"]
        self.ntype = 1
        self.nspin = 1
        self.lspinorb = False
        
        # Initialize without pseudos
        self.op2c = Op2c(self.ntype, self.nspin, self.lspinorb, 
                         self.orb_dir, self.orb_name, 
                         "", [],  # No pseudos
                         "test_diff.log", 0)

    def test_overlap_value(self):
        itype = 0
        jtype = 0
        Rij = [2.0, 0.0, 0.0]
        
        # Reference value from pre-refactor/smoke test run
        expected_norm = 2.208489944340288
        
        v = self.op2c.overlap(itype, jtype, Rij, False)
        calculated_norm = np.linalg.norm(v)
        
        print(f"Calculated: {calculated_norm}, Expected: {expected_norm}")
        self.assertAlmostEqual(calculated_norm, expected_norm, places=15)

    def test_position_value(self):
        itype = 0
        jtype = 0
        Rij = [2.0, 0.0, 0.0]
        
        R1 = [0.0, 0.0, 0.0]
        R2 = Rij
        S, Rx, Ry, Rz = self.op2c.overlap_position(itype, jtype, R1, R2, False)
        
        norm_S  = np.linalg.norm(S)
        norm_Rx = np.linalg.norm(Rx)
        norm_Ry = np.linalg.norm(Ry)
        norm_Rz = np.linalg.norm(Rz)
        
        print(f"Position Norms - S: {norm_S}, Rx: {norm_Rx}, Ry: {norm_Ry}, Rz: {norm_Rz}")
        
        expected_norm_S = 2.208489944340288
        self.assertAlmostEqual(norm_S, expected_norm_S, places=15)
        
        expected_norm_Rx = 3.107137526811723
        expected_norm_Ry = 2.753701504677152
        expected_norm_Rz = 2.753701504677152
        
        self.assertAlmostEqual(norm_Rx, expected_norm_Rx, places=15)
        self.assertAlmostEqual(norm_Ry, expected_norm_Ry, places=15)
        self.assertAlmostEqual(norm_Rz, expected_norm_Rz, places=14)

if __name__ == '__main__':
    unittest.main()
