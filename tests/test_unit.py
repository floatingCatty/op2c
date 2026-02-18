import unittest
import sys
import os
import numpy as np

from _op2c import Op2c


class TestOp2CUnit(unittest.TestCase):
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
                         "test_unit.log", 0)

    def test_overlap(self):
        itype = 0
        jtype = 0
        Rij = [2.0, 0.0, 0.0]
        v = self.op2c.overlap(itype, jtype, Rij, False)
        self.assertTrue(v.size > 0)
        # Check return type is numpy array
        self.assertIsInstance(v, np.ndarray)

    def test_overlap_deriv(self):
        itype = 0
        jtype = 0
        Rij = [2.0, 0.0, 0.0]
        v, dvx, dvy, dvz = self.op2c.overlap_deriv(itype, jtype, Rij, False)
        self.assertTrue(v.size > 0)
        self.assertEqual(v.shape, dvx.shape)
        self.assertEqual(v.shape, dvy.shape)
        self.assertEqual(v.shape, dvz.shape)

    def test_overlap_position(self):
        itype = 0
        jtype = 0
        Ri = [0.0, 0.0, 0.0]
        Rj = [2.0, 0.0, 0.0]
        v, vx, vy, vz = self.op2c.overlap_position(itype, jtype, Ri, Rj, False)
        self.assertTrue(v.size > 0)
        self.assertEqual(v.shape, vx.shape)
    
    def test_orb_r_beta(self):
        itypes = [0]
        ktype = 0
        Ri = [[0.0, 0.0, 0.0]]
        Rk = [0.5, 0.0, 0.0]
        # returns tuple of lists of matrices/arrays
        ob, oxb, oyb, ozb = self.op2c.orb_r_beta(itypes, ktype, Ri, Rk, False)
        self.assertEqual(len(ob), 1)
        self.assertTrue(ob[0].size > 0)

    def test_ncomm_IKJ(self):
        itype = 0
        ktype = 0
        jtype = 0
        
        Ri = [[0.0, 0.0, 0.0]]
        Rk = [0.5, 0.0, 0.0]
        
        ob, oxb, oyb, ozb = self.op2c.orb_r_beta([itype], ktype, Ri, Rk, False)
        
        # args: itype, idx, ktype, jtype, jdx, ob, oxb, oyb, ozb, npol, is_transpose
        # returns: vx, vy, vz
        vx, vy, vz = self.op2c.ncomm_IKJ(itype, 0, ktype, jtype, 0, ob, oxb, oyb, ozb, 1, False)
        
        self.assertEqual(len(vx), len(vy))
        self.assertEqual(len(vy), len(vz))
        self.assertTrue(len(vx) > 0)
        self.assertIsInstance(vx, list)

if __name__ == '__main__':
    unittest.main()
