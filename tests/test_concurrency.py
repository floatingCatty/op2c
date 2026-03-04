
import unittest
import numpy as np
import concurrent.futures
import time
from op2c import Op2c

class TestOp2cConcurrency(unittest.TestCase):
    def setUp(self):
        # Use existing paths found in environment
        self.orb_dir = "/home/zhanghao/softwares/nao-abacus/test/pporb/"
        self.orb_name = ["C_gga_7au_100Ry_2s2p1d.orb"]
        self.psd_dir = self.orb_dir
        self.psd_name = ["C_ONCV_PBE-1.0.upf"]
        self.ntype = 1
        self.nspin = 1
        self.lspinorb = False
        
        # Initialize Op2c object (shared resource)
        self.op2c = Op2c.from_files(self.orb_dir, self.orb_name, 
                         self.psd_dir, self.psd_name, 
                         self.nspin, self.lspinorb, 
                         "test_concurrency.log", 0)

    def worker_overlap(self, task_id):
        """Worker function to compute overlap integrals."""
        try:
            # Randomize input slightly to simulate different tasks
            itype = 0
            jtype = 0
            # Shift Rj based on task_id to avoid identical calls (though redundant calls are also good for testing race conditions)
            Rij = [2.0 + task_id * 0.01, 0.0, 0.0]
            
            # Call overlapping computation
            S = self.op2c.overlap(itype, jtype, Rij, False)
            
            # Simple check
            norm = np.linalg.norm(S)
            return norm
        except Exception as e:
            return e

    def worker_deriv(self, task_id):
        """Worker function to compute overlap derivatives."""
        try:
            itype = 0
            jtype = 0
            Rij = [2.0 + task_id * 0.01, 0.1, 0.0]
            S, dSx, dSy, dSz = self.op2c.overlap_deriv(itype, jtype, Rij, False)
            return np.linalg.norm(dSx)
        except Exception as e:
            return e

    def test_concurrent_overlap(self):
        """Test overlap calculation with multiple threads."""
        n_threads = 8
        n_tasks = 100
        
        print(f"\nRunning {n_tasks} overlap tasks with {n_threads} threads...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            # Submit tasks
            futures = [executor.submit(self.worker_overlap, i) for i in range(n_tasks)]
            
            # Wait for results
            results = []
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if isinstance(res, Exception):
                    self.fail(f"Thread worker failed with exception: {res}")
                results.append(res)
        
        elapsed = time.time() - start_time
        print(f"Completed in {elapsed:.4f}s")
        
        # Check if we got results (norms should be positive)
        self.assertTrue(all(r > 0 for r in results if isinstance(r, float)))
        self.assertEqual(len(results), n_tasks)

    def test_concurrent_mixed(self):
        """Test mixed calls (overlap + deriv) concurrently."""
        n_threads = 8
        n_tasks = 50
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures_ovlp = [executor.submit(self.worker_overlap, i) for i in range(n_tasks)]
            futures_deriv = [executor.submit(self.worker_deriv, i) for i in range(n_tasks)]
            
            all_futures = futures_ovlp + futures_deriv
            
            for future in concurrent.futures.as_completed(all_futures):
                res = future.result()
                if isinstance(res, Exception):
                    self.fail(f"Thread worker failed: {res}")

if __name__ == '__main__':
    unittest.main()
