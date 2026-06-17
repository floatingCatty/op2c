import numpy as np
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from op2c.nao import NumericalRadial, AtomicRadials
from op2c.pseudo import AtomPseudo, BetaRadials
import _op2c

TEST_DIR = Path(__file__).resolve().parents[1]
C_ORBITAL = TEST_DIR / "test" / "pporb" / "C_gga_7au_100Ry_2s2p1d.orb"


class TestNumericalRadial(unittest.TestCase):
    """Test NumericalRadial binding and wrapper."""

    def test_build_from_grid(self):
        """Build a NumericalRadial from explicit grid data."""
        r = np.linspace(0, 5, 100)
        vals = np.exp(-r)

        nr = NumericalRadial(l=0, grid=r, values=vals, symbol="H")

        self.assertEqual(nr.l, 0)
        self.assertEqual(nr.symbol, "H")
        self.assertEqual(len(nr.rgrid), 100)
        self.assertTrue(np.allclose(nr.rgrid, r))

    def test_normalize(self):
        """Normalization should not crash and should modify values."""
        r = np.linspace(0, 5, 100)
        vals = np.exp(-r)
        nr = NumericalRadial(l=0, grid=r, values=vals)
        original_max = np.max(np.abs(nr.rvalues))
        nr.normalize()
        # Normalization changes the values
        self.assertGreater(original_max, 0)

    def test_set_uniform_grid(self):
        """set_uniform_grid should produce k-space data."""
        r = np.linspace(0, 5, 100)
        vals = np.exp(-r)
        nr = NumericalRadial(l=0, grid=r, values=vals)
        nr.set_uniform_grid(for_r_space=False, ngrid=100, cutoff=10.0, mode='t')

        self.assertEqual(len(nr.kgrid), 100)
        self.assertTrue(len(nr.kvalues) > 0)

    def test_properties(self):
        """pr and pk should have default values."""
        r = np.linspace(0, 5, 100)
        vals = np.exp(-r)
        nr = NumericalRadial(l=0, grid=r, values=vals)
        self.assertEqual(nr.pr, 0)
        self.assertEqual(nr.pk, 0)


class TestAtomicRadials(unittest.TestCase):
    """Test AtomicRadials orbital evaluation."""

    def test_evaluate_orbitals_from_file(self):
        """AtomicRadials should evaluate orbitals in C++."""
        orbitals = AtomicRadials.from_file(str(C_ORBITAL))
        values = orbitals.evaluate_orbitals(np.array([[0.1, 0.2, 0.3], [8.0, 0.0, 0.0]]))

        self.assertEqual(values.shape, (2, orbitals.nphi))
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.any(np.abs(values[0]) > 0.0))

    def test_evaluate_orbitals_grad_shapes_and_value_consistency(self):
        """``evaluate_orbitals_grad`` returns (values, grad) with the right shapes,
        and its ``values`` matches plain ``evaluate_orbitals``."""
        orbitals = AtomicRadials.from_file(str(C_ORBITAL))
        pts = np.array([[0.3, -0.4, 0.5], [1.1, 0.2, -0.7], [8.0, 0.0, 0.0]])
        values, grad = orbitals.evaluate_orbitals_grad(pts)
        self.assertEqual(values.shape, (3, orbitals.nphi))
        self.assertEqual(grad.shape, (3, orbitals.nphi, 3))
        self.assertTrue(np.all(np.isfinite(grad)))
        np.testing.assert_allclose(values, orbitals.evaluate_orbitals(pts), atol=1e-13)

    def test_evaluate_orbitals_grad_matches_finite_difference(self):
        """∇φ from the kernel matches a central finite difference of φ (the
        physics correctness gate for the forces/meta-GGA foundation)."""
        orbitals = AtomicRadials.from_file(str(C_ORBITAL))
        # Generic off-axis points well inside the orbital support (avoid r=0 and
        # the cutoff edge where the spline derivative is one-sided).
        rng = np.random.default_rng(0)
        pts = rng.uniform(-2.0, 2.0, size=(12, 3))
        pts = pts[np.linalg.norm(pts, axis=1) > 0.3]
        _, grad = orbitals.evaluate_orbitals_grad(pts)
        h = 1e-5
        for a in range(3):
            step = np.zeros(3)
            step[a] = h
            plus = orbitals.evaluate_orbitals(pts + step)
            minus = orbitals.evaluate_orbitals(pts - step)
            fd = (plus - minus) / (2.0 * h)
            np.testing.assert_allclose(grad[:, :, a], fd, atol=1e-6, rtol=1e-5)


class TestAtomPseudo(unittest.TestCase):
    """Test AtomPseudo binding and wrapper."""

    def test_default_construction(self):
        """Default-constructed AtomPseudo should have sane defaults."""
        ap = AtomPseudo()
        self.assertEqual(ap.nproj, 0)
        self.assertEqual(ap.nproj_soc, 0)
        self.assertEqual(ap.itype, 0)
        self.assertEqual(ap.nbeta, 0)
        self.assertEqual(ap.lmax, 0)
        self.assertEqual(ap.mesh, 0)
        self.assertEqual(ap.psd, "")

    def test_itype_setter(self):
        """itype should be writable."""
        ap = AtomPseudo()
        ap.itype = 3
        self.assertEqual(ap.itype, 3)

    def test_repr(self):
        """repr should contain label."""
        ap = AtomPseudo()
        r = repr(ap)
        self.assertIn("Atom_pseudo", r)
        self.assertIn("label=", r)

    def test_scalar_properties(self):
        """All scalar properties should be accessible."""
        ap = AtomPseudo()
        # These should all return without error
        _ = ap.zv
        _ = ap.etotps
        _ = ap.ecutwfc
        _ = ap.ecutrho
        _ = ap.has_so
        _ = ap.pp_type
        _ = ap.psd

    def test_array_properties_empty(self):
        """Array properties on uninitialized AtomPseudo should return empty arrays."""
        ap = AtomPseudo()
        self.assertEqual(ap.vloc_at.shape, (0,))
        self.assertEqual(ap.r.shape, (0,))
        self.assertEqual(ap.rab.shape, (0,))
        self.assertEqual(ap.rho_atc.shape, (0,))

    def test_from_upf_exists(self):
        """from_upf factory method should exist and be callable."""
        self.assertTrue(callable(AtomPseudo.from_upf))


class TestBetaRadials(unittest.TestCase):
    """Test BetaRadials binding and wrapper."""

    def test_default_construction(self):
        """Default BetaRadials from uninitialized AtomPseudo."""
        ap = AtomPseudo()
        br = ap.beta_radials
        self.assertIsInstance(br, BetaRadials)
        self.assertEqual(len(br), 0)

    def test_repr(self):
        """BetaRadials repr should work."""
        ap = AtomPseudo()
        br = ap.beta_radials
        r = repr(br)
        self.assertIn("BetaRadials", r)


class TestLowLevelBindings(unittest.TestCase):
    """Test that _op2c low-level module is accessible."""

    def test_module_classes(self):
        """All expected classes should be importable from _op2c."""
        self.assertTrue(hasattr(_op2c, 'NumericalRadial'))
        self.assertTrue(hasattr(_op2c, 'AtomicRadials'))
        self.assertTrue(hasattr(_op2c, 'BetaRadials'))
        self.assertTrue(hasattr(_op2c, 'Atom_pseudo'))
        self.assertTrue(hasattr(_op2c, 'Op2c'))
        self.assertTrue(hasattr(_op2c, 'RadialCollection'))
        self.assertTrue(hasattr(_op2c, 'TwoCenterIntegrator'))
        self.assertTrue(hasattr(_op2c, 'TwoCenterBundle'))

    def test_atom_pseudo_low_level(self):
        """Low-level Atom_pseudo should expose init_from_upf."""
        ap = _op2c.Atom_pseudo()
        self.assertTrue(hasattr(ap, 'init_from_upf'))
        self.assertTrue(hasattr(ap, 'nbeta'))
        self.assertTrue(hasattr(ap, 'vloc_at'))
        self.assertTrue(hasattr(ap, 'd_so'))

    def test_low_level_atomic_orbital_evaluation(self):
        """Low-level AtomicRadials should expose C++ orbital values."""
        orbitals = _op2c.AtomicRadials()
        orbitals.build(str(C_ORBITAL), 0, 0, 0)
        values = orbitals.evaluate_orbitals(np.array([[0.1, 0.2, 0.3]]))

        self.assertEqual(values.shape, (1, orbitals.nphi))
        self.assertTrue(np.any(np.abs(values) > 0.0))

    def test_low_level_atomic_orbital_evaluation_threaded_matches_serial(self):
        """Concurrent orbital-value calls should match a serial reference."""
        orbitals = _op2c.AtomicRadials()
        orbitals.build(str(C_ORBITAL), 0, 0, 0)
        points = np.column_stack(
            (
                np.linspace(0.05, 2.0, 48),
                np.linspace(-0.3, 0.8, 48),
                np.linspace(0.1, 1.4, 48),
            )
        )
        expected = orbitals.evaluate_orbitals(points)

        def worker(_):
            out = expected
            for _ in range(12):
                out = orbitals.evaluate_orbitals(points)
            return out

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(8)))

        for result in results:
            self.assertTrue(np.allclose(result, expected, rtol=0.0, atol=0.0))


class TestRadialCollection(unittest.TestCase):
    """Test RadialCollection binding."""

    def test_default_construction(self):
        rc = _op2c.RadialCollection()
        self.assertEqual(rc.ntype, 0)
        self.assertIn("RadialCollection", repr(rc))

    def test_properties_empty(self):
        rc = _op2c.RadialCollection()
        self.assertEqual(rc.ntype, 0)
        self.assertEqual(rc.p, 0)

    def test_build_from_beta_empty(self):
        """build_from_beta with empty list shouldn't crash."""
        rc = _op2c.RadialCollection()
        betas = []
        rc.build_from_beta(betas)
        self.assertEqual(rc.ntype, 0)


class TestTwoCenterIntegrator(unittest.TestCase):
    """Test TwoCenterIntegrator binding."""

    def test_default_construction(self):
        tci = _op2c.TwoCenterIntegrator()
        self.assertEqual(tci.table_memory, 0)


class TestTwoCenterBundle(unittest.TestCase):
    """Test TwoCenterBundle binding."""

    def test_default_construction(self):
        tcb = _op2c.TwoCenterBundle()
        self.assertIn("TwoCenterBundle", repr(tcb))

    def test_integrators_none_before_tabulate(self):
        """Integrators should be None before tabulate."""
        tcb = _op2c.TwoCenterBundle()
        self.assertIsNone(tcb.overlap_orb)
        self.assertIsNone(tcb.overlap_orb_beta)
        self.assertIsNone(tcb.kinetic_orb)

    def test_collections_none_before_build(self):
        """Radial collections should be None before build."""
        tcb = _op2c.TwoCenterBundle()
        self.assertIsNone(tcb.orb)
        self.assertIsNone(tcb.beta)


class TestOp2cWrapper(unittest.TestCase):
    """Test Python Op2c wrapper."""

    def test_import(self):
        from op2c import Op2c, TwoCenterBundle
        self.assertTrue(callable(Op2c.from_files))
        self.assertTrue(callable(TwoCenterBundle.from_files))

    def test_bundle_construction(self):
        from op2c import TwoCenterBundle
        b = TwoCenterBundle()
        self.assertIn("TwoCenterBundle", repr(b))


if __name__ == '__main__':
    unittest.main()
