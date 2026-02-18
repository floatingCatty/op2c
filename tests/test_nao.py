import numpy as np
import unittest
from op2c.nao import NumericalRadial, AtomicRadials
from op2c.pseudo import AtomPseudo, BetaRadials
import _op2c


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

    def test_atom_pseudo_low_level(self):
        """Low-level Atom_pseudo should expose init_from_upf."""
        ap = _op2c.Atom_pseudo()
        self.assertTrue(hasattr(ap, 'init_from_upf'))
        self.assertTrue(hasattr(ap, 'nbeta'))
        self.assertTrue(hasattr(ap, 'vloc_at'))
        self.assertTrue(hasattr(ap, 'd_so'))


if __name__ == '__main__':
    unittest.main()
