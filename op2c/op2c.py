"""Pythonic wrapper for Op2c two-center integral engine."""
import numpy as np
import _op2c


class TwoCenterBundle:
    """Wrapper for C++ TwoCenterBundle.
    
    Manages radial function collections and two-center integration tables.
    """
    
    def __init__(self, handle=None):
        self._handle = handle or _op2c.TwoCenterBundle()
    
    @classmethod
    def from_files(cls, orb_files, orb_dir, betas=None):
        """Build bundle from orbital files and optional beta projectors.
        
        Parameters
        ----------
        orb_files : list[str]
            Orbital file names (one per atom type).
        orb_dir : str
            Directory containing orbital files.
        betas : list[BetaRadials], optional
            Beta projectors from Atom_pseudo objects.
        """
        obj = cls()
        obj._handle.build_orb(orb_files, orb_dir)
        if betas:
            obj._handle.build_beta(betas)
        return obj
    
    def tabulate(self):
        """Build two-center integration tables (automatic grid)."""
        self._handle.tabulate()
    
    def tabulate_with_params(self, lcao_ecut, lcao_dk, lcao_dr, lcao_rmax):
        """Build two-center integration tables with explicit parameters."""
        self._handle.tabulate_with_params(lcao_ecut, lcao_dk, lcao_dr, lcao_rmax)
    
    @property
    def overlap_orb(self):
        return self._handle.overlap_orb
    
    @property
    def overlap_orb_beta(self):
        return self._handle.overlap_orb_beta
    
    @property
    def kinetic_orb(self):
        return self._handle.kinetic_orb
    
    @property
    def orb(self):
        return self._handle.orb
    
    @property
    def beta(self):
        return self._handle.beta
    
    def __repr__(self):
        return repr(self._handle)


class Op2c:
    """High-level interface for two-center integral calculations.
    
    Provides overlap, position operator, and non-local pseudopotential
    integrals between numerical atomic orbital pairs.
    """
    
    def __init__(self, handle):
        self._handle = handle
    
    @classmethod
    def from_files(cls, orb_dir, orb_names, psd_dir="", psd_names=None,
                   nspin=1, lspinorb=False, log_file="", mpi_handle=0):
        """Create Op2c from orbital and pseudopotential file paths.
        
        Parameters
        ----------
        orb_dir : str
            Directory containing orbital files.
        orb_names : list[str]
            Orbital file names (one per atom type).
        psd_dir : str, optional
            Directory containing pseudopotential files.
        psd_names : list[str], optional
            Pseudopotential file names (one per atom type).
        nspin : int
            Number of spin components (1 or 2).
        lspinorb : bool
            Enable spin-orbit coupling.
        log_file : str
            Path to log file.
        mpi_handle : int
            MPI communicator handle (Fortran).
        """
        psd_names = psd_names or []
        ntype = len(orb_names)
        h = _op2c.Op2c(ntype, nspin, lspinorb, 
                        orb_dir, orb_names,
                        psd_dir, psd_names,
                        log_file, mpi_handle)
        return cls(h)
    
    @classmethod
    def from_objects(cls, orbitals, pseudos=None, nspin=1, lspinorb=False):
        """Create Op2c from pre-loaded AtomicRadials and AtomPseudo objects.
        
        No file I/O is performed. Both orbitals and pseudopotentials are
        passed as pre-loaded Python objects.
        
        Parameters
        ----------
        orbitals : list[AtomicRadials]
            Pre-loaded AtomicRadials objects (one per atom type).
            Can be either raw _op2c.AtomicRadials or op2c.nao.AtomicRadials wrappers.
        pseudos : list[Atom_pseudo | AtomPseudo], optional
            Pre-loaded pseudopotential objects. Can be either raw 
            _op2c.Atom_pseudo handles or op2c.pseudo.AtomPseudo wrappers.
        nspin : int
            Number of spin components (1 or 2).
        lspinorb : bool
            Enable spin-orbit coupling.
        """
        # Convert wrappers to raw handles if needed
        raw_orbs = []
        for o in orbitals:
            if hasattr(o, '_handle'):
                raw_orbs.append(o._handle)
            else:
                raw_orbs.append(o)
        
        raw_pseudos = []
        if pseudos:
            for p in pseudos:
                if hasattr(p, '_handle'):
                    raw_pseudos.append(p._handle)
                else:
                    raw_pseudos.append(p)
        
        h = _op2c.Op2c(raw_orbs, raw_pseudos, nspin, lspinorb)
        return cls(h)
    
    @property
    def bundle(self):
        """Access the underlying TwoCenterBundle."""
        return TwoCenterBundle(self._handle.tcbd)
    
    def get_orb_rcut_max(self, itype):
        """Maximum orbital cutoff radius for given atom type."""
        return self._handle.get_orb_rcut_max(itype)
    
    def get_beta_rcut_max(self, itype):
        """Maximum beta projector cutoff radius for given atom type."""
        return self._handle.get_beta_rcut_max(itype)
    
    def overlap(self, itype, jtype, Rij, transpose=False):
        """Compute overlap integral <phi_i | phi_j>.
        
        Parameters
        ----------
        itype, jtype : int
            Element type indices.
        Rij : list[float]
            Displacement vector R_j - R_i (length 3).
        transpose : bool
            If True, transpose the result block.
        
        Returns
        -------
        np.ndarray
            Flattened overlap matrix.
        """
        return self._handle.overlap(itype, jtype, Rij, transpose)
    
    def overlap_deriv(self, itype, jtype, Rij, transpose=False):
        """Compute overlap integral and its gradient.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (S, dS/dx, dS/dy, dS/dz)
        """
        return self._handle.overlap_deriv(itype, jtype, Rij, transpose)
    
    def overlap_position(self, itype, jtype, Ri, Rj, transpose=False):
        """Compute overlap and position operator integrals.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (S, <x>, <y>, <z>)
        """
        return self._handle.overlap_position(itype, jtype, Ri, Rj, transpose)
    
    def orb_r_beta(self, itypes, ktype, Ri, Rk, transpose=False):
        """Compute orbital-beta projector integrals <phi_i | beta_k>.
        
        Returns
        -------
        tuple[list, list, list, list]
            (ob, oxb, oyb, ozb) matrices for each atom.
        """
        return self._handle.orb_r_beta(itypes, ktype, Ri, Rk, transpose)
    
    def ncomm_IKJ(self, itype, idx, ktype, jtype, jdx, 
                  ob, oxb, oyb, ozb, npol=1, transpose=False):
        """Compute non-local commutator integrals [r, V_nl].
        
        Returns
        -------
        tuple[list, list, list]
            (vx, vy, vz) complex commutator integrals.
        """
        return self._handle.ncomm_IKJ(itype, idx, ktype, jtype, jdx,
                                       ob, oxb, oyb, ozb, npol, transpose)
    
    def __repr__(self):
        return "<Op2c>"
