import _op2c
import numpy as np
from .beta_radials import BetaRadials

class AtomPseudo:
    """
    A class representing the pseudopotential of an atom.
    Wraps _op2c.Atom_pseudo.
    """
    def __init__(self, handle=None):
        if handle is None:
            self._handle = _op2c.Atom_pseudo()
        else:
            self._handle = handle

    @classmethod
    def from_upf(cls, filename: str, type: str = "upf", rcut: float = 0.0, 
                 lspinorb: bool = False, soc_lambda: float = 0.0):
        """
        Initialize AtomPseudo from a UPF (or other supported) file.
        
        Args:
            filename (str): Path to pseudopotential file.
            type (str): Type of pseudopotential ("upf", "vwr", "blps").
            rcut (float): Cutoff radius.
            lspinorb (bool): Whether spin-orbit coupling is enabled.
            soc_lambda (float): Spin-orbit coupling parameter.
        """
        obj = cls()
        obj._handle.init_from_upf(filename, type, rcut, lspinorb, soc_lambda)
        return obj

    @property
    def nproj(self):
        return self._handle.nproj
    
    @nproj.setter
    def nproj(self, value):
        self._handle.nproj = value

    @property
    def nproj_soc(self):
        return self._handle.nproj_soc
        
    @nproj_soc.setter
    def nproj_soc(self, value):
        self._handle.nproj_soc = value

    @property
    def itype(self):
        return self._handle.itype
    
    @itype.setter
    def itype(self, value):
        self._handle.itype = value

    @property
    def beta_radials(self):
        """Returns the BetaRadials (projectors) object wrapper."""
        return BetaRadials(self._handle.beta_radials)
    
    @property
    def d_real(self):
        """Returns the non-SOC coupling matrix D as a numpy array."""
        return np.array(self._handle.d_real, copy=False)

    @d_real.setter
    def d_real(self, value):
        self._handle.d_real = value
        
    @property
    def d_so(self):
        """Returns the spin-orbit part of the D matrix as a complex numpy array."""
        return self._handle.d_so
        
    # Read-only properties from base class
    @property
    def psd(self): return self._handle.psd
    
    @property
    def pp_type(self): return self._handle.pp_type
    
    @property
    def zv(self): return self._handle.zv
    
    @property
    def etotps(self): return self._handle.etotps
    
    @property
    def lmax(self): return self._handle.lmax
    
    @property
    def mesh(self): return self._handle.mesh
    
    @property
    def nbeta(self): return self._handle.nbeta
    
    @property
    def ecutwfc(self): return self._handle.ecutwfc
    
    @property
    def ecutrho(self): return self._handle.ecutrho
    
    @property
    def has_so(self): return self._handle.has_so
    
    # Array properties
    @property
    def vloc_at(self): return self._handle.vloc_at
    
    @property
    def r(self): return self._handle.r
    
    @property
    def rab(self): return self._handle.rab
    
    @property
    def rho_atc(self): return self._handle.rho_atc
    
    @property
    def lll(self): return self._handle.lll

    def __repr__(self):
        return self._handle.__repr__()
