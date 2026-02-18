from __future__ import annotations
import _op2c
from .numerical_radial import NumericalRadial
from typing import Union, List, Iterator, Optional

class AtomicRadials:
    """
    A collection of numerical atomic orbitals for a single element.
    Wraps _op2c.AtomicRadials.
    Implements the Sequence protocol.
    """
    def __init__(self, handle=None):
        if handle is None:
            self._handle = _op2c.AtomicRadials()
        else:
            self._handle = handle

    @classmethod
    def from_file(cls, filename: str, itype: int = 0, p: int = 0, pm: int = 0) -> AtomicRadials:
        """
        Create an AtomicRadials object from an ABACUS orbital file.
        
        Args:
            filename (str): Path to the .orb file.
            itype (int): Element type index.
            p (int): Implicit exponent power.
            pm (int): Angular momentum shift.
        """
        obj = cls()
        obj._handle.build(filename, itype, p, pm)
        return obj

    @property
    def symbol(self) -> str:
        return self._handle.symbol

    @property
    def itype(self) -> int:
        return self._handle.itype

    @property
    def lmax(self) -> int:
        return self._handle.lmax

    @property
    def rcut_max(self) -> float:
        return self._handle.rcut_max

    @property
    def orb_ecut(self) -> float:
        """Energy cutoff suggested by the orbital file."""
        return self._handle.orb_ecut()
    
    def nzeta(self, l: int) -> int:
        return self._handle.nzeta(l)

    def norb(self, l: int) -> int:
        return self._handle.norb(l)

    def __len__(self) -> int:
        return len(self._handle)

    def __getitem__(self, index: Union[int, slice]) -> Union[NumericalRadial, List[NumericalRadial]]:
        if isinstance(index, slice):
            # Return a list of wrapped NumericalRadial objects
            return [NumericalRadial(handle=orb) for orb in self._handle[index]]
        
        # Return a single wrapped object
        orb_handle = self._handle[index]
        return NumericalRadial(handle=orb_handle)

    def __iter__(self) -> Iterator[NumericalRadial]:
        for orb in self._handle:
            yield NumericalRadial(handle=orb)

    def __repr__(self) -> str:
        return f"<AtomicRadials symbol='{self.symbol}' nchi={len(self)} lmax={self.lmax}>"
