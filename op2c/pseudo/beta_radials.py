from __future__ import annotations
from op2c.nao.numerical_radial import NumericalRadial
import _op2c
from typing import Union, List, Iterator

class BetaRadials:
    """
    Wrapper for BetaRadials class (collection of non-local projectors).
    """
    def __init__(self, handle=None):
        if handle:
            self._handle = handle
        else:
            self._handle = _op2c.BetaRadials()

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
    def nchi(self) -> int:
        return self._handle.nchi

    @property
    def nphi(self) -> int:
        return self._handle.nphi
        
    def nzeta(self, l: int) -> int:
        return self._handle.nzeta(l)

    def norb(self, l: int) -> int:
        return self._handle.norb(l)

    def __len__(self) -> int:
        return len(self._handle)

    def __getitem__(self, index: Union[int, slice]) -> Union[NumericalRadial, List[NumericalRadial]]:
        if isinstance(index, slice):
            return [NumericalRadial(handle=orb) for orb in self._handle[index]]
        return NumericalRadial(handle=self._handle[index])

    def __iter__(self) -> Iterator[NumericalRadial]:
        for orb in self._handle:
            yield NumericalRadial(handle=orb)

    def __repr__(self) -> str:
        return f"<BetaRadials symbol='{self.symbol}' nchi={len(self)}>"
