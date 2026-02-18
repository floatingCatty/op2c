from op2c.nao.numerical_radial import NumericalRadial
import _op2c

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
    def symbol(self):
        return self._handle.symbol

    @property
    def itype(self):
        return self._handle.itype

    @property
    def lmax(self):
        return self._handle.lmax

    @property
    def nchi(self):
        return self._handle.nchi

    @property
    def nphi(self):
        return self._handle.nphi
        
    def nzeta(self, l: int):
        return self._handle.nzeta(l)

    def norb(self, l: int):
        return self._handle.norb(l)

    def __len__(self):
        return len(self._handle)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [NumericalRadial(handle=orb) for orb in self._handle[index]]
        return NumericalRadial(handle=self._handle[index])

    def __iter__(self):
        for orb in self._handle:
            yield NumericalRadial(handle=orb)

    def __repr__(self):
        return f"<BetaRadials symbol='{self.symbol}' nchi={len(self)}>"
