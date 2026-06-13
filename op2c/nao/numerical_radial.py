from __future__ import annotations
import numpy as np
import _op2c
from typing import Optional, Any

class NumericalRadial:
    """
    A class representing a numerical radial function defined on a grid.
    Wraps the low-level C++ _op2c.NumericalRadial class.
    """
    def __init__(self, l: int = 0, grid: Optional[np.ndarray] = None, values: Optional[np.ndarray] = None, 
                 itype: int = 0, symbol: str = "", handle=None):
        """
        Initialize a NumericalRadial object.

        Args:
            l (int): Angular momentum.
            grid (np.ndarray): The radial grid points r (1D array).
            values (np.ndarray): The radial function values f(r) (1D array).
            itype (int): Element type index.
            symbol (str): Chemical symbol.
            handle (_op2c.NumericalRadial, optional): Wrap an existing low-level object. 
                                                      If provided, other args are ignored.
        """
        if handle is not None:
            self._handle = handle
        else:
            if grid is None or values is None:
                raise ValueError("Both 'grid' and 'values' must be provided if handle is None.")
            
            # Ensure proper types for C++ binding
            grid = np.ascontiguousarray(grid, dtype=np.float64)
            values = np.ascontiguousarray(values, dtype=np.float64)
            
            self._handle = _op2c.NumericalRadial()
            # Default to building in r-space
            self._handle.build(l, True, grid, values, 0, 0, symbol, itype, True)

    @property
    def l(self) -> int:
        return self._handle.l

    @property
    def symbol(self) -> str:
        return self._handle.symbol

    @property
    def itype(self) -> int:
        return self._handle.itype

    @property
    def rgrid(self) -> np.ndarray:
        """Returns the r-space grid as a numpy array."""
        return self._handle.rgrid

    @property
    def rvalues(self) -> np.ndarray:
        """Returns the r-space values as a numpy array."""
        return self._handle.rvalues

    @property
    def kgrid(self) -> np.ndarray:
        """Returns the k-space grid as a numpy array."""
        return self._handle.kgrid

    @property
    def kvalues(self) -> np.ndarray:
        """Returns the k-space values as a numpy array."""
        return self._handle.kvalues
        
    @property
    def pr(self) -> int:
        """Implicit exponent in r-values."""
        return self._handle.pr

    @property
    def pk(self) -> int:
        """Implicit exponent in k-values."""
        return self._handle.pk

    @property
    def kcut(self) -> float:
        """K-space cutoff."""
        return self._handle.kcut

    def set_grid(self, for_r_space: bool, grid: np.ndarray, mode: str = 'i') -> None:
        """
        Sets up the grid in the specified space.
        
        Args:
            for_r_space (bool): True for r-space, False for k-space.
            grid (np.ndarray): 1D array of grid points.
            mode (str): 'i' for interpolation, 't' for transform.
        """
        grid = np.ascontiguousarray(grid, dtype=np.float64)
        self._handle.set_grid(for_r_space, grid, mode)

    def set_uniform_grid(self, for_r_space: bool, ngrid: int, cutoff: float, mode: str = 'i', enable_fft: bool = False) -> None:
        """
        Sets up a uniform grid.
        
        Args:
            for_r_space (bool): True for r-space, False for k-space.
            ngrid (int): Number of grid points.
            cutoff (float): Cutoff radius/k-vector.
            mode (str): 'i' for interpolation, 't' for transform.
            enable_fft (bool): If True, ensure grid is FFT compliant.
        """
        self._handle.set_uniform_grid(for_r_space, ngrid, cutoff, mode, enable_fft)

    def normalize(self) -> None:
        """Normalizes the radial function."""
        self._handle.normalize(True) # Normalize in r-space

    def plot(self, ax: Optional[Any] = None, label: Optional[str] = None, **kwargs) -> Any:
        """
        Plot the radial function using matplotlib.
        
        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            label (str, optional): Label for the plot.
            **kwargs: Additional keyword arguments passed to ax.plot.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        
        lbl = label if label else f"{self.symbol} (l={self.l})"
        ax.plot(self.rgrid, self.rvalues, label=lbl, **kwargs)
        ax.set_xlabel("r (Bohr)")
        ax.set_ylabel("f(r)")
        ax.legend()
        return ax

    def __repr__(self) -> str:
        return f"<NumericalRadial symbol='{self.symbol}' l={self.l} rmax={self._handle.rmax:.2f}>"
