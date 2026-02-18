# Op2c C++ API Guide

This document provides a high-level overview of the `op2c` C++ interface. The source code is maximizingly commented with Doxygen-style annotations.

## Generating Full Documentation

To generate the complete HTML API reference, install Doxygen and run:
```bash
sudo apt-get install doxygen graphviz
cd /path/to/op2c
doxygen Doxyfile
```
Then open `doc/html/index.html` in your browser.

## Core Classes

### [Op2c](src/int2c/op2c.hpp)
The main entry point for the library. It manages the high-level logic for computing integrals between atoms.

*   **Header**: `src/int2c/op2c.hpp`
*   **Key Methods**:
    *   `overlap(...)`: Computes overlap integrals $S_{ij}$.
    *   `overlap_position(...)`: Computes $S_{ij}$ and dipole integrals $\mathbf{D}_{ij} = \langle \phi_i | \mathbf{r} | \phi_j \rangle$.
    *   `orb_r_beta(...)`: Computes integrals involving non-local pseudopotential projectors.

### [TwoCenterIntegrator](src/int2c/integrator.h)
The mathematical engine exposed to `Op2c`. It handles the details of radial tabulation and spherical harmonic summation.

*   **Header**: `src/int2c/integrator.h`
*   **Key Methods**:
    *   `tabulate(...)`: Pre-computes radial integrals on a grid.
    *   `calculate(...)`: Evaluates the integral for a specific atomic pair vector $\mathbf{R}$.

### [RadialCollection](src/nao/atomic_basis.h)
Container for numerical atomic orbitals.

*   **Header**: `src/nao/atomic_basis.h`
*   **Purpose**: Loads and stores radial functions $\chi(r)$.

## Conventions
*   **Units**: Atomic units (Bohr for length, Hartree for energy) are used internally unless otherwise noted.
*   **Coordinates**: Cartesian coordinates $(x, y, z)$.
*   **Harmonics**: Real Spherical Harmonics are used throughout.
