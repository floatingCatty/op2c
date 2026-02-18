# Grid Integration Framework Development Plan (VBCSR + Intrinsic Basis)

## 1. Objective
Develop a modular, object-oriented grid integration framework for `op2c` that:
1.  **Integrates seamlessly** with `op2c` and `vbcsr`.
2.  **Improves logical structure**: Merges basis evaluation logic into `RadialCollection` / `NumericalRadial` as **intrinsic properties**.
3.  **Generalizes application**: Supports diverse tasks (Density, LDOS, Potential Integrals).

## 2. Design Philosophy
We adopt a **Component-Based Architecture**:

-   **Atomic Input**: `vbcsr::atomic::AtomicData` (Distributed atoms).
-   **Matrix Output**: `vbcsr::atomic::ImageContainer` (Handles PBC logic).
-   **Grid**: Represents the physical discretization.
-   **Basis Evaluation**: **Intrinsic**. The basis classes know how to evaluate themselves.
-   **Integrator**: Orchestrates the loop over Grid and Atoms.

## 3. Core Component Enhancements

### 3.1. `NumericalRadial` Class (Enhanced)
Modify `src/nao/radial_function.h` to support point-wise evaluation.

```cpp
class NumericalRadial {
public:
    // ... existing ...

    // NEW: Evaluate radial function at distance r
    // Uses cubic spline interpolation (matches ABACUS/SIESTA accuracy)
    double evaluate(double r) const;
    
    // NEW: Evaluate derivative
    double evaluate_deriv(double r) const;
};
```

### 3.2. `RadialCollection` Class (Enhanced)
Modify `src/nao/atomic_basis.h` to support full orbital evaluation.

```cpp
class RadialCollection {
public:
    // ... existing ...

    // NEW: Evaluate single orbital (itype, ia) at vector dr (relative to atom)
    double evaluate(int itype, int ia, const Vec3& dr) const;

    // NEW: Batch evaluation for efficiency
    void evaluate_batch(
        int itype, 
        int ia,
        const std::vector<Vec3>& points, 
        std::vector<double>& out_values
    ) const;
};
```

## 4. `GridIntegrator` Class
The high-level driver.

```cpp
class GridIntegrator {
public:
    // Compute Matrix Element: M_ij(R) = \int \phi_i(r) * V(r) * \phi_j(r-R) dr
    // Output: ImageContainer (stores sparse matrices for each periodic shift R)
    void integrate_matrix(
        const Grid& grid,
        const vbcsr::atomic::AtomicData& atoms,
        const RadialCollection& basis,
        const std::vector<double>& field,
        vbcsr::atomic::ImageContainer<double>& out_container
    );
};
```

## 5. Numerical Accuracy Strategy
To ensure high accuracy and efficiency, we adopt best practices from established codes like **SIESTA**, **OpenMX**, and **FHI-aims**:

### 5.1. Grid Density & Convergence (The "Mesh Cutoff")
*   **Strategy**: Instead of arbitrary grid spacing, we define grid density using an equivalent **Energy Cutoff (Ry)**, similar to **SIESTA** and **OpenMX**.
    *   $E_{cut} \propto 1/h^2$, where $h$ is grid spacing.
*   **Implementation**: The `Grid` class will allow initialization via `MeshCutoff`. Users can systematically increase this cutoff (e.g., 100 Ry -> 200 Ry -> 400 Ry) to converge integrals.
*   **Goal**: Enable sub-meV accuracy by allowing users to request grids fine enough for their specific pseudopotentials.

### 5.2. Interpolation Accuracy
*   **Strategy**: **Cubic Spline Interpolation**.
*   **Justification**: Linear interpolation requires extremely dense radial grids (orders of magnitude more points). Cubic splines provide $O(h^4)$ error scaling, allowing standard logarithmic radial grids (used in ABACUS/SIESTA) to be interpolated accurately onto the uniform integration grid.
*   **Smoothing**: Ensure radial functions decay smoothly to exactly zero at $r_{cut}$ to avoid numerical noise at block boundaries (similar to **FHI-aims**' "tight" settings).

### 5.3. Egg-Box Effect Mitigation
*   **Problem**: Invariance of total energy with respect to grid translation (breaking continuous translation symmetry).
*   **Strategy**: Implement **Grid Cell Sampling** (optional): Shift the grid randomly or systematically and average results, or use high-order stencils for derivatives. For standard integrals, using a sufficiently high `MeshCutoff` (typically >300 Ry for hard potentials) is the primary solution.

### 5.4. High-Order Quadrature (Future Proofing)
*   **Current**: Riemann Sum (product of values $\times$ volume element).
*   **Future**: Enable **Simpson’s Rule** weighted integration for non-uniform atom-centered grids if we implement them later. For the current uniform grid, Reimann sum is consistent with the FFT-based dual-space logic of standard Plane-Wave/LCAO codes.

## 6. Implementation Steps

### Phase 1: Foundation (Intrinsic Evaluation)
1.  Implement `NumericalRadial::evaluate` (Cubic Spline).
2.  Implement `RadialCollection::evaluate` (Radial + $Y_{lm}$).

### Phase 2: Interface & Grid
1.  Define C++ interfaces for `Grid`.
2.  Implement `UniformGrid` obeying `MeshCutoff` logic.

### Phase 3: Integrator & Application
1.  Implement `GridIntegrator` using `ImageContainer`.
2.  Bind to Python via `pybind11`.
