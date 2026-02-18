# Theory: Position Operator Evaluation

This document details the algorithm for computing matrix elements of the position operator $\hat{\mathbf{r}}$ between two Numerical Atomic Orbitals (NAOs).

## Mathematical Formulation

We want to evaluate the matrix element:
$$
\mathbf{D}_{ab}(\mathbf{R}) = \langle \phi_a(\mathbf{r}) | \hat{\mathbf{r}} | \phi_b(\mathbf{r} - \mathbf{R}) \rangle
$$
where $\phi_a$ is centered at origin $\mathbf{0}$ and $\phi_b$ is centered at $\mathbf{R}$.

### 1. Decomposition of Coordinates
The position operator $\hat{\mathbf{r}}$ acts on the wavefunctions in the global coordinate system. However, the orbitals are defined relative to their centers.
Let $\mathbf{r}' = \mathbf{r} - \mathbf{R}$ be the coordinate relative to the second center. Then $\mathbf{r} = \mathbf{r}' + \mathbf{R}$.
Substitution yields:
$$
\langle \phi_a(\mathbf{r}) | \hat{\mathbf{r}} | \phi_b(\mathbf{r}') \rangle = \langle \phi_a(\mathbf{r}) | \mathbf{r}' + \mathbf{R} | \phi_b(\mathbf{r}') \rangle
$$
$$
\mathbf{D}_{ab}(\mathbf{R}) = \langle \phi_a(\mathbf{r}) | \mathbf{r}' | \phi_b(\mathbf{r}') \rangle + \mathbf{R} \langle \phi_a(\mathbf{r}) | \phi_b(\mathbf{r}') \rangle
$$
The second term is simply the overlap integral $S_{ab}(\mathbf{R})$ multiplied by the vector $\mathbf{R}$.
$$
\mathbf{D}_{ab}(\mathbf{R}) = \mathbf{D}'_{ab}(\mathbf{R}) + \mathbf{R} S_{ab}(\mathbf{R})
$$
where $\mathbf{D}'_{ab} = \langle \phi_a | \mathbf{r}' | \phi_b \rangle$ is the "local" dipole moment relative to center $b$.

### 2. Evaluation of Local Dipole Term
The term $\mathbf{r}' | \phi_b(\mathbf{r}') \rangle$ involves multiplying the orbital $\phi_b$ by its relative coordinate vector.
Recall that $\phi_b(\mathbf{r}') = \chi_b(r') Y_{l_b m_b}(\hat{\mathbf{r}}')$.
The vector $\mathbf{r}'$ can be expressed in terms of spherical harmonics with $L=1$:
$$
x = r \sqrt{\frac{2\pi}{3}} (Y_{1,-1} - Y_{1,1})
$$
$$
y = i r \sqrt{\frac{2\pi}{3}} (Y_{1,-1} + Y_{1,1})
$$
$$
z = r \sqrt{\frac{4\pi}{3}} Y_{1,0}
$$
The product valid for $L=1$ acting on orbital $l_b$:
$$
r Y_{1,m} \chi_b(r) Y_{l_b m_b}
$$
This results in a linear combination of new "effective" orbitals with angular momentum $l' = l_b \pm 1$.
$$
\mathbf{r}' \phi_b(\mathbf{r}') = \sum_{l' = l_b \pm 1} C(l_b, m_b; l') \chi'_b(r') Y_{l' m'}(\hat{\mathbf{r}}')
$$
where $\chi'_b(r') = r \chi_b(r)$.

### 3. Implementation Strategy
Instead of creating explicit new orbitals, `op2c` handles this via **Spherical Harmonic Recurrence Relations** directly in the integration routine.

Inside `TwoCenterIntegrator::calculate`:
1.  **Selection Rules**: The dipole operator couples $l_b$ only to $l_b \pm 1$.
2.  **Radial Tables**: We pre-compute radial integrals for these "shifted" angular momenta.
    *   `tablep_`: Corresponds to $l_b + 1$ (Plus table).
    *   `tablem_`: Corresponds to $l_b - 1$ (Minus table).
3.  **Gaunt Coefficients**: We sum over the contributions from $l_b \pm 1$ weighted by the appropriate Gaunt coefficients for the dipole transition.

#### Code Flow
*   `op2c` API calls `overlap_position`.
*   It computes the overlap $S$ using the standard table.
*   It calculates the term $\mathbf{R} S_{ab}$.
*   It invokes `TwoCenterIntegrator` with `tablep_` and `tablem_` to find the dipole components $(x, y, z)$ corresponding to the $\Delta l = \pm 1$ shifts.
*   Final result is summed and returned.

This avoids constructing explicit new basis functions in memory, keeping the memory footprint low while handling the operator exactly.
