#ifndef MATH_ATOMIC_GRID_H
#define MATH_ATOMIC_GRID_H

#include "math/linalg/vector3.h"
#include <vector>

namespace ModuleBase
{

/**
 * @brief Treutler-Ahlrichs M4 radial quadrature (Gauss-Chebyshev 2nd kind base
 *        convolved with the TA M4 transformation). J. Chem. Phys. 102, 346 (1995).
 *
 * The base rule is the ``npts``-point Gauss-Chebyshev quadrature of the second
 * kind on (-1, 1), scaled to a unit weight factor (nodes x_i = cos(i*pi/(n+1)),
 * weights w_i = pi/(n+1) * sin(i*pi/(n+1))). The TA M4 map (a = 1, alpha = 0.6)
 *     r(x) = R * (1 + x)^alpha * ln(2 / (1 - x)) / ln 2
 * sends (-1, 1) -> (0, inf); the returned weight already carries the transform
 * Jacobian dr/dx (but NOT the r^2 spherical Jacobian -- that belongs to the
 * spherical composition in AtomicGrid).
 *
 * @param[in]  npts  number of radial nodes
 * @param[in]  R     radial scaling factor (Bohr) -- sets the node density; for a
 *                   confined orbital take R ~ rcut / 2 so nodes cluster over support
 * @param[out] r     radial nodes r_i (Bohr), ascending in i is NOT guaranteed
 * @param[out] w     radial weights w_i = w_gc2_i * (dr/dx)_i
 * @param[in]  alpha TA exponential factor (0.6 = M4, 0.0 = M3)
 */
void treutler_ahlrichs_radial(int npts, double R,
                              std::vector<double>& r, std::vector<double>& w,
                              double alpha = 0.6);

/**
 * @brief Atom-centered spherical quadrature grid: a radial (Treutler-Ahlrichs M4)
 *        quadrature x an angular (Lebedev-Laikov) quadrature, rigidly centered at
 *        ``center``. The grid moves with its center -> integration of an
 *        atom-localized function is translationally covariant (egg-box-free).
 *
 * Produces ``n_radial * degree`` points p_k = center + r_i * omega_j with VOLUME
 * weights w_k = w_r_i * r_i^2 * w_ang_j, so that
 *     sum_k w_k f(p_k)  ~=  integral f(r) d^3r
 * for a function localized near ``center``. (The Lebedev-Laikov weights already
 * carry integral dOmega, i.e. they sum to 4*pi.)
 *
 * @param center   grid center (Bohr)
 * @param n_radial number of radial nodes (TA M4)
 * @param R        radial scaling factor (Bohr), see treutler_ahlrichs_radial
 * @param degree   Lebedev-Laikov order (allowed: 6,14,26,...,5810)
 * @param alpha    TA exponential factor (default 0.6 = M4)
 */
class AtomicGrid
{
public:
    AtomicGrid(const Vector3<double>& center, int n_radial, double R, int degree,
               double alpha = 0.6);

    /*!
     * @brief PRUNED atom grid: a per-radial-shell Lebedev order (angular pruning).
     *
     * Same TA-M4 radial nodes as the uniform ctor, but shell ``i`` carries a Lebedev
     * order ``shell_degree[i]`` instead of one global ``degree``. Lets the caller use a
     * coarse angular grid where the integrand is smooth/small (inner shells: low angular
     * content of the pseudo-NAO products near r->0; outer shells: the weight V_na(r)->0)
     * and a full order only in the valence band — ~2-4x fewer points at fixed accuracy
     * (doc 50 §4.1). ``shell_degree.size()`` must equal ``n_radial``; each entry must be an
     * allowed Lebedev order. Points are laid out radial-outer, angular-inner.
     */
    AtomicGrid(const Vector3<double>& center, int n_radial, double R,
               const std::vector<int>& shell_degree, double alpha = 0.6);

    int npts() const { return static_cast<int>(weights_.size()); }
    const std::vector<Vector3<double>>& points() const { return points_; }
    const std::vector<double>& weights() const { return weights_; }
    const Vector3<double>& center() const { return center_; }

    /// Rigidly translate the grid to a new center (points shift, weights unchanged).
    void recenter(const Vector3<double>& new_center);

private:
    Vector3<double> center_;
    std::vector<Vector3<double>> points_;
    std::vector<double> weights_;
};

/**
 * @brief Treutler-style angular-pruning schedule for the pruned @ref AtomicGrid ctor.
 *
 * Returns a per-shell Lebedev order for ``n_radial`` TA shells: the inner 25% (near r=0) get
 * ``inner_degree``, the valence band (25%-70%) ``valence_degree``, and the outer 30% (near rcut,
 * where the short-ranged weight decays to 0) ``outer_degree``. Keep ``valence_degree`` at the full
 * order needed for the off-centre orbital content; coarsen only the inner/outer bands (doc 50 §4.1).
 */
std::vector<int> treutler_prune_schedule(int n_radial, int inner_degree,
                                         int valence_degree, int outer_degree);

/**
 * @brief SSF (Stratmann-Scuseria-Frisch) nuclear partition of unity at a point.
 *
 * For a point r and atom centers {R_A}, returns the normalized partition weights
 * ``p[A]`` with sum_A p[A] = 1. The SSF cell function is
 *     P_A(r) = prod_{B != A} s(mu_AB),   mu_AB = (|r-R_A| - |r-R_B|) / |R_A-R_B|,
 * with the finite-support switch s(mu) = 1 for mu <= -a, 0 for mu >= a, and
 * 0.5*(1 - g(mu)) otherwise, g(s)=(35(s-s^3)+21 s^5-5 s^7)/16, s=mu/a, a=0.64
 * (Stratmann, Scuseria, Frisch, Chem. Phys. Lett. 257, 213 (1996)). p[A] = P_A / sum_C P_C.
 *
 * Because s has compact support in mu (|mu|<a), a point well inside one atom's cell gets
 * p=1 there and 0 elsewhere WITHOUT evaluating distant atoms -> strictly O(N) with the
 * nearest-atom screen (see ssf_parent_weight). This O(N^2)-per-point reference form is for
 * validation and small systems; it matches GauXC's reference_ssf_weights_host pairwise branch.
 */
void ssf_partition_weights(const Vector3<double>& point,
                           const std::vector<Vector3<double>>& centers,
                           std::vector<double>& p);

/**
 * @brief SSF partition weight of ONE atom (``parent``) at a point, with the nearest-atom
 *        screen. ``dist_nearest`` is the parent's nearest-neighbour distance
 *        min_{B != parent} |R_parent - R_B|. If the point lies within
 *        ``0.5*(1-a)*dist_nearest`` of the parent it is deep inside the parent's cell and the
 *        weight is exactly 1 (no pair loop). This is the production form (GauXC's screened
 *        SSF): the grid weight w_k for a point of atom ``parent`` is multiplied by this.
 */
double ssf_parent_weight(const Vector3<double>& point,
                         const std::vector<Vector3<double>>& centers,
                         int parent, double dist_nearest);

} // namespace ModuleBase

#endif // MATH_ATOMIC_GRID_H
