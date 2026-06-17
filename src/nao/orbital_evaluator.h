#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "math/interpolation/cubic_spline.h"
#include "nao/atomic_radials.h"
#include "nao/radial_function.h"

/*!
 * @brief Cached evaluator of an element's numerical atomic orbitals on points.
 *
 * The single C++ home for "evaluate phi_{l,izeta,m}(r) = R(|r|) * Y_lm(r_hat)"
 * on real-space points. It owns a copy of the element's @ref AtomicRadials and
 * builds one cubic spline per radial **once** in the constructor, so repeated
 * point evaluation (the SCF grid projector hot path) reuses them. The angular
 * part is `ModuleBase::Ylm::get_ylm_real`. Both the `_op2c` Python binding
 * (`AtomicRadials.evaluate_orbitals`) and rescu++'s `_core` grid projector use
 * this class — there is no second copy of the orbital-on-grid wheel.
 *
 * Column / orbital order is the natural (l, izeta, m=-l..l), matching nphi().
 */
class OrbitalEvaluator
{
  public:
    explicit OrbitalEvaluator(AtomicRadials radials);

    int nphi() const { return nphi_; }
    int lmax() const { return lmax_; }
    double rcut_max() const { return rcut_max_; }

    /*!
     * @brief Orbitals whose radial cutoff can reach a box.
     *
     * Appends to @p active the indices of orbitals with `|center| <= rcut +
     * box_radius` (the centre is relative to the atom, Bohr). Used to skip
     * distant orbitals when integrating over a grid box.
     */
    void active_orbitals_for_box(double cx, double cy, double cz,
                                 double box_radius_bohr,
                                 std::vector<int>& active) const;

    /*!
     * @brief Evaluate the @p active orbitals at one relative point (Bohr).
     *
     * Fills @p values (length nphi(), inactive/out-of-range entries set to 0);
     * returns true if any value is non-zero. Early-outs when the point is beyond
     * rcut_max() (the grid-projector convention).
     */
    bool evaluate_active(double x, double y, double z,
                         const std::vector<int>& active,
                         std::vector<double>& values) const;

    /*!
     * @brief Evaluate **all** orbitals at @p npoint points (row-major xyz).
     *
     * Writes @p out as (npoint, nphi()) row-major. Uses only the per-orbital
     * [rgrid(0), rmax()] support test (no rcut_max early-out), so this matches a
     * plain "evaluate every orbital at every point" reference.
     */
    void evaluate_all(int npoint, const double* xyz, double* out) const;

    /*!
     * @brief Evaluate the @p active orbitals AND their Cartesian gradients at one
     * relative point (Bohr).
     *
     * Fills @p values (length nphi(), inactive entries 0) and @p grad (length
     * nphi()*3, row-major: `grad[3*iorb + a] = d phi_iorb / d x_a`, inactive 0).
     * ∇φ = (dR/dr)(r_hat) Y_lm + R ∇Y_lm — the radial spline derivative composed
     * with the Ylm gradient (``ModuleBase::Ylm::get_ylm_real`` value+grad). Same
     * rcut_max early-out as :func:`evaluate_active`. Foundation for forces/stress
     * (Pulay grid term) and meta-GGA τ (doc 32 §5 gap 5).
     */
    bool evaluate_active_grad(double x, double y, double z,
                              const std::vector<int>& active,
                              std::vector<double>& values,
                              std::vector<double>& grad) const;

    /*!
     * @brief Evaluate **all** orbitals + gradients at @p npoint points.
     *
     * @p values is (npoint, nphi()) and @p grad is (npoint, nphi(), 3), both
     * row-major. Reference path for the gradient (matches :func:`evaluate_all`).
     */
    void evaluate_all_grad(int npoint, const double* xyz,
                           double* values, double* grad) const;

  private:
    struct RadialEval
    {
        const NumericalRadial* radial = nullptr;
        std::unique_ptr<ModuleBase::CubicSpline> spline;
    };
    struct OrbitalEntry
    {
        std::size_t radial_index = 0;
        int l = 0;
        int m = 0;
    };

    // Core per-point evaluation shared by evaluate_active / evaluate_all.
    // @p values must be pre-zeroed by the caller; only listed orbitals in range
    // are written. Returns true if any written value is non-zero.
    bool evaluate_point(double x, double y, double z,
                        const int* active, int n_active,
                        double* values) const;

    // Core per-point value+gradient evaluation shared by evaluate_active_grad /
    // evaluate_all_grad. @p values (len nphi) and @p grad (len nphi*3) must be
    // pre-zeroed by the caller; only listed in-range orbitals are written.
    bool evaluate_point_grad(double x, double y, double z,
                             const int* active, int n_active,
                             double* values, double* grad) const;

    AtomicRadials radials_;
    int nphi_ = 0;
    int lmax_ = 0;
    double rcut_max_ = 0.0;
    std::vector<RadialEval> radial_evals_;
    std::vector<OrbitalEntry> orbitals_;
};
