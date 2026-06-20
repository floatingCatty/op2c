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
     * @brief Evaluate the @p active orbitals at a BATCH of relative points (Bohr).
     *
     * Atom-batched hot path (ABACUS ``GintAtom::set_phi``): one call per atom-image
     * sweeps all @p npoint box points, so per-atom setup is amortized and Φ is written
     * straight into its column block — no per-point zero-fill or scratch copy. @p xyz is
     * (npoint, 3) row-major relative coords; for each point the active orbitals are
     * written to ``out[ip*out_stride + iorb]``. The caller pre-zeros @p out; points
     * beyond ``rcut_max()`` and inactive/out-of-range orbitals are left untouched.
     * Numerically identical to looping :func:`evaluate_active` over the points.
     */
    void evaluate_active_batch(int npoint, const double* xyz, int out_stride,
                               const std::vector<int>& active, double* out) const;

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

    // Per-thread scratch for the point loop, hoisted out of fill_row so a batch sweep
    // pays the buffer setup once instead of per point (no per-point thread_local access).
    struct EvalScratch
    {
        std::vector<double> ylm;      // (lmax+1)^2 real spherical harmonics at the point
        std::vector<double> rad_val;  // one radial value per distinct radial (radial_evals_)
    };

    // Core per-point evaluation shared by evaluate_active / evaluate_all.
    // @p values must be pre-zeroed by the caller; only listed orbitals in range
    // are written. Returns true if any written value is non-zero.
    bool evaluate_point(double x, double y, double z,
                        const int* active, int n_active,
                        double* values, EvalScratch& scratch) const;

    // Fill one point's active-orbital values into @p row (pre-zeroed by caller).
    // @p r = |(x,y,z)|. The hot kernel: (A) Cartesian real spherical harmonics via
    // Ylm::sph_harm (no atan/pow/Legendre); (B) all radials in one CubicSpline::multi_eval
    // (index/weights computed once, shared across radials); (C) flat per-orbital
    // {radial, ylm-slot, sign} tables (no per-point struct lookups). @p scratch.rad_val
    // must be pre-sized to radial_evals_.size() by the caller.
    void fill_row(double x, double y, double z, double r,
                  const int* active, int n_active, double* row,
                  EvalScratch& scratch) const;

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
    double rcut_max2_ = 0.0;  // rcut_max_^2, for the squared-distance cutoff (skip sqrt)
    std::vector<RadialEval> radial_evals_;
    std::vector<OrbitalEntry> orbitals_;

    // Optimization B: all radials in ONE shared-knot CubicSpline (multi_eval evaluates
    // them at a point with the index/weights computed once). Built only when every radial
    // shares the same uniform knots; else shared_radials_ stays false and fill_row uses
    // the per-radial radial_evals_ splines.
    std::unique_ptr<ModuleBase::CubicSpline> shared_spline_;
    bool shared_radials_ = false;
    double shared_rmax_ = 0.0;

    // Optimization C: flat per-orbital tables (size nphi_) so the per-point loop is
    // row[iorb] = orb_sign_[iorb] * rad_val[orb_radial_[iorb]] * ylm[orb_slot_[iorb]].
    std::vector<int> orb_radial_;   // radial (spline-interpolant) index
    std::vector<int> orb_slot_;     // ylm array slot = ylm_index(l, m)
    std::vector<double> orb_sign_;  // sign factor for the sph_harm convention
};
