#include "nao/orbital_evaluator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "math/linalg/vector3.h"
#include "math/ylm.h"

namespace {
// Index of the real spherical harmonic (l, m) in ModuleBase::Ylm::get_ylm_real
// output ordering: per l, [m=0, +1, -1, +2, -2, ...].
inline int ylm_index(const int l, const int m)
{
    const int base = l * l;
    if (m == 0) return base;
    if (m > 0) return base + 2 * m - 1;
    return base + 2 * (-m);
}
inline double m_phase(const int m) { return (m % 2 == 0) ? 1.0 : -1.0; }

// True iff radial @p r lies on the uniform knot set {r0 + i*dr : i = 0..nr-1} (within a
// relative tolerance). Checks knot POSITIONS against the model, not consecutive gaps, so
// tiny per-gap errors can't accumulate undetected. dr <= 0 (malformed/decreasing) -> false.
// The single source of truth for "is this radial on the common uniform grid?".
bool radial_on_uniform_grid(const NumericalRadial& r, double r0, double dr, int nr)
{
    if (r.nr() != nr || dr <= 0.0) return false;
    const double* g = r.rgrid();
    const double tol = dr * 1e-6;
    if (std::abs(g[0] - r0) > tol) return false;
    for (int i = 1; i < nr; ++i)
    {
        if (std::abs(g[i] - (r0 + i * dr)) > tol) return false;
    }
    return true;
}
}  // namespace

OrbitalEvaluator::OrbitalEvaluator(AtomicRadials radials)
    : radials_(std::move(radials))
{
    nphi_ = radials_.nphi();
    lmax_ = radials_.lmax();
    rcut_max_ = radials_.rcut_max();
    rcut_max2_ = rcut_max_ * rcut_max_;
    // Pass 1: collect the element's radials (in (l, izeta) order) and the per-m orbital
    // entries referencing them. Splines are built in pass 2, once the grid model is known.
    std::vector<const NumericalRadial*> rads;
    orbitals_.reserve(static_cast<std::size_t>(nphi_));
    for (int l = 0; l <= lmax_; ++l)
    {
        for (int izeta = 0; izeta < radials_.nzeta(l); ++izeta)
        {
            const NumericalRadial& radial = radials_.chi(l, izeta);
            if (radial.nr() < 2)
            {
                throw std::runtime_error("OrbitalEvaluator requires r-space radial grids");
            }
            const std::size_t radial_index = rads.size();
            rads.push_back(&radial);
            for (int m = -l; m <= l; ++m)
            {
                orbitals_.push_back({radial_index, l, m});
            }
        }
    }
    if (static_cast<int>(orbitals_.size()) != nphi_)
    {
        throw std::runtime_error("OrbitalEvaluator orbital count does not match nphi()");
    }

    // Class-level grid model, decided ONCE: do ALL radials share one uniform knot grid?
    // NAO .orb grids do (so the fast paths below + uniform_grid() switch on it); arbitrary
    // grids take the general binary-search path. Reference grid = the first radial's.
    if (!rads.empty())
    {
        const double* g0 = rads[0]->rgrid();
        grid_nr_ = rads[0]->nr();
        grid_r0_ = g0[0];
        grid_dr_ = (grid_nr_ > 1) ? g0[1] - g0[0] : 0.0;
        uniform_grid_ = true;
        for (const NumericalRadial* r : rads)
        {
            if (!radial_on_uniform_grid(*r, grid_r0_, grid_dr_, grid_nr_))
            {
                uniform_grid_ = false;
                break;
            }
        }
    }

    // Pass 2: build the per-radial spline. uniform_grid_ -> evenly-spaced-knot constructor so
    // CubicSpline::eval takes its O(1) index path (xmin + p*dr) instead of a binary search
    // over the ~800-point grid (the ABACUS-gint dr_uniform trick); else the general one.
    radial_evals_.reserve(rads.size());
    for (const NumericalRadial* r : rads)
    {
        auto spline = uniform_grid_
            ? std::make_unique<ModuleBase::CubicSpline>(grid_nr_, grid_r0_, grid_dr_, r->rvalue())
            : std::make_unique<ModuleBase::CubicSpline>(r->nr(), r->rgrid(), r->rvalue());
        radial_evals_.push_back({r, std::move(spline)});
    }

    // Optimization C + A: flat per-orbital tables. fill_row uses the fast Cartesian
    // Ylm::sph_harm, but it assigns the m<->(cos/sin) real-harmonic channels differently
    // from the get_ylm_real convention the operators (S/T/Vnl/DM) are built in. So we map
    // each orbital to the sph_harm (slot, sign) that REPRODUCES m_phase*get_ylm_real exactly,
    // matched at a generic direction (no two same-l harmonics coincide there). One-time.
    orb_radial_.resize(static_cast<std::size_t>(nphi_));
    orb_slot_.resize(static_cast<std::size_t>(nphi_));
    orb_sign_.resize(static_cast<std::size_t>(nphi_));
    {
        // Match each harmonic at SEVERAL generic directions so the (slot, sign) is unique
        // even when two same-l harmonics nearly coincide at one direction (robust to high l).
        static const double kDirs[][3] = {
            {0.3699, 0.5102, 0.7766}, {0.8112, -0.2731, 0.5168}, {-0.4459, 0.6231, 0.6428}};
        constexpr int kNdir = 3;
        const int nylm = (lmax_ + 1) * (lmax_ + 1);
        std::vector<std::vector<double>> gy(kNdir), sy(kNdir);
        for (int d = 0; d < kNdir; ++d)
        {
            gy[static_cast<std::size_t>(d)].resize(static_cast<std::size_t>(nylm));
            const double tx = kDirs[d][0], ty = kDirs[d][1], tz = kDirs[d][2];
            const double tn = std::sqrt(tx * tx + ty * ty + tz * tz);
            ModuleBase::Vector3<double> tv(tx, ty, tz);
            ModuleBase::Ylm::get_ylm_real(lmax_ + 1, tv, gy[static_cast<std::size_t>(d)].data());
            ModuleBase::Ylm::sph_harm(lmax_, tx / tn, ty / tn, tz / tn, sy[static_cast<std::size_t>(d)]);
        }
        for (int i = 0; i < nphi_; ++i)
        {
            const OrbitalEntry& o = orbitals_[static_cast<std::size_t>(i)];
            orb_radial_[static_cast<std::size_t>(i)] = static_cast<int>(o.radial_index);
            const int slot0 = ylm_index(o.l, o.m);
            int best = o.l * o.l;
            double best_sign = 1.0, best_err = 1e300;
            for (int s = o.l * o.l; s < (o.l + 1) * (o.l + 1); ++s)
            {
                for (double sgn : {1.0, -1.0})
                {
                    double err = 0.0;
                    for (int d = 0; d < kNdir; ++d)
                    {
                        const double ref = m_phase(o.m) * gy[static_cast<std::size_t>(d)][static_cast<std::size_t>(slot0)];
                        err += std::abs(sgn * sy[static_cast<std::size_t>(d)][static_cast<std::size_t>(s)] - ref);
                    }
                    if (err < best_err) { best_err = err; best = s; best_sign = sgn; }
                }
            }
            if (best_err > 1e-9)
            {
                throw std::runtime_error("OrbitalEvaluator: sph_harm/get_ylm_real mismatch beyond sign/order");
            }
            orb_slot_[static_cast<std::size_t>(i)] = best;
            orb_sign_[static_cast<std::size_t>(i)] = best_sign;
        }
    }

    // Optimization B (uniform_grid_ only): hold all radials in one shared CubicSpline so
    // multi_eval evaluates them at a point with the index/weights computed once. The grid
    // model (grid_nr_/grid_r0_/grid_dr_) is already validated above — no re-check here.
    if (uniform_grid_ && !radial_evals_.empty())
    {
        shared_spline_ = std::make_unique<ModuleBase::CubicSpline>(grid_nr_, grid_r0_, grid_dr_);
        shared_spline_->reserve(static_cast<int>(radial_evals_.size()));
        for (const RadialEval& re : radial_evals_)
        {
            shared_spline_->add(re.radial->rvalue());
        }
        shared_rmax_ = grid_r0_ + (grid_nr_ - 1) * grid_dr_;
    }
}

void OrbitalEvaluator::active_orbitals_for_box(double cx, double cy, double cz,
                                               double box_radius_bohr,
                                               std::vector<int>& active) const
{
    active.clear();
    const double center_distance = std::sqrt(cx * cx + cy * cy + cz * cz);
    for (int iorb = 0; iorb < nphi_; ++iorb)
    {
        const OrbitalEntry& orbital = orbitals_[static_cast<std::size_t>(iorb)];
        const NumericalRadial& radial = *radial_evals_[orbital.radial_index].radial;
        if (center_distance <= radial.rcut() + box_radius_bohr)
        {
            active.push_back(iorb);
        }
    }
}

void OrbitalEvaluator::fill_row(double x, double y, double z, double r,
                                const int* active, int n_active, double* row,
                                EvalScratch& scratch) const
{
    // (A) Cartesian real spherical harmonics: no atan / pow / Legendre / redundant norm.
    // r==0 -> unit vector (0,0,0): sph_harm gives l=0 only, l>0 -> 0 (and R(0)=0 there).
    const double inv_r = (r > 1e-12) ? 1.0 / r : 0.0;
    ModuleBase::Ylm::sph_harm(lmax_, x * inv_r, y * inv_r, z * inv_r, scratch.ylm);  // resizes

    // (B) all radials at r in one pass — index/weights computed once and shared.
    // scratch.rad_val is pre-sized to radial_evals_.size() by the batch caller.
    const std::size_t n_rad = radial_evals_.size();
    double* rad_val = scratch.rad_val.data();
    if (uniform_grid_)
    {
        if (r <= shared_rmax_) { shared_spline_->multi_eval(r, rad_val); }
        else { std::fill(rad_val, rad_val + n_rad, 0.0); }  // past rmax -> 0
    }
    else
    {
        for (std::size_t ri = 0; ri < n_rad; ++ri)
        {
            const NumericalRadial& radial = *radial_evals_[ri].radial;
            double rv = 0.0;
            if (r >= radial.rgrid(0) && r <= radial.rmax())
            {
                radial_evals_[ri].spline->eval(1, &r, &rv);
            }
            rad_val[ri] = rv;
        }
    }

    // (C) flat per-orbital combine: no struct lookups / index math in the hot loop.
    const double* ylm = scratch.ylm.data();
    for (int a = 0; a < n_active; ++a)
    {
        const std::size_t iorb = static_cast<std::size_t>(active[a]);
        row[iorb] = orb_sign_[iorb]
                    * rad_val[static_cast<std::size_t>(orb_radial_[iorb])]
                    * ylm[static_cast<std::size_t>(orb_slot_[iorb])];
    }
}

bool OrbitalEvaluator::evaluate_point(double x, double y, double z,
                                      const int* active, int n_active,
                                      double* values, EvalScratch& scratch) const
{
    const double r = std::sqrt(x * x + y * y + z * z);
    fill_row(x, y, z, r, active, n_active, values, scratch);
    for (int a = 0; a < n_active; ++a)
    {
        if (values[static_cast<std::size_t>(active[a])] != 0.0) return true;
    }
    return false;
}

void OrbitalEvaluator::evaluate_active_batch(int npoint, const double* xyz, int out_stride,
                                             const std::vector<int>& active, double* out) const
{
    if (active.empty())
    {
        return;
    }
    const int n_active = static_cast<int>(active.size());
    const int* act = active.data();
    EvalScratch scratch;                              // one scratch per batch sweep
    scratch.rad_val.resize(radial_evals_.size());     // sized once, reused per point
    for (int ip = 0; ip < npoint; ++ip)
    {
        const double x = xyz[3 * ip];
        const double y = xyz[3 * ip + 1];
        const double z = xyz[3 * ip + 2];
        const double r2 = x * x + y * y + z * z;
        if (r2 > rcut_max2_)
        {
            continue;  // grid-projector convention: skip past the largest cutoff (no sqrt)
        }
        const double r = std::sqrt(r2);
        fill_row(x, y, z, r, act, n_active,
                 out + static_cast<std::size_t>(ip) * static_cast<std::size_t>(out_stride),
                 scratch);
    }
}

void OrbitalEvaluator::evaluate_all(int npoint, const double* xyz, double* out) const
{
    std::vector<int> all(static_cast<std::size_t>(nphi_));
    std::iota(all.begin(), all.end(), 0);
    EvalScratch scratch;
    scratch.rad_val.resize(radial_evals_.size());
    for (int ip = 0; ip < npoint; ++ip)
    {
        double* row = out + static_cast<std::size_t>(ip) * nphi_;
        std::fill(row, row + nphi_, 0.0);
        evaluate_point(xyz[3 * ip + 0], xyz[3 * ip + 1], xyz[3 * ip + 2], all.data(), nphi_, row,
                       scratch);
    }
}

bool OrbitalEvaluator::evaluate_point_grad(double x, double y, double z,
                                           const int* active, int n_active,
                                           double* values, double* grad) const
{
    const double r = std::sqrt(x * x + y * y + z * z);
    // ∇φ at (or essentially at) an orbital centre: the analytic 1/r form and the
    // Ylm gradient at the zero vector are singular there. r=0 occurs whenever an
    // atom sits exactly on a grid point (symmetric crystals on a commensurate grid),
    // so fall back to a central finite difference of φ — the correct limit for all l
    // and consistent with how the value kernel treats the centre.
    constexpr double kCenterEps = 1.0e-6;
    if (r < kCenterEps)
    {
        // Cold path (grid point on an atom centre); a local scratch is fine here.
        thread_local EvalScratch scratch;
        scratch.rad_val.resize(radial_evals_.size());
        evaluate_point(x, y, z, active, n_active, values, scratch);
        const double delta = 1.0e-3;
        // thread_local scratch: this central-difference branch fires once per
        // grid point that coincides with an atom centre (cold force path), but
        // reuse the buffers rather than heap-allocate two nphi vectors each hit.
        thread_local std::vector<double> vp;
        thread_local std::vector<double> vm;
        vp.assign(static_cast<std::size_t>(nphi_), 0.0);
        vm.assign(static_cast<std::size_t>(nphi_), 0.0);
        for (int d = 0; d < 3; ++d)
        {
            double xp[3] = {x, y, z};
            double xm[3] = {x, y, z};
            xp[d] += delta;
            xm[d] -= delta;
            std::fill(vp.begin(), vp.end(), 0.0);
            std::fill(vm.begin(), vm.end(), 0.0);
            evaluate_point(xp[0], xp[1], xp[2], active, n_active, vp.data(), scratch);
            evaluate_point(xm[0], xm[1], xm[2], active, n_active, vm.data(), scratch);
            for (int a = 0; a < n_active; ++a)
            {
                const int iorb = active[a];
                grad[static_cast<std::size_t>(iorb) * 3 + d] =
                    (vp[static_cast<std::size_t>(iorb)] - vm[static_cast<std::size_t>(iorb)])
                    / (2.0 * delta);
            }
        }
        bool nz = false;
        for (int a = 0; a < n_active; ++a)
        {
            nz = nz || (std::abs(values[static_cast<std::size_t>(active[a])]) > 0.0);
        }
        return nz;
    }

    const int nlm = (lmax_ + 1) * (lmax_ + 1);
    // Hoisted per-thread scratch: this runs inside the OMP box loop of the ∇φ force
    // kernel, so reuse the buffers across the whole sweep instead of heap-allocating
    // ylm/dylm (and the radial value/deriv tables) at every grid point.
    thread_local std::vector<double> ylm;
    thread_local std::vector<double> dylm;   // dylm[lm][a] = dY_lm/dx_a, flat (nlm, 3)
    thread_local std::vector<double> rad_val;
    thread_local std::vector<double> rad_deriv;
    ylm.assign(static_cast<std::size_t>(nlm), 0.0);
    dylm.assign(static_cast<std::size_t>(nlm) * 3, 0.0);
    ModuleBase::Vector3<double> vec(x, y, z);
    ModuleBase::Ylm::get_ylm_real(lmax_ + 1, vec, ylm.data(),
                                  reinterpret_cast<double(*)[3]>(dylm.data()));

    const double inv_r = 1.0 / r;   // r >= kCenterEps on this path
    const double coord[3] = {x, y, z};

    // (B) All radials' values AND first derivatives in ONE shared multi_eval (uniform
    // grid) — the gradient sibling of fill_row's optimization. Replaces the per-orbital
    // spline->eval below, which redundantly re-evaluated each radial once per m-degenerate
    // copy (nphi evals -> 1 multi_eval; e.g. 13 -> 1 for a 2s2p1d Si set).
    const bool use_shared = uniform_grid_ && shared_spline_;
    if (use_shared)
    {
        const std::size_t n_rad = radial_evals_.size();
        rad_val.assign(n_rad, 0.0);
        rad_deriv.assign(n_rad, 0.0);
        if (r <= shared_rmax_)
        {
            shared_spline_->multi_eval(r, rad_val.data(), rad_deriv.data());
        }
    }

    bool has_nonzero = false;
    for (int a = 0; a < n_active; ++a)
    {
        const int iorb = active[a];
        const OrbitalEntry& orbital = orbitals_[static_cast<std::size_t>(iorb)];
        double radial_value;
        double radial_deriv;
        if (use_shared)
        {
            const std::size_t ri =
                static_cast<std::size_t>(orb_radial_[static_cast<std::size_t>(iorb)]);
            radial_value = rad_val[ri];     // 0 past the radial's rcut (shared spline zero-pad)
            radial_deriv = rad_deriv[ri];
        }
        else
        {
            const RadialEval& radial_eval = radial_evals_[orbital.radial_index];
            const NumericalRadial& radial = *radial_eval.radial;
            if (r < radial.rgrid(0) || r > radial.rmax())
            {
                continue;
            }
            radial_value = 0.0;
            radial_deriv = 0.0;
            radial_eval.spline->eval(1, &r, &radial_value, &radial_deriv);
        }
        const std::size_t idx = static_cast<std::size_t>(ylm_index(orbital.l, orbital.m));
        const double c = m_phase(orbital.m);
        const double Y = ylm[idx];
        values[static_cast<std::size_t>(iorb)] = c * radial_value * Y;
        // ∇φ = c [ dR/dr (x_a/r) Y + R dY/dx_a ]
        for (int d = 0; d < 3; ++d)
        {
            grad[static_cast<std::size_t>(iorb) * 3 + d] =
                c * (radial_deriv * coord[d] * inv_r * Y + radial_value * dylm[idx * 3 + d]);
        }
        has_nonzero = has_nonzero || (std::abs(values[static_cast<std::size_t>(iorb)]) > 0.0);
    }
    return has_nonzero;
}

bool OrbitalEvaluator::evaluate_active_grad(double x, double y, double z,
                                            const std::vector<int>& active,
                                            std::vector<double>& values,
                                            std::vector<double>& grad) const
{
    values.assign(static_cast<std::size_t>(nphi_), 0.0);
    grad.assign(static_cast<std::size_t>(nphi_) * 3, 0.0);
    if (active.empty())
    {
        return false;
    }
    // Grid-projector convention: a point past the largest cutoff touches no orbital.
    if (std::sqrt(x * x + y * y + z * z) > rcut_max_)
    {
        return false;
    }
    return evaluate_point_grad(x, y, z, active.data(), static_cast<int>(active.size()),
                               values.data(), grad.data());
}

void OrbitalEvaluator::evaluate_all_grad(int npoint, const double* xyz,
                                         double* values, double* grad) const
{
    std::vector<int> all(static_cast<std::size_t>(nphi_));
    std::iota(all.begin(), all.end(), 0);
    for (int ip = 0; ip < npoint; ++ip)
    {
        double* vrow = values + static_cast<std::size_t>(ip) * nphi_;
        double* grow = grad + static_cast<std::size_t>(ip) * nphi_ * 3;
        std::fill(vrow, vrow + nphi_, 0.0);
        std::fill(grow, grow + static_cast<std::size_t>(nphi_) * 3, 0.0);
        evaluate_point_grad(xyz[3 * ip + 0], xyz[3 * ip + 1], xyz[3 * ip + 2],
                            all.data(), nphi_, vrow, grow);
    }
}
