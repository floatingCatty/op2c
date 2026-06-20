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
}  // namespace

OrbitalEvaluator::OrbitalEvaluator(AtomicRadials radials)
    : radials_(std::move(radials))
{
    nphi_ = radials_.nphi();
    lmax_ = radials_.lmax();
    rcut_max_ = radials_.rcut_max();
    rcut_max2_ = rcut_max_ * rcut_max_;
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
            const std::size_t radial_index = radial_evals_.size();
            // Build the radial spline. If the r-grid is uniform (NAO .orb grids are,
            // dr const), use the evenly-spaced-knot constructor so CubicSpline::eval takes
            // its O(1) index path (xmin + p·dr) instead of a per-eval binary search over the
            // ~800-point grid — the ABACUS-gint ``dr_uniform`` trick. Fall back to the
            // general (binary-search) constructor for a non-uniform grid.
            const int nr = radial.nr();
            const double* rg = radial.rgrid();
            const double r0 = rg[0];
            const double dr = rg[1] - rg[0];   // candidate spacing (verified below)
            // Use the uniform-knot constructor ONLY if the grid really is evenly spaced:
            // the fast constructor places knots at r0 + i*dr, so check that EVERY real knot
            // rg[i] matches that model (checking knot positions, not consecutive gaps, so
            // tiny per-gap errors can't accumulate undetected). dr > 0 is required by the
            // constructor and rules out a malformed/decreasing grid.
            bool uniform = (dr > 0.0);
            if (uniform)
            {
                const double tol = dr * 1e-6;
                for (int i = 2; i < nr; ++i)   // i=1 matches r0+dr by definition of dr
                {
                    if (std::abs(rg[i] - (r0 + i * dr)) > tol)
                    {
                        uniform = false;
                        break;
                    }
                }
            }
            auto spline = uniform
                ? std::make_unique<ModuleBase::CubicSpline>(nr, r0, dr, radial.rvalue())  // O(1) eval
                : std::make_unique<ModuleBase::CubicSpline>(nr, rg, radial.rvalue());      // general (binary search)
            radial_evals_.push_back({&radial, std::move(spline)});
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

    // Optimization B: if every radial shares the same uniform knots, hold them all in one
    // CubicSpline so multi_eval evaluates them with the index/weights computed once.
    if (!radial_evals_.empty())
    {
        const NumericalRadial& r0r = *radial_evals_[0].radial;
        const int nr0 = r0r.nr();
        const double* g0 = r0r.rgrid();
        const double dr0 = (nr0 > 1) ? g0[1] - g0[0] : 0.0;
        bool share = dr0 > 0.0;
        const double tol = dr0 * 1e-6;
        for (std::size_t ri = 0; ri < radial_evals_.size() && share; ++ri)
        {
            const NumericalRadial& rr = *radial_evals_[ri].radial;
            const double* gi = rr.rgrid();
            if (rr.nr() != nr0 || std::abs(gi[0] - g0[0]) > tol) { share = false; break; }
            for (int k = 1; k < nr0; ++k)
            {
                if (std::abs(gi[k] - (g0[0] + k * dr0)) > tol) { share = false; break; }
            }
        }
        if (share)
        {
            shared_spline_ = std::make_unique<ModuleBase::CubicSpline>(nr0, g0[0], dr0);
            shared_spline_->reserve(static_cast<int>(radial_evals_.size()));
            for (std::size_t ri = 0; ri < radial_evals_.size(); ++ri)
            {
                shared_spline_->add(radial_evals_[ri].radial->rvalue());
            }
            shared_radials_ = true;
            shared_rmax_ = g0[0] + (nr0 - 1) * dr0;
        }
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
    if (shared_radials_)
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
    std::vector<double> ylm(static_cast<std::size_t>(nlm));
    // dylm[lm][a] = d Y_lm(r_hat) / d x_a (flat (nlm, 3), passed as double[][3]).
    std::vector<double> dylm(static_cast<std::size_t>(nlm) * 3);
    ModuleBase::Vector3<double> vec(x, y, z);
    ModuleBase::Ylm::get_ylm_real(lmax_ + 1, vec, ylm.data(),
                                  reinterpret_cast<double(*)[3]>(dylm.data()));

    const double inv_r = (r > 0.0) ? 1.0 / r : 0.0;
    const double coord[3] = {x, y, z};
    bool has_nonzero = false;
    for (int a = 0; a < n_active; ++a)
    {
        const int iorb = active[a];
        const OrbitalEntry& orbital = orbitals_[static_cast<std::size_t>(iorb)];
        const RadialEval& radial_eval = radial_evals_[orbital.radial_index];
        const NumericalRadial& radial = *radial_eval.radial;
        if (r < radial.rgrid(0) || r > radial.rmax())
        {
            continue;
        }
        double radial_value = 0.0;
        double radial_deriv = 0.0;
        radial_eval.spline->eval(1, &r, &radial_value, &radial_deriv);
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
