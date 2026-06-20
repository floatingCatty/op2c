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

bool OrbitalEvaluator::evaluate_point(double x, double y, double z,
                                      const int* active, int n_active,
                                      double* values) const
{
    const double r = std::sqrt(x * x + y * y + z * z);
    // Reused per-thread Ylm buffer: avoids a heap allocation on EVERY (atom, point)
    // call (build_box_phi calls this np * n_images times per box -> millions per grid).
    thread_local std::vector<double> ylm;
    ylm.resize(static_cast<std::size_t>((lmax_ + 1) * (lmax_ + 1)));
    ModuleBase::Vector3<double> vec(x, y, z);
    ModuleBase::Ylm::get_ylm_real(lmax_ + 1, vec, ylm.data());

    // Evaluate each DISTINCT radial spline ONCE per point: the 2l+1 m-channels of a
    // shell share one radial R_nl(r), so caching R by radial_index cuts spline evals
    // from n_active down to the few distinct radials (13 -> 5 for Si). Byte-identical.
    const std::size_t n_rad = radial_evals_.size();
    thread_local std::vector<double> rad_val;
    thread_local std::vector<char> rad_done;
    rad_val.resize(n_rad);
    rad_done.assign(n_rad, 0);

    bool has_nonzero = false;
    for (int a = 0; a < n_active; ++a)
    {
        const int iorb = active[a];
        const OrbitalEntry& orbital = orbitals_[static_cast<std::size_t>(iorb)];
        const std::size_t ri = orbital.radial_index;
        if (!rad_done[ri])
        {
            const RadialEval& radial_eval = radial_evals_[ri];
            const NumericalRadial& radial = *radial_eval.radial;
            double rv = 0.0;
            if (r >= radial.rgrid(0) && r <= radial.rmax())
            {
                radial_eval.spline->eval(1, &r, &rv);
            }
            rad_val[ri] = rv;
            rad_done[ri] = 1;
        }
        const double radial_value = rad_val[ri];
        if (radial_value == 0.0)
        {
            continue;  // out-of-range / zero radial: values[iorb] stays pre-zeroed (as before)
        }
        const double value =
            m_phase(orbital.m) * radial_value * ylm[static_cast<std::size_t>(ylm_index(orbital.l, orbital.m))];
        values[static_cast<std::size_t>(iorb)] = value;
        has_nonzero = has_nonzero || (std::abs(value) > 0.0);
    }
    return has_nonzero;
}

bool OrbitalEvaluator::evaluate_active(double x, double y, double z,
                                       const std::vector<int>& active,
                                       std::vector<double>& values) const
{
    values.assign(static_cast<std::size_t>(nphi_), 0.0);
    if (active.empty())
    {
        return false;
    }
    // Grid-projector convention: a point past the largest cutoff touches no orbital.
    if (std::sqrt(x * x + y * y + z * z) > rcut_max_)
    {
        return false;
    }
    return evaluate_point(x, y, z, active.data(), static_cast<int>(active.size()), values.data());
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
    const std::size_t n_rad = radial_evals_.size();
    // Per-thread scratch reused across points AND across calls (no per-point alloc).
    thread_local std::vector<double> ylm;
    thread_local std::vector<double> rad_val;
    thread_local std::vector<char> rad_done;
    ylm.resize(static_cast<std::size_t>((lmax_ + 1) * (lmax_ + 1)));
    rad_val.resize(n_rad);

    for (int ip = 0; ip < npoint; ++ip)
    {
        const double x = xyz[3 * ip];
        const double y = xyz[3 * ip + 1];
        const double z = xyz[3 * ip + 2];
        const double r = std::sqrt(x * x + y * y + z * z);
        if (r > rcut_max_)
        {
            continue;  // grid-projector convention: point past the largest cutoff
        }
        double* row = out + static_cast<std::size_t>(ip) * static_cast<std::size_t>(out_stride);
        ModuleBase::Vector3<double> vec(x, y, z);
        ModuleBase::Ylm::get_ylm_real(lmax_ + 1, vec, ylm.data());
        rad_done.assign(n_rad, 0);   // distinct radial evaluated once per point
        for (int a = 0; a < n_active; ++a)
        {
            const int iorb = act[a];
            const OrbitalEntry& orbital = orbitals_[static_cast<std::size_t>(iorb)];
            const std::size_t ri = orbital.radial_index;
            if (!rad_done[ri])
            {
                const RadialEval& radial_eval = radial_evals_[ri];
                const NumericalRadial& radial = *radial_eval.radial;
                double rv = 0.0;
                if (r >= radial.rgrid(0) && r <= radial.rmax())
                {
                    radial_eval.spline->eval(1, &r, &rv);
                }
                rad_val[ri] = rv;
                rad_done[ri] = 1;
            }
            const double radial_value = rad_val[ri];
            if (radial_value == 0.0)
            {
                continue;  // out-of-range/zero radial -> column stays pre-zeroed
            }
            row[iorb] = m_phase(orbital.m) * radial_value
                        * ylm[static_cast<std::size_t>(ylm_index(orbital.l, orbital.m))];
        }
    }
}

void OrbitalEvaluator::evaluate_all(int npoint, const double* xyz, double* out) const
{
    std::vector<int> all(static_cast<std::size_t>(nphi_));
    std::iota(all.begin(), all.end(), 0);
    for (int ip = 0; ip < npoint; ++ip)
    {
        double* row = out + static_cast<std::size_t>(ip) * nphi_;
        std::fill(row, row + nphi_, 0.0);
        evaluate_point(xyz[3 * ip + 0], xyz[3 * ip + 1], xyz[3 * ip + 2], all.data(), nphi_, row);
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
        evaluate_point(x, y, z, active, n_active, values);
        const double delta = 1.0e-3;
        std::vector<double> vp(static_cast<std::size_t>(nphi_));
        std::vector<double> vm(static_cast<std::size_t>(nphi_));
        for (int d = 0; d < 3; ++d)
        {
            double xp[3] = {x, y, z};
            double xm[3] = {x, y, z};
            xp[d] += delta;
            xm[d] -= delta;
            std::fill(vp.begin(), vp.end(), 0.0);
            std::fill(vm.begin(), vm.end(), 0.0);
            evaluate_point(xp[0], xp[1], xp[2], active, n_active, vp.data());
            evaluate_point(xm[0], xm[1], xm[2], active, n_active, vm.data());
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
