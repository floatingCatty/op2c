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
            radial_evals_.push_back(
                {&radial,
                 std::make_unique<ModuleBase::CubicSpline>(radial.nr(), radial.rgrid(), radial.rvalue())});
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
    std::vector<double> ylm(static_cast<std::size_t>((lmax_ + 1) * (lmax_ + 1)));
    ModuleBase::Vector3<double> vec(x, y, z);
    ModuleBase::Ylm::get_ylm_real(lmax_ + 1, vec, ylm.data());

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
        radial_eval.spline->eval(1, &r, &radial_value);
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
