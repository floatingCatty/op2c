#include "nao/radial_function.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "math/constants.h"
#include "math/interpolation/cubic_spline.h"
#include "math/math_integral.h"
#include "math/spherical_bessel_transformer.h"

using ModuleBase::PI;


void NumericalRadial::build(const int l,
                            const bool for_r_space,
                            const int ngrid,
                            const double* const grid,
                            const double* const value,
                            const int p,
                            const int izeta,
                            const std::string symbol,
                            const int itype,
                            const bool init_sbt)
{
#ifdef __DEBUG
    assert(l >= 0);
    assert(ngrid > 1);
    assert(grid && value);

    // grid must be strictly increasing and every element must be non-negative
    assert(std::is_sorted(grid, grid + ngrid, std::less_equal<double>())); // using less_equal forbids equal values
    assert(grid[0] >= 0.0);
#endif

    // wipe off any existing r & k space data
    wipe(true, true);

    symbol_ = symbol;
    itype_ = itype;
    izeta_ = izeta;
    l_ = l;

    if (for_r_space)
    {
        nr_ = ngrid;
        pr_ = p;
        rgrid_.resize(nr_);
        rvalue_.resize(nr_);
        std::memcpy(rgrid_.data(), grid, nr_ * sizeof(double));
        std::memcpy(rvalue_.data(), value, nr_ * sizeof(double));
    }
    else
    {
        nk_ = ngrid;
        pk_ = p;
        kgrid_.resize(nk_);
        kvalue_.resize(nk_);
        std::memcpy(kgrid_.data(), grid, nk_ * sizeof(double));
        std::memcpy(kvalue_.data(), value, nk_ * sizeof(double));
    }

    set_icut(for_r_space, !for_r_space);
}

void NumericalRadial::set_transformer(ModuleBase::SphericalBesselTransformer sbt, int update)
{
    sbt_ = sbt;

#ifdef __DEBUG
    assert(update == 0 || update == 1 || update == -1);
#endif
    switch (update)
    {
    case 1:
        transform(true); // forward transform r -> k
        break;
    case -1:
        transform(false); // backward transform k -> r
        break;
    default:; // do nothing
    }
}

void NumericalRadial::set_grid(const bool for_r_space, const int ngrid, const double* const grid, const char mode)
{
#ifdef __DEBUG
    assert(mode == 'i' || mode == 't');
    assert(ngrid > 1);

    // grid must be strictly increasing and every element must be non-negative
    assert(std::is_sorted(grid, grid + ngrid, std::less_equal<double>())); // using less_equal forbids equal values
    assert(grid[0] >= 0.0);
#endif

    // tbu stands for "to be updated"
    std::vector<double>* grid_tbu = (for_r_space ? &rgrid_ : &kgrid_);
    std::vector<double>* value_tbu = (for_r_space ? &rvalue_ : &kvalue_);
    int& ngrid_tbu = (for_r_space ? nr_ : nk_);

    if (mode == 't')
    { // obtain new values by a transform from the other space
        // make sure a transform from the other space is available
#ifdef __DEBUG
        assert(for_r_space ? (!kgrid_.empty() && !kvalue_.empty()) : (!rgrid_.empty() && !rvalue_.empty()));
#endif

        grid_tbu->resize(ngrid);
        value_tbu->resize(ngrid);
        ngrid_tbu = ngrid;
        std::memcpy(grid_tbu->data(), grid, ngrid * sizeof(double));

        is_fft_compliant_ = is_fft_compliant(nr_, rgrid_.data(), nk_, kgrid_.data());
        transform(!for_r_space); // transform(true): r -> k; transform(false): k -> r
        // ircut_ or ikcut_ is updated in transform()
    }
    else
    { // obtain new values by interpolation in the current space
        // make sure an interpolation in the current space is available
#ifdef __DEBUG
        assert(!grid_tbu->empty() && !value_tbu->empty());
#endif

        // cubic spline interpolation
        ModuleBase::CubicSpline cubspl(ngrid_tbu, grid_tbu->data(), value_tbu->data()); // not-a-knot boundary condition

        std::vector<double> grid_new(ngrid);
        std::vector<double> value_new(ngrid, 0.0);

        std::memcpy(grid_new.data(), grid, ngrid * sizeof(double));

        // do interpolation for grid points within the range of the origional grid
        // for grid points outside the original range, simply set the values to zero

        // grid_start is the first grid point that is greater than or equal to grid_tbu[0]
        double* grid_start = std::lower_bound(grid_new.data(), grid_new.data() + ngrid, (*grid_tbu)[0]);

        // grid_end is the first grid point that is strictly greater than grid_tbu[ngrid_tbu-1]
        double* grid_end = std::upper_bound(grid_new.data(), grid_new.data() + ngrid, (*grid_tbu)[ngrid_tbu - 1]);

        cubspl.eval(std::distance(grid_start, grid_end), grid_start, value_new.data() + std::distance(grid_new.data(), grid_start));

        *grid_tbu = grid_new;
        *value_tbu = value_new;
        ngrid_tbu = ngrid;

        is_fft_compliant_ = is_fft_compliant(nr_, rgrid_.data(), nk_, kgrid_.data());
        set_icut(for_r_space, !for_r_space);
        transform(for_r_space); // transform(true): r -> k; transform(false): k -> r
    }
}

void NumericalRadial::set_uniform_grid(const bool for_r_space,
                                       const int ngrid,
                                       const double cutoff,
                                       const char mode,
                                       const bool enable_fft)
{
    std::vector<double> grid(ngrid);
    double dx = cutoff / (ngrid - 1);
    for (int i = 0; i != ngrid; ++i)
    {
        grid[i] = i * dx;
    }

    set_grid(for_r_space, ngrid, grid.data(), mode);

    if (enable_fft)
    {
        set_uniform_grid(!for_r_space, ngrid, PI / dx, 't', false);
    }
}

void NumericalRadial::set_value(const bool for_r_space, const double* const value, const int p)
{
#ifdef __DEBUG
    assert(for_r_space ? rvalue_ : kvalue_);
#endif
    if (for_r_space)
    {
        std::memcpy(rvalue_.data(), value, nr_ * sizeof(double));
        pr_ = p;
        transform(true);
        set_icut(true, false);
    }
    else
    {
        std::memcpy(kvalue_.data(), value, nk_ * sizeof(double));
        pk_ = p;
        transform(false);
        set_icut(false, true);
    }
}

void NumericalRadial::wipe(const bool r_space, const bool k_space)
{
#ifdef __DEBUG
    assert(r_space || k_space);
#endif

    // wipe the grid and value in r/k space
    if (r_space)
    {
        rgrid_.clear();
        rvalue_.clear();
        nr_ = 0;
        pr_ = 0;
        ircut_ = 0;
    }

    if (k_space)
    {
        kgrid_.clear();
        kvalue_.clear();
        nk_ = 0;
        pk_ = 0;
        ikcut_ = 0;
    }
    is_fft_compliant_ = false;
}


void NumericalRadial::normalize(bool for_r_space)
{
    int& ngrid = for_r_space ? nr_ : nk_;

    // tbu stands for "to be updated"
    double* grid_tbu = for_r_space ? rgrid_.data() : kgrid_.data();
    double* value_tbu = for_r_space ? rvalue_.data() : kvalue_.data();

    double factor = 0.0;
    std::vector<double> integrand(ngrid);
    std::vector<double> rab(ngrid);

    std::adjacent_difference(grid_tbu, grid_tbu + ngrid, rab.begin());
    std::transform(value_tbu, value_tbu + ngrid, grid_tbu, integrand.begin(), std::multiplies<double>());
    std::for_each(integrand.begin(), integrand.end(), [](double& x) { x *= x; });

    factor = ModuleBase::Integral::simpson(ngrid, integrand.data(), &rab[1]);
    factor = 1. / std::sqrt(factor);

    std::for_each(value_tbu, value_tbu + ngrid, [factor](double& x) { x *= factor; });
    transform(for_r_space);
}

void NumericalRadial::transform(const bool forward)
{
#ifdef __DEBUG
    // grid & value must exist in the initial space
    assert(forward ? (!rgrid_.empty() && !rvalue_.empty()) : (!kgrid_.empty() && !kvalue_.empty()));
#endif

    // do nothing if there is no grid in the destination space
    if ((forward && kgrid_.empty()) || (!forward && rgrid_.empty()))
    {
        return;
    }

    if (forward)
    { // r -> k
        if (is_fft_compliant_)
        {
            sbt_.radrfft(l_, nr_, rgrid_[nr_ - 1], rvalue_.data(), kvalue_.data(), pr_);
        }
        else
        {
            sbt_.direct(l_, nr_, rgrid_.data(), rvalue_.data(), nk_, kgrid_.data(), kvalue_.data(), pr_);
        }
        pk_ = 0;
        set_icut(false, true);
    }
    else
    { // k -> r
        if (is_fft_compliant_)
        {
            sbt_.radrfft(l_, nk_, kgrid_[nk_ - 1], kvalue_.data(), rvalue_.data(), pk_);
        }
        else
        {
            sbt_.direct(l_, nk_, kgrid_.data(), kvalue_.data(), nr_, rgrid_.data(), rvalue_.data(), pk_);
        }
        pr_ = 0;
        set_icut(true, false);
    }
}

void NumericalRadial::set_icut(const bool for_r_space, const bool for_k_space, const double tol)
{
    if (for_r_space)
    {
#ifdef __DEBUG
        assert(!rgrid_.empty() && !rvalue_.empty());
#endif
        ircut_ = nr_;
        while (ircut_ && std::abs(rvalue_[ircut_ - 1]) <= tol) { --ircut_; }
    }

    if (for_k_space)
    {
#ifdef __DEBUG
        assert(!kgrid_.empty() && !kvalue_.empty());
#endif
        ikcut_ = nk_;
        while (ikcut_ && std::abs(kvalue_[ikcut_ - 1]) <= tol) { --ikcut_; }
    }
}

bool NumericalRadial::is_uniform(const int n, const double* const x, const double tol)
{
    double dx = (x[n - 1] - x[0]) / (n - 1);
    return std::all_of(x, x + n,
            [&](const double& xi) { return std::abs(x[0] + (&xi - x) * dx - xi) < tol; });
}

bool NumericalRadial::is_fft_compliant(const int nr,
                                       const double* const rgrid,
                                       const int nk,
                                       const double* const kgrid,
                                       const double tol
                                       )
{
    if (!rgrid || !kgrid || nr != nk || nr < 2)
    {
        return false;
    }

    double dr = rgrid[nr - 1] / (nr - 1);
    double dk = kgrid[nk - 1] / (nk - 1);

    return nr * std::abs(dr * dk - PI / (nr - 1)) < tol
           && rgrid[0] == 0.0 && is_uniform(nr, rgrid, tol)
           && kgrid[0] == 0.0 && is_uniform(nk, kgrid, tol);
}
