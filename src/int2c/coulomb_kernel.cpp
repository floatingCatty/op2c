#include "int2c/coulomb_kernel.h"

#include "math/math_integral.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace ExxKernel
{

void kernel_transform(const NumericalRadial& P, const CoulombWeight& w, NumericalRadial& Q)
{
    const int nk = P.nk();
    if (nk <= 0)
    {
        throw std::runtime_error("ExxKernel::kernel_transform: source radial has no k-space grid");
    }

    const double* kgrid = P.kgrid();
    const double* kval = P.kvalue();

    // kvalue() is k^pk * Ptilde(k); the weight multiplies Ptilde, so the implicit
    // exponent is carried through unchanged and handed back to build() as p.
    std::vector<double> qk(nk);
    for (int ik = 0; ik < nk; ++ik)
    {
        qk[ik] = kval[ik] * w(kgrid[ik]);
    }

    Q.build(P.l(), false, nk, kgrid, qk.data(), P.pk(), P.izeta(), P.symbol(), P.itype(), true);

    // build() populates only the space it was given, so Q would come back k-space-only and
    // every r-space consumer -- the (Q.phi) product of doc 27 §3.2, and the two-center
    // tables, which read rvalue() -- would silently see an empty grid. Transform back onto
    // the source's own r-grid ('t' = compute values by spherical Bessel transform).
    const int nr = P.nr();
    if (nr > 0)
    {
        Q.set_grid(true, nr, P.rgrid(), 't');
    }
}

double one_center_gram(const NumericalRadial& a, const NumericalRadial& b, const CoulombWeight& w)
{
    if (a.l() != b.l())
    {
        return 0.0; // orthogonal angular parts
    }
    const int nk = a.nk();
    if (nk <= 0 || nk != b.nk())
    {
        throw std::runtime_error("ExxKernel::one_center_gram: radials need a common k-space grid");
    }

    const double* kgrid = a.kgrid();
    const double* ka = a.kvalue();
    const double* kb = b.kvalue();

    // integrand = k^2 chi~_a chi~_b vtilde(k); undo the implicit k^pk exponents.
    const int pk_total = a.pk() + b.pk();
    std::vector<double> f(nk);
    for (int ik = 0; ik < nk; ++ik)
    {
        const double k = kgrid[ik];
        double v = ka[ik] * kb[ik] * w(k) * k * k;
        if (pk_total != 0)
        {
            v = (k > 0.0) ? v / std::pow(k, pk_total) : 0.0;
        }
        f[ik] = v;
    }

    std::vector<double> h(nk);
    std::adjacent_difference(kgrid, kgrid + nk, h.begin());
    return ModuleBase::Integral::simpson(nk, f.data(), &h[1]);
}

} // namespace ExxKernel
