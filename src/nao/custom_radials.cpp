#include "nao/custom_radials.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

void CustomRadials::build(const NumericalRadial* chi, const int nchi, const int itype,
                          const std::string& symbol)
{
    cleanup();

    if (nchi <= 0 || chi == nullptr)
    {
        throw std::runtime_error("CustomRadials::build: no radials given");
    }

    itype_ = itype;
    nchi_ = nchi;
    symbol_ = symbol;

    lmax_ = chi[0].l();
    lmin_ = chi[0].l();
    for (int i = 0; i < nchi; ++i)
    {
        lmax_ = std::max(lmax_, chi[i].l());
        lmin_ = std::min(lmin_, chi[i].l());
    }

    nzeta_ = new int[lmax_ + 1];
    norb_ = new int[lmax_ + 1];
    std::fill(nzeta_, nzeta_ + lmax_ + 1, 0);
    std::fill(norb_, norb_ + lmax_ + 1, 0);
    nphi_ = 0;

    // chi_ MUST be stored sorted by (l, izeta): RadialSet::indexing() numbers the entries
    // l-ascending then izeta-ascending, and index(l, izeta) reads that numbering straight
    // out. Storing in the caller's registration order instead silently permutes the set --
    // chi(1,0) would hand back whatever radial happened to be registered second -- so a
    // caller that lists, say, a p radial before an s one gets a different (wrong) integral
    // for the same input. Sorting here is STABLE, so izeta within an l is still order of
    // appearance, which is what callers index by.
    std::vector<int> order(nchi_);
    for (int i = 0; i < nchi_; ++i)
    {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(),
                     [&chi](int a, int b) { return chi[a].l() < chi[b].l(); });

    chi_ = new NumericalRadial[nchi_];
    for (int ichi = 0; ichi != nchi_; ++ichi)
    {
        const NumericalRadial& src = chi[order[ichi]];
        const int l = src.l();
        if (src.nr() <= 0)
        {
            throw std::runtime_error("CustomRadials::build: radial has no r-space grid");
        }
        // izeta is assigned by order of appearance within l, and symbol/itype are stamped
        // on, because the two-center table index_map_ keys on (itype, l, izeta).
        chi_[ichi].build(l, true, src.nr(), src.rgrid(), src.rvalue(), src.pr(), nzeta_[l], symbol_,
                         itype_, false);
        nzeta_[l] += 1;
        norb_[l] = nzeta_[l] * (2 * l + 1);
        nphi_ += 2 * l + 1;
    }
    nzeta_max_ = *std::max_element(nzeta_, nzeta_ + lmax_ + 1);

    indexing();
    set_rcut_max();
}

void CustomRadials::build(const RadialSet* const other, const int itype, const int p, const int pm,
                          const double rcut)
{
    cleanup();

    if (other == nullptr || other->nchi() <= 0)
    {
        throw std::runtime_error("CustomRadials::build: source RadialSet is empty");
    }

    itype_ = itype;
    symbol_ = other->symbol();
    lmax_ = -1;
    lmin_ = 99999;
    for (int ichi = 0; ichi != other->nchi(); ++ichi)
    {
        const int l_new = other->cbegin()[ichi].l() + pm;
        if (l_new < 0)
        {
            continue;
        }
        lmax_ = std::max(lmax_, l_new);
        lmin_ = std::min(lmin_, l_new);
    }
    if (lmax_ < 0)
    {
        throw std::runtime_error("CustomRadials::build: angular-momentum shift left no radials");
    }

    nzeta_ = new int[lmax_ + 1];
    norb_ = new int[lmax_ + 1];
    std::fill(nzeta_, nzeta_ + lmax_ + 1, 0);
    std::fill(norb_, norb_ + lmax_ + 1, 0);
    nphi_ = 0;

    std::vector<int> order(other->nchi());
    for (int i = 0; i < other->nchi(); ++i)
    {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [other, pm](int a, int b) {
        return other->cbegin()[a].l() + pm < other->cbegin()[b].l() + pm;
    });

    chi_ = new NumericalRadial[other->nchi()];
    int ichi_this = 0;
    for (int ichi = 0; ichi != other->nchi(); ++ichi)
    {
        const NumericalRadial& src = other->cbegin()[order[ichi]];
        const int l_new = src.l() + pm;
        if (l_new < 0)
        {
            continue;
        }
        int ngrid = src.nr();
        if (rcut > 0.0)
        {
            // truncate the r-grid rather than resampling: the auxiliary radials this class
            // holds are already on the common uniform grid.
            while (ngrid > 1 && src.rgrid()[ngrid - 1] > rcut)
            {
                --ngrid;
            }
        }
        chi_[ichi_this].build(l_new, true, ngrid, src.rgrid(), src.rvalue(), p, nzeta_[l_new],
                              symbol_, itype_, false);
        nzeta_[l_new] += 1;
        norb_[l_new] = nzeta_[l_new] * (2 * l_new + 1);
        nphi_ += 2 * l_new + 1;
        ++ichi_this;
    }
    nchi_ = ichi_this;
    nzeta_max_ = *std::max_element(nzeta_, nzeta_ + lmax_ + 1);

    indexing();
    set_rcut_max();
}
