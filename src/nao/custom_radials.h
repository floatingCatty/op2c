#ifndef OP2C_CUSTOM_RADIALS_H_
#define OP2C_CUSTOM_RADIALS_H_

#include "nao/orbital_shell.h"
#include "nao/radial_function.h"

#include <vector>

/**
 * @brief A RadialSet assembled from radials that were *computed*, not read from a file.
 *
 * AtomicRadials comes from an orbital file and BetaRadials insists on the pr = 1
 * convention of KB projectors, so neither can host a set that is generated in memory.
 * This class is the plain container: hand it NumericalRadial objects, it works out
 * lmax/nzeta/indexing and behaves like any other RadialSet, so a RadialCollection built
 * from it drives the two-center engine unchanged.
 *
 * Used by the exact-exchange work (rescu++ doc/design/27 §4) to hold the auxiliary basis
 * -- product-PAO radials after the pivoted-Cholesky reduction, and the kernel-transformed
 * partners Q of §3.2 -- and by its tests to build analytically known radials for which a
 * closed-form reference integral exists.
 */
class CustomRadials : public RadialSet
{
  public:
    CustomRadials() {}
    CustomRadials(const CustomRadials& other) : RadialSet(other) {} //!< deep copy

    using RadialSet::operator=;
    CustomRadials* clone() const override { return new CustomRadials(*this); }
    CustomRadials* create_empty() const override { return new CustomRadials(); }

    ~CustomRadials() {}

    /**
     * @brief Build from a list of radials.
     *
     * The izeta of each radial is reassigned by order of appearance within its l, so
     * callers need only supply (l, values); @p symbol and @p itype are stamped onto every
     * radial because the two-center tables key on them.
     *
     * @param[in] chi     radials to adopt (deep-copied)
     * @param[in] nchi    number of radials
     * @param[in] itype   element index this set represents
     * @param[in] symbol  element label
     *
     * @note Like every other RadialSet builder this adopts the R-SPACE values only. Call
     *       RadialCollection::set_uniform_grid(false, ...) on the collection before handing
     *       it to the two-center engine, exactly as TwoCenterBundle does for orbitals.
     */
    void build(const NumericalRadial* chi, const int nchi, const int itype,
               const std::string& symbol);

    /// Required by RadialSet; rebuilds this set from another with p/pm modifications.
    void build(const RadialSet* const other, const int itype, const int p = 0, const int pm = 0,
               const double rcut = -1.0) override;
};

#endif
