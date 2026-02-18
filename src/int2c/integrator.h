#ifndef TWO_CENTER_INTEGRATOR_H_
#define TWO_CENTER_INTEGRATOR_H_

#include "int2c/table.h"
#include <string>
#include <vector>
#include <memory>
#include "int2c/gaunt_table.h"
#include "nao/atomic_basis.h"
#include "math/linalg/vector3.h"
#include "utils/array_pool.h"

/*!
 * @class TwoCenterIntegrator
 * @brief Computes two-center integrals and their derivatives using tabulated radial functions.
 *
 * This class handles the evaluation of integrals of the form:
 * \f[ I(\mathbf{R}) = \int d^3\mathbf{r} \, \phi_a(\mathbf{r}) \hat{O} \phi_b(\mathbf{r} - \mathbf{R}) \f]
 * where \f$ \phi(\mathbf{r}) = \chi(r) Y_{lm}(\hat{\mathbf{r}}) \f$ are numerical atomic orbitals.
 *
 * It uses the **Talman Method** (Fourier Transform) to pre-calculate radial integrals on a logarithmic grid,
 * creating a `TwoCenterTable`. During the evaluation phase (`calculate`), it interpolates these tables
 * and combines them with Spherical Harmonics and Gaunt coefficients to produce the final result.
 *
 * Supported Operators:
 * - Overlap (S): \f$ \hat{O} = 1 \f$
 * - Kinetic (T): \f$ \hat{O} = -\frac{1}{2}\nabla^2 \f$
 * - Position (r): handled via \f$ \Delta L = \pm 1 \f$ shifts in `calculate`.
 */
class TwoCenterIntegrator
{
  public:
    TwoCenterIntegrator(std::shared_ptr<RealGauntTable> gt);
    TwoCenterIntegrator();
    TwoCenterIntegrator(const TwoCenterIntegrator&) = delete;
    TwoCenterIntegrator& operator=(const TwoCenterIntegrator&) = delete;

    /*!
     * @brief Tabulates radial integrals for a given operator.
     *
     * Pre-computes the radial integral \f$ S_L(R) \f$ for all pairs of orbitals in `bra` and `ket` sets.
     *
     * @param[in] bra Collection of radial functions for the first center.
     * @param[in] ket Collection of radial functions for the second center.
     * @param[in] op Operator type code: 'S' (Overlap), 'T' (Kinetic), 'R' (Radial only).
     * @param[in] nr Number of radial grid points for the table.
     * @param[in] cutoff Maximum radius \f$ R_{cut} \f$ for the table (Bohr).
     */
    void tabulate(const RadialCollection& bra,
                  const RadialCollection& ket,
                  const char op,
                  const int nr,
                  const double cutoff
    );

    ~TwoCenterIntegrator();

    void tabulate(const RadialCollection& bra,
                  const RadialCollection& ketp,
                  const RadialCollection& ketm,
                  const int nr,
                  const double cutoff
    );

    /*!
     * @brief Tabulates radial integrals specifically for Position Operator calculations.
     *
     * This prepares tables for the standard overlap (`ket`) as well as the 
     * \f$ \Delta L = +1 \f$ (`ketp`) and \f$ \Delta L = -1 \f$ (`ketm`) shifted orbitals
     * required for the position operator dipole selection rules.
     *
     * @param[in] bra First orbital collection.
     * @param[in] ket Standard second orbital collection (for Overlap part).
     * @param[in] ketp Second collection with \f$ \times r \f$ and \f$ L+1 \f$.
     * @param[in] ketm Second collection with \f$ \times r \f$ and \f$ L-1 \f$.
     * @param[in] nr Grid size.
     * @param[in] cutoff Cutoff radius.
     */
    void tabulate(const RadialCollection& bra,
                  const RadialCollection& ket,
                  const RadialCollection& ketp,
                  const RadialCollection& ketm,
                  const int nr,
                  const double cutoff
    );

    /*!
     * @brief Compute the two-center integrals.
     *
     * This function calculates the two-center integral
     *
     *                     /    
     *              I(R) = | dr phi1(r) (op_) phi2(r - R)
     *                     /               
     *
     * or its gradient by using the tabulated radial part and real Gaunt coefficients.
     *
     * @param[in] itype1       Element index of orbital 1.
     * @param[in] l1           Angular momentum of orbital 1.
     * @param[in] izeta1       Zeta number of orbital 1.
     * @param[in] m1           Magnetic quantum number of orbital 1.
     * @param[in] itype2       Element index of orbital 2.
     * @param[in] l2           Angular momentum of orbital 2.
     * @param[in] izeta2       Zeta number of orbital 2.
     * @param[in] m2           Magnetic quantum number of orbital 2.
     * @param[in] vR           R2 - R1.
     * @param[out] out         Two-center integral. The integral will not be computed
     *                         if out is nullptr.
     * @param[out] grad_out    Gradient of the integral. grad_out[0], grad_out[1] and
     *                         grad_out[2] are the x, y, z components of the gradient.
     *                         The gradient will not be computed if grad_out is nullptr.
     *
     * @note out and grad_out cannot be both nullptr.
     *                                                                                  */
    void calculate(const int itype1, 
                   const int l1, 
                   const int izeta1, 
                   const int m1, 
                   const int itype2,
                   const int l2,
                   const int izeta2,
                   const int m2,
	                 const ModuleBase::Vector3<double>& vR, // vR = R2 - R1
                   double* out = nullptr,
                   double* grad_out = nullptr
    ) const;

    /*!
     * @brief Advanced calculation method for joint Overlap and Position integrals.
     *
     * Simultaneously computes \f$ S \f$ and \f$ \langle \mathbf{r} \rangle \f$.
     *
     * @param itype1, l1, izeta1, m1 Quantum numbers for first orbital.
     * @param itype2, l2, izeta2, m2 Quantum numbers for second orbital.
     * @param vR Relative position \f$ \mathbf{R}_2 - \mathbf{R}_1 \f$.
     * @param R2 Absolute position of center 2 (needed for \f$ \langle \mathbf{r} \rangle = \langle \mathbf{r}' \rangle + \mathbf{R}_2 S \f$).
     * @param[out] outS Output Overlap integral.
     * @param[out] outRx Output Position x-component.
     * @param[out] outRy Output Position y-component.
     * @param[out] outRz Output Position z-component.
     */
    void calculate(const int itype1, 
                   const int l1, 
                   const int izeta1, 
                   const int m1, 
                   const int itype2,
                   const int l2,
                   const int izeta2,
                   const int m2,
                   const ModuleBase::Vector3<double>& vR, // R = R2 - R1
                   const ModuleBase::Vector3<double>& R2,
                   double* outS = nullptr,
                   double* outRx = nullptr,
                   double* outRy = nullptr,
                   double* outRz = nullptr
    ) const;


    /*!
     * @brief Compute a batch of two-center integrals (optimized).
     *
     * This overload allows reusing the output buffer to avoid re-allocation.
     *                                                                                  */
    void snap(const int itype1, 
              const int l1, 
              const int izeta1, 
              const int m1, 
              const int itype2,
	            const ModuleBase::Vector3<double>& vR, // vR = R2 - R1
              const bool deriv,
              std::vector<std::vector<double>>& out
    ) const;

    /// Returns the amount of heap memory used by table_ (in bytes).
    size_t table_memory() const {
          size_t mem = table_.memory();
          if (is_tabulated_pos_) { 
              mem += tablep_.memory();
              mem += tablem_.memory();
          }

          return mem;
       }

  private:
    /*!
     * @brief Core kernel to compute overlap/integral between two single orbitals
     * 
     * Computes <phi1 | O | phi2> where phi2 is at -vR relative to phi1.
     * 
     * @param[in] l1, m1 angular momentum of first orbital
     * @param[in] l2, m2 angular momentum of second orbital
     * @param[in] table Reference to the radial table (table_, tablep_, or tablem_)
     * @param[in] itype1, izeta1 indices for first orbital radial part
     * @param[in] itype2, izeta2 indices for second orbital radial part
     * @param[in] R distance
     * @param[in] uR unit vector R
     * @param[in] Rl_Y precomputed spherical harmonics R^l * Y_lm(uR)
     * @param[in] grad_Rl_Y precomputed gradients of spherical harmonics (optional)
     * @param[in,out] out accumulated result
     * @param[in,out] grad_out accumulated gradient (optional)
     */
    void calculate_kernel(
        int l1, int m1, 
        int l2, int m2,
        const TwoCenterTable& table,
        int itype1, int izeta1,
        int itype2, int izeta2,
        double R,
        const ModuleBase::Vector3<double>& uR,
        const std::vector<double>& Rl_Y,
        const ModuleBase::Array_Pool<double>* grad_Rl_Y,
        double* out,
        double* grad_out
    ) const;

    bool is_tabulated_, is_tabulated_pos_;
    char op_;
    TwoCenterTable table_, tablep_, tablem_;

    /*!
     * @brief Returns the index of (l,m) in the array of spherical harmonics.
     *
     * Spherical harmonics in ABACUS are stored in the following order:
     *
     * index  0   1   2   3   4   5   6   7   8   9  10  11  12 ...
     *   l    0   1   1   1   2   2   2   2   2   3   3   3   3 ...
     *   m    0   0   1  -1   0   1  -1   2  -2   0   1  -1   2 ...
     *                                                                                  */
    int ylm_index(const int l, const int m) const;
    
    std::shared_ptr<RealGauntTable> gaunt_table_;
};

#endif
