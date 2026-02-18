#include "int2c/integrator.h"

#include "math/linalg/vector3.h"
#include "math/ylm.h"
#include "utils/array_pool.h"
#include "math/constants.h"
#include <iostream>

TwoCenterIntegrator::TwoCenterIntegrator():
    is_tabulated_(false),
    is_tabulated_pos_(false),
    op_('\0')
{
    // Default fallback if no table provided (though Bundle should provide one)
    gaunt_table_ = std::make_shared<RealGauntTable>();
}

TwoCenterIntegrator::TwoCenterIntegrator(std::shared_ptr<RealGauntTable> gt):
    is_tabulated_(false),
    is_tabulated_pos_(false),
    op_('\0'),
    gaunt_table_(gt)
{
}

TwoCenterIntegrator::~TwoCenterIntegrator()
{
}

void TwoCenterIntegrator::tabulate(const RadialCollection& bra,
                                   const RadialCollection& ket,
                                   const char op,
                                   const int nr,
                                   const double cutoff)
{
    op_ = op;
    table_.build(bra, ket, op, nr, cutoff);
    table_.build(bra, ket, op, nr, cutoff);
    gaunt_table_->build(std::max(bra.lmax(), ket.lmax()));
    is_tabulated_ = true;
}

void TwoCenterIntegrator::tabulate(const RadialCollection& bra,
                                   const RadialCollection& ketp,
                                   const RadialCollection& ketm,
                                   const int nr,
                                   const double cutoff)
{
    op_ = 'S';
    tablep_.build(bra, ketp, op_, nr, cutoff);
    tablem_.build(bra, ketm, op_, nr, cutoff);
    table_.build(bra, bra, op_, nr, cutoff);
    table_.build(bra, bra, op_, nr, cutoff);
    gaunt_table_->build(std::max(bra.lmax(), ketp.lmax())); // ketp's lmax should always larger than ketm's.
    is_tabulated_pos_ = true;
    is_tabulated_ = true;
}

void TwoCenterIntegrator::tabulate(const RadialCollection& bra,
                                   const RadialCollection& ket,
                                   const RadialCollection& ketp,
                                   const RadialCollection& ketm,
                                   const int nr,
                                   const double cutoff)
{
    op_ = 'S';
    tablep_.build(bra, ketp, op_, nr, cutoff);
    tablem_.build(bra, ketm, op_, nr, cutoff);
    table_.build(bra, ket, op_, nr, cutoff);
    table_.build(bra, ket, op_, nr, cutoff);
    gaunt_table_->build(std::max(bra.lmax(), ketp.lmax())); 
    is_tabulated_pos_ = true;
    is_tabulated_ = true;
}

void TwoCenterIntegrator::calculate(const int itype1, 
                                    const int l1, 
                                    const int izeta1, 
                                    const int m1, 
                                    const int itype2,
                                    const int l2,
                                    const int izeta2,
                                    const int m2,
	                                const ModuleBase::Vector3<double>& vR,
                                    double* out,
                                    double* grad_out) const
{
#ifdef __DEBUG
    assert( is_tabulated_ );
    assert( out || grad_out );
#endif

    if (out) *out = 0.0;
    if (grad_out) std::fill(grad_out, grad_out + 3, 0.0);

    double R = vR.norm();
    
    if (R > table_.rmax())
    {
        return;
    }

    // unit vector along R
    ModuleBase::Vector3<double> uR = (R < 1e-12 ? ModuleBase::Vector3<double>(0., 0., 1.) : vR / R);

    // generate all necessary real (solid) spherical harmonics
    const int lmax = l1 + l2;
	std::vector<double> Rl_Y((lmax+1) * (lmax+1));
	ModuleBase::Array_Pool<double> grad_Rl_Y;
    
    ModuleBase::Ylm::rl_sph_harm(lmax, vR[0], vR[1], vR[2], Rl_Y);
    if (grad_out) {
        grad_Rl_Y = ModuleBase::Array_Pool<double>((lmax+1) * (lmax+1), 3);
        ModuleBase::Ylm::grad_rl_sph_harm(lmax, vR[0], vR[1], vR[2], Rl_Y.data(), grad_Rl_Y.get_ptr_2D());
    }

    // Use kernel
    calculate_kernel(l1, m1, l2, m2, table_, itype1, izeta1, itype2, izeta2, R, uR, Rl_Y, grad_out ? &grad_Rl_Y : nullptr, out, grad_out);
}

void TwoCenterIntegrator::calculate(const int itype1, 
                                    const int l1, 
                                    const int izeta1, 
                                    const int m1, 
                                    const int itype2,
                                    const int l2,
                                    const int izeta2,
                                    const int m2,
	                                const ModuleBase::Vector3<double>& vR, // R = R2 - R1
                                    const ModuleBase::Vector3<double>& R2,
                                    double* outS,
                                    double* outRx,
                                    double* outRy,
                                    double* outRz
                                    ) const
{
#ifdef __DEBUG
    assert( is_tabulated_pos_ );
    assert( outS );
#endif

    if (outRx) *outRx = 0.0;
    if (outRy) *outRy = 0.0;
    if (outRz) *outRz = 0.0;

    // 1. Calculate Overlap (S)
    double R = vR.norm();
    // Max R needed is max of overlap and dipole tables.
    double max_rcut = table_.rmax();
    if (is_tabulated_pos_) max_rcut = std::max(max_rcut, std::max(tablep_.rmax(), tablem_.rmax()));

    if (R > max_rcut) {
        *outS = 0.0;
        return;
    }

    ModuleBase::Vector3<double> uR = (R < 1e-12 ? ModuleBase::Vector3<double>(0., 0., 1.) : vR / R);

    // Max L needed: 
    // For S: l1 + l2
    // For r: l1 + (l2 +/- 1) => l1 + l2 + 1
    int lmax_S = l1 + l2;
    int lmax_r = l1 + l2 + 1;
    int lmax_overall = std::max(lmax_S, lmax_r);

    std::vector<double> Rl_Y((lmax_overall+1) * (lmax_overall+1));
    ModuleBase::Ylm::rl_sph_harm(lmax_overall, vR[0], vR[1], vR[2], Rl_Y);

    // S calculation
    if (R <= table_.rmax()) {
        calculate_kernel(l1, m1, l2, m2, table_, itype1, izeta1, itype2, izeta2, 
                         R, uR, Rl_Y, nullptr, outS, nullptr);
    } else {
        *outS = 0.0;
    }

    // 2. Calculate Position (r) parts
    bool do_pos = (outRx || outRy || outRz);
    
    if (do_pos && R <= std::min(tablep_.rmax(), tablem_.rmax())) 
    {
        double pref = std::sqrt(ModuleBase::FOUR_PI / 3.0);
        
        // Loop over l3 = l2 +/- 1
        for (int l3 = (l2 == 0 ? 1 : l2 - 1); l3 <= l2 + 1; l3 += 2) {
            
            // Choose table
            const TwoCenterTable* tab_ptr = nullptr;
            if (l3 == l2 - 1) tab_ptr = &tablem_;
            else if (l3 == l2 + 1) tab_ptr = &tablep_;
            else continue; 

            // Iterate valid m3 given m2 and dipole selection rules
            // Dipole (L=1, mm=0, +/-1) pairs with l2 to give l3. m changes by -1, 0, 1.
            for (int dm = -1; dm <= 1; ++dm) {
                int m3 = m2 + dm;
                if (m3 < -l3 || m3 > l3) continue;

                // Compute overlap kernel value <l1 m1 | | l3 m3>
                double kernel_val = 0.0;
                calculate_kernel(l1, m1, l3, m3, *tab_ptr, itype1, izeta1, itype2, izeta2,
                                 R, uR, Rl_Y, nullptr, &kernel_val, nullptr);
                
                if (std::abs(kernel_val) < 1e-15) continue;

                // Z component (alpha='z', mm=0)
                if (outRz && dm == 0) {
                     double Gr = (*gaunt_table_)(l2, l3, 1, m2, m3, 0); 
                     *outRz += kernel_val * Gr * pref;
                }

                // X component (alpha='x', mm=1, sign -1, involves m2+/-1)
                // Y component (alpha='y', mm=-1, sign -1, involves m2+/-1)
                
                if (dm == -1 || dm == 1) {
                    if (outRx) {
                         double Gr = (*gaunt_table_)(l2, l3, 1, m2, m3, 1);
                         *outRx += kernel_val * Gr * pref * (-1.0);
                    }
                    if (outRy) {
                        double Gr = (*gaunt_table_)(l2, l3, 1, m2, m3, -1);
                        *outRy += kernel_val * Gr * pref * (-1.0);
                    }
                }
            }
        }
    }

    if (outRx) *outRx += R2[0] * (*outS);
    if (outRy) *outRy += R2[1] * (*outS);
    if (outRz) *outRz += R2[2] * (*outS);
}

void TwoCenterIntegrator::calculate_kernel(
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
    ) const
{
    double tmp[2] = {0.0, 0.0};
    double* S_by_Rl = tmp;
    double* d_S_by_Rl = grad_out ? tmp + 1 : nullptr;

    // the sign is given by i^(l1-l2-l) = (-1)^((l1-l2-l)/2)
    int sign = (l1 - l2 - std::abs(l1 - l2)) % 4 == 0 ? 1 : -1;
    
    for (int l = std::abs(l1 - l2); l <= l1 + l2; l += 2)
    {
        // look up S/R^l and (d/dR)(S/R^l) from the radial table
        table.lookup(itype1, l1, izeta1, itype2, l2, izeta2, l, R, S_by_Rl, d_S_by_Rl);

		for (int m = -l; m <= l; ++m)
        {
            double G = (*gaunt_table_)(l1, l2, l, m1, m2, m);
            
            if (std::abs(G) < 1e-12) continue;

            if (out)
            {
                *out += sign * G * (*S_by_Rl) * Rl_Y[ylm_index(l, m)];
            }

            if (grad_out)
            {
                // grad_out is double[3]
                // uR is Vector3
                // grad_Rl_Y is Array_Pool (2D accessor)
                const double* grad_Y_ptr = (*grad_Rl_Y)[ylm_index(l, m)];
                
                for (int i = 0; i < 3; ++i)
                {
                    grad_out[i] += sign * G * ( (*d_S_by_Rl) * uR[i] * Rl_Y[ylm_index(l, m)]
                                                + (*S_by_Rl) * grad_Y_ptr[i] );
                }
            }
        }
        sign = -sign;
    }
}


void TwoCenterIntegrator::snap(const int itype1, 
                               const int l1, 
                               const int izeta1, 
                               const int m1, 
                               const int itype2,
	                           const ModuleBase::Vector3<double>& vR,
                               const bool deriv,
                               std::vector<std::vector<double>>& out) const
{
#ifdef __DEBUG
    assert( is_tabulated_ );
#endif

    // Optimization check: resize only if needed
    int num_needed = deriv ? 4 : 1;
    if (out.size() != num_needed) {
        out.resize(num_needed);
    }

    // total number of ket functions (including all m!)
    // We can cache this count on tabulate()? 
    // For now, simple optimization: calc once
    int num_ket = 0;
    for (int l2 = 0; l2 <= table_.lmax_ket(); ++l2)
    {
        num_ket += (2 * l2 + 1) * table_.nchi_ket(itype2, l2);
    }

    if (num_ket == 0) return;

	for(size_t i = 0; i < out.size(); ++i)
	{
		if (out[i].size() != num_ket) out[i].resize(num_ket);
        std::fill(out[i].begin(), out[i].end(), 0.0);
	}

    int index = 0;
    double tmp[3] = {0.0, 0.0, 0.0};
    for (int l2 = 0; l2 <= table_.lmax_ket(); ++l2)
    {
        for (int izeta2 = 0; izeta2 < table_.nchi_ket(itype2, l2); ++izeta2)
        {
            for (int mm2 = 0; mm2 <= 2*l2; ++mm2)
            {
                int m2 = (mm2 % 2 == 0) ? -mm2 / 2 : (mm2 + 1) / 2;
                
                // Using the optimized scalar calculate
                // Note: we can further optimize by hoisting Rl_Y calculation out of the loop
                // But for now, just calling the kernel-optimized calculate is good.
                calculate(itype1, l1, izeta1, m1, itype2, l2, izeta2, m2, vR, &out[0][index], deriv ? tmp : nullptr);

                if (deriv)
                {
                    out[1][index] = tmp[0];
                    out[2][index] = tmp[1];
                    out[3][index] = tmp[2];
                }

                ++index;
            }
        }
    }
}

int TwoCenterIntegrator::ylm_index(const int l, const int m) const
{
    return l * l + (m > 0 ? 2 * m - 1 : -2 * m);
}
