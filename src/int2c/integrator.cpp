#include "int2c/integrator.h"

#include "math/linalg/vector3.h"
#include "math/ylm.h"
#include "utils/array_pool.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
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
                                   const double cutoff,
                                   const double omega,
                                   const double r_cut,
                                   const std::function<bool(int, int)>& tabulate_pair)
{
    op_ = op;
    table_.build(bra, ket, op, nr, cutoff, omega, r_cut, tabulate_pair);
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

    if (R > table_.rmax()) {
        *outS = 0.0;
        return;
    }

    // unit vector along R
    ModuleBase::Vector3<double> uR = (R < 1e-12 ? ModuleBase::Vector3<double>(0., 0., 1.) : vR / R);

    // Max L: for S: l1+l2, for position: l1+l2+1
    int lmax_overall = l1 + l2 + 1;
    std::vector<double> Rl_Y((lmax_overall+1) * (lmax_overall+1));
    ModuleBase::Ylm::rl_sph_harm(lmax_overall, vR[0], vR[1], vR[2], Rl_Y);

    // S calculation using calculate_kernel
    calculate_kernel(l1, m1, l2, m2, table_, itype1, izeta1, itype2, izeta2, 
                     R, uR, Rl_Y, nullptr, outS, nullptr);

    // 2. Position operator dipole parts
    // Following the ABACUS reference: each Cartesian component has
    // its own m3 selection rules and sign (Condon-Shortley phase).
    bool do_pos = (outRx || outRy || outRz);
    
    if (do_pos && R <= std::min(tablep_.rmax(), tablem_.rmax())) 
    {
        double pref_base = std::sqrt(ModuleBase::FOUR_PI / 3.0);
        double S_by_Rl_val = 0.0;
        double* S_by_Rl = &S_by_Rl_val;

        // Process each Cartesian component separately
        double* out_ptrs[3] = {outRx, outRy, outRz};
        
        for (int alpha = 0; alpha < 3; ++alpha) {
            if (!out_ptrs[alpha]) continue;
            
            double pref = pref_base;
            int mm;     // dipole m quantum number
            int mr[2];  // m3 candidates
            
            if (alpha == 0) {        // X: mm = +1
                mr[0] = m2 - 1;
                mr[1] = m2 + 1;
                mm = 1;
                pref *= -1;          // Condon-Shortley phase for mm=+1
            } else if (alpha == 1) { // Y: mm = -1
                mr[0] = -m2 - 1;     // KEY: -m2, not m2!
                mr[1] = -m2 + 1;
                mm = -1;
                pref *= -1;
            } else {                 // Z: mm = 0
                mr[0] = m2;
                mr[1] = m2;
                mm = 0;
            }
            
            for (int l3 = (l2 == 0 ? 1 : l2 - 1); l3 <= l2 + 1; l3 += 2) {
                for (int im = 0; im <= std::abs(mm); ++im) {
                    int m3 = mr[im];
                    if (m3 < -l3 || m3 > l3) continue;
                    
                    double Gr = (*gaunt_table_)(l2, l3, 1, m2, m3, mm);
                    
                    // Radial expansion: sum over l = |l1-l3| ... l1+l3
                    int sign = (l1 - l3 - std::abs(l1 - l3)) % 4 == 0 ? 1 : -1;
                    for (int l = std::abs(l1 - l3); l <= l1 + l3; l += 2) {
                        const TwoCenterTable* tab_ptr = (l3 == l2 - 1) ? &tablem_ : &tablep_;
                        tab_ptr->lookup(itype1, l1, izeta1, itype2, l3, izeta2, l, R, S_by_Rl, nullptr);
                        
                        for (int m = -l; m <= l; ++m) {
                            double G = (*gaunt_table_)(l1, l3, l, m1, m3, m);
                            *out_ptrs[alpha] += sign * pref * Gr * G * (*S_by_Rl) * Rl_Y[ylm_index(l, m)];
                        }
                        sign = -sign;
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

void TwoCenterIntegrator::calculate_block(const int itype1, const int n1, const int* f1,
                                          const int itype2, const int n2, const int* f2,
                                          const int nv, const double* vecs, double* out) const
{
    block_kernel(itype1, n1, f1, itype2, n2, f2, nv, vecs, out, nullptr);
}

void TwoCenterIntegrator::calculate_block_grad(const int itype1, const int n1, const int* f1,
                                               const int itype2, const int n2, const int* f2,
                                               const int nv, const double* vecs, double* grad) const
{
    block_kernel(itype1, n1, f1, itype2, n2, f2, nv, vecs, nullptr, grad);
}

void TwoCenterIntegrator::block_kernel(const int itype1, const int n1, const int* f1,
                                       const int itype2, const int n2, const int* f2,
                                       const int nv, const double* vecs, double* out,
                                       double* grad) const
{
    // channels: runs of equal (l, izeta), with the member rows and their m
    struct Chan { int l, iz; std::vector<int> row, m; };
    auto channels = [](const int n, const int* f) {
        std::vector<Chan> ch;
        for (int i = 0; i < n; ++i) {
            const int l = f[3 * i], iz = f[3 * i + 1], m = f[3 * i + 2];
            auto it = std::find_if(ch.begin(), ch.end(), [&](const Chan& c) { return c.l == l && c.iz == iz; });
            if (it == ch.end()) { ch.push_back(Chan{l, iz, {}, {}}); it = ch.end() - 1; }
            it->row.push_back(i);
            it->m.push_back(m);
        }
        return ch;
    };
    const std::vector<Chan> c1 = channels(n1, f1), c2 = channels(n2, f2);
    int l1max = 0, l2max = 0;
    for (const Chan& c : c1) l1max = std::max(l1max, c.l);
    for (const Chan& c : c2) l2max = std::max(l2max, c.l);
    const int lmax = l1max + l2max;

    // the non-zero real Gaunt coefficients of each (l1, l2, l), built once per call: real
    // harmonics couple only |m| in {|m1| + |m2|, ||m1| - |m2||}, so every other m is skipped
    struct GEntry { int m1, m2, y; double G; };
    std::vector<std::vector<GEntry>> glist((l1max + 1) * (l2max + 1) * (lmax + 1));
    std::vector<char> gbuilt(glist.size(), 0);
    auto gaunts = [&](const int l1, const int l2, const int l) -> const std::vector<GEntry>& {
        const std::size_t key = (static_cast<std::size_t>(l1) * (l2max + 1) + l2) * (lmax + 1) + l;
        if (!gbuilt[key]) {
            gbuilt[key] = 1;
            for (int m1 = -l1; m1 <= l1; ++m1)
                for (int m2 = -l2; m2 <= l2; ++m2) {
                    const int s = std::abs(m1) + std::abs(m2), d = std::abs(std::abs(m1) - std::abs(m2));
                    const int cand[4] = {s, -s, d, -d};
                    for (int c = 0; c < 4; ++c) {
                        const int m = cand[c];
                        if (std::abs(m) > l) continue;
                        bool seen = false;
                        for (int e = 0; e < c; ++e) seen = seen || cand[e] == m;
                        if (seen) continue;
                        const double G = (*gaunt_table_)(l1, l2, l, m1, m2, m);
                        if (std::abs(G) >= 1e-12) glist[key].push_back(GEntry{m1, m2, ylm_index(l, m), G});
                    }
                }
        }
        return glist[key];
    };
    // member row of each m in a channel (-1: absent)
    auto mpos = [](const Chan& c) {
        std::vector<int> pos(2 * c.l + 1, -1);
        for (std::size_t p = 0; p < c.m.size(); ++p) pos[c.m[p] + c.l] = c.row[p];
        return pos;
    };
    std::vector<std::vector<int>> pos1, pos2;
    for (const Chan& c : c1) pos1.push_back(mpos(c));
    for (const Chan& c : c2) pos2.push_back(mpos(c));

    // The images first: R, its unit vector and R^l Y_lm (and gradients) of every image within the
    // table. Then per channel pair and l: the table row once, its radial value at every image, and
    // the image sum of each harmonic component, scattered through the Gaunt list ONCE -- not once per
    // image (a periodized kernel block sums ~30 images; the scatter was most of the kernel's time).
    const int ny = (lmax + 1) * (lmax + 1);
    std::vector<double> Rk, uk, Yk, gYk;
    Rk.reserve(nv);
    for (int k = 0; k < nv; ++k)
    {
        const ModuleBase::Vector3<double> vR(vecs[3 * k], vecs[3 * k + 1], vecs[3 * k + 2]);
        const double R = vR.norm();
        if (R > table_.rmax()) continue;
        std::vector<double> Rl_Y(ny);
        ModuleBase::Ylm::rl_sph_harm(lmax, vR[0], vR[1], vR[2], Rl_Y);
        Rk.push_back(R);
        Yk.insert(Yk.end(), Rl_Y.begin(), Rl_Y.end());
        // unit vector (the table's d(S/R^l)/dR carries the radial direction); at R = 0 only
        // the l = 0 channel survives and its radial derivative term vanishes by symmetry
        uk.push_back(R > 1e-12 ? vR[0] / R : 0.0);
        uk.push_back(R > 1e-12 ? vR[1] / R : 0.0);
        uk.push_back(R > 1e-12 ? vR[2] / R : 0.0);
        if (grad)
        {
            ModuleBase::Array_Pool<double> gY(ny, 3);
            ModuleBase::Ylm::grad_rl_sph_harm(lmax, vR[0], vR[1], vR[2], Rl_Y.data(), gY.get_ptr_2D());
            for (int y = 0; y < ny; ++y)
                for (int c = 0; c < 3; ++c) gYk.push_back(gY[y][c]);
        }
    }
    const int nk = static_cast<int>(Rk.size());
    if (nk == 0) return;
    std::vector<double> f(nk), df(nk), Sy(2 * lmax + 1), Gy(3 * (2 * lmax + 1));
    // An element pair the table leaves to direct evaluation: S_l(R) / R^l = 4 pi sum_q w_q q^2 v(q)
    // kv_a(q) kv_b(q) j_l(qR) / R^l on the radials' k grid. j_l(qR) / R^l once per image and l (shared
    // by every channel pair), each bra channel's values times the weights once per call.
    const bool direct = table_.direct(itype1, itype2);
    if (direct && grad)
        throw std::runtime_error("TwoCenterIntegrator::calculate_block_grad: no gradient for a directly integrated pair");
    int nq = 0;
    std::vector<double> Jl, wa, kvb_f;
    std::vector<const double*> kvb;
    if (direct)
    {
        // the quadrature's reach (pi / dk: its first alias image stays beyond the image)
        const double rfar = *std::max_element(Rk.begin(), Rk.end());
        if (rfar > table_.direct_rmax())
            throw std::runtime_error("TwoCenterIntegrator: a directly integrated pair at " + std::to_string(rfar)
                                     + " bohr, past the quadrature's range (" + std::to_string(table_.direct_rmax()) + ")");
        const std::vector<double>& kw = table_.direct_weight();
        const double* q = table_.direct_grid().data();
        nq = static_cast<int>(kw.size());
        auto on_grid = [&](const int side, const int itype, const Chan& c, double* dst) {
            const double* kv = (*(side == 0 ? table_.bra() : table_.ket()))(itype, c.l, c.iz).kvalue();
            std::copy(kv, kv + nq, dst);
        };
        Jl.assign(static_cast<std::size_t>(nk) * (lmax + 1) * nq, 0.0);
        for (int k = 0; k < nk; ++k)
            for (int l = 0; l <= lmax; ++l)
            {
                double* J = &Jl[(static_cast<std::size_t>(k) * (lmax + 1) + l) * nq];
                if (Rk[k] > 1e-12)
                {
                    const double Rl = std::pow(Rk[k], l);
                    for (int i = 0; i < nq; ++i) J[i] = std::sph_bessel(static_cast<unsigned>(l), q[i] * Rk[k]) / Rl;
                }
                else if (l == 0)
                    std::fill(J, J + nq, 1.0);        // at R = 0 only l = 0 survives (R^l Y_lm = 0 above)
            }
        wa.assign(c1.size() * static_cast<std::size_t>(nq), 0.0);
        for (std::size_t ia = 0; ia < c1.size(); ++ia)
        {
            on_grid(0, itype1, c1[ia], &wa[ia * nq]);
            for (int i = 0; i < nq; ++i) wa[ia * nq + i] *= kw[i];
        }
        kvb_f.assign(c2.size() * static_cast<std::size_t>(nq), 0.0);
        for (std::size_t ib = 0; ib < c2.size(); ++ib)
        {
            on_grid(1, itype2, c2[ib], &kvb_f[ib * nq]);
            kvb.push_back(&kvb_f[ib * nq]);
        }
    }
    for (std::size_t ia = 0; ia < c1.size(); ++ia)
        for (std::size_t ib = 0; ib < c2.size(); ++ib)
        {
            const Chan& a = c1[ia];
            const Chan& b = c2[ib];
            const std::vector<int>& p1 = pos1[ia];
            const std::vector<int>& p2 = pos2[ib];
            int sign = (a.l - b.l - std::abs(a.l - b.l)) % 4 == 0 ? 1 : -1;
            for (int l = std::abs(a.l - b.l); l <= a.l + b.l; l += 2)
            {
                bool any = false;
                for (int k = 0; k < nk; ++k)
                {
                    double S_by_Rl = 0.0, dS_by_Rl = 0.0;
                    if (direct)
                    {
                        const double* J = &Jl[(static_cast<std::size_t>(k) * (lmax + 1) + l) * nq];
                        const double* w = &wa[ia * nq];
                        const double* kb = kvb[ib];
                        double acc = 0.0;
                        for (int i = 0; i < nq; ++i) acc += w[i] * kb[i] * J[i];
                        S_by_Rl = acc;
                    }
                    else
                        table_.lookup(itype1, a.l, a.iz, itype2, b.l, b.iz, l, Rk[k], &S_by_Rl,
                                      grad ? &dS_by_Rl : nullptr);
                    f[k] = sign * S_by_Rl;
                    df[k] = sign * dS_by_Rl;
                    any = any || S_by_Rl != 0.0 || dS_by_Rl != 0.0;
                }
                sign = -sign;
                if (!any) continue;
                // image sums of the degree-l components: Sy[m] = sum_k f_k R^l Y_lm(R_k), and for the
                // gradient Gy[m][c] = sum_k (df_k u_k[c] R^l Y_lm + f_k d(R^l Y_lm)/dR_c)
                const int y0 = l * l;
                for (int m = 0; m <= 2 * l; ++m)
                {
                    double acc = 0.0;
                    for (int k = 0; k < nk; ++k) acc += f[k] * Yk[static_cast<std::size_t>(k) * ny + y0 + m];
                    Sy[m] = acc;
                    if (grad)
                        for (int c = 0; c < 3; ++c)
                        {
                            double g = 0.0;
                            for (int k = 0; k < nk; ++k)
                                g += df[k] * uk[3 * k + c] * Yk[static_cast<std::size_t>(k) * ny + y0 + m]
                                     + f[k] * gYk[(static_cast<std::size_t>(k) * ny + y0 + m) * 3 + c];
                            Gy[3 * m + c] = g;
                        }
                }
                for (const GEntry& g : gaunts(a.l, b.l, l))
                {
                    const int i = p1[g.m1 + a.l], j = p2[g.m2 + b.l];
                    if (i < 0 || j < 0) continue;
                    const std::size_t e = static_cast<std::size_t>(i) * n2 + j;
                    const int m = g.y - y0;
                    if (out) out[e] += g.G * Sy[m];
                    if (grad)
                        for (int c = 0; c < 3; ++c) grad[3 * e + c] += g.G * Gy[3 * m + c];
                }
            }
        }
}

int TwoCenterIntegrator::ylm_index(const int l, const int m) const
{
    return l * l + (m > 0 ? 2 * m - 1 : -2 * m);
}
