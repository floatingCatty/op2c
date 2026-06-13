#include "int2c/op2c.hpp"
#include "math/linalg/matrix.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <array>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <vector>

Op2c::Op2c(size_t ntype, int nspin, bool lspinorb,
        const std::string& orb_dir, const std::vector<std::string> orb_name, const std::string& psd_dir, const std::vector<std::string> psd_name,
        MPI_Comm comm, const std::string& log_file, bool pm_build
        ):
        comm(comm), nspin(nspin), lspinorb(lspinorb)
{
    int rank = 0;
#ifdef __MPI
    MPI_Comm_rank(comm, &rank);
#endif
    if (orb_name.size() != ntype){
        throw std::runtime_error("orb_name.size() != ntype");
    }
    if (!psd_name.empty() && psd_name.size() != ntype){
        throw std::runtime_error("psd_name.size() != ntype");
    }

    std::ofstream ofs;
    if (!log_file.empty()) {
        ofs.open(log_file);
    }
    ModuleBase::Logger logger(log_file.empty() ? std::cout : ofs);

    tcbd.build_orb(ntype, orb_name.data(), orb_dir, pm_build, comm);

    if(!psd_name.empty()) {
        psds.resize(ntype);
        read_pseudo(psd_dir, psd_name, "auto", 10.0, false, 0.0, logger, psds, comm);

        std::vector<BetaRadials> beta_radials(ntype);
        for(size_t itype=0; itype<ntype; ++itype){
            beta_radials[itype] = psds[itype].beta_radials;
        }
        tcbd.build_beta(ntype, beta_radials.data(), pm_build);
    }

    build_maps();
};

Op2c::Op2c(
    std::vector<AtomicRadials> orbitals,
    std::vector<Atom_pseudo> pseudos,
    int nspin, bool lspinorb,
    bool pm_build
    ):
    comm(0), nspin(nspin), lspinorb(lspinorb),
    psds(std::move(pseudos))
{
    size_t ntype = orbitals.size();

    tcbd.build_orb(ntype, orbitals.data(), pm_build);

    if(!psds.empty()) {
        if (psds.size() != ntype){
            throw std::runtime_error("pseudos.size() != orbitals.size()");
        }
        std::vector<BetaRadials> beta_radials(ntype);
        for(size_t itype=0; itype<ntype; ++itype){
            beta_radials[itype] = psds[itype].beta_radials;
        }
        tcbd.build_beta(ntype, beta_radials.data(), pm_build);
    }

    build_maps();
}

void Op2c::build_maps()
{
    size_t ntype = tcbd.orb_->ntype();

    tcbd.tabulate();

    // build orbital mapping table: [itype][iorb] -> (l, zeta, m)
    orb_map.resize({(int)ntype, tcbd.orb_->nphi_max(), 3});
    for(size_t itype=0; itype<ntype; ++itype){
        int iphi = 0;
        for(int l=0; l<=tcbd.orb_->lmax(itype); ++l){
            for(int izeta=0; izeta<tcbd.orb_->nzeta(itype, l); ++izeta){
                for(int m=-l; m<=l; ++m){
                    orb_map.get_value<int>(itype, iphi, 0) = l;
                    orb_map.get_value<int>(itype, iphi, 1) = izeta;
                    orb_map.get_value<int>(itype, iphi, 2) = m;
                    iphi++;
                }
            }
        }
    }

    if(tcbd.beta_) {
        // build beta mapping table: [itype][ibeta] -> (l, zeta, m)
        beta_map.resize({(int)ntype, tcbd.beta_->nphi_max(), 3});
        for(size_t itype=0; itype<ntype; ++itype){
            int iphi = 0;
            for(int l=0; l<=tcbd.beta_->lmax(itype); ++l){
                for(int izeta=0; izeta<tcbd.beta_->nzeta(itype, l); ++izeta){
                    for(int m=-l; m<=l; ++m){
                        beta_map.get_value<int>(itype, iphi, 0) = l;
                        beta_map.get_value<int>(itype, iphi, 1) = izeta;
                        beta_map.get_value<int>(itype, iphi, 2) = m;
                        iphi++;
                    }
                }
            }
        }
    }
}

void Op2c::overlap(size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose, std::vector<double>& v, std::vector<double>* dvx, std::vector<double>* dvy, std::vector<double>* dvz)
{
    int inorb, jnorb;
    inorb = tcbd.orb_->nphi(itype);
    jnorb = tcbd.orb_->nphi(jtype);
    int il, izeta, im;
    int jl, jzeta, jm;
    int shift, m_phase;
    double* grad_out=nullptr;

    if (dvx!=nullptr || dvy!=nullptr || dvz!=nullptr){
        grad_out = new double[3];
    }

    if (v.size() < inorb * jnorb){
        std::cout << "Warning: v.size() " << v.size() << " < inorb * jnorb: " << inorb * jnorb << std::endl;
        v.resize(inorb * jnorb);
    } else if (v.size() > inorb * jnorb){
        std::cout << "Warning: v.size() " << v.size() << " > inorb * jnorb: " << inorb * jnorb << std::endl;
    }
    if (dvx != nullptr){
        if (dvx->size() < inorb * jnorb){
            std::cout << "Warning: dvx.size() " << dvx->size() << " < inorb * jnorb: " << inorb * jnorb << std::endl;
            dvx->resize(inorb * jnorb);
        } else if (dvx->size() > inorb * jnorb){
            std::cout << "Warning: dvx.size() " << dvx->size() << " > inorb * jnorb: " << inorb * jnorb << std::endl;
        }
    }
    if (dvy != nullptr){
        if (dvy->size() < inorb * jnorb){
            std::cout << "Warning: dvy.size() " << dvy->size() << " < inorb * jnorb: " << inorb * jnorb << std::endl;
            dvy->resize(inorb * jnorb);
        } else if (dvy->size() > inorb * jnorb){
            std::cout << "Warning: dvy.size() " << dvy->size() << " > inorb * jnorb: " << inorb * jnorb << std::endl;
        }
    }
    if (dvz != nullptr){
        if (dvz->size() < inorb * jnorb){
            std::cout << "Warning: dvz.size() < inorb * jnorb" << std::endl;
            dvz->resize(inorb * jnorb);
        } else if (dvz->size() > inorb * jnorb){
            std::cout << "Warning: dvz.size() > inorb * jnorb" << std::endl;
        }
    }

    for(int i=0; i<inorb; ++i){
        for(int j=0; j<jnorb; ++j){

            if (is_transpose){
                shift = j * inorb + i;
            } else {
                shift = i * jnorb + j;
            }

            il = orb_map.get_value<int>(itype, i, 0);
            izeta = orb_map.get_value<int>(itype, i, 1);
            im = orb_map.get_value<int>(itype, i, 2);
            jl = orb_map.get_value<int>(jtype, j, 0);
            jzeta = orb_map.get_value<int>(jtype, j, 1);
            jm = orb_map.get_value<int>(jtype, j, 2);
            tcbd.overlap_orb->calculate(itype, il, izeta, im, jtype, jl, jzeta, jm, Rij, v.data()+shift, grad_out);
            
            m_phase = (im+jm) % 2 == 0 ? 1 : -1;
            v.data()[shift] *= m_phase;
            if (grad_out != nullptr) {
                grad_out[0] *= m_phase;
                grad_out[1] *= m_phase;
                grad_out[2] *= m_phase;
            }

            if (dvx != nullptr) dvx->data()[shift] = grad_out[0];
            if (dvy != nullptr) dvy->data()[shift] = grad_out[1];
            if (dvz != nullptr) dvz->data()[shift] = grad_out[2];
        }
    }
    if (grad_out != nullptr) delete[] grad_out;
}

void Op2c::kinetic(size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose, std::vector<double>& v, std::vector<double>* dvx, std::vector<double>* dvy, std::vector<double>* dvz)
{
    // Block-by-block kinetic integral, mirroring Op2c::overlap. The only
    // difference is the tabulator: tcbd.kinetic_orb was tabulated with the
    // 'T' tag (op_pk = -2 in table.cpp), so calculate(...) returns
    // < phi_i | -1/2 nabla^2 | phi_j > in Hartree.
    int inorb, jnorb;
    inorb = tcbd.orb_->nphi(itype);
    jnorb = tcbd.orb_->nphi(jtype);
    int il, izeta, im;
    int jl, jzeta, jm;
    int shift, m_phase;
    double* grad_out=nullptr;

    if (dvx!=nullptr || dvy!=nullptr || dvz!=nullptr){
        grad_out = new double[3];
    }

    if (v.size() < inorb * jnorb){
        std::cout << "Warning: v.size() " << v.size() << " < inorb * jnorb: " << inorb * jnorb << std::endl;
        v.resize(inorb * jnorb);
    } else if (v.size() > inorb * jnorb){
        std::cout << "Warning: v.size() " << v.size() << " > inorb * jnorb: " << inorb * jnorb << std::endl;
    }
    if (dvx != nullptr){
        if (dvx->size() < inorb * jnorb){
            dvx->resize(inorb * jnorb);
        }
    }
    if (dvy != nullptr){
        if (dvy->size() < inorb * jnorb){
            dvy->resize(inorb * jnorb);
        }
    }
    if (dvz != nullptr){
        if (dvz->size() < inorb * jnorb){
            dvz->resize(inorb * jnorb);
        }
    }

    for(int i=0; i<inorb; ++i){
        for(int j=0; j<jnorb; ++j){
            if (is_transpose){
                shift = j * inorb + i;
            } else {
                shift = i * jnorb + j;
            }

            il = orb_map.get_value<int>(itype, i, 0);
            izeta = orb_map.get_value<int>(itype, i, 1);
            im = orb_map.get_value<int>(itype, i, 2);
            jl = orb_map.get_value<int>(jtype, j, 0);
            jzeta = orb_map.get_value<int>(jtype, j, 1);
            jm = orb_map.get_value<int>(jtype, j, 2);
            tcbd.kinetic_orb->calculate(itype, il, izeta, im, jtype, jl, jzeta, jm, Rij, v.data()+shift, grad_out);

            m_phase = (im+jm) % 2 == 0 ? 1 : -1;
            v.data()[shift] *= m_phase;
            if (grad_out != nullptr) {
                grad_out[0] *= m_phase;
                grad_out[1] *= m_phase;
                grad_out[2] *= m_phase;
            }

            if (dvx != nullptr) dvx->data()[shift] = grad_out[0];
            if (dvy != nullptr) dvy->data()[shift] = grad_out[1];
            if (dvz != nullptr) dvz->data()[shift] = grad_out[2];
        }
    }
    if (grad_out != nullptr) delete[] grad_out;
}

void Op2c::two_center_batch(int kind,
                            const int* itypes, const int* jtypes, const double* rij,
                            size_t npair,
                            std::vector<double>& flat_out,
                            std::vector<long>& offsets_out)
{
    if (kind != 0 && kind != 1) {
        throw std::invalid_argument(
            "Op2c::two_center_batch: kind must be 0 (overlap) or 1 (kinetic)");
    }

    // Per-pair block sizes + prefix offsets (cheap, serial).
    offsets_out.assign(npair + 1, 0);
    for (size_t p = 0; p < npair; ++p) {
        const int ni = tcbd.orb_->nphi(static_cast<size_t>(itypes[p]));
        const int nj = tcbd.orb_->nphi(static_cast<size_t>(jtypes[p]));
        offsets_out[p + 1] = offsets_out[p] + static_cast<long>(ni) * nj;
    }
    flat_out.assign(static_cast<size_t>(offsets_out[npair]), 0.0);

    // Threaded per-pair evaluation. op2c's two-center eval is thread-safe
    // (const tables; see test_concurrency). An exception in any worker is
    // captured and rethrown after the region (throwing across an OpenMP region
    // is undefined).
    std::atomic<bool> failed{false};
    std::exception_ptr eptr;
    #pragma omp parallel for schedule(dynamic)
    for (long long p = 0; p < static_cast<long long>(npair); ++p) {
        if (failed.load(std::memory_order_relaxed)) continue;
        try {
            const size_t itp = static_cast<size_t>(itypes[p]);
            const size_t jtp = static_cast<size_t>(jtypes[p]);
            const int ni = tcbd.orb_->nphi(itp);
            const int nj = tcbd.orb_->nphi(jtp);
            ModuleBase::Vector3<double> R(rij[3 * p + 0], rij[3 * p + 1], rij[3 * p + 2]);
            std::vector<double> v(static_cast<size_t>(ni) * nj, 0.0);
            if (kind == 0) {
                overlap(itp, jtp, R, false, v, nullptr, nullptr, nullptr);
            } else {
                kinetic(itp, jtp, R, false, v, nullptr, nullptr, nullptr);
            }
            std::copy(v.begin(), v.end(), flat_out.begin() + offsets_out[p]);
        } catch (...) {
            #pragma omp critical
            {
                if (!eptr) eptr = std::current_exception();
            }
            failed.store(true, std::memory_order_relaxed);
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }
}

namespace {
// (i, j, R) block key for V_nl accumulation across projector atoms.
struct VnlKey {
    int i, j, sx, sy, sz;
    bool operator==(const VnlKey& o) const {
        return i == o.i && j == o.j && sx == o.sx && sy == o.sy && sz == o.sz;
    }
};
struct VnlKeyHash {
    std::size_t operator()(const VnlKey& k) const {
        std::size_t h = 1469598103934665603ull;  // FNV-ish combine
        for (int v : {k.i, k.j, k.sx, k.sy, k.sz}) {
            h = (h ^ static_cast<std::size_t>(static_cast<unsigned>(v))) * 1099511628211ull;
        }
        return h;
    }
};
using VnlMap = std::unordered_map<VnlKey, std::vector<double>, VnlKeyHash>;

inline void accumulate_block(VnlMap& acc, const VnlKey& key, const ModuleBase::matrix& block) {
    const std::size_t n = static_cast<std::size_t>(block.nr) * block.nc;
    std::vector<double>& dst = acc[key];
    if (dst.empty()) {
        dst.assign(n, 0.0);
    }
    for (std::size_t e = 0; e < n; ++e) {
        dst[e] += block.c[e];
    }
}
}  // namespace

void Op2c::vnl_batch(const int* k_types, size_t n_K,
                     const long* neigh_off,
                     const int* neigh_gidx, const int* neigh_type,
                     const int* neigh_shift, const double* neigh_disp,
                     int n_types, const int* dm_dim, const double* dm_flat,
                     std::vector<int>& out_i, std::vector<int>& out_j,
                     std::vector<int>& out_shift,
                     std::vector<double>& out_flat,
                     std::vector<long>& out_off)
{
    // Rebuild the per-type m-expanded D matrices (caller passes them as flat
    // row-major data; they depend only on element type).
    std::vector<ModuleBase::matrix> Dmat(static_cast<size_t>(n_types));
    {
        long off = 0;
        for (int t = 0; t < n_types; ++t) {
            const int d = dm_dim[t];
            if (d > 0) {
                Dmat[t].create(d, d, false);
                std::memcpy(Dmat[t].c, dm_flat + off, static_cast<size_t>(d) * d * sizeof(double));
                off += static_cast<long>(d) * d;
            }
        }
    }

    VnlMap global;
    std::atomic<bool> failed{false};
    std::exception_ptr eptr;

    #pragma omp parallel
    {
        VnlMap local;
        #pragma omp for schedule(dynamic)
        for (long long k = 0; k < static_cast<long long>(n_K); ++k) {
            if (failed.load(std::memory_order_relaxed)) continue;
            try {
                const size_t type_K = static_cast<size_t>(k_types[k]);
                const long beg = neigh_off[k];
                const long end = neigh_off[k + 1];
                const int m = static_cast<int>(end - beg);
                if (m <= 0) continue;

                std::vector<size_t> itypes_v(static_cast<size_t>(m));
                std::vector<ModuleBase::Vector3<double>> Ri(static_cast<size_t>(m));
                for (int a = 0; a < m; ++a) {
                    itypes_v[a] = static_cast<size_t>(neigh_type[beg + a]);
                    Ri[a].set(neigh_disp[3 * (beg + a) + 0],
                              neigh_disp[3 * (beg + a) + 1],
                              neigh_disp[3 * (beg + a) + 2]);
                }
                std::vector<ModuleBase::matrix> ob(m), oxb(m), oyb(m), ozb(m);
                ModuleBase::Vector3<double> Rk(0.0, 0.0, 0.0);
                // V_nl needs only the <phi|beta> values, not the position blocks.
                orb_r_beta(itypes_v, type_K, Ri, Rk, false, ob, oxb, oyb, ozb, /*with_grad=*/false);

                const ModuleBase::matrix& D = Dmat[type_K];
                // W[a] = <phi_a|beta_K> * D_K (skip orbitals beyond the table cutoff).
                std::vector<ModuleBase::matrix> W(static_cast<size_t>(m));
                for (int a = 0; a < m; ++a) {
                    if (ob[a].nc > 0) W[a] = ob[a] * D;
                }
                for (int in = 0; in < m; ++in) {
                    if (ob[in].nc == 0) continue;
                    const int gi = neigh_gidx[beg + in];
                    const int six = neigh_shift[3 * (beg + in) + 0];
                    const int siy = neigh_shift[3 * (beg + in) + 1];
                    const int siz = neigh_shift[3 * (beg + in) + 2];
                    for (int jn = 0; jn < m; ++jn) {
                        if (ob[jn].nc == 0) continue;
                        ModuleBase::matrix block = W[in] * transpose(ob[jn]);
                        VnlKey key{gi, neigh_gidx[beg + jn],
                                   neigh_shift[3 * (beg + jn) + 0] - six,
                                   neigh_shift[3 * (beg + jn) + 1] - siy,
                                   neigh_shift[3 * (beg + jn) + 2] - siz};
                        accumulate_block(local, key, block);
                    }
                }
            } catch (...) {
                #pragma omp critical
                {
                    if (!eptr) eptr = std::current_exception();
                }
                failed.store(true, std::memory_order_relaxed);
            }
        }
        #pragma omp critical
        {
            for (auto& kv : local) {
                std::vector<double>& dst = global[kv.first];
                if (dst.empty()) {
                    dst = std::move(kv.second);
                } else {
                    for (std::size_t e = 0; e < dst.size(); ++e) dst[e] += kv.second[e];
                }
            }
        }
    }
    if (eptr) {
        std::rethrow_exception(eptr);
    }

    // Emit the accumulated blocks as flat arrays.
    const size_t n_out = global.size();
    out_i.clear(); out_i.reserve(n_out);
    out_j.clear(); out_j.reserve(n_out);
    out_shift.clear(); out_shift.reserve(3 * n_out);
    out_off.clear(); out_off.reserve(n_out + 1);
    out_flat.clear();
    out_off.push_back(0);
    for (auto& kv : global) {
        out_i.push_back(kv.first.i);
        out_j.push_back(kv.first.j);
        out_shift.push_back(kv.first.sx);
        out_shift.push_back(kv.first.sy);
        out_shift.push_back(kv.first.sz);
        out_flat.insert(out_flat.end(), kv.second.begin(), kv.second.end());
        out_off.push_back(static_cast<long>(out_flat.size()));
    }
}

void Op2c::overlap_position(
    size_t itype, size_t jtype,
    ModuleBase::Vector3<double> Ri, ModuleBase::Vector3<double> Rj,
    bool is_transpose,
    // output
    std::vector<double>& v, std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz
){
    int inorb, jnorb;
    inorb = tcbd.orb_->nphi(itype);
    jnorb = tcbd.orb_->nphi(jtype);
    int il, izeta, im;
    int jl, jzeta, jm;
    int m_phase;
    double *ptr, *ptrx, *ptry, *ptrz;

    if (v.size() < inorb * jnorb){
        v.resize(inorb * jnorb);
    }
    if (vx.size() < inorb * jnorb){
        vx.resize(inorb * jnorb);
    }
    if (vy.size() < inorb * jnorb){
        vy.resize(inorb * jnorb);
    }
    if (vz.size() < inorb * jnorb){
        vz.resize(inorb * jnorb);
    }

    for(int i=0; i<inorb; ++i){
        for(int j=0; j<jnorb; ++j){

            if (is_transpose){
                ptr = v.data() + j * inorb + i;
                ptrx = vx.data() + j * inorb + i;
                ptry = vy.data() + j * inorb + i;
                ptrz = vz.data() + j * inorb + i;
            } else {
                ptr = v.data() + i * jnorb + j;
                ptrx = vx.data() + i * jnorb + j;
                ptry = vy.data() + i * jnorb + j;
                ptrz = vz.data() + i * jnorb + j;
            }

            il = orb_map.get_value<int>(itype, i, 0);
            izeta = orb_map.get_value<int>(itype, i, 1);
            im = orb_map.get_value<int>(itype, i, 2);
            jl = orb_map.get_value<int>(jtype, j, 0);
            jzeta = orb_map.get_value<int>(jtype, j, 1);
            jm = orb_map.get_value<int>(jtype, j, 2);
            tcbd.overlap_orb->calculate(itype, il, izeta, im, jtype, jl, jzeta, jm, Rj-Ri, Rj, ptr, ptrx, ptry, ptrz);

            m_phase = (im+jm) % 2 == 0 ? 1 : -1;
            *ptr *= m_phase;
            *ptrx *= m_phase;
            *ptry *= m_phase;
            *ptrz *= m_phase;
        }
    }
}

void Op2c::orb_r_beta(
    std::vector<size_t>& itype, size_t ktype,
    std::vector<ModuleBase::Vector3<double>> Ri, ModuleBase::Vector3<double> Rk,
    bool is_transpose,
    // output
    std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb,
    std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
    bool with_grad
){
    int inorb, knorb;
    int il, izeta, im;
    int kl, kzeta, km;
    int shift;
    int m_phase;
    double cutoff;
    
    if(!tcbd.beta_) {
        std::cout << "Error: Beta radials not initialized (missing pseudo?)" << std::endl;
        return;
    }
    
    knorb = tcbd.beta_->nphi(ktype);

    // v, vx, vy, vz: itype.size() * (inorb, knorb)
    for (size_t i=0; i<itype.size(); ++i){

        // compute cutoff, and check for the distance of edge ik
        cutoff = this->get_orb_rcut_max(itype[i]) + this->get_beta_rcut_max(ktype);
        if (cutoff < (Rk-Ri[i]).norm()){
            continue;
        }

        inorb = tcbd.orb_->nphi(itype[i]);
        ob[i].create(inorb, knorb, true);
        if (with_grad) {
            oxb[i].create(inorb, knorb, true);
            oyb[i].create(inorb, knorb, true);
            ozb[i].create(inorb, knorb, true);
        }

        for (int io=0; io<inorb; ++io){
            for (int ko=0; ko<knorb; ++ko){
                il = orb_map.get_value<int>(itype[i], io, 0);
                izeta = orb_map.get_value<int>(itype[i], io, 1);
                im = orb_map.get_value<int>(itype[i], io, 2);

                kl = beta_map.get_value<int>(ktype, ko, 0);
                kzeta = beta_map.get_value<int>(ktype, ko, 1);
                km = beta_map.get_value<int>(ktype, ko, 2);

                if (is_transpose){
                    shift = ko * inorb + io;
                } else {
                    shift = io * knorb + ko;
                }

                m_phase = (im+km) % 2 == 0 ? 1 : -1;

                if (with_grad) {
                    // ob = <phi|beta>, oxb/oyb/ozb = <phi|r|beta> (position op,
                    // needs the position-augmented beta tables).
                    tcbd.overlap_orb_beta->calculate(itype[i], il, izeta, im, ktype, kl, kzeta, km, Rk-Ri[i], Rk, ob[i].c+shift, oxb[i].c+shift, oyb[i].c+shift, ozb[i].c+shift);
                    ob[i].c[shift] *= m_phase;
                    oxb[i].c[shift] *= m_phase;
                    oyb[i].c[shift] *= m_phase;
                    ozb[i].c[shift] *= m_phase;
                } else {
                    // values only: <phi|beta>, no position blocks. Uses the plain
                    // overlap path so it works with pm_build=false.
                    tcbd.overlap_orb_beta->calculate(itype[i], il, izeta, im, ktype, kl, kzeta, km, Rk-Ri[i], ob[i].c+shift, nullptr);
                    ob[i].c[shift] *= m_phase;
                }
            }
        }
    }
}

void Op2c::ncomm_IKJ(
    size_t itype, size_t idx, size_t ktype, size_t jtype, size_t jdx, // here idx and jdx are the index of i and j atom 's position in the neighbour list of k
    std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb,  // i and j are atoms with orbitals, k are the atom with beta projectors
    std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
    int npol, bool is_transpose,
    // output
    std::vector<std::complex<double>>& vx, std::vector<std::complex<double>>& vy, std::vector<std::complex<double>>& vz
){
    /*
         <I|K,R><K,R|r|J,R'>-<I|r|K,R><K,R|J,R'> = <I,-R|K>(<K|r-R|J,R'-R>)-(<I,-R|r-R|K>)<K|J,R'-R>
        =<I,-R|K><K|r|J,R'-R>-R<I,-R|K><K|J,R'-R>-<I,-R|r|K><K|J,R'-R>+R<I,-R|K><K|J,R'-R>
        =<I,-R|K><K|r|J,R'-R>-<I,-R|r|K><K|J,R'-R>
        */
    int inorb, jnorb, knorb, shift, s1, s2;
    inorb = tcbd.orb_->nphi(itype);
    jnorb = tcbd.orb_->nphi(jtype);
    
    if(!tcbd.beta_) {
        std::cout << "Error: Beta radials not initialized (missing pseudo?)" << std::endl;
        return;
    }

    knorb = tcbd.beta_->nphi(ktype);


    std::complex<double> imag_unit = std::complex<double>{0, 1};

    // npol is the number of polarizations

    if (vx.size() < inorb * jnorb * npol * npol){
        std::cout << "Warning: vx.size() < inorb * jnorb * npol * npol" << std::endl;
        vx.resize(inorb * jnorb * npol * npol);
    } else if (vx.size() > inorb * jnorb * npol * npol){
        std::cout << "Warning: vx.size() > inorb * jnorb * npol * npol" << std::endl;
    }
    if (vy.size() < inorb * jnorb * npol * npol){
        std::cout << "Warning: vy.size() < inorb * jnorb * npol * npol" << std::endl;
        vy.resize(inorb * jnorb * npol * npol);
    } else if (vy.size() > inorb * jnorb * npol * npol){
        std::cout << "Warning: vy.size() > inorb * jnorb * npol * npol" << std::endl;
        vy.resize(inorb * jnorb * npol * npol);
    }
    if (vz.size() < inorb * jnorb * npol * npol){
        std::cout << "Warning: vz.size() < inorb * jnorb * npol * npol" << std::endl;
        vz.resize(inorb * jnorb * npol * npol);
    } else if (vz.size() > inorb * jnorb * npol * npol){
        std::cout << "Warning: vz.size() > inorb * jnorb * npol * npol" << std::endl;
    }

    for(int i=0; i<inorb; ++i){
        for(int j=0; j<jnorb; ++j){
            for(int is=0; is<npol*npol; ++is){ //(uu, ud, du, dd)
                s1 = is / npol;
                s2 = is % npol;
                std::complex<double> vx_tmp = 0;
                std::complex<double> vy_tmp = 0;
                std::complex<double> vz_tmp = 0;
                for(int s=0; s<psds[ktype].non_zero_count_soc[is]; ++s){
                    const int p1 = psds[ktype].index1_soc[is][s]; // p1 and p2 are spatial orbitals index, without spin
                    const int p2 = psds[ktype].index2_soc[is][s];
                    double tmp = 0;
                    const std::complex<double>* tmp_d_c = nullptr;
                    const double* tmp_d_r = nullptr;
                    
                    if (npol == 2) {
                        this->psds[ktype].get_d(is, p1, p2, tmp_d_c);
                    } else {
                        this->psds[ktype].get_d(is, p1, p2, tmp_d_r);
                    }
                    
                    //<psi|rexp(-iAr)|beta><beta|exp(iAr)|psi>-<psi|exp(-iAr)|beta><beta|rexp(iAr)|psi>
                    // multiply d in the end
                    int idx_i_p1, idx_j_p2;
                    if (is_transpose) {
                        idx_i_p1 = p1 * inorb + i;
                        idx_j_p2 = p2 * jnorb + j;
                    } else {
                        idx_i_p1 = i * knorb + p1;
                        idx_j_p2 = j * knorb + p2;
                    }

                    tmp = (ob[idx].c[idx_i_p1] * oxb[jdx].c[idx_j_p2]
                                    - ob[jdx].c[idx_j_p2] * oxb[idx].c[idx_i_p1]);
                    
                    if (npol == 2 && tmp_d_c) {
                        vx_tmp += tmp * (*tmp_d_c);
                    } else if (tmp_d_r) {
                        vx_tmp += tmp * (*tmp_d_r);
                    }

                    tmp = (ob[idx].c[idx_i_p1] * oyb[jdx].c[idx_j_p2]
                                    - ob[jdx].c[idx_j_p2] * oyb[idx].c[idx_i_p1]);
                    if (npol == 2 && tmp_d_c) {
                        vy_tmp += tmp * (*tmp_d_c);
                    } else if (tmp_d_r) {
                        vy_tmp += tmp * (*tmp_d_r);
                    }

                    tmp = (ob[idx].c[idx_i_p1] * ozb[jdx].c[idx_j_p2]
                                    - ob[jdx].c[idx_j_p2] * ozb[idx].c[idx_i_p1]);
                    if (npol == 2 && tmp_d_c) {
                        vz_tmp += tmp * (*tmp_d_c);
                    } else if (tmp_d_r) {
                        vz_tmp += tmp * (*tmp_d_r);

                        // std::cout << "p1: " << p1 << " p2: " << p2 << "Dnl: " << *tmp_d_r << std::endl;
                    }

                    
                }
                if (is_transpose){
                    shift = (j * npol + s2) * inorb * npol + i * npol + s1;
                } else {
                    shift = (i * npol + s1) * jnorb * npol + j * npol + s2;
                }
                vx[shift] = (imag_unit * vx_tmp / 2.0); // Taking real part as requested by user warning
                vy[shift] = (imag_unit * vy_tmp / 2.0); // Ry to hartree
                vz[shift] = (imag_unit * vz_tmp / 2.0);
            }
        }
    }
    
}

double Op2c::get_orb_rcut_max(int itype) const {
    return tcbd.orb_->rcut_max(itype);
}

double Op2c::get_beta_rcut_max(int itype) const {
    if(tcbd.beta_)
        return tcbd.beta_->rcut_max(itype);
    return 0.0;
}

double Op2c::valence_charge(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::valence_charge: itype out of range");
    }
    return psds[itype].zv;
}

std::vector<double> Op2c::vloc_rgrid(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::vloc_rgrid: itype out of range");
    }
    const Atom_pseudo& pp = psds[itype];
    return pp.vloc_r.empty() ? pp.r : pp.vloc_r;
}

std::vector<double> Op2c::vloc_rab(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::vloc_rab: itype out of range");
    }
    const Atom_pseudo& pp = psds[itype];
    return pp.vloc_rab.empty() ? pp.rab : pp.vloc_rab;
}

int Op2c::vloc_msh(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::vloc_msh: itype out of range");
    }
    const Atom_pseudo& pp = psds[itype];
    if (!pp.vloc_r.empty()) {
        return static_cast<int>(pp.vloc_r.size());
    }
    if (pp.msh > 0) {
        return pp.msh;
    }
    return static_cast<int>(pp.r.size());
}

std::vector<double> Op2c::vloc_at(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::vloc_at: itype out of range");
    }
    const Atom_pseudo& pp = psds[itype];
    return pp.vloc_at_radial.empty() ? pp.vloc_at : pp.vloc_at_radial;
}

std::vector<double> Op2c::atomic_density_rgrid(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::atomic_density_rgrid: itype out of range");
    }
    return psds[itype].r;
}

std::vector<double> Op2c::atomic_density_rab(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::atomic_density_rab: itype out of range");
    }
    return psds[itype].rab;
}

std::vector<double> Op2c::atomic_density_at(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::atomic_density_at: itype out of range");
    }
    return psds[itype].rho_at;
}

double Op2c::short_range_radius(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::short_range_radius: itype out of range");
    }
    return psds[itype].short_range_radius;
}

double Op2c::short_range_charge(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::short_range_charge: itype out of range");
    }
    return psds[itype].short_range_charge;
}

std::vector<double> Op2c::short_range_q_grid(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::short_range_q_grid: itype out of range");
    }
    return psds[itype].short_range_q_grid;
}

std::vector<double> Op2c::short_range_q_weights(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::short_range_q_weights: itype out of range");
    }
    return psds[itype].short_range_q_weights;
}

std::vector<double> Op2c::short_range_fq(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::short_range_fq: itype out of range");
    }
    return psds[itype].short_range_fq;
}

bool Op2c::has_partial_core(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::has_partial_core: itype out of range");
    }
    return psds[itype].nlcc;
}

std::vector<double> Op2c::partial_core_rgrid(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::partial_core_rgrid: itype out of range");
    }
    return psds[itype].r;
}

std::vector<double> Op2c::partial_core_at(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::partial_core_at: itype out of range");
    }
    return psds[itype].rho_atc;
}

int Op2c::beta_nbeta(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::beta_nbeta: itype out of range");
    }
    return psds[itype].nbeta;
}

std::vector<int> Op2c::beta_lll(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::beta_lll: itype out of range");
    }
    const auto& pp = psds[itype];
    std::vector<int> out(pp.nbeta);
    for (int i = 0; i < pp.nbeta; ++i) {
        out[i] = pp.lll[i];
    }
    return out;
}

std::vector<double> Op2c::beta_dion(int itype) const {
    if (itype < 0 || static_cast<size_t>(itype) >= psds.size()) {
        throw std::out_of_range("Op2c::beta_dion: itype out of range");
    }
    const auto& pp = psds[itype];
    const int n = pp.nbeta;
    std::vector<double> out(n * n);
    // Return the channel-space projector strength matrix. ``d_real`` is
    // m-resolved after ``Atom_pseudo::setup_nonlocal``; callers that need the
    // m-resolved matrix expand this channel-space ``dion`` explicitly.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i * n + j] = pp.dion(i, j);
        }
    }
    return out;
}
