#include "int2c/op2c.hpp"
#include "math/linalg/matrix.h"
#include <iostream>
#include <fstream>

Op2c::Op2c(size_t ntype, int nspin, bool lspinorb, 
        const std::string& orb_dir, const std::vector<std::string> orb_name, const std::string& psd_dir, const std::vector<std::string> psd_name,
        MPI_Comm comm, const std::string& log_file
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

    tcbd.build_orb(ntype, orb_name.data(), orb_dir, comm);

    if(!psd_name.empty()) {
        psds.resize(ntype);
        read_pseudo(psd_dir, psd_name, "auto", 10.0, false, 0.0, logger, psds, comm);

        std::vector<BetaRadials> beta_radials(ntype);
        for(size_t itype=0; itype<ntype; ++itype){
            beta_radials[itype] = psds[itype].beta_radials;
        }
        tcbd.build_beta(ntype, beta_radials.data());
    }

    build_maps();
};

Op2c::Op2c(
    std::vector<AtomicRadials> orbitals,
    std::vector<Atom_pseudo> pseudos,
    int nspin, bool lspinorb
    ):
    comm(0), nspin(nspin), lspinorb(lspinorb),
    psds(std::move(pseudos))
{
    size_t ntype = orbitals.size();

    tcbd.build_orb(ntype, orbitals.data());

    if(!psds.empty()) {
        if (psds.size() != ntype){
            throw std::runtime_error("pseudos.size() != orbitals.size()");
        }
        std::vector<BetaRadials> beta_radials(ntype);
        for(size_t itype=0; itype<ntype; ++itype){
            beta_radials[itype] = psds[itype].beta_radials;
        }
        tcbd.build_beta(ntype, beta_radials.data());
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
    std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb
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
        oxb[i].create(inorb, knorb, true);
        oyb[i].create(inorb, knorb, true);
        ozb[i].create(inorb, knorb, true);
        

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
                
                tcbd.overlap_orb_beta->calculate(itype[i], il, izeta, im, ktype, kl, kzeta, km, Rk-Ri[i], Rk, ob[i].c+shift, oxb[i].c+shift, oyb[i].c+shift, ozb[i].c+shift);

                ob[i].c[shift] *= m_phase;
                oxb[i].c[shift] *= m_phase;
                oyb[i].c[shift] *= m_phase;
                ozb[i].c[shift] *= m_phase;
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
    // psds[itype].d_real is a ModuleBase::matrix sized (nbeta, nbeta); copy
    // into a row-major std::vector so the binding can produce a numpy array
    // with shape (nbeta, nbeta).
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i * n + j] = pp.d_real(i, j);
        }
    }
    return out;
}