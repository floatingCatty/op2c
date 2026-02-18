#include "pseudopotential/beta_projectors.h"
#include "math/projgen.h"

// #include "source_io/module_parameter/parameter.h"
// #include "source_base/global_variable.h"
#include "utils/parallel_common.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <cstring>
#include <cmath>
// #include "source_base/tool_quit.h"
#ifdef __MPI
#include <mpi.h>
#endif

void BetaRadials::build(const NumericalRadial* nl, const int nchi, const int itype, const ModuleBase::Logger* const ptr_logger, MPI_Comm comm)
{
    cleanup();

    itype_ = itype;
    nchi_ = nchi;

    symbol_ = nl[0].symbol();
    lmax_ = nl[0].l();
    lmin_ = nl[0].l();
    for (int i=0; i<nchi; ++i){
        if (symbol_!=nl[i].symbol()) {
            throw std::runtime_error("The collected numerical radial contains more than 1 element type, please check!");
        }
        lmax_ = std::max(lmax_, nl[i].l());
        lmin_ = std::min(lmin_, nl[i].l());
    }

    chi_ = new NumericalRadial[nchi_];
    nzeta_ = new int[lmax_ + 1];
    std::fill(nzeta_, nzeta_ + lmax_ + 1, 0);

    norb_ = new int[lmax_ + 1];
    std::fill(norb_, norb_ + lmax_ + 1, 0);
    nphi_ = 0;

    for (int ichi = 0; ichi != nchi_; ++ichi)
    {
        chi_[ichi] = nl[ichi];
        int l = chi_[ichi].l();
        // skip the initialization of sbt_ in this stage
        // check the numerical radials if it is correct.
        if (chi_[ichi].nr() < 0 || chi_[ichi].pr() != 1) {
            throw std::runtime_error("The collected numerical radial are not strictly for beta functions.");
        }
        nzeta_[l] += 1;
        norb_[l] = nzeta_[l] * (2*l+1);
        nphi_ += 2*l+1;
    }
    nzeta_max_ = *std::max_element(nzeta_, nzeta_ + lmax_ + 1);

    indexing();
    set_rcut_max();
}

void BetaRadials::build(const RadialSet* const other, const int itype, const int p, const int pm, const double rcut)
{
    cleanup();

    itype_ = itype;
    nchi_ = other->nchi();

    symbol_ = other->symbol();
    lmax_ = other->lmax() + pm;
    lmin_ = std::max(0, other->lmin() + pm);
    
    // Check if lmax is valid
    if (lmax_ < 0) {
        // This might happen if all l+pm < 0
        // Handle gracefully or throw?
        // For now, proceed, nzeta_ will be empty or small
    }

    chi_ = new NumericalRadial[nchi_];
    nzeta_ = new int[lmax_ + 1];
    std::fill(nzeta_, nzeta_ + lmax_ + 1, 0);

    norb_ = new int[lmax_ + 1];
    std::fill(norb_, norb_ + lmax_ + 1, 0);
    nphi_ = 0;

    int ichi_this = 0;
    for (int ichi = 0; ichi != nchi_; ++ichi)
    {
        const NumericalRadial& chi_other = other->cbegin()[ichi];
        int l_old = chi_other.l();
        int l_new = l_old + pm;
        
        if (l_new < 0) continue;

        int ngrid = chi_other.nr();
        const double* rgrid = chi_other.rgrid();
        const double* rvalue = chi_other.rvalue();
        const int izeta = chi_other.izeta();
        
        // Use provided p
        int p_new = p;

        // if the cutoff radius is larger than the original one, just copy the orbitals
        if (rcut < 0.0 || rcut >= chi_other.rcut())
        {
            chi_[ichi_this].build(l_new, true, ngrid, rgrid, rvalue, p_new, izeta, symbol_, itype_, false);
        }
        else
        {
            // call smoothgen to modify the orbitals to the local projections
            std::vector<double> rvalue_new;
            smoothgen(ngrid, rgrid, rvalue, rcut, rvalue_new);
            ngrid = rvalue_new.size();
            // projgen(l, ngrid, rgrid, rvalue, rcut, 20, rvalue_new);
            // build the new on-site orbitals
            chi_[ichi_this].build(l_new, true, ngrid, rgrid, rvalue_new.data(), p_new, izeta, symbol_, itype_, false);
        }
        
        // Check if build was successful (e.g. valid l)
        // NumericalRadial::build asserts l>=0
        
        nzeta_[l_new] += 1;
        norb_[l_new] = nzeta_[l_new] * (2*l_new+1);
        nphi_ += 2*l_new+1;
        ichi_this++;
    }
    
    // Update nchi_
    nchi_ = ichi_this;
    
    nzeta_max_ = 0;
    if (lmax_ >= 0) {
        nzeta_max_ = *std::max_element(nzeta_, nzeta_ + lmax_ + 1);
    }
    
    // Re-calculate lmin_ properly from the built orbitals
    lmin_ = 99999;
    for(int i=0; i<nchi_; ++i) {
        lmin_ = std::min(lmin_, chi_[i].l());
    }
    if (nchi_ == 0) lmin_ = 0; // or -1?

    indexing();
    set_rcut_max();
}

