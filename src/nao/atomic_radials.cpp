#include "nao/atomic_radials.h"

#include "io/rescumat_mat.h"
#include "math/math_integral.h"

// FIXME: should update with pyabacus
// #include "source_io/orb_io.h"

#include "math/projgen.h"

#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <utility>
#include <sstream>
#include <cmath>
#include <cctype>
#include <vector>

namespace
{

#ifdef __MPI
void bcast_vector(std::vector<double>& values, MPI_Comm comm)
{
    int size = static_cast<int>(values.size());
    Parallel_Common::bcast_int(size, comm);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    if (rank != 0)
    {
        values.resize(size);
    }
    if (size > 0)
    {
        Parallel_Common::bcast_double(values.data(), size, comm);
    }
}

void bcast_rescumat_radial(RescumatMat::RadialData& radial, MPI_Comm comm)
{
    Parallel_Common::bcast_int(radial.angular_momentum, comm);
    Parallel_Common::bcast_int(radial.principal_quantum_number, comm);
    Parallel_Common::bcast_double(radial.energy, comm);
    Parallel_Common::bcast_double(radial.population, comm);
    Parallel_Common::bcast_double(radial.kb_energy, comm);
    Parallel_Common::bcast_double(radial.kb_cosine, comm);
    Parallel_Common::bcast_bool(radial.is_ghost, comm);
    bcast_vector(radial.r_grid, comm);
    bcast_vector(radial.r_values, comm);
    bcast_vector(radial.q_grid, comm);
}

void bcast_rescumat_atomic_data(RescumatMat::AtomicData& data, MPI_Comm comm)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    Parallel_Common::bcast_string(data.symbol, comm);
    Parallel_Common::bcast_int(data.atomic_number, comm);
    Parallel_Common::bcast_double(data.valence_electrons, comm);

    int orbital_count = static_cast<int>(data.orbitals.size());
    Parallel_Common::bcast_int(orbital_count, comm);
    if (rank != 0)
    {
        data.orbitals.resize(orbital_count);
    }
    for (auto& orbital : data.orbitals)
    {
        bcast_rescumat_radial(orbital, comm);
    }

    int projector_count = static_cast<int>(data.projectors.size());
    Parallel_Common::bcast_int(projector_count, comm);
    if (rank != 0)
    {
        data.projectors.resize(projector_count);
    }
    for (auto& projector : data.projectors)
    {
        bcast_rescumat_radial(projector, comm);
    }
}
#endif

} // namespace

AtomicRadials& AtomicRadials::operator=(const AtomicRadials& rhs)
{
    RadialSet::operator=(rhs);
    orb_ecut_ = rhs.orb_ecut_;
    return *this;
}

void AtomicRadials::build(const std::string& file, const int itype, const int p, const int pm, const ModuleBase::Logger* ptr_logger, MPI_Comm comm)
{
    // deallocates all arrays and reset variables (excluding sbt_)
    cleanup();
    itype_ = itype;

    if (RescumatMat::has_mat_suffix(file))
    {
        read_rescumat_mat(file, p, pm, ptr_logger, comm);
        set_rcut_max();
        return;
    }

    {
        std::string lowered = file;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (lowered.size() >= 4 && lowered.substr(lowered.size() - 4) == ".ion")
        {
            read_siesta_ion(file, p, pm, ptr_logger, comm);
            set_rcut_max();
            return;
        }
    }

    std::ifstream ifs;
    bool is_open = false;
    int rank = 0;
#ifdef __MPI
    MPI_Comm_rank(comm, &rank);
#endif

    if (rank == 0)
    {
        ifs.open(file);
        is_open = ifs.is_open();
    }

#ifdef __MPI
    Parallel_Common::bcast_bool(is_open, comm);
#endif

    if (!is_open)
    {
        std::cout << "AtomicRadials::build, Couldn't open orbital file: " + file << std::endl;
    }

    if (ptr_logger)
    {
        ptr_logger->info() << "\n\n\n\n";
        ptr_logger->info() << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        ptr_logger->info() << " |                                                                   |" << std::endl;
        ptr_logger->info() << " |               SETUP NUMERICAL ATOMIC ORBITALS                     |" << std::endl;
        ptr_logger->info() << " |                                                                   |" << std::endl;
        ptr_logger->info() << " | Orbital information includes the cutoff radius, angular momentum, |" << std::endl;
        ptr_logger->info() << " | zeta number and numerical values on a radial grid.                |" << std::endl;
        ptr_logger->info() << " |                                                                   |" << std::endl;
        ptr_logger->info() << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        ptr_logger->info() << "\n\n\n\n";
    }

    read_abacus_orb(ifs, p, pm, ptr_logger, comm);
    set_rcut_max();

    if (rank == 0)
    {
        ifs.close();
    }
}

void AtomicRadials::read_rescumat_mat(const std::string& file,
                                      const int p,
                                      const int pm,
                                      const ModuleBase::Logger* ptr_logger,
                                      MPI_Comm comm)
{
    int rank = 0;
#ifdef __MPI
    MPI_Comm_rank(comm, &rank);
#endif

    if (ptr_logger)
    {
        ptr_logger->info() << "\n\n\n\n";
        ptr_logger->info() << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        ptr_logger->info() << " |               SETUP RESCUMAT MAT ORBITALS                       |" << std::endl;
        ptr_logger->info() << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        ptr_logger->info() << "\n\n\n\n";
    }

    RescumatMat::AtomicData data;
    if (rank == 0)
    {
        data = RescumatMat::read_atomic_data(file);
    }
#ifdef __MPI
    bcast_rescumat_atomic_data(data, comm);
#endif

    symbol_ = data.symbol;
    orb_ecut_ = RescumatMat::infer_energy_cutoff_ry(data.orbitals);

    std::vector<RescumatMat::RadialData> shifted_orbitals;
    shifted_orbitals.reserve(data.orbitals.size());
    for (const auto& orbital : data.orbitals)
    {
        RescumatMat::RadialData shifted = orbital;
        shifted.angular_momentum += pm;
        if (shifted.angular_momentum >= 0)
        {
            shifted_orbitals.push_back(std::move(shifted));
        }
    }
    if (shifted_orbitals.empty())
    {
        throw std::runtime_error("rescumat MAT orbital file has no orbitals after angular momentum shift: " + file);
    }

    lmax_ = -1;
    lmin_ = 99999;
    for (const auto& orbital : shifted_orbitals)
    {
        lmax_ = std::max(lmax_, orbital.angular_momentum);
        lmin_ = std::min(lmin_, orbital.angular_momentum);
    }

    nzeta_ = new int[lmax_ + 1];
    norb_ = new int[lmax_ + 1];
    std::fill(nzeta_, nzeta_ + lmax_ + 1, 0);
    std::fill(norb_, norb_ + lmax_ + 1, 0);
    for (const auto& orbital : shifted_orbitals)
    {
        nzeta_[orbital.angular_momentum] += 1;
    }

    nchi_ = 0;
    nphi_ = 0;
    nzeta_max_ = 0;
    for (int l = 0; l <= lmax_; ++l)
    {
        nchi_ += nzeta_[l];
        norb_[l] = nzeta_[l] * (2 * l + 1);
        nphi_ += norb_[l];
        nzeta_max_ = std::max(nzeta_max_, nzeta_[l]);
    }
    indexing();

    chi_ = new NumericalRadial[nchi_];
    std::vector<int> next_zeta(lmax_ + 1, 0);
    for (const auto& orbital : shifted_orbitals)
    {
        const int l = orbital.angular_momentum;
        const int izeta = next_zeta[l]++;
        chi_[index(l, izeta)].build(
            l,
            true,
            static_cast<int>(orbital.r_grid.size()),
            orbital.r_grid.data(),
            orbital.r_values.data(),
            p,
            izeta,
            symbol_,
            itype_,
            false
        );
        chi_[index(l, izeta)].normalize();
    }
}

void AtomicRadials::read_siesta_ion(const std::string& file,
                                    const int p,
                                    const int pm,
                                    const ModuleBase::Logger* ptr_logger,
                                    MPI_Comm comm)
{
    int rank = 0;
#ifdef __MPI
    MPI_Comm_rank(comm, &rank);
#endif

    if (ptr_logger)
    {
        ptr_logger->info() << "\n\n\n\n";
        ptr_logger->info() << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        ptr_logger->info() << " |               SETUP SIESTA .ion PAO ORBITALS                    |" << std::endl;
        ptr_logger->info() << " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
        ptr_logger->info() << "\n\n\n\n";
    }

    // Per-orbital parsed data (rank 0 fills, then broadcasts). chi = f(r) * r^l is
    // the true radial function (SIESTA .ion stores the reduced f with chi = f*r^l,
    // normalized int chi^2 r^2 dr = 1 — same convention as ABACUS .orb stores chi).
    std::string symbol;
    std::vector<int> orb_l;                     // physical angular momentum per orbital
    std::vector<std::vector<double>> orb_r;     // native (uniform) r grid per orbital
    std::vector<std::vector<double>> orb_chi;   // chi(r) = f(r) * r^l per orbital

    if (rank == 0)
    {
        std::ifstream ifs(file);
        if (!ifs)
        {
            throw std::runtime_error("AtomicRadials::read_siesta_ion: cannot open " + file);
        }
        std::string line;
        bool in_paos = false;
        while (std::getline(ifs, line))
        {
            // Element symbol: the "<Sym>   # Symbol" line in the .ion header.
            if (symbol.empty() && line.find("# Symbol") != std::string::npos)
            {
                std::istringstream iss(line);
                iss >> symbol;
            }
            if (line.find("# PAOs") != std::string::npos)
            {
                in_paos = true;
                continue;
            }
            if (!in_paos)
            {
                continue;
            }
            // A PAO header is "l n z is_polarized population"; the PAO section ends at
            // the next "#"-prefixed marker (e.g. "# KBs:") or a non-conforming line.
            {
                std::string trimmed = line;
                std::size_t first = trimmed.find_first_not_of(" \t");
                if (first == std::string::npos || trimmed[first] == '#')
                {
                    break;
                }
            }
            int l = 0, n = 0, z = 0, ispol = 0;
            double pop = 0.0;
            {
                std::istringstream iss(line);
                if (!(iss >> l >> n >> z >> ispol >> pop))
                {
                    break; // not a PAO header -> end of PAO section
                }
            }
            // "npts  delta  cutoff"
            int npts = 0;
            double delta = 0.0, cutoff = 0.0;
            if (!std::getline(ifs, line))
            {
                throw std::runtime_error("read_siesta_ion: truncated PAO header in " + file);
            }
            {
                std::istringstream iss(line);
                iss >> npts >> delta >> cutoff;
            }
            if (npts < 2)
            {
                throw std::runtime_error("read_siesta_ion: bad npts in " + file);
            }
            std::vector<double> rg(npts), chi(npts);
            for (int i = 0; i < npts; ++i)
            {
                if (!std::getline(ifs, line))
                {
                    throw std::runtime_error("read_siesta_ion: truncated PAO data in " + file);
                }
                std::istringstream iss(line);
                double rr = 0.0, ff = 0.0;
                iss >> rr >> ff;
                rg[i] = rr;
                chi[i] = ff * std::pow(rr, l); // f(r) * r^l = chi(r)
            }
            orb_l.push_back(l);
            orb_r.push_back(std::move(rg));
            orb_chi.push_back(std::move(chi));
        }
        if (orb_l.empty())
        {
            throw std::runtime_error("read_siesta_ion: no PAO orbitals found in " + file);
        }
    }

    int n_orb = static_cast<int>(orb_l.size());
#ifdef __MPI
    Parallel_Common::bcast_string(symbol, comm);
    Parallel_Common::bcast_int(n_orb, comm);
    if (rank != 0)
    {
        orb_l.assign(n_orb, 0);
        orb_r.assign(n_orb, {});
        orb_chi.assign(n_orb, {});
    }
    for (int i = 0; i < n_orb; ++i)
    {
        int ng = (rank == 0) ? static_cast<int>(orb_r[i].size()) : 0;
        Parallel_Common::bcast_int(orb_l[i], comm);
        Parallel_Common::bcast_int(ng, comm);
        if (rank != 0)
        {
            orb_r[i].assign(ng, 0.0);
            orb_chi[i].assign(ng, 0.0);
        }
        Parallel_Common::bcast_double(orb_r[i].data(), ng, comm);
        Parallel_Common::bcast_double(orb_chi[i].data(), ng, comm);
    }
#endif

    symbol_ = symbol;
    orb_ecut_ = 0.0; // SIESTA .ion carries no ABACUS-style energy cutoff (informational only)

    // Apply the angular-momentum shift (pm), dropping any orbital pushed below l=0. The
    // radial chi is unchanged by the shift (it was computed with the physical l), matching
    // read_rescumat_mat.
    std::vector<int> L;
    std::vector<int> src;
    L.reserve(n_orb);
    src.reserve(n_orb);
    for (int i = 0; i < n_orb; ++i)
    {
        int l = orb_l[i] + pm;
        if (l < 0)
        {
            continue;
        }
        L.push_back(l);
        src.push_back(i);
    }
    if (L.empty())
    {
        throw std::runtime_error("read_siesta_ion: no orbitals left after angular-momentum shift: " + file);
    }

    lmax_ = -1;
    lmin_ = 99999;
    for (int l : L)
    {
        lmax_ = std::max(lmax_, l);
        lmin_ = std::min(lmin_, l);
    }

    nzeta_ = new int[lmax_ + 1];
    norb_ = new int[lmax_ + 1];
    std::fill(nzeta_, nzeta_ + lmax_ + 1, 0);
    std::fill(norb_, norb_ + lmax_ + 1, 0);
    for (int l : L)
    {
        nzeta_[l] += 1;
    }

    nchi_ = 0;
    nphi_ = 0;
    nzeta_max_ = 0;
    for (int l = 0; l <= lmax_; ++l)
    {
        nchi_ += nzeta_[l];
        norb_[l] = nzeta_[l] * (2 * l + 1);
        nphi_ += norb_[l];
        nzeta_max_ = std::max(nzeta_max_, nzeta_[l]);
    }
    indexing();

    // Common uniform grid for ALL orbitals (r0=0, dr=0.01, cutoff = max orbital cutoff
    // rounded up to a dr multiple). The shared uniform grid is required for the fast paths:
    // the FFT-based SBT in the two-center tables and the O(1)-index orbital evaluator
    // (orbital_evaluator.cpp). set_uniform_grid resamples each native PAO onto it via op2c's
    // own cubic spline (the same operation the bundle does for .orb).
    const double table_dr = 0.01;
    double max_cut = 0.0;
    for (int i : src)
    {
        if (!orb_r[i].empty())
        {
            max_cut = std::max(max_cut, orb_r[i].back());
        }
    }
    const int common_nr = static_cast<int>(std::ceil(max_cut / table_dr)) + 1;
    const double common_cut = (common_nr - 1) * table_dr;

    chi_ = new NumericalRadial[nchi_];
    std::vector<int> next_zeta(lmax_ + 1, 0);
    for (std::size_t k = 0; k < L.size(); ++k)
    {
        const int l = L[k];
        const int i = src[k];
        const int izeta = next_zeta[l]++;
        const int idx = index(l, izeta);
        chi_[idx].build(l,
                        true,
                        static_cast<int>(orb_r[i].size()),
                        orb_r[i].data(),
                        orb_chi[i].data(),
                        p,
                        izeta,
                        symbol_,
                        itype_,
                        false);
        // Normalize onto the shared uniform grid (cubic-spline interpolation, zero-padded
        // beyond each orbital's own cutoff).
        chi_[idx].set_uniform_grid(true, common_nr, common_cut, 'i', false);
    }
}

void AtomicRadials::build(const RadialSet* const other, const int itype, const int p, const int pm, const double rcut)
{
    this->symbol_ = other->symbol();
    this->lmax_ = other->lmax() + pm; // Adjust lmax
    this->lmin_ = std::max(0, other->lmin() + pm); // Adjust lmin
    this->nchi_ = other->nchi();
    this->nphi_ = other->nphi(); // This might need adjustment if norb changes due to l shift
    this->nzeta_max_ = other->nzeta_max();
    this->itype_ = itype;
    this->symbol_ = other->symbol();
    this->nzeta_ = new int[this->lmax_ + 1];
    std::fill(this->nzeta_, this->nzeta_ + this->lmax_ + 1, 0);
    
    // Recalculate nzeta and norb based on shifted l
    // Note: We assume the input 'other' has orbitals that can be mapped to l+pm
    // If l+pm < 0, we should probably discard or handle error.
    // For now, assume valid shifts.
    for (int l = 0; l <= other->lmax(); ++l)
    {
        int l_new = l + pm;
        if (l_new >= 0) {
            this->nzeta_[l_new] = other->nzeta(l);
        }
    }
    
    this->indexing();
    this->chi_ = new NumericalRadial[nchi_];
    
    // We need to iterate over 'other' and map to 'this'
    int ichi_this = 0;
    for (int ichi = 0; ichi < other->nchi(); ichi++)
    {
        const NumericalRadial& chi_other = other->cbegin()[ichi];
        int l_old = chi_other.l();
        int l_new = l_old + pm;
        
        if (l_new < 0) continue; // Skip if l becomes negative
        
        int ngrid = chi_other.nr();
        const double* rgrid = chi_other.rgrid();
        const double* rvalue = chi_other.rvalue();
        const int izeta = chi_other.izeta();
        
        // Use the provided p, or keep original? 
        // The requirement implies we want to SET p.
        int p_new = p; 

        // if the cutoff radius is larger than the original one, just copy the orbitals
        if (rcut < 0.0 || rcut >= chi_other.rcut())
        {
            this->chi_[ichi_this].build(l_new, true, ngrid, rgrid, rvalue, p_new, izeta, symbol_, itype, false);
        }
        else
        {
            // call smoothgen to modify the orbitals to the local projections
            std::vector<double> rvalue_new;
            smoothgen(ngrid, rgrid, rvalue, rcut, rvalue_new);
            ngrid = rvalue_new.size();
            // projgen(l, ngrid, rgrid, rvalue, rcut, 20, rvalue_new);
            // build the new on-site orbitals
            this->chi_[ichi_this].build(l_new, true, ngrid, rgrid, rvalue_new.data(), p_new, izeta, symbol_, itype, false);
        }
        ichi_this++;
    }
    // Update nchi_ to actual number of orbitals (in case some were skipped)
    this->nchi_ = ichi_this;
    
    // Recalculate nphi
    this->nphi_ = 0;
    this->norb_ = new int[this->lmax_ + 1];
    for(int l=0; l<=this->lmax_; ++l) {
        this->norb_[l] = this->nzeta_[l] * (2*l+1);
        this->nphi_ += this->norb_[l];
    }
}

void AtomicRadials::read_abacus_orb(std::ifstream& ifs, const int p, const int pm, const ModuleBase::Logger* ptr_logger, MPI_Comm comm)
{
    /*
     * Read the orbital file.
     *
     * For orbital file format, see
     * (new) abacus-develop/tools/SIAB/PyTorchGradient/source/IO/print_orbital.py
     * (old) abacus-develop/tools/SIAB/SimulatedAnnealing/source/src_spillage/Plot_Psi.cpp
     *                                                                                  */
    int ngrid = 0; // number of grid points
    double dr = 0; // grid spacing
    std::string tmp;
    int rank = 0;
#ifdef __MPI
    MPI_Comm_rank(comm, &rank);
#endif

    if (rank == 0)
    {
        /*
         * read the header & grid information, including
         *
         * 1. element symbol --> symbol_
         * 2. energy cutoff --> orb_ecut_
         * 3. maximum angular momentum --> lmax_
         * 4. number of radial functions for each angular momentum --> nzeta_
         * 5. number of grid points --> ngrid
         * 6. grid spacing --> dr
         *                                                                              */
        while (ifs >> tmp)
        {
            if (tmp == "Element")
            {
                ifs >> symbol_;
            }
            else if (tmp == "Cutoff(Ry)")
            {
                ifs >> orb_ecut_;
            }
            else if (tmp == "Lmax")
            {
                ifs >> lmax_;
#ifdef __DEBUG
                assert(lmax_ >= 0);
#endif
                lmax_ += pm;
                lmin_ = (pm >= 0) ? pm : 0;

                nzeta_ = new int[lmax_ + 1];
                norb_ = new int[lmax_ + 1];
                std::fill(nzeta_, nzeta_+lmax_+1, 0);
                std::fill(norb_, norb_+lmax_+1, 0);
                for (int l = pm; l <= lmax_; ++l)
                {
                    if (l >= 0)
                    {
                        ifs >> tmp >> tmp >> tmp >> nzeta_[l]; // skip "Number" "of" "Xorbital-->"
                        norb_[l] = nzeta_[l] * (2*l+1);
                    } else {
                        ifs >> tmp >> tmp >> tmp >> tmp; // tmp ignore this orbital
                    }
                }
            }
            else if (tmp == "Mesh")
            {
                ifs >> ngrid;
                continue;
            }
            else if (tmp == "dr")
            {
                ifs >> dr;
                break;
            }
        }

        /*
         * calculate:
         *
         * 1. the total number of radial functions --> nchi_
         * 2. maximum number of radial functions for each angular momentum --> nzeta_max_
         * 3. a map from (l, izeta) to 1-d array index in chi_
         *                                                                              */
        nchi_ = 0;
        nphi_ = 0;
        for (int l = lmin_; l <= lmax_; ++l)
        {
            nchi_ += nzeta_[l];
            nphi_ += norb_[l];
        }
        nzeta_max_ = *std::max_element(nzeta_+lmin_, nzeta_ + lmax_ + 1);
        indexing(); // build index_map_
    }

#ifdef __MPI
    Parallel_Common::bcast_string(symbol_, comm);
    Parallel_Common::bcast_double(orb_ecut_, comm);
    Parallel_Common::bcast_int(lmax_, comm);
    Parallel_Common::bcast_int(lmin_, comm);

    Parallel_Common::bcast_int(nchi_, comm);
    Parallel_Common::bcast_int(nphi_, comm);
    Parallel_Common::bcast_int(nzeta_max_, comm);

    Parallel_Common::bcast_int(ngrid, comm);
    Parallel_Common::bcast_double(dr, comm);
#endif

    if (rank != 0)
    {
        nzeta_ = new int[lmax_ + 1];
        norb_ = new int[lmax_ + 1];
        index_map_ = new int[(lmax_ + 1) * nzeta_max_];
    }

#ifdef __MPI
    Parallel_Common::bcast_int(nzeta_, lmax_ + 1, comm);
    Parallel_Common::bcast_int(norb_, lmax_ + 1, comm);
    Parallel_Common::bcast_int(index_map_, (lmax_ + 1) * nzeta_max_, comm);
#endif

    std::vector<double> rvalue(ngrid);
    std::vector<double> rgrid(ngrid);
    for (int ir = 0; ir != ngrid; ++ir)
    {
        rgrid[ir] = ir * dr;
    }
    chi_ = new NumericalRadial[nchi_];

    // record whether an orbital has been read or not
    bool* is_read = new bool[nchi_];
    for (int i = 0; i != nchi_; ++i)
    {
        is_read[i] = false;
    }

    for (int l = lmin_; l <= lmax_; ++l)
    {
        for (int izeta = 0; izeta < nzeta_[l]; ++izeta)
        {
            if (rank == 0)
            {
                /*
                 * read the orbital information, including
                 *
                 * 1. angular momentum
                 * 2. zeta number
                 * 3. values on the grid
                 *                                                                              */
                while (ifs.good())
                {
                    while (ifs >> tmp)
                    {
                        if (tmp == "N")
                        {
                            break;
                        }
                    }
                    int read_l, read_izeta;
                    ifs >> tmp >> read_l >> read_izeta;
                    if (l == read_l+pm && izeta == read_izeta)
                    {
                        break;
                    }
                }

                for (int ir = 0; ir != ngrid; ++ir)
                {
                    ifs >> rvalue[ir];
                }
            }

#ifdef __MPI
            Parallel_Common::bcast_double(rvalue.data(), ngrid, comm);
#endif
#ifdef __DEBUG
            assert(index(l, izeta) >= 0 && index(l, izeta) < nchi_);
            assert(!is_read[index(l, izeta)]);
#endif
            is_read[index(l, izeta)] = true;

            // skip the initialization of sbt_ in this stage
            chi_[index(l, izeta)].build(l, true, ngrid, rgrid.data(), rvalue.data(), p, izeta, symbol_, itype_, false);
            chi_[index(l, izeta)].normalize();
        }
    }

    delete[] is_read;
}

// FIXME: should update with pyabacus
// void AtomicRadials::read_abacus_orb(std::ifstream& ifs, std::ofstream* ptr_log, const int rank)
// {
//     /*
//      * Read the orbital file.
//      *
//      * For orbital file format, see
//      * (new) abacus-develop/tools/SIAB/PyTorchGradient/source/IO/print_orbital.py
//      * (old) abacus-develop/tools/SIAB/SimulatedAnnealing/source/src_spillage/Plot_Psi.cpp
//      *                                                                                  */
//     int ngrid = 0; // number of grid points
//     double dr = 0; // grid spacing
//     std::string tmp;

//     int nr;
//     std::vector<int> nzeta;
//     std::vector<std::vector<double>> radials;

//     ModuleIO::read_abacus_orb(ifs, symbol_, orb_ecut_, nr, dr, nzeta, radials, rank);

//     lmax_ = nzeta.size() - 1;
//     nzeta_ = new int[lmax_ + 1];
//     std::copy(nzeta.begin(), nzeta.end(), nzeta_);

//     nchi_ = std::accumulate(nzeta.begin(), nzeta.end(), 0);
//     nzeta_max_ = *std::max_element(nzeta.begin(), nzeta.end());

//     indexing();
    
//     std::vector<double> rgrid(nr);
//     std::iota(rgrid.begin(), rgrid.end(), 0);
//     std::for_each(rgrid.begin(), rgrid.end(), [dr](double& r) { r *= dr; });
//     chi_ = new NumericalRadial[nchi_];
//     int ichi = 0;
//     for (int l = 0; l <= lmax_; ++l)
//     {
//         for (int izeta = 0; izeta < nzeta[l]; ++izeta)
//         {
//             chi_[index(l, izeta)].build(l, true, nr, rgrid.data(), radials[ichi].data(), 0, izeta, symbol_, itype_, false);
//             chi_[index(l, izeta)].normalize();
//             ++ichi;
//         }
//     }
// }
