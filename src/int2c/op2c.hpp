/*
    This file implement the operators that the 2-center integrals compute, includeing:
    1. The overlap operator
    2. The position operator
    3. The velocity operator

    More can be computed, such as:
    1. The non-local pesudopotential operator
    2. The kinetic operator
    ...

    The operator will be computed with a real space format, which gives a index - subblock relation:
    For example: 
        overlap, position, kinetic: IJR -> block
        velocity: IKJRR' -> block
*/
#include "int2c/bundle.h"
#include "pseudopotential/pseudo_atom.h"
#include "pseudopotential/io/read_pseudo.h"
#include "utils/container/ATen/tensor.h"
#include "nao/atomic_radials.h"
#include <vector>
#include <complex>
#ifdef __MPI
    #include <mpi.h>
#else
    typedef int MPI_Comm;
    #define MPI_COMM_WORLD 0
#endif


/*!
 * @class Op2c
 * @brief High-level interface for computing two-center integrals in ABACUS.
 *
 * This class coordinates the calculation of:
 * - Overlap integrals: < phi_i | phi_j >
 * - Position operator integrals: < phi_i | r | phi_j >
 * - Non-local pseudopotential interactions: < phi_i | beta >
 * - Kinetic energy and other operators (extensible).
 *
 * It manages the `TwoCenterBundle` which handles lower-level radial tabulation and integration.
 * The operators are computed in a real-space block format, suitable for assembling the 
 * Hamiltonian and Overlap matrices in linear-scaling or plane-wave DFT.
 */
class Op2c
{
private:
    std::vector<Atom_pseudo> psds;
    container::Tensor orb_map{container::DataType::DT_INT, container::TensorShape({0})};
    container::Tensor beta_map{container::DataType::DT_INT, container::TensorShape({0})};
    MPI_Comm comm;
    int rank;
    int nspin;
    bool lspinorb;

    /// Shared init: tabulate + build orb_map/beta_map
    void build_maps();
    
public:
    TwoCenterBundle tcbd;

    /*!
     * @brief Constructs the Op2c operator handler.
     *
     * @param ntype Number of atom types (elements).
     * @param nspin Number of spin components (1 or 2).
     * @param lspinorb True if spin-orbit coupling is enabled.
     * @param orb_dir Directory containing numerical atomic orbital files.
     * @param orb_name List of filenames for orbitals of each element type.
     * @param psd_dir Directory containing pseudopotential files.
     * @param psd_name List of filenames for pseudopotentials of each element type.
     * @param comm MPI Communicator for parallel execution.
     * @param log_file Path to local log file (optional).
     */
    Op2c(
        size_t ntype, int nspin, bool lspinorb,
        const std::string& orb_dir, const std::vector<std::string> orb_name, const std::string& psd_dir, const std::vector<std::string> psd_name,
        MPI_Comm comm, const std::string& log_file,
        bool pm_build = true
    );

    /*!
     * @brief Constructs Op2c from pre-loaded orbitals and pseudopotentials.
     *
     * @param orbitals Pre-loaded AtomicRadials objects (one per atom type).
     * @param pseudos Pre-loaded Atom_pseudo objects (moved in).
     * @param nspin Number of spin components.
     * @param lspinorb Spin-orbit coupling flag.
     */
    Op2c(
        std::vector<AtomicRadials> orbitals,
        std::vector<Atom_pseudo> pseudos,
        int nspin, bool lspinorb,
        bool pm_build = true
    );

    ~Op2c() = default;

    /*!
     * @brief Computes the overlap integral S_ij = < phi_i | phi_j >.
     *
     * @param itype Element type index of atom i.
     * @param jtype Element type index of atom j.
     * @param Rij Displacement vector R_j - R_i.
     * @param is_transpose If true, the result blocks are transposed.
     * @param[out] v Flattened output array for S values. Size mapping: [block_i, block_j].
     * @param[out] dvx (Optional) x-component of the gradient dS/dR.
     * @param[out] dvy (Optional) y-component of the gradient dS/dR.
     * @param[out] dvz (Optional) z-component of the gradient dS/dR.
     */
    void overlap(size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose, std::vector<double>& v, std::vector<double>* dvx = nullptr, std::vector<double>* dvy = nullptr, std::vector<double>* dvz = nullptr);

    /*!
     * @brief Computes the kinetic-energy integral T_ij = < phi_i | -1/2 nabla^2 | phi_j >.
     *
     * Hartree atomic units; the prefactor -1/2 is included in the 'T' tabulation
     * tag of TwoCenterIntegrator. The block layout mirrors overlap() exactly so
     * callers can sum H = T + V_loc + V_nl block-by-block.
     *
     * @param itype Element type index of atom i.
     * @param jtype Element type index of atom j.
     * @param Rij Displacement vector R_j - R_i.
     * @param is_transpose If true, the result blocks are transposed.
     * @param[out] v Flattened output array for T values. Size [block_i * block_j].
     * @param[out] dvx (Optional) x-component of the gradient dT/dR.
     * @param[out] dvy (Optional) y-component of the gradient dT/dR.
     * @param[out] dvz (Optional) z-component of the gradient dT/dR.
     */
    void kinetic(size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose, std::vector<double>& v, std::vector<double>* dvx = nullptr, std::vector<double>* dvy = nullptr, std::vector<double>* dvz = nullptr);

    /*!
     * @brief Batched, OpenMP-threaded two-center build over a whole pair list.
     *
     * Evaluates overlap (kind=0) or kinetic (kind=1) for @p npair atom pairs,
     * parallelizing the per-pair op2c evaluation with OpenMP. Each pair's
     * RowMajor inorb*jnorb block values are concatenated into @p flat_out;
     * @p offsets_out[p]..offsets_out[p+1] slices pair p. No 0.5 is applied to
     * the kinetic result (this mirrors kinetic(); callers convert to Hartree).
     *
     * This is the C++ home of the batch routine; the pybind layer is only a thin
     * numpy<->vector marshalling wrapper around it (it releases the GIL).
     *
     * @param kind 0 = overlap S_ij, 1 = kinetic T_ij.
     * @param itypes npair element types of the row (bra) atom.
     * @param jtypes npair element types of the col (ket) atom.
     * @param rij    npair*3 displacements R_col - R_row (row-major, Bohr).
     * @param npair  number of pairs.
     * @param[out] flat_out    concatenated block values (resized inside).
     * @param[out] offsets_out npair+1 prefix offsets into flat_out (resized inside).
     */
    void two_center_batch(int kind,
                          const int* itypes, const int* jtypes, const double* rij,
                          size_t npair,
                          std::vector<double>& flat_out,
                          std::vector<long>& offsets_out);

    /*!
     * @brief Batched, OpenMP-threaded non-local (V_nl) build over projector atoms.
     *
     * Assembles the 3-center sum V_nl,ij(R) = sum_K <phi_i|beta_K> D_K
     * <beta_K|phi_j>, accumulated over every projector atom K, parallelizing the
     * per-K work (one orb_r_beta batch + the neighbour (i,j) products) over K.
     * Different K contribute to the same (i,j,R) block, so accumulation uses
     * thread-local maps merged after the parallel region. The per-type m-expanded
     * D matrix is supplied by the caller (it depends only on the element type and
     * is cheap to build once). This is the C++ home of the V_nl loop; the pybind
     * layer is a thin numpy<->vector wrapper.
     *
     * Neighbour structure (CSR over the n_K owned projector atoms):
     * @param k_types     op2c type of each projector atom K (size n_K).
     * @param n_K         number of projector atoms.
     * @param neigh_off   CSR offsets, size n_K+1.
     * @param neigh_gidx  global atom index of each neighbour orbital atom i.
     * @param neigh_type  op2c type of each neighbour i.
     * @param neigh_shift 3 ints per neighbour: lattice shift of i relative to K.
     * @param neigh_disp  3 doubles per neighbour: displacement r_i - r_K (Bohr).
     * Per-type m-expanded D (only types with beta projectors are used):
     * @param n_types     number of element types.
     * @param dm_dim      n_phi_beta (D dimension) per type, size n_types.
     * @param dm_flat     concatenated dm_dim[t]^2 row-major D matrices.
     * Outputs (one entry per unique non-zero (i,j,R) block, accumulated):
     * @param[out] out_i, out_j   global block indices.
     * @param[out] out_shift      3 ints per block (R_j - R_i).
     * @param[out] out_flat       concatenated RowMajor n_orb_i*n_orb_j values.
     * @param[out] out_off        prefix offsets into out_flat, size n_out+1.
     */
    void vnl_batch(const int* k_types, size_t n_K,
                   const long* neigh_off,
                   const int* neigh_gidx, const int* neigh_type,
                   const int* neigh_shift, const double* neigh_disp,
                   int n_types, const int* dm_dim, const double* dm_flat,
                   std::vector<int>& out_i, std::vector<int>& out_j,
                   std::vector<int>& out_shift,
                   std::vector<double>& out_flat,
                   std::vector<long>& out_off);

    /*!
     * @brief Computes overlap and position operator integrals simultaneously.
     *
     * Computes < phi_i | phi_j > and < phi_i | r | phi_j >.
     * Note that the position operator acts in the global frame.
     *
     * @param itype Element type index of atom i.
     * @param jtype Element type index of atom j.
     * @param Ri Absolute position of atom i.
     * @param Rj Absolute position of atom j.
     * @param is_transpose If true, the result blocks are transposed.
     * @param[out] v Overlap integral S.
     * @param[out] vx x-component of position integral <x>.
     * @param[out] vy y-component of position integral <y>.
     * @param[out] vz z-component of position integral <z>.
     */
    void overlap_position(
        size_t itype, size_t jtype, 
        ModuleBase::Vector3<double> Ri, ModuleBase::Vector3<double> Rj, 
        bool is_transpose,
        std::vector<double>& v, std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz
    );

    /*!
     * @brief Computes integrals between atomic orbitals and beta projectors < phi_i | beta_k >.
     *
     * Used for constructing the non-local pseudopotential part of the Hamiltonian.
     *
     * @param itype List of atom types for the bra side (orbitals).
     * @param ktype Atom type for the ket side (beta projector).
     * @param Ri List of positions for orbital atoms.
     * @param Rk Position of the beta projector atom.
     * @param is_transpose If true, transpose the result.
     * @param[out] ob Overlap < phi | beta >.
     * @param[out] oxb x-derivative < d_phi/dx | beta > (or gradient w.r.t R).
     * @param[out] oyb y-derivative.
     * @param[out] ozb z-derivative.
     */
    // with_grad=true computes ob = <phi|beta> plus the position blocks
    // oxb/oyb/ozb = <phi|r|beta> (needs the position-augmented beta tables, i.e.
    // pm_build=true). with_grad=false computes ONLY ob (values), via the plain
    // overlap path — valid with pm_build=false and all V_nl needs.
    void orb_r_beta(
        std::vector<size_t>& itype, size_t ktype,
        std::vector<ModuleBase::Vector3<double>> Ri, ModuleBase::Vector3<double> Rk,
        bool is_transpose,
        std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb,
        std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
        bool with_grad = true
    );

    /*!
     * @brief Computes non-local commutator integrals [r, V_nl].
     *
     * Specifically computes the matrix element involving the position operator and non-local projectors.
     * Used for velocity or Berry phase calculations.
     *
     * @param itype Atom type of orbital i.
     * @param idx Index of atom i in neighbor list.
     * @param ktype Atom type of projector k.
     * @param jtype Atom type of orbital j.
     * @param jdx Index of atom j in neighbor list.
     * @param ob Overlap <phi|beta> matrices.
     * @param oxb x-component matrices.
     * @param oyb y-component matrices.
     * @param ozb z-component matrices.
     * @param npol POLARIZATION dimension (1 or 2).
     * @param is_transpose Transpose flag.
     * @param[out] vx, vy, vz Resulting commutator integrals.
     */
    void ncomm_IKJ(
        size_t itype, size_t idx, size_t ktype, size_t jtype, size_t jdx, 
        std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb, 
        std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
        int npol, bool is_transpose,
        std::vector<std::complex<double>>& vx, std::vector<std::complex<double>>& vy, std::vector<std::complex<double>>& vz
    );

    double get_orb_rcut_max(int itype) const;
    double get_beta_rcut_max(int itype) const;

    /*!
     * @brief Valence charge for element ``itype`` from the loaded pseudopotential.
     *
     * Returns ``Atom_pseudo::zv``. rescu++ uses this value as the ionic charge
     * in pseudopotential ion-ion Ewald sums.
     */
    double valence_charge(int itype) const;

    /*!
     * @brief Radial grid for the local pseudopotential of element ``itype``.
     *
     * Returns the ``Atom_pseudo::r`` vector in Bohr. Requires a pseudopotential
     * to be loaded for every element type.
     */
    std::vector<double> vloc_rgrid(int itype) const;

    /*!
     * @brief Radial integration weights for ``vloc_rgrid``.
     *
     * Returns ``Atom_pseudo::rab`` in Bohr. The weights are paired with the
     * local-potential radial grid and are used by Simpson-style radial
     * Fourier transforms.
     */
    std::vector<double> vloc_rab(int itype) const;

    /*!
     * @brief Number of local-potential radial points active for integration.
     *
     * Returns ``Atom_pseudo::msh`` when set, otherwise the full mesh size.
     */
    int vloc_msh(int itype) const;

    /*!
     * @brief Local pseudopotential values for element ``itype`` on ``vloc_rgrid``.
     *
     * Returns ``Atom_pseudo::vloc_at`` in the op2c pseudopotential convention
     * (Rydberg). For rescumat MAT files this slot contains ``data.Vna``;
     * Python wrappers that combine it with rescu++ Hartree-valued grids divide
     * by two.
     */
    std::vector<double> vloc_at(int itype) const;

    /*!
     * @brief Radial grid for the neutral atomic density of element ``itype``.
     *
     * Returns the ``Atom_pseudo::r`` vector in Bohr. The companion values are
     * returned by ``atomic_density_at``.
     */
    std::vector<double> atomic_density_rgrid(int itype) const;

    /*!
     * @brief Radial integration weights for ``atomic_density_rgrid``.
     *
     * Returns ``Atom_pseudo::rab`` in Bohr. These weights are paired with the
     * neutral atomic density mesh and are used by radial Fourier transforms.
     */
    std::vector<double> atomic_density_rab(int itype) const;

    /*!
     * @brief Neutral atomic density values for element ``itype``.
     *
     * Returns ``Atom_pseudo::rho_at`` on ``atomic_density_rgrid``. UPF files
     * fill this from ``PP_RHOATOM``; rescumat MAT files fill this from
     * ``data.Rna.rhoData``.
     */
    std::vector<double> atomic_density_at(int itype) const;

    /*!
     * @brief Rescumat short-range neutral-density radius for element ``itype``.
     *
     * Returns the final value in ``data.Rna.rrData`` from rescumat MAT pseudo
     * input. This radius is used as the overlap support in
     * ``runtime/energy.py::short_range_ion_energy``.
     */
    double short_range_radius(int itype) const;

    /*!
     * @brief Rescumat short-range neutral charge for element ``itype``.
     *
     * Returns ``4*pi*trapz(r^2*rho, r)`` from ``data.Rna.rrData`` and
     * ``data.Rna.rhoData``.
     */
    double short_range_charge(int itype) const;

    /*!
     * @brief Rescumat short-range q grid for element ``itype``.
     *
     * Returns ``OrbitalSet(1).qqData`` from rescumat MAT pseudo input.
     */
    std::vector<double> short_range_q_grid(int itype) const;

    /*!
     * @brief Rescumat short-range q weights for element ``itype``.
     *
     * Returns ``OrbitalSet(1).qwData`` from rescumat MAT pseudo input.
     */
    std::vector<double> short_range_q_weights(int itype) const;

    /*!
     * @brief Fourier transform of rescumat ``data.Rna.rhoData``.
     *
     * Values are evaluated on ``short_range_q_grid(itype)`` with
     * ``data.Rna.drData`` weights and the same l=0 transform used by
     * ``runtime/energy.py::_radial_ft_l0``.
     */
    std::vector<double> short_range_fq(int itype) const;

    /*!
     * @brief Whether element ``itype`` carries a partial core (NLCC) charge.
     *
     * Returns ``Atom_pseudo::nlcc``. rescumat MAT files set this true when
     * ``data.Rpc`` is populated (e.g. P) and false otherwise (e.g. Si, Al).
     */
    bool has_partial_core(int itype) const;

    /*!
     * @brief Radial grid for the partial core density of element ``itype``.
     *
     * Returns the ``Atom_pseudo::r`` vector in Bohr. The companion values are
     * returned by ``partial_core_at``.
     */
    std::vector<double> partial_core_rgrid(int itype) const;

    /*!
     * @brief Partial core (NLCC) charge density values for element ``itype``.
     *
     * Returns ``Atom_pseudo::rho_atc`` (a volume density rho_core(r)) on
     * ``partial_core_rgrid``. rescumat MAT files fill this from
     * ``data.Rpc.rhoData``; it is all zero for elements without NLCC.
     */
    std::vector<double> partial_core_at(int itype) const;

    /*!
     * @brief Number of distinct (l, zeta) projector channels for element ``itype``.
     *
     * For an ONCV C pseudo with 2 s + 2 p projectors this returns 4; the
     * total m-resolved projector count is ``beta_radials.nphi(itype)`` which
     * equals 8 (= 2*1 + 2*3).
     */
    int beta_nbeta(int itype) const;

    /*!
     * @brief Angular momentum (l) of each projector channel for element ``itype``.
     *
     * Length is ``beta_nbeta(itype)``. Used to expand the channel-space D
     * matrix into the m-resolved space matching ``orb_r_beta`` output blocks.
     */
    std::vector<int> beta_lll(int itype) const;

    /*!
     * @brief Channel-space projector strength matrix ``D`` for element ``itype``.
     *
     * Stored in the ``Atom_pseudo::d_real`` field (sized
     * ``beta_nbeta * beta_nbeta``). Caller flattens row-major; the matrix is
     * used in ``V_nl = sum_K |beta_K> D_K <beta_K|`` after m-expansion. The
     * flattening matches the binding convention used by ``Op2c::overlap``.
     */
    std::vector<double> beta_dion(int itype) const;
};
