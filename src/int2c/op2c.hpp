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
        MPI_Comm comm, const std::string& log_file
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
        int nspin, bool lspinorb
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
    void orb_r_beta(
        std::vector<size_t>& itype, size_t ktype, 
        std::vector<ModuleBase::Vector3<double>> Ri, ModuleBase::Vector3<double> Rk,
        bool is_transpose,
        std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb, 
        std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb
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