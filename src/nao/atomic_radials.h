#ifndef ATOMIC_RADIALS_H_
#define ATOMIC_RADIALS_H_

#include "nao/orbital_shell.h"
#include "utils/parallel_common.h"
#include "utils/log.h"

//! The radial part of numerical atomic orbitals of a single element
/*!
 *  This class represents the radial part of all numerical atomic orbitals
 *  of a single element as read from an orbital file.
 *
 *  @see RadialSet
 *
 *  Usage:
 *
 *      int element_index = 1;
 *      std::ofstream ofs_log("/path/to/log/file");
 *      std::string orb_file = "/path/to/orbital/file";
 *
 *      AtomicRadials O_radials;
 *      O_radials.build(orb_file, element_index, ofs_log, GlobalV::MY_RANK);
 *
 *                                                                          */
class AtomicRadials : public RadialSet
{
  public:
    AtomicRadials() {}
    AtomicRadials(const AtomicRadials& other) : RadialSet(other), orb_ecut_(other.orb_ecut_) {}

    AtomicRadials& operator=(const AtomicRadials& rhs);
    AtomicRadials* clone() const { return new AtomicRadials(*this); } // covariant return type
    AtomicRadials* create_empty() const { return new AtomicRadials(); }

    ~AtomicRadials() {} // ~RadialSet() is called automatically

    //! Build the class from an orbital file
    void build(const std::string& file,          //!< orbital file name
               const int itype = 0,              //!< element index in calculation
               const int p = 0,
               const int pm = 0,                 //!< add an extra pm integer value to the readed l
               const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
               MPI_Comm comm = MPI_COMM_WORLD   //!< MPI comm
    );

    void build(const RadialSet* const other, const int itype, const int p = 0, const int pm = 0, const double rcut = -1.0) override;

    //! Multiply every radial in place by a spherical function w(r), turning each orbital phi_i
    //! into (w . phi_i) with the SAME (l, izeta). Used to build the "screened orbital" R_i.dV for
    //! the two-centre neutral-atom-potential-residual integral <phi_i|dV|phi_j> = <(dV.phi_i)|phi_j>
    //! (moment-cancelling Vna, doc 52 Inc 3b). ``w`` is tabulated on ``(w_r, w_v)`` (linear interp,
    //! 0 outside its range); a compact ``w`` (e.g. dV, support < r_delta) makes the screened radial
    //! compact so the overlap support shrinks accordingly.
    void screen_by_radial(const double* w_r, const double* w_v, int w_n);

    //! Get the energy cutoff as given by the orbital file
    double orb_ecut() const { return orb_ecut_; }

  private:
    double orb_ecut_; //!< energy cutoff as given by the orbital file

    //! Read the orbital file in the ABACUS format
    void read_abacus_orb(std::ifstream& ifs,               //!< input file stream from orbital file
                         const int p = 0, 
                         const int pm = 0,        // !< add pm to the readed l
                         const ModuleBase::Logger* ptr_logger = nullptr, //!< output file stream for logging
                         MPI_Comm comm = MPI_COMM_WORLD   //!< MPI comm
    );

    //! Read the rescumat *_AtomicData.mat format
    void read_rescumat_mat(const std::string& file,
                           const int p = 0,
                           const int pm = 0,
                           const ModuleBase::Logger* ptr_logger = nullptr,
                           MPI_Comm comm = MPI_COMM_WORLD);

    //! Read a SIESTA <El>.ion basis file (the PAO blocks only; KB/Vna ignored —
    //! the pseudopotential is supplied separately by the UPF/PSML reader).
    void read_siesta_ion(const std::string& file,
                         const int p = 0,
                         const int pm = 0,
                         const ModuleBase::Logger* ptr_logger = nullptr,
                         MPI_Comm comm = MPI_COMM_WORLD);

};

#endif
