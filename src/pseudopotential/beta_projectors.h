#ifndef BETA_RADIALS_H_
#define BETA_RADIALS_H_

#include "nao/orbital_shell.h"
#include "utils/log.h"

//! The radial part of beta functions of a single element
/*!
 *  This class represents the radial part of all Kleinman-Bylander beta
 *  functions of a single element as read from a pseudopotential file.
 *
 *  @see RadialSet
 *
 *  Usage:
 *
 *      int element_index = 1;
 *      std::ofstream ofs_log("/path/to/log/file");
 *      std::string upf_file = "/path/to/pseudopotential/file";
 *
 *      BetaRadials O_beta;
 *      O_beta.build(orb_file, element_index, ofs_log, GlobalV::MY_RANK);
 *
 *                                                                          */
class BetaRadials : public RadialSet
{
  public:
    BetaRadials() {}
    BetaRadials(const BetaRadials& other) : RadialSet(other) {} //!< deep copy

    using RadialSet::operator=;
    BetaRadials* clone() const { return new BetaRadials(*this); } // covariant return type
    BetaRadials* create_empty() const { return new BetaRadials(); }

    ~BetaRadials() {}

    /// Build the class from a Numerical_Nonlocal object
    // void build(const Numerical_Nonlocal& nl,
    //            const int itype = 0,
    //            const ModuleBase::Logger* const ptr_logger = nullptr);

    // zyzh: added for avoid the inclusion of previous 2c implementation.
    void build(const NumericalRadial* nl, const int nchi, const int itype, const ModuleBase::Logger* const ptr_logger, MPI_Comm comm = MPI_COMM_WORLD);

    void build(const RadialSet* const other, const int itype, const int p = 0, const int pm = 0, const double rcut = -1.0) override;

    

  private:
    ModuleBase::SphericalBesselTransformer sbt_;
    
};

#endif