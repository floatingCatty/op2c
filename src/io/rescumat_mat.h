#ifndef RESCUMAT_MAT_H_
#define RESCUMAT_MAT_H_

#include <string>
#include <vector>

class Atom_pseudo;

namespace RescumatMat
{

struct RadialData
{
    int angular_momentum = 0;
    int principal_quantum_number = 0;
    double energy = 0.0;
    double population = 0.0;
    double kb_energy = 0.0;
    double kb_cosine = 0.0;
    bool is_ghost = false;
    std::vector<double> r_grid;
    std::vector<double> r_values;
    std::vector<double> q_grid;
};

struct AtomicData
{
    std::string symbol;
    int atomic_number = 0;
    double valence_electrons = 0.0;
    std::vector<RadialData> orbitals;
    std::vector<RadialData> projectors;
};

bool has_mat_suffix(const std::string& file);

AtomicData read_atomic_data(const std::string& file);

double infer_energy_cutoff_ry(const std::vector<RadialData>& orbitals);

void fill_atom_pseudo_from_atomic_data(const AtomicData& data, Atom_pseudo& pseudo, double rcut);

} // namespace RescumatMat

#endif
