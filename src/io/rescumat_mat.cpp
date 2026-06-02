#include "io/rescumat_mat.h"

#include "pseudopotential/pseudo_atom.h"

#include <matio.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

namespace
{

using MatFilePtr = std::unique_ptr<mat_t, decltype(&Mat_Close)>;
using MatVarPtr = std::unique_ptr<matvar_t, decltype(&Mat_VarFree)>;

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

size_t element_count(const matvar_t* variable)
{
    if (variable == nullptr || variable->rank <= 0 || variable->dims == nullptr)
    {
        return 0;
    }
    size_t count = 1;
    for (int idim = 0; idim < variable->rank; ++idim)
    {
        count *= variable->dims[idim];
    }
    return count;
}

matvar_t* struct_field(const matvar_t* structure,
                       const char* field_name,
                       size_t index,
                       const std::string& owner)
{
    if (structure == nullptr || structure->class_type != MAT_C_STRUCT)
    {
        throw std::runtime_error(owner + " is not a MAT struct");
    }
    matvar_t* field = Mat_VarGetStructFieldByName(structure, field_name, index);
    if (field == nullptr)
    {
        throw std::runtime_error(owner + " is missing field '" + std::string(field_name) + "'");
    }
    return field;
}

matvar_t* optional_struct_field(const matvar_t* structure, const char* field_name, size_t index)
{
    if (structure == nullptr || structure->class_type != MAT_C_STRUCT)
    {
        return nullptr;
    }
    return Mat_VarGetStructFieldByName(structure, field_name, index);
}

template <class T>
double scalar_from_typed_data(const matvar_t* variable)
{
    return static_cast<double>(static_cast<const T*>(variable->data)[0]);
}

template <class T>
std::vector<double> vector_from_typed_data(const matvar_t* variable, size_t count)
{
    const T* values = static_cast<const T*>(variable->data);
    std::vector<double> output(count);
    for (size_t index = 0; index < count; ++index)
    {
        output[index] = static_cast<double>(values[index]);
    }
    return output;
}

double numeric_scalar(const matvar_t* variable,
                      const std::string& owner,
                      double default_value = std::numeric_limits<double>::quiet_NaN())
{
    if (variable == nullptr || variable->data == nullptr || element_count(variable) == 0)
    {
        if (std::isnan(default_value))
        {
            throw std::runtime_error(owner + " is empty");
        }
        return default_value;
    }

    switch (variable->data_type)
    {
    case MAT_T_DOUBLE:
        return scalar_from_typed_data<double>(variable);
    case MAT_T_SINGLE:
        return scalar_from_typed_data<float>(variable);
    case MAT_T_INT8:
        return scalar_from_typed_data<std::int8_t>(variable);
    case MAT_T_UINT8:
        return scalar_from_typed_data<std::uint8_t>(variable);
    case MAT_T_INT16:
        return scalar_from_typed_data<std::int16_t>(variable);
    case MAT_T_UINT16:
        return scalar_from_typed_data<std::uint16_t>(variable);
    case MAT_T_INT32:
        return scalar_from_typed_data<std::int32_t>(variable);
    case MAT_T_UINT32:
        return scalar_from_typed_data<std::uint32_t>(variable);
    case MAT_T_INT64:
        return scalar_from_typed_data<std::int64_t>(variable);
    case MAT_T_UINT64:
        return scalar_from_typed_data<std::uint64_t>(variable);
    default:
        if (std::isnan(default_value))
        {
            throw std::runtime_error(owner + " is not numeric");
        }
        return default_value;
    }
}

std::vector<double> numeric_vector(const matvar_t* variable, const std::string& owner)
{
    if (variable == nullptr || variable->data == nullptr)
    {
        throw std::runtime_error(owner + " is empty");
    }

    const size_t count = element_count(variable);
    switch (variable->data_type)
    {
    case MAT_T_DOUBLE:
        return vector_from_typed_data<double>(variable, count);
    case MAT_T_SINGLE:
        return vector_from_typed_data<float>(variable, count);
    case MAT_T_INT8:
        return vector_from_typed_data<std::int8_t>(variable, count);
    case MAT_T_UINT8:
        return vector_from_typed_data<std::uint8_t>(variable, count);
    case MAT_T_INT16:
        return vector_from_typed_data<std::int16_t>(variable, count);
    case MAT_T_UINT16:
        return vector_from_typed_data<std::uint16_t>(variable, count);
    case MAT_T_INT32:
        return vector_from_typed_data<std::int32_t>(variable, count);
    case MAT_T_UINT32:
        return vector_from_typed_data<std::uint32_t>(variable, count);
    case MAT_T_INT64:
        return vector_from_typed_data<std::int64_t>(variable, count);
    case MAT_T_UINT64:
        return vector_from_typed_data<std::uint64_t>(variable, count);
    default:
        throw std::runtime_error(owner + " is not a numeric vector");
    }
}

std::string mat_string(const matvar_t* variable, const std::string& owner)
{
    if (variable == nullptr || variable->data == nullptr)
    {
        throw std::runtime_error(owner + " is empty");
    }

    std::string output;
    const size_t count = element_count(variable);
    output.reserve(count);
    if (variable->data_type == MAT_T_UINT8 || variable->data_type == MAT_T_INT8
        || variable->data_type == MAT_T_UTF8)
    {
        const char* chars = static_cast<const char*>(variable->data);
        for (size_t index = 0; index < count; ++index)
        {
            if (chars[index] != '\0')
            {
                output.push_back(chars[index]);
            }
        }
    }
    else if (variable->data_type == MAT_T_UINT16 || variable->data_type == MAT_T_UTF16)
    {
        const std::uint16_t* chars = static_cast<const std::uint16_t*>(variable->data);
        for (size_t index = 0; index < count; ++index)
        {
            if (chars[index] != 0)
            {
                output.push_back(static_cast<char>(chars[index]));
            }
        }
    }
    else
    {
        throw std::runtime_error(owner + " is not a MAT char array");
    }

    while (!output.empty() && std::isspace(static_cast<unsigned char>(output.back())))
    {
        output.pop_back();
    }
    return output;
}

RescumatMat::RadialData read_radial_entry(const matvar_t* entries,
                                          size_t index,
                                          const std::string& owner,
                                          const char* values_field)
{
    RescumatMat::RadialData output;
    matvar_t* params = struct_field(entries, "Parameter", index, owner);
    output.angular_momentum = static_cast<int>(
        std::llround(numeric_scalar(struct_field(params, "L", 0, owner + ".Parameter"), owner + ".Parameter.L"))
    );
    output.principal_quantum_number = static_cast<int>(std::llround(
        numeric_scalar(optional_struct_field(params, "N", 0), owner + ".Parameter.N", 0.0)
    ));
    output.energy = numeric_scalar(optional_struct_field(params, "E", 0), owner + ".Parameter.E", 0.0);
    output.population =
        numeric_scalar(optional_struct_field(params, "Population", 0), owner + ".Parameter.Population", 0.0);
    output.kb_energy =
        numeric_scalar(optional_struct_field(params, "KBenergy", 0), owner + ".Parameter.KBenergy", 0.0);
    output.kb_cosine =
        numeric_scalar(optional_struct_field(params, "KBcosine", 0), owner + ".Parameter.KBcosine", 0.0);
    output.is_ghost =
        std::llround(numeric_scalar(optional_struct_field(params, "isGhost", 0), owner + ".Parameter.isGhost", 0.0))
        != 0;
    output.r_grid = numeric_vector(struct_field(entries, "rrData", index, owner), owner + ".rrData");
    output.r_values = numeric_vector(struct_field(entries, values_field, index, owner), owner + "." + values_field);
    matvar_t* q_grid = optional_struct_field(entries, "qqData", index);
    if (q_grid != nullptr)
    {
        output.q_grid = numeric_vector(q_grid, owner + ".qqData");
    }
    if (output.r_grid.size() != output.r_values.size())
    {
        throw std::runtime_error(owner + " rrData and radial values have different lengths");
    }
    if (output.r_grid.size() < 2)
    {
        throw std::runtime_error(owner + " must contain at least two radial grid points");
    }
    return output;
}

std::vector<RescumatMat::RadialData> read_radial_array(const matvar_t* array,
                                                       const std::string& owner,
                                                       const char* values_field)
{
    if (array == nullptr || array->class_type != MAT_C_STRUCT)
    {
        throw std::runtime_error(owner + " is not a MAT struct array");
    }
    const size_t count = element_count(array);
    std::vector<RescumatMat::RadialData> output;
    output.reserve(count);
    for (size_t index = 0; index < count; ++index)
    {
        output.push_back(read_radial_entry(array, index, owner + "(" + std::to_string(index) + ")", values_field));
    }
    return output;
}

std::vector<double> make_common_grid(const std::vector<RescumatMat::RadialData>& radials)
{
    double min_dr = std::numeric_limits<double>::max();
    double max_r = 0.0;
    for (const auto& radial : radials)
    {
        for (size_t index = 1; index < radial.r_grid.size(); ++index)
        {
            const double dr = radial.r_grid[index] - radial.r_grid[index - 1];
            if (dr > 0.0)
            {
                min_dr = std::min(min_dr, dr);
            }
        }
        max_r = std::max(max_r, radial.r_grid.back());
    }
    if (!std::isfinite(min_dr) || min_dr <= 0.0)
    {
        throw std::runtime_error("rescumat MAT radial grid spacing is invalid");
    }
    const int mesh = static_cast<int>(std::ceil(max_r / min_dr)) + 1;
    std::vector<double> grid(mesh, 0.0);
    for (int index = 0; index < mesh; ++index)
    {
        grid[index] = index * min_dr;
    }
    return grid;
}

std::vector<double> interpolate_linear(const std::vector<double>& source_grid,
                                       const std::vector<double>& source_values,
                                       const std::vector<double>& target_grid)
{
    std::vector<double> output(target_grid.size(), 0.0);
    size_t source_index = 0;
    for (size_t target_index = 0; target_index < target_grid.size(); ++target_index)
    {
        const double r = target_grid[target_index];
        if (r < source_grid.front() || r > source_grid.back())
        {
            continue;
        }
        while (source_index + 1 < source_grid.size() && source_grid[source_index + 1] < r)
        {
            ++source_index;
        }
        if (source_index + 1 >= source_grid.size())
        {
            output[target_index] = source_values.back();
            continue;
        }
        const double x0 = source_grid[source_index];
        const double x1 = source_grid[source_index + 1];
        const double y0 = source_values[source_index];
        const double y1 = source_values[source_index + 1];
        if (x1 == x0)
        {
            output[target_index] = y0;
        }
        else
        {
            const double t = (r - x0) / (x1 - x0);
            output[target_index] = y0 + t * (y1 - y0);
        }
    }
    return output;
}

std::vector<double> radial_steps(const std::vector<double>& grid)
{
    std::vector<double> steps(grid.size(), 0.0);
    if (grid.size() < 2)
    {
        return steps;
    }
    steps[0] = grid[1] - grid[0];
    for (size_t index = 1; index < grid.size(); ++index)
    {
        steps[index] = grid[index] - grid[index - 1];
    }
    return steps;
}

int maximum_l(const std::vector<RescumatMat::RadialData>& radials)
{
    int output = -1;
    for (const auto& radial : radials)
    {
        output = std::max(output, radial.angular_momentum);
    }
    return output;
}

} // namespace

namespace RescumatMat
{

bool has_mat_suffix(const std::string& file)
{
    const std::string lowered = lower_copy(file);
    return lowered.size() >= 4 && lowered.substr(lowered.size() - 4) == ".mat";
}

AtomicData read_atomic_data(const std::string& file)
{
    MatFilePtr mat(Mat_Open(file.c_str(), MAT_ACC_RDONLY), &Mat_Close);
    if (!mat)
    {
        throw std::runtime_error("could not open rescumat MAT file: " + file);
    }

    MatVarPtr data(Mat_VarRead(mat.get(), "data"), &Mat_VarFree);
    if (!data)
    {
        throw std::runtime_error(file + " is missing top-level MAT variable 'data'");
    }
    if (data->class_type != MAT_C_STRUCT)
    {
        throw std::runtime_error(file + " top-level MAT variable 'data' is not a struct");
    }

    matvar_t* atom = struct_field(data.get(), "atom", 0, "data");
    AtomicData output;
    output.symbol = mat_string(struct_field(atom, "symbol", 0, "data.atom"), "data.atom.symbol");
    output.atomic_number =
        static_cast<int>(std::llround(numeric_scalar(struct_field(atom, "Z", 0, "data.atom"), "data.atom.Z")));
    output.valence_electrons = numeric_scalar(struct_field(atom, "N", 0, "data.atom"), "data.atom.N");
    output.orbitals = read_radial_array(struct_field(data.get(), "OrbitalSet", 0, "data"), "data.OrbitalSet", "frData");

    matvar_t* projectors = optional_struct_field(data.get(), "Vnl", 0);
    if (projectors != nullptr && element_count(projectors) > 0)
    {
        output.projectors = read_radial_array(projectors, "data.Vnl", "vvData");
    }
    return output;
}

double infer_energy_cutoff_ry(const std::vector<RadialData>& orbitals)
{
    double qmax = 0.0;
    for (const auto& orbital : orbitals)
    {
        if (!orbital.q_grid.empty())
        {
            qmax = std::max(qmax, orbital.q_grid.back());
        }
    }
    return qmax > 0.0 ? qmax * qmax : 100.0;
}

void fill_atom_pseudo_from_atomic_data(const AtomicData& data, Atom_pseudo& pseudo, double rcut)
{
    const std::vector<double> grid = data.projectors.empty()
        ? make_common_grid(data.orbitals)
        : make_common_grid(data.projectors);
    const int mesh = static_cast<int>(grid.size());

    pseudo.has_so = false;
    pseudo.nv = 0;
    pseudo.psd = data.symbol;
    pseudo.pp_type = "NC";
    pseudo.tvanp = false;
    pseudo.nlcc = false;
    pseudo.xc_func = "";
    pseudo.zv = data.valence_electrons;
    pseudo.etotps = 0.0;
    pseudo.ecutwfc = infer_energy_cutoff_ry(data.orbitals);
    pseudo.ecutrho = 4.0 * pseudo.ecutwfc;
    pseudo.lmax = std::max(maximum_l(data.orbitals), maximum_l(data.projectors));
    pseudo.mesh = mesh;
    pseudo.nchi = static_cast<int>(data.orbitals.size());
    pseudo.nbeta = static_cast<int>(data.projectors.size());
    pseudo.nqlc = 0;
    pseudo.kkbeta = mesh;
    pseudo.nh = 0;
    for (const auto& projector : data.projectors)
    {
        pseudo.nh += 2 * projector.angular_momentum + 1;
    }

    pseudo.els.assign(pseudo.nchi, data.symbol);
    pseudo.lchi.resize(pseudo.nchi);
    pseudo.oc.resize(pseudo.nchi);
    pseudo.jchi.assign(pseudo.nchi, 0.0);
    pseudo.nn.resize(pseudo.nchi);
    for (int index = 0; index < pseudo.nchi; ++index)
    {
        pseudo.lchi[index] = data.orbitals[index].angular_momentum;
        pseudo.oc[index] = data.orbitals[index].population;
        pseudo.nn[index] = data.orbitals[index].principal_quantum_number;
    }

    pseudo.r = grid;
    pseudo.rab = radial_steps(grid);
    pseudo.rho_atc.assign(mesh, 0.0);
    pseudo.rho_at.assign(mesh, 0.0);
    pseudo.vloc_at.assign(mesh, 0.0);
    pseudo.rcut = rcut > 0.0 ? rcut : grid.back();
    pseudo.msh = mesh;
    for (int index = 0; index < mesh; ++index)
    {
        if (grid[index] > pseudo.rcut)
        {
            pseudo.msh = 2 * ((index + 2) / 2) - 1;
            break;
        }
    }

    pseudo.chi.create(pseudo.nchi, mesh);
    for (int iorb = 0; iorb < pseudo.nchi; ++iorb)
    {
        const auto values =
            interpolate_linear(data.orbitals[iorb].r_grid, data.orbitals[iorb].r_values, grid);
        for (int ir = 0; ir < mesh; ++ir)
        {
            pseudo.chi(iorb, ir) = values[ir];
        }
    }

    pseudo.lll.resize(pseudo.nbeta);
    pseudo.jjj.resize(pseudo.nbeta);
    pseudo.dion.create(pseudo.nbeta, pseudo.nbeta);
    pseudo.betar.create(pseudo.nbeta, mesh);
    for (int ibeta = 0; ibeta < pseudo.nbeta; ++ibeta)
    {
        const auto& projector = data.projectors[ibeta];
        pseudo.lll[ibeta] = projector.angular_momentum;
        pseudo.jjj[ibeta] = static_cast<double>(projector.angular_momentum);
        pseudo.dion(ibeta, ibeta) = projector.kb_energy;
        const auto values = interpolate_linear(projector.r_grid, projector.r_values, grid);
        for (int ir = 0; ir < mesh; ++ir)
        {
            pseudo.betar(ibeta, ir) = grid[ir] * values[ir];
        }
    }
}

} // namespace RescumatMat
