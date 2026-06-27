#include "io/rescumat_mat.h"

#include "math/interpolation/cubic_spline.h"
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
    // const_cast: Mat_VarGetStructFieldByName is a read-only accessor (returns an internal
    // field pointer without mutating the parent struct). matio only made its first parameter
    // ``const matvar_t*`` in newer releases (>=1.5.28); older matio (e.g. 1.5.26) declares it
    // ``matvar_t*``. Casting away const here binds to BOTH signatures, so op2c builds against
    // whatever matio the toolchain provides instead of requiring >=1.5.30.
    matvar_t* field = Mat_VarGetStructFieldByName(const_cast<matvar_t*>(structure), field_name, index);
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
    return Mat_VarGetStructFieldByName(const_cast<matvar_t*>(structure), field_name, index);
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
    matvar_t* r_weights = optional_struct_field(entries, "drData", index);
    if (r_weights != nullptr)
    {
        output.r_weights = numeric_vector(r_weights, owner + ".drData");
    }
    matvar_t* q_grid = optional_struct_field(entries, "qqData", index);
    if (q_grid != nullptr)
    {
        output.q_grid = numeric_vector(q_grid, owner + ".qqData");
    }
    matvar_t* q_weights = optional_struct_field(entries, "qwData", index);
    if (q_weights != nullptr)
    {
        output.q_weights = numeric_vector(q_weights, owner + ".qwData");
    }
    if (output.r_grid.size() != output.r_values.size())
    {
        throw std::runtime_error(owner + " rrData and radial values have different lengths");
    }
    if (!output.r_weights.empty() && output.r_grid.size() != output.r_weights.size())
    {
        throw std::runtime_error(owner + " rrData and drData have different lengths");
    }
    if (!output.q_weights.empty() && output.q_grid.size() != output.q_weights.size())
    {
        throw std::runtime_error(owner + " qqData and qwData have different lengths");
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

std::vector<double> interpolate_spline_zero(const std::vector<double>& source_grid,
                                            const std::vector<double>& source_values,
                                            const std::vector<double>& target_grid)
{
    std::vector<double> output(target_grid.size(), 0.0);
    std::vector<double> eval_grid;
    std::vector<size_t> eval_indices;
    eval_grid.reserve(target_grid.size());
    eval_indices.reserve(target_grid.size());

    const double r_min = source_grid.front();
    const double r_max = source_grid.back();
    for (size_t target_index = 0; target_index < target_grid.size(); ++target_index)
    {
        const double r = target_grid[target_index];
        if (r < r_min || r > r_max)
        {
            continue;
        }
        eval_indices.push_back(target_index);
        eval_grid.push_back(r);
    }
    if (eval_grid.empty())
    {
        return output;
    }

    ModuleBase::CubicSpline spline(
        static_cast<int>(source_grid.size()),
        source_grid.data(),
        source_values.data()
    );
    std::vector<double> eval_values(eval_grid.size(), 0.0);
    spline.eval(
        static_cast<int>(eval_grid.size()),
        eval_grid.data(),
        eval_values.data()
    );
    for (size_t index = 0; index < eval_indices.size(); ++index)
    {
        output[eval_indices[index]] = eval_values[index];
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

double spherical_j0(const double x)
{
    return std::abs(x) < 1.0e-12 ? 1.0 : std::sin(x) / x;
}

std::vector<double> radial_ft_l0(
    const std::vector<double>& r_grid,
    const std::vector<double>& rho,
    const std::vector<double>& r_weights,
    const std::vector<double>& q_grid
)
{
    constexpr double four_pi = 12.56637061435917295385;
    constexpr double sqrt_two_over_pi = 0.79788456080286535588;
    if (r_grid.size() != rho.size() || r_grid.size() != r_weights.size())
    {
        throw std::runtime_error("data.Rna radial arrays have inconsistent lengths");
    }
    std::vector<double> fq(q_grid.size(), 0.0);
    for (std::size_t iq = 0; iq < q_grid.size(); ++iq)
    {
        const double q = q_grid[iq];
        double accum = 0.0;
        for (std::size_t ir = 0; ir < r_grid.size(); ++ir)
        {
            const double r = r_grid[ir];
            accum += four_pi * rho[ir] * r * r * spherical_j0(q * r) * r_weights[ir];
        }
        fq[iq] = sqrt_two_over_pi * accum;
    }
    return fq;
}

double radial_charge(
    const std::vector<double>& r_grid,
    const std::vector<double>& rho
)
{
    constexpr double four_pi = 12.56637061435917295385;
    if (r_grid.size() != rho.size())
    {
        throw std::runtime_error("data.Rna charge arrays have inconsistent lengths");
    }
    if (r_grid.size() < 2)
    {
        return 0.0;
    }
    double accum = 0.0;
    for (std::size_t index = 1; index < r_grid.size(); ++index)
    {
        const double r_left = r_grid[index - 1];
        const double r_right = r_grid[index];
        const double y_left = r_left * r_left * rho[index - 1];
        const double y_right = r_right * r_right * rho[index];
        accum += 0.5 * (y_left + y_right) * (r_right - r_left);
    }
    return four_pi * accum;
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

    matvar_t* neutral_potential = optional_struct_field(data.get(), "Vna", 0);
    if (neutral_potential != nullptr && element_count(neutral_potential) > 0)
    {
        output.local_potential.r_grid =
            numeric_vector(struct_field(neutral_potential, "rrData", 0, "data.Vna"), "data.Vna.rrData");
        output.local_potential.r_values =
            numeric_vector(struct_field(neutral_potential, "vvData", 0, "data.Vna"), "data.Vna.vvData");
        output.local_potential.r_weights =
            numeric_vector(struct_field(neutral_potential, "drData", 0, "data.Vna"), "data.Vna.drData");
        if (output.local_potential.r_grid.size() != output.local_potential.r_values.size())
        {
            throw std::runtime_error("data.Vna rrData and vvData have different lengths");
        }
        if (output.local_potential.r_grid.size() != output.local_potential.r_weights.size())
        {
            throw std::runtime_error("data.Vna rrData and drData have different lengths");
        }
        if (output.local_potential.r_grid.size() < 2)
        {
            throw std::runtime_error("data.Vna must contain at least two radial grid points");
        }
        output.has_local_potential = true;
    }

    matvar_t* atomic_density = optional_struct_field(data.get(), "Rna", 0);
    if (atomic_density != nullptr && element_count(atomic_density) > 0)
    {
        output.atomic_density.r_grid =
            numeric_vector(struct_field(atomic_density, "rrData", 0, "data.Rna"), "data.Rna.rrData");
        output.atomic_density.r_values =
            numeric_vector(struct_field(atomic_density, "rhoData", 0, "data.Rna"), "data.Rna.rhoData");
        matvar_t* rna_weights = optional_struct_field(atomic_density, "drData", 0);
        if (rna_weights != nullptr)
        {
            output.atomic_density.r_weights = numeric_vector(rna_weights, "data.Rna.drData");
        }
        if (output.atomic_density.r_grid.size() != output.atomic_density.r_values.size())
        {
            throw std::runtime_error("data.Rna rrData and rhoData have different lengths");
        }
        if (!output.atomic_density.r_weights.empty()
            && output.atomic_density.r_grid.size() != output.atomic_density.r_weights.size())
        {
            throw std::runtime_error("data.Rna rrData and drData have different lengths");
        }
        if (output.atomic_density.r_grid.size() < 2)
        {
            throw std::runtime_error("data.Rna must contain at least two radial grid points");
        }
        output.has_atomic_density = true;
    }

    // data.Rpc holds the partial core (nonlinear core correction) charge
    // density as a volume density rho_core(r). It is empty for elements
    // without NLCC (e.g. Si, Al); populated (rrData/rhoData) for e.g. P.
    matvar_t* partial_core = optional_struct_field(data.get(), "Rpc", 0);
    if (partial_core != nullptr && element_count(partial_core) > 0)
    {
        matvar_t* rr = optional_struct_field(partial_core, "rrData", 0);
        matvar_t* rho = optional_struct_field(partial_core, "rhoData", 0);
        if (rr != nullptr && rho != nullptr && element_count(rr) > 0 && element_count(rho) > 0)
        {
            output.partial_core.r_grid = numeric_vector(rr, "data.Rpc.rrData");
            output.partial_core.r_values = numeric_vector(rho, "data.Rpc.rhoData");
            if (output.partial_core.r_grid.size() != output.partial_core.r_values.size())
            {
                throw std::runtime_error("data.Rpc rrData and rhoData have different lengths");
            }
            if (output.partial_core.r_grid.size() >= 2)
            {
                output.has_partial_core = true;
            }
        }
    }

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
    std::vector<RadialData> pseudo_radials = data.projectors.empty() ? data.orbitals : data.projectors;
    if (data.has_local_potential)
    {
        pseudo_radials.push_back(data.local_potential);
    }
    if (data.has_atomic_density)
    {
        pseudo_radials.push_back(data.atomic_density);
    }
    if (data.has_partial_core)
    {
        pseudo_radials.push_back(data.partial_core);
    }
    const std::vector<double> grid = make_common_grid(pseudo_radials);
    const int mesh = static_cast<int>(grid.size());

    pseudo.has_so = false;
    pseudo.nv = 0;
    pseudo.psd = data.symbol;
    pseudo.pp_type = "NC";
    pseudo.tvanp = false;
    pseudo.nlcc = data.has_partial_core;
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
    if (data.has_atomic_density)
    {
        const auto values = interpolate_spline_zero(
            data.atomic_density.r_grid,
            data.atomic_density.r_values,
            grid
        );
        for (int ir = 0; ir < mesh; ++ir)
        {
            pseudo.rho_at[ir] = values[ir];
        }
        if (!data.orbitals.empty()
            && !data.orbitals.front().q_grid.empty()
            && !data.orbitals.front().q_weights.empty()
            && !data.atomic_density.r_weights.empty())
        {
            pseudo.short_range_radius = data.atomic_density.r_grid.empty()
                ? 0.0
                : data.atomic_density.r_grid.back();
            pseudo.short_range_charge = radial_charge(
                data.atomic_density.r_grid,
                data.atomic_density.r_values
            );
            pseudo.short_range_q_grid = data.orbitals.front().q_grid;
            pseudo.short_range_q_weights = data.orbitals.front().q_weights;
            pseudo.short_range_fq = radial_ft_l0(
                data.atomic_density.r_grid,
                data.atomic_density.r_values,
                data.atomic_density.r_weights,
                pseudo.short_range_q_grid
            );
        }
    }
    if (data.has_partial_core)
    {
        const auto values = interpolate_spline_zero(
            data.partial_core.r_grid,
            data.partial_core.r_values,
            grid
        );
        for (int ir = 0; ir < mesh; ++ir)
        {
            pseudo.rho_atc[ir] = values[ir];
        }
    }
    pseudo.vloc_at.assign(mesh, 0.0);
    if (data.has_local_potential)
    {
        pseudo.vloc_r = data.local_potential.r_grid;
        pseudo.vloc_rab = data.local_potential.r_weights.empty()
            ? radial_steps(data.local_potential.r_grid)
            : data.local_potential.r_weights;
        pseudo.vloc_at_radial.resize(data.local_potential.r_values.size());
        for (size_t ir = 0; ir < data.local_potential.r_values.size(); ++ir)
        {
            pseudo.vloc_at_radial[ir] = 2.0 * data.local_potential.r_values[ir];
        }
        const auto values = interpolate_spline_zero(
            data.local_potential.r_grid,
            data.local_potential.r_values,
            grid
        );
        for (int ir = 0; ir < mesh; ++ir)
        {
            pseudo.vloc_at[ir] = 2.0 * values[ir];
        }
    }
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
            interpolate_spline_zero(data.orbitals[iorb].r_grid, data.orbitals[iorb].r_values, grid);
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
        const auto values = interpolate_spline_zero(projector.r_grid, projector.r_values, grid);
        for (int ir = 0; ir < mesh; ++ir)
        {
            pseudo.betar(ibeta, ir) = grid[ir] * values[ir];
        }
    }
}

} // namespace RescumatMat
