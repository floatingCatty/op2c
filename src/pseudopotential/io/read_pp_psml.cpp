// PSML (PSeudopotential Markup Language) reader.
//
// PSML is the XML interchange format produced by oncvpsp / PseudoDojo and read
// natively by SIESTA & ABINIT. This reader populates the same Atom_pseudo model
// as the UPF readers, so a deck can point at a `.psml` and both rescu++ and
// SIESTA can share one pseudopotential file. PSML is regular machine-generated
// XML, so a small self-contained scanner suffices (no XML library dependency).
//
// Conventions (validated against the matching PseudoDojo UPF via op2c, see
// tests/unit/test_psml_pseudo.py):
//   * energies are in Hartree in the file; op2c stores vloc / Dij in Rydberg
//     (the UPF convention, halved back to Hartree downstream), so vloc and the
//     KB energies are multiplied by 2.
//   * the local potential carries the -Z/r tail; beyond the tabulated range it
//     is extended analytically as -2*Zval/r (Rydberg).
//   * PSML charge radfuncs store 4*pi*rho(r):
//       valence  -> rho_at  (UPF 4*pi*r^2*rho)  = r^2 * (PSML value)
//       core     -> rho_atc (UPF rho_core)      = (PSML value) / (4*pi)
//   * projectors: betar = r * (PSML projector); Dij = diag(2*ekb).

#include "pseudopotential/io/read_pp.h"
#include "pseudopotential/pseudo_atom.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr double FOUR_PI = 4.0 * 3.14159265358979323846;

// Parse all whitespace-separated reals in [s] (tolerates Fortran 'D' exponents).
std::vector<double> parse_doubles(const std::string& s)
{
    std::vector<double> out;
    std::string tok;
    std::istringstream iss(s);
    while (iss >> tok)
    {
        for (char& ch : tok)
        {
            if (ch == 'D' || ch == 'd') ch = 'E';
        }
        try
        {
            out.push_back(std::stod(tok));
        }
        catch (...)
        {
        }
    }
    return out;
}

// The "<tag ...>" header substring at position [pos].
std::string tag_header(const std::string& c, std::size_t pos)
{
    std::size_t e = c.find('>', pos);
    return c.substr(pos, e == std::string::npos ? std::string::npos : e - pos + 1);
}

// Value of attribute [name] in an opening-tag string.
std::string attr(const std::string& open_tag, const std::string& name)
{
    std::size_t p = open_tag.find(name + "=\"");
    if (p == std::string::npos) return "";
    p += name.size() + 2;
    std::size_t e = open_tag.find('"', p);
    return open_tag.substr(p, e - p);
}

// Numbers inside the first <data ...>...</data> at/after [from].
std::vector<double> read_data(const std::string& c, std::size_t from)
{
    std::size_t d = c.find("<data", from);
    if (d == std::string::npos) return {};
    std::size_t b = c.find('>', d) + 1;
    std::size_t e = c.find("</data>", b);
    return parse_doubles(c.substr(b, e - b));
}
} // namespace

int Pseudopot_upf::read_pseudo_psml(std::ifstream& ifs, Atom_pseudo& pp)
{
    std::stringstream ss;
    ss << ifs.rdbuf();
    const std::string c = ss.str();

    // --- atom spec / header ---
    std::size_t sp = c.find("<pseudo-atom-spec");
    if (sp == std::string::npos)
    {
        std::cerr << "read_pseudo_psml: missing <pseudo-atom-spec>" << std::endl;
        return 1;
    }
    std::string spec = tag_header(c, sp);
    pp.psd = attr(spec, "atomic-label");
    pp.zv = std::stod(attr(spec, "z-pseudo"));
    pp.pp_type = "NC";
    pp.tvanp = false;
    pp.has_so = false;
    pp.nv = 0;

    // --- radial grid (one global log/linear grid; short functions use a prefix) ---
    std::size_t g = c.find("<grid-data>");
    std::size_t ge = c.find("</grid-data>", g);
    if (g == std::string::npos || ge == std::string::npos)
    {
        std::cerr << "read_pseudo_psml: missing <grid-data>" << std::endl;
        return 1;
    }
    std::vector<double> r = parse_doubles(c.substr(g + 11, ge - (g + 11)));
    int mesh = static_cast<int>(r.size());
    if (mesh < 2)
    {
        std::cerr << "read_pseudo_psml: empty grid" << std::endl;
        return 1;
    }

    // --- valence charge: PSML stores 4*pi*rho -> rho_at = r^2 * (PSML value) ---
    std::vector<double> val = read_data(c, c.find("<valence-charge"));

    // The oncvpsp log grid runs out to ~40 Bohr but the pseudo data (and the
    // neutral atom) is ~0 well before that. The neutral-atom short-range ion energy
    // uses the grid extent as the pair cutoff, so an over-long grid makes the
    // "short-range" sum span many spurious cells. Truncate to the physical extent —
    // the largest radius where the valence density 4*pi*r^2*rho is non-negligible —
    // matching how UPF files are already truncated.
    {
        double rho_max = 0.0;
        for (int i = 0; i < mesh && i < static_cast<int>(val.size()); ++i)
            rho_max = std::max(rho_max, std::abs(r[i] * r[i] * val[i]));
        const double thr = 1e-6 * rho_max;
        int last = 1;
        for (int i = 0; i < mesh && i < static_cast<int>(val.size()); ++i)
            if (std::abs(r[i] * r[i] * val[i]) > thr) last = i;
        const int mesh_t = std::min(mesh, last + 4);
        if (mesh_t >= 2) mesh = mesh_t;
        r.resize(mesh);
    }

    pp.mesh = mesh;
    pp.msh = mesh;
    pp.r = r;
    pp.rab.assign(mesh, 0.0);
    for (int i = 0; i < mesh; ++i)
    {
        if (i == 0)
            pp.rab[i] = r[1] - r[0];
        else if (i == mesh - 1)
            pp.rab[i] = r[i] - r[i - 1];
        else
            pp.rab[i] = 0.5 * (r[i + 1] - r[i - 1]);
    }
    pp.rcut = r[mesh - 1];

    pp.rho_at.assign(mesh, 0.0);
    for (int i = 0; i < static_cast<int>(val.size()) && i < mesh; ++i)
        pp.rho_at[i] = r[i] * r[i] * val[i];

    // --- pseudo-core charge (NLCC): rho_atc = (PSML value) / (4*pi) ---
    std::size_t pc = c.find("<pseudocore-charge");
    pp.nlcc = (pc != std::string::npos);
    pp.rho_atc.assign(mesh, 0.0);
    if (pp.nlcc)
    {
        std::vector<double> core = read_data(c, pc);
        for (int i = 0; i < static_cast<int>(core.size()) && i < mesh; ++i)
            pp.rho_atc[i] = core[i] / FOUR_PI;
    }

    // --- local potential (Hartree -> Rydberg), -2*Zval/r tail beyond the table ---
    {
        std::size_t lp = c.find("<local-potential");
        std::vector<double> vloc = read_data(c, lp);
        const int nv = static_cast<int>(vloc.size());
        pp.vloc_at.assign(mesh, 0.0);
        for (int i = 0; i < mesh; ++i)
        {
            if (i < nv)
                pp.vloc_at[i] = 2.0 * vloc[i];
            else
                pp.vloc_at[i] = (r[i] > 0.0) ? (-2.0 * pp.zv / r[i]) : 0.0;
        }
    }

    // --- nonlocal KB projectors: betar = r * proj, Dij = diag(2*ekb) ---
    {
        std::size_t np = c.find("<nonlocal-projectors");
        std::size_t np_end = c.find("</nonlocal-projectors>", np);
        std::vector<std::size_t> proj_pos;
        std::vector<int> proj_l;
        std::vector<double> proj_ekb;
        std::size_t q = np;
        while (np != std::string::npos)
        {
            q = c.find("<proj ", q + 1);
            if (q == std::string::npos || q > np_end) break;
            std::string h = tag_header(c, q);
            const std::string ls = attr(h, "l");
            const int l = (ls == "s") ? 0 : (ls == "p") ? 1 : (ls == "d") ? 2 : (ls == "f") ? 3 : (ls == "g") ? 4 : -1;
            if (l < 0)
            {
                std::cerr << "read_pseudo_psml: unknown projector l='" << ls << "'" << std::endl;
                return 1;
            }
            proj_pos.push_back(q);
            proj_l.push_back(l);
            proj_ekb.push_back(std::stod(attr(h, "ekb")));
        }
        const int nbeta = static_cast<int>(proj_pos.size());
        pp.nbeta = nbeta;
        pp.lll.assign(nbeta, 0);
        this->kbeta = std::vector<int>(nbeta, mesh);
        this->els_beta = std::vector<std::string>(nbeta, "Xn");
        this->rcut = std::vector<double>(nbeta, 0.0);
        this->rcutus = std::vector<double>(nbeta, 0.0);
        pp.kkbeta = mesh;
        pp.jjj = std::vector<double>(nbeta, 0.0);
        pp.betar.create(nbeta, mesh);
        pp.dion.create(nbeta, nbeta);
        for (int i = 0; i < nbeta; ++i)
            for (int j = 0; j < nbeta; ++j) pp.dion(i, j) = 0.0;
        int lmax = 0;
        for (int ib = 0; ib < nbeta; ++ib)
        {
            pp.lll[ib] = proj_l[ib];
            lmax = std::max(lmax, proj_l[ib]);
            pp.dion(ib, ib) = 2.0 * proj_ekb[ib]; // Hartree -> Rydberg
            std::vector<double> proj = read_data(c, proj_pos[ib]);
            for (int i = 0; i < mesh; ++i)
                pp.betar(ib, i) = (i < static_cast<int>(proj.size())) ? r[i] * proj[i] : 0.0;
        }
        pp.lmax = lmax;
        pp.nh = nbeta;
        this->nd = nbeta * nbeta;
    }
    pp.check_betar();

    // No pseudo-wavefunctions are consumed: the atomic-density guess uses rho_at.
    pp.nchi = 0;
    pp.nqlc = 0;

    return 0;
}
