#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "int2c/op2c.hpp"
#include "nao/radial_function.h"
#include "nao/atomic_radials.h"
#include "pseudopotential/pseudo_atom.h"
#include "pseudopotential/beta_projectors.h"
#include "math/linalg/matrix.h"
#include "math/linalg/vector3.h"
#include "pseudopotential/io/read_pseudo.h"
#include "utils/log.h"
#include <filesystem>

namespace py = pybind11;

// Type caster for ModuleBase::matrix <-> numpy array
// We treat ModuleBase::matrix as a row-major dense matrix.
namespace pybind11 { namespace detail {
    template <> struct type_caster<ModuleBase::matrix> {
    public:
        PYBIND11_TYPE_CASTER(ModuleBase::matrix, _("numpy.ndarray"));

        // Conversion from Python -> C++
        bool load(handle src, bool convert) {
            if (!convert && !array_t<double>::check_(src))
                return false;

            auto buf = array_t<double, array::c_style | array::forcecast>::ensure(src);
            if (!buf) return false;

            auto dims = buf.ndim();
            if (dims != 2) return false;

            auto shape = buf.shape();
            value.create(shape[0], shape[1]); // Ensure matrix is allocated
            
            // Copy data
            std::memcpy(value.c, buf.data(), sizeof(double) * value.nr * value.nc);

            return true;
        }

        // Conversion from C++ -> Python
        static handle cast(const ModuleBase::matrix& src, return_value_policy /* policy */, handle /* parent */) {
            // Allocates a new numpy array and copies data
            auto arr = array_t<double>({src.nr, src.nc});
            auto req = arr.request();
            std::memcpy(req.ptr, src.c, sizeof(double) * src.nr * src.nc);
            return arr.release();
        }
    };
    
    // Type caster for ModuleBase::Vector3<double>
    template <> struct type_caster<ModuleBase::Vector3<double>> {
    public:
        PYBIND11_TYPE_CASTER(ModuleBase::Vector3<double>, _("list"));

        bool load(handle src, bool convert) {
            if (!isinstance<sequence>(src)) return false;
            auto seq = reinterpret_borrow<sequence>(src);
            if (seq.size() != 3) return false;
            value.x = seq[0].cast<double>();
            value.y = seq[1].cast<double>();
            value.z = seq[2].cast<double>();
            return true;
        }

        static handle cast(const ModuleBase::Vector3<double>& src, return_value_policy /* policy */, handle /* parent */) {
            py::list lst;
            lst.append(src.x);
            lst.append(src.y);
            lst.append(src.z);
            return lst.release();
        }
    };
}} // namespace pybind11::detail

void bind_numerical_radial(py::module &m) {
    py::class_<NumericalRadial>(m, "NumericalRadial")
        .def(py::init<>())
        .def("build", [](NumericalRadial& self, int l, bool for_r_space, 
                         py::array_t<double> grid, py::array_t<double> value, 
                         int p, int izeta, std::string symbol, int itype, bool init_sbt) {
            
            auto buf_grid = grid.request();
            auto buf_value = value.request();
            
            if (buf_grid.ndim != 1 || buf_value.ndim != 1)
                throw std::runtime_error("Grid and Values must be 1D arrays");

            self.build(l, for_r_space, buf_grid.shape[0], 
                       static_cast<double*>(buf_grid.ptr), 
                       static_cast<double*>(buf_value.ptr), 
                       p, izeta, symbol, itype, init_sbt);
        },
        py::arg("l"), py::arg("for_r_space"), 
        py::arg("grid"), py::arg("value"), 
        py::arg("p")=0, py::arg("izeta")=0, 
        py::arg("symbol")="", py::arg("itype")=0, py::arg("init_sbt")=true)
        .def("normalize", &NumericalRadial::normalize, py::arg("for_r_space")=true)
        .def("wipe", &NumericalRadial::wipe, py::arg("r_space")=true, py::arg("k_space")=true)
        .def("set_grid", [](NumericalRadial& self, bool for_r_space, py::array_t<double> grid, char mode) {
            auto buf = grid.request();
            if (buf.ndim != 1) throw std::runtime_error("Grid must be 1D array");
            self.set_grid(for_r_space, buf.shape[0], static_cast<double*>(buf.ptr), mode);
        }, py::arg("for_r_space"), py::arg("grid"), py::arg("mode")='i')
        
        .def("set_uniform_grid", &NumericalRadial::set_uniform_grid,
             py::arg("for_r_space"), py::arg("ngrid"), py::arg("cutoff"), 
             py::arg("mode")='i', py::arg("enable_fft")=false)
             
        .def_property_readonly("pr", &NumericalRadial::pr)
        .def_property_readonly("pk", &NumericalRadial::pk)
        .def_property_readonly("kcut", &NumericalRadial::kcut)
        
        // Properties
        .def_property_readonly("l", &NumericalRadial::l)
        .def_property_readonly("nr", &NumericalRadial::nr)
        .def_property_readonly("nk", &NumericalRadial::nk)
        .def_property_readonly("symbol", &NumericalRadial::symbol)
        .def_property_readonly("itype", &NumericalRadial::itype)
        .def_property_readonly("izeta", &NumericalRadial::izeta)
        .def_property_readonly("rcut", &NumericalRadial::rcut)
        .def_property_readonly("rmax", &NumericalRadial::rmax)
        .def_property_readonly("kmax", &NumericalRadial::kmax)
        // Access data as numpy arrays (copy for safety)
        .def_property_readonly("rgrid", [](const NumericalRadial& self) {
            if (self.nr() == 0) return py::array(0, (double*)nullptr);
            return py::array(self.nr(), self.rgrid());
        })
        .def_property_readonly("rvalues", [](const NumericalRadial& self) {
            if (self.nr() == 0) return py::array(0, (double*)nullptr);
            return py::array(self.nr(), self.rvalue());
        })
        .def_property_readonly("kgrid", [](const NumericalRadial& self) {
            if (self.nk() == 0) return py::array(0, (double*)nullptr);
            return py::array(self.nk(), self.kgrid());
        })
        .def_property_readonly("kvalues", [](const NumericalRadial& self) {
            if (self.nk() == 0) return py::array(0, (double*)nullptr);
            return py::array(self.nk(), self.kvalue());
        })
        .def("__repr__", [](const NumericalRadial& self) {
            return "<NumericalRadial l=" + std::to_string(self.l()) + 
                   " symbol='" + self.symbol() + "'>";
        });
}

void bind_atomic_radials(py::module &m) {
    py::class_<AtomicRadials>(m, "AtomicRadials")
        .def(py::init<>())
        .def("build", [](AtomicRadials& self, const std::string& file, int itype, int p, int pm) {
            self.build(file, itype, p, pm);
        }, py::arg("file"), py::arg("itype")=0, py::arg("p")=0, py::arg("pm")=0)
        
        // RadialSet properties
        .def_property_readonly("symbol", &AtomicRadials::symbol)
        .def_property_readonly("itype", &AtomicRadials::itype)
        .def_property_readonly("lmax", &AtomicRadials::lmax)
        .def_property_readonly("lmin", &AtomicRadials::lmin)
        .def_property_readonly("nzeta_max", &AtomicRadials::nzeta_max)
        .def_property_readonly("rcut_max", &AtomicRadials::rcut_max)
        .def_property_readonly("nchi", &AtomicRadials::nchi)
        .def_property_readonly("nphi", &AtomicRadials::nphi)
        
        // Specific AtomicRadials properties
        .def("orb_ecut", &AtomicRadials::orb_ecut)
        .def("nzeta", &AtomicRadials::nzeta)
        .def("norb", &AtomicRadials::norb)
        
        // Sequence protocol
        .def("__len__", &AtomicRadials::nchi)
        .def("__getitem__", [](const AtomicRadials& self, int i) {
            if (i < 0 || i >= self.nchi()) throw py::index_error();
            return self.cbegin()[i]; 
        })
        .def("__iter__", [](const AtomicRadials& self) {
            return py::make_iterator(self.cbegin(), self.cend());
        }, py::keep_alive<0, 1>()); 
}

void bind_beta_radials(py::module &m) {
    py::class_<BetaRadials>(m, "BetaRadials")
        .def(py::init<>())
        // RadialSet properties
        .def_property_readonly("symbol", &BetaRadials::symbol)
        .def_property_readonly("itype", &BetaRadials::itype)
        .def_property_readonly("lmax", &BetaRadials::lmax)
        .def_property_readonly("lmin", &BetaRadials::lmin)
        .def_property_readonly("nzeta_max", &BetaRadials::nzeta_max)
        .def_property_readonly("rcut_max", &BetaRadials::rcut_max)
        .def_property_readonly("nchi", &BetaRadials::nchi)
        .def_property_readonly("nphi", &BetaRadials::nphi)
        
        .def("nzeta", &BetaRadials::nzeta)
        .def("norb", &BetaRadials::norb)

        .def("__len__", &BetaRadials::nchi)
        .def("__getitem__", [](const BetaRadials& self, int i) {
            if (i < 0 || i >= self.nchi()) throw py::index_error();
            return self.cbegin()[i];
        })
        .def("__iter__", [](const BetaRadials& self) {
            return py::make_iterator(self.cbegin(), self.cend());
        }, py::keep_alive<0, 1>());
}

void bind_atom_pseudo(py::module &m) {
    py::class_<Atom_pseudo>(m, "Atom_pseudo")
        .def(py::init<>())
        .def("init_from_upf", [](Atom_pseudo& self, std::string filename, std::string type, double rcut, bool lspinorb, double soc_lambda) {
             ModuleBase::Logger logger; // Default logger to stdout
             int comm = 0; // Serial mode
             
             // Split filename into directory and file name
             std::filesystem::path p(filename);
             std::string dir = p.parent_path().string();
             if (!dir.empty() && dir.back() != '/') dir += "/";
             std::string name = p.filename().string();
             
             read_atom_pseudopots(dir, name, type, rcut, lspinorb, soc_lambda, logger, self, comm);
             
             // After reading, setup nonlocal part (projectors)
             self.setup_nonlocal(logger, lspinorb, 1, comm);
             
        }, py::arg("filename"), py::arg("type")="upf", py::arg("rcut")=0.0, py::arg("lspinorb")=false, py::arg("soc_lambda")=0.0)
        
        // Properties
        .def_readwrite("nproj", &Atom_pseudo::nproj)
        .def_readwrite("nproj_soc", &Atom_pseudo::nproj_soc)
        .def_readwrite("itype", &Atom_pseudo::itype)
        .def_readwrite("beta_radials", &Atom_pseudo::beta_radials)
        .def_readwrite("d_real", &Atom_pseudo::d_real)
        
        // Base pseudo class properties (use lambdas since base class 'pseudo' is not registered)
        .def_property_readonly("psd", [](const Atom_pseudo& s) { return s.psd; })
        .def_property_readonly("pp_type", [](const Atom_pseudo& s) { return s.pp_type; })
        .def_property_readonly("zv", [](const Atom_pseudo& s) { return s.zv; })
        .def_property_readonly("etotps", [](const Atom_pseudo& s) { return s.etotps; })
        .def_property_readonly("lmax", [](const Atom_pseudo& s) { return s.lmax; })
        .def_property_readonly("mesh", [](const Atom_pseudo& s) { return s.mesh; })
        .def_property_readonly("nbeta", [](const Atom_pseudo& s) { return s.nbeta; })
        .def_property_readonly("nchi", [](const Atom_pseudo& s) { return s.nchi; })
        .def_property_readonly("ecutwfc", [](const Atom_pseudo& s) { return s.ecutwfc; })
        .def_property_readonly("ecutrho", [](const Atom_pseudo& s) { return s.ecutrho; })
        .def_property_readonly("has_so", [](const Atom_pseudo& s) { return s.has_so; })
        .def_property_readonly("nlcc", [](const Atom_pseudo& s) { return s.nlcc; })
        .def_property_readonly("xc_func", [](const Atom_pseudo& s) { return s.xc_func; })
        .def_property_readonly("rcut", [](const Atom_pseudo& s) { return s.rcut; })
        .def_property_readonly("msh", [](const Atom_pseudo& s) { return s.msh; })
        
        // Array accessors (copies)
        .def_property_readonly("vloc_at", [](const Atom_pseudo& self) {
            return py::array(self.vloc_at.size(), self.vloc_at.data());
        })
        .def_property_readonly("r", [](const Atom_pseudo& self) {
            return py::array(self.r.size(), self.r.data());
        })
        .def_property_readonly("rab", [](const Atom_pseudo& self) {
            return py::array(self.rab.size(), self.rab.data());
        })
        .def_property_readonly("rho_atc", [](const Atom_pseudo& self) {
            return py::array(self.rho_atc.size(), self.rho_atc.data());
        })
        .def_property_readonly("lll", [](const Atom_pseudo& self) {
            return py::array(self.lll.size(), self.lll.data());
        })
        .def_property_readonly("d_so", [](const Atom_pseudo& self) {
            auto& ca = self.d_so;
            std::vector<ssize_t> shape;
            // Determine dimensionality from bounds
            if (ca.getSize() > 0) {
                 if (ca.getBound1() > 0) shape.push_back(ca.getBound1());
                 if (ca.getBound2() > 0) shape.push_back(ca.getBound2());
                 if (ca.getBound3() > 0) shape.push_back(ca.getBound3());
                 if (ca.getBound4() > 0) shape.push_back(ca.getBound4());
            }
            return py::array(py::dtype("complex128"), shape, ca.ptr);
        })

        .def("__repr__", [](const Atom_pseudo &a) -> std::string {
            return "<Atom_pseudo label='" + a.psd + "'>";
        });
}

void bind_op2c(py::module &m) {
    py::class_<Op2c>(m, "Op2c")
        .def(py::init([](size_t ntype, int nspin, bool lspinorb,
                         const std::string& orb_dir, const std::vector<std::string> orb_name,
                         const std::string& psd_dir, const std::vector<std::string> psd_name,
                         const std::string& log_file, int mpi_handle) {
            MPI_Comm comm = MPI_COMM_WORLD;
#ifdef __MPI
            int flag = 0;
            MPI_Initialized(&flag);
            if (!flag) {
                // MPI not initialized, initialize standard serial mode
                MPI_Init(NULL, NULL);
                std::atexit([](){ 
                    int f; MPI_Finalized(&f); 
                    if(!f) MPI_Finalize(); 
                });
            }

            if (mpi_handle != 0) {
                comm = MPI_Comm_f2c((MPI_Fint)mpi_handle);
            }
#else
            // Avoid unused warning
            (void)mpi_handle;
            comm = 0;
#endif
            return new Op2c(ntype, nspin, lspinorb, orb_dir, orb_name, psd_dir, psd_name, comm, log_file);
        }),
        py::arg("ntype"), py::arg("nspin"), py::arg("lspinorb"),
        py::arg("orb_dir"), py::arg("orb_name"),
        py::arg("psd_dir") = "", py::arg("psd_name") = std::vector<std::string>(),
        py::arg("log_file") = "", py::arg("mpi_handle") = 0)
        
        .def("overlap", [](Op2c& self, size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose) {
            int inorb = self.tcbd.orb_->nphi(itype);
            int jnorb = self.tcbd.orb_->nphi(jtype);
            std::vector<double> v;
            v.resize(inorb * jnorb);
            
            self.overlap(itype, jtype, Rij, is_transpose, v, nullptr, nullptr, nullptr);
            
            return py::array(v.size(), v.data());
        })
        
        .def("overlap_deriv", [](Op2c& self, size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose) {
            int inorb = self.tcbd.orb_->nphi(itype);
            int jnorb = self.tcbd.orb_->nphi(jtype);
            std::vector<double> v(inorb * jnorb);
            std::vector<double> dvx(inorb * jnorb);
            std::vector<double> dvy(inorb * jnorb);
            std::vector<double> dvz(inorb * jnorb);

            self.overlap(itype, jtype, Rij, is_transpose, v, &dvx, &dvy, &dvz);
            return py::make_tuple(
                py::array(v.size(), v.data()),
                py::array(dvx.size(), dvx.data()),
                py::array(dvy.size(), dvy.data()),
                py::array(dvz.size(), dvz.data())
            );
        })

        .def("overlap_position", [](Op2c& self, size_t itype, size_t jtype, 
                                    ModuleBase::Vector3<double> Ri, ModuleBase::Vector3<double> Rj, 
                                    bool is_transpose) {
            int inorb = self.tcbd.orb_->nphi(itype);
            int jnorb = self.tcbd.orb_->nphi(jtype);
            std::vector<double> v(inorb * jnorb);
            std::vector<double> vx(inorb * jnorb);
            std::vector<double> vy(inorb * jnorb);
            std::vector<double> vz(inorb * jnorb);

            self.overlap_position(itype, jtype, Ri, Rj, is_transpose, v, vx, vy, vz);
            return py::make_tuple(
                py::array(v.size(), v.data()),
                py::array(vx.size(), vx.data()),
                py::array(vy.size(), vy.data()),
                py::array(vz.size(), vz.data())
            );
        })

        .def("orb_r_beta", [](Op2c& self, std::vector<size_t>& itype, size_t ktype, 
                              std::vector<ModuleBase::Vector3<double>> Ri, ModuleBase::Vector3<double> Rk,
                              bool is_transpose) {
            std::vector<ModuleBase::matrix> ob(itype.size()), oxb(itype.size()), oyb(itype.size()), ozb(itype.size());

            self.orb_r_beta(itype, ktype, Ri, Rk, is_transpose, ob, oxb, oyb, ozb);
            return py::make_tuple(ob, oxb, oyb, ozb);
        })
        
        .def("ncomm_IKJ", [](Op2c& self, size_t itype, size_t idx, size_t ktype, size_t jtype, size_t jdx,
                             std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb,
                             std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
                             int npol, bool is_transpose) {
            std::vector<std::complex<double>> vx, vy, vz;
            
            self.ncomm_IKJ(itype, idx, ktype, jtype, jdx, ob, oxb, oyb, ozb, npol, is_transpose, vx, vy, vz);
            
            return py::make_tuple(vx, vy, vz);
        })
        ;
}

PYBIND11_MODULE(_op2c, m) {
    m.doc() = "Python bindings for Op2c (NAO, Pseudo, and TwoCenter Integrals)";
    
    bind_numerical_radial(m);
    bind_atomic_radials(m);
    bind_beta_radials(m);
    bind_atom_pseudo(m);
    bind_op2c(m);
}
