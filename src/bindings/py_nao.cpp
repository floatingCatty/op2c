#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "int2c/op2c.hpp"
#include "math/linalg/matrix.h"
#include "math/linalg/vector3.h"

namespace py = pybind11;

// Type caster for ModuleBase::matrix <-> numpy array
// We treat ModuleBase::matrix as a row-major dense matrix.
// Existing class has: int nr, nc; double *c;
// NOTE: pybind11 might not automatically handle this custom class, we need custom type casters or just copy data.
// For simplicity/robustness in this step, I will use manual conversion helper functions if needed,
// or use buffer protocol if I can modify the class. But here I cannot easily modify ModuleBase::matrix to be a py::buffer.
// So binding it as an opaque type or converting to/from numpy is best.
// The user wants "easy access", implying numpy arrays for data.
// I'll add a helper to copy data out/in, or use py::buffer_info if I bind the class itself.
// However, the Op2c methods take std::vector<double> or std::vector<ModuleBase::matrix>.
// I should inspect Op2c methods again.
// overlap: std::vector<double>& v -> output.
// orb_r_beta: std::vector<ModuleBase::matrix>& ob -> output.

// Let's rely on stl wrappers for std::vector.
// For ModuleBase::matrix, we need a way to convert it to numpy.

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
            
            // The C++ signature:
            // void overlap(size_t itype, size_t jtype, ModuleBase::Vector3<double> Rij, bool is_transpose, 
            //              std::vector<double>& v, std::vector<double>* dvx, ...);
            self.overlap(itype, jtype, Rij, is_transpose, v, nullptr, nullptr, nullptr);
            
            // Convert vector<double> to numpy array
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
            // No need to resize internal matrices, Op2c::orb_r_beta calls create() on them.

            self.orb_r_beta(itype, ktype, Ri, Rk, is_transpose, ob, oxb, oyb, ozb);
            // Type caster for vector<matrix> will kick in (list of numpy arrays)
            return py::make_tuple(ob, oxb, oyb, ozb);
        })
        
        .def("ncomm_IKJ", [](Op2c& self, size_t itype, size_t idx, size_t ktype, size_t jtype, size_t jdx,
                             std::vector<ModuleBase::matrix>& ob, std::vector<ModuleBase::matrix>& oxb,
                             std::vector<ModuleBase::matrix>& oyb, std::vector<ModuleBase::matrix>& ozb,
                             int npol, bool is_transpose) {
            std::vector<std::complex<double>> vx, vy, vz;
            
            // C++ signature expects non-const references.
            // pybind11 will convert python lists of numpy arrays to std::vector<ModuleBase::matrix>
            // These are temporary copies, so it is safe to pass them as references.
            
            self.ncomm_IKJ(itype, idx, ktype, jtype, jdx, ob, oxb, oyb, ozb, npol, is_transpose, vx, vy, vz);
            
            return py::make_tuple(vx, vy, vz);
        })
        ;
}

PYBIND11_MODULE(_estate, m) {
    m.doc() = "Python bindings for estate Op2c";
    bind_op2c(m);
}
