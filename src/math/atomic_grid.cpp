#include "math/atomic_grid.h"
#include "math/math_lebedev_laikov.h"
#include "math/constants.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace ModuleBase
{

namespace
{
// The Lebedev-Laikov orders the generator supports. Validated up front (in release builds the
// generator's own assert() is compiled out and it would otherwise emit an uninitialized grid).
const std::set<int>& allowed_lebedev_degrees()
{
    static const std::set<int> allowed = {
        6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434,
        590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890,
        4334, 4802, 5294, 5810};
    return allowed;
}
} // namespace

void treutler_ahlrichs_radial(int npts, double R,
                              std::vector<double>& r, std::vector<double>& w,
                              double alpha)
{
    r.assign(npts, 0.0);
    w.assign(npts, 0.0);
    const double ln2 = 0.693147180559945309417232;
    const double pi_over = PI / (npts + 1.0);
    for (int idx = 0; idx < npts; ++idx)
    {
        const int i = idx + 1;                    // 1 <= i <= npts
        const double ti = i * pi_over;
        const double xi = std::cos(ti);
        // Gauss-Chebyshev (2nd kind) unit weight: pi/(n+1) * sqrt(1-x^2) = pi/(n+1) sin(i pi/(n+1)).
        const double wx = pi_over * std::sin(ti);
        // Treutler-Ahlrichs M4 map (a = 1): r = R (1+x)^alpha ln(2/(1-x)) / ln2, and dr/dx.
        const double pow_term = std::pow(1.0 + xi, alpha);
        const double log_term = std::log(2.0 / (1.0 - xi));
        r[idx] = R * pow_term * log_term / ln2;
        w[idx] = wx * R * pow_term / ln2 * (alpha * log_term / (1.0 + xi) + 1.0 / (1.0 - xi));
    }
}

AtomicGrid::AtomicGrid(const Vector3<double>& center, int n_radial, double R,
                       int degree, double alpha)
    : center_(center)
{
    if (allowed_lebedev_degrees().count(degree) == 0)
        throw std::invalid_argument("AtomicGrid: unsupported Lebedev degree " +
                                    std::to_string(degree));

    std::vector<double> r, wr;
    treutler_ahlrichs_radial(n_radial, R, r, wr, alpha);

    Lebedev_laikov_grid leb(degree);
    leb.generate_grid_points();
    const Vector3<double>* omega = leb.get_grid_coor();  // unit-sphere directions
    const double* wa = leb.get_weight();                 // Lebedev weights (sum to 4pi = int dOmega)
    const int nang = degree;                             // #angular points == degree

    const int ntot = n_radial * nang;
    points_.resize(ntot);
    weights_.resize(ntot);
    int k = 0;
    for (int j = 0; j < nang; ++j)
    {
        const double waj = wa[j];                        // already carries int dOmega (sum = 4pi)
        for (int i = 0; i < n_radial; ++i)
        {
            const double ri = r[i];
            points_[k] = Vector3<double>(center_.x + ri * omega[j].x,
                                         center_.y + ri * omega[j].y,
                                         center_.z + ri * omega[j].z);
            weights_[k] = wr[i] * ri * ri * waj;         // w_r * r^2 * w_ang
            ++k;
        }
    }
}

AtomicGrid::AtomicGrid(const Vector3<double>& center, int n_radial, double R,
                       const std::vector<int>& shell_degree, double alpha)
    : center_(center)
{
    if (static_cast<int>(shell_degree.size()) != n_radial)
        throw std::invalid_argument("AtomicGrid(pruned): shell_degree.size() != n_radial");
    for (int d : shell_degree)
        if (allowed_lebedev_degrees().count(d) == 0)
            throw std::invalid_argument("AtomicGrid(pruned): unsupported Lebedev degree " +
                                        std::to_string(d));

    std::vector<double> r, wr;
    treutler_ahlrichs_radial(n_radial, R, r, wr, alpha);

    // Generate each DISTINCT Lebedev order once; shells of the same order share the cached
    // directions/weights (a pruned grid uses only ~3 distinct orders).
    using AngGrid = std::pair<std::vector<Vector3<double>>, std::vector<double>>;
    std::map<int, AngGrid> cache;
    auto leb_for = [&cache](int deg) -> const AngGrid& {
        auto it = cache.find(deg);
        if (it == cache.end())
        {
            Lebedev_laikov_grid leb(deg);
            leb.generate_grid_points();
            const Vector3<double>* om = leb.get_grid_coor();
            const double* wa = leb.get_weight();
            AngGrid g{std::vector<Vector3<double>>(om, om + deg),
                      std::vector<double>(wa, wa + deg)};
            it = cache.emplace(deg, std::move(g)).first;
        }
        return it->second;
    };

    int ntot = 0;
    for (int d : shell_degree) ntot += d;
    points_.resize(ntot);
    weights_.resize(ntot);
    int k = 0;
    for (int i = 0; i < n_radial; ++i)
    {
        const AngGrid& g = leb_for(shell_degree[i]);
        const std::vector<Vector3<double>>& om = g.first;
        const std::vector<double>& wa = g.second;
        const double ri = r[i];
        const double wri = wr[i] * ri * ri;                 // w_r * r^2
        const int deg = shell_degree[i];
        for (int j = 0; j < deg; ++j)
        {
            points_[k] = Vector3<double>(center_.x + ri * om[j].x,
                                         center_.y + ri * om[j].y,
                                         center_.z + ri * om[j].z);
            weights_[k] = wri * wa[j];
            ++k;
        }
    }
}

std::vector<int> treutler_prune_schedule(int n_radial, int inner_degree,
                                         int valence_degree, int outer_degree)
{
    std::vector<int> deg(std::max(0, n_radial), valence_degree);
    const int a = n_radial / 4;            // inner 25% (near r=0)
    const int b = (n_radial * 7) / 10;     // valence band up to 70%
    for (int i = 0; i < n_radial; ++i)
        deg[i] = (i < a) ? inner_degree : (i < b ? valence_degree : outer_degree);
    return deg;
}

void AtomicGrid::recenter(const Vector3<double>& new_center)
{
    const Vector3<double> shift = new_center - center_;
    for (auto& p : points_) p += shift;
    center_ = new_center;
}

namespace
{
constexpr double kSsfA = 0.64;        // SSF finite-support parameter
constexpr double kSsfTol = 1.0e-13;   // GauXC ssf_weight_tol

// SSF/Frisch smooth step polynomial g(mu), s = mu/a: g = (35(s-s^3)+21 s^5-5 s^7)/16.
inline double ssf_gfrisch(double mu)
{
    const double s = mu / kSsfA;
    const double s2 = s * s, s3 = s * s2, s5 = s3 * s2, s7 = s5 * s2;
    return (35.0 * (s - s3) + 21.0 * s5 - 5.0 * s7) / 16.0;
}
} // namespace

void ssf_partition_weights(const Vector3<double>& point,
                           const std::vector<Vector3<double>>& centers,
                           std::vector<double>& p)
{
    const int n = static_cast<int>(centers.size());
    std::vector<double> dist(n);
    for (int A = 0; A < n; ++A) dist[A] = (point - centers[A]).norm();

    p.assign(n, 1.0);
    // Pairwise SSF (GauXC reference_ssf_weights_host, jA < iA branch).
    for (int iA = 0; iA < n; ++iA)
        for (int jA = 0; jA < iA; ++jA)
        {
            if (p[iA] <= kSsfTol && p[jA] <= kSsfTol) continue;
            const double rab = (centers[iA] - centers[jA]).norm();
            const double mu = (dist[iA] - dist[jA]) / rab;
            if (mu <= -kSsfA)
                p[jA] = 0.0;
            else if (mu >= kSsfA)
                p[iA] = 0.0;
            else
            {
                const double g = 0.5 * (1.0 - ssf_gfrisch(mu));
                p[iA] *= g;
                p[jA] *= 1.0 - g;
            }
        }

    double sum = 0.0;
    for (double v : p) sum += v;
    if (sum > 0.0)
        for (double& v : p) v /= sum;
}

double ssf_parent_weight(const Vector3<double>& point,
                         const std::vector<Vector3<double>>& centers,
                         int parent, double dist_nearest)
{
    const double d_parent = (point - centers[parent]).norm();
    // Deep inside the parent's cell -> partition weight is exactly 1 (SSF screen).
    if (d_parent < 0.5 * (1.0 - kSsfA) * dist_nearest) return 1.0;

    const int n = static_cast<int>(centers.size());
    std::vector<double> dist(n);
    for (int A = 0; A < n; ++A) dist[A] = (A == parent) ? d_parent : (point - centers[A]).norm();

    std::vector<double> p(n, 1.0);
    for (int iA = 0; iA < n; ++iA)
        for (int jA = 0; jA < iA; ++jA)
        {
            if (p[iA] <= kSsfTol && p[jA] <= kSsfTol) continue;
            const double rab = (centers[iA] - centers[jA]).norm();
            const double mu = (dist[iA] - dist[jA]) / rab;
            if (mu <= -kSsfA)
                p[jA] = 0.0;
            else if (mu >= kSsfA)
                p[iA] = 0.0;
            else
            {
                const double g = 0.5 * (1.0 - ssf_gfrisch(mu));
                p[iA] *= g;
                p[jA] *= 1.0 - g;
            }
        }
    if (p[parent] < kSsfTol) return 0.0;
    double sum = 0.0;
    for (double v : p) sum += v;
    return sum > 0.0 ? p[parent] / sum : 0.0;
}

} // namespace ModuleBase
