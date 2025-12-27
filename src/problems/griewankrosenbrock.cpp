#include "griewankrosenbrock.h"
#include <cmath>

namespace optimsolution {

GriewankRosenbrock::GriewankRosenbrock()
{
    setName("griewankrosenbrock");
    setFullName("Griewank–Rosenbrock Composition Function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    // Global minimum f = 0 at x_i = 1 for all i.
}

void GriewankRosenbrock::init(int dim)
{
    if (dim < 2) dim = 2;
    Problem::init(dim);

    Vec lo(dim, -5.0), hi(dim, 5.0);
    setBounds(lo, hi);

    // x* = (1,1,...,1)
    Vec xopt(dim, 1.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double GriewankRosenbrock::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double f = 0.0;

    for (int i = 0; i < D - 1; ++i) {
        const double xi  = x[i];
        const double xip = x[i + 1];
        const double r   = xi * xi - xip;             // Rosenbrock chain
        const double z   = 100.0 * r * r + (xi - 1.0) * (xi - 1.0);
        f += (z / 4000.0) - std::cos(z) + 1.0;        // Griewank over z_i
    }

    return std::isfinite(f) ? f : 1e12;
}

void GriewankRosenbrock::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    for (int i = 0; i < D - 1; ++i) {
        const double xi  = x[i];
        const double xip = x[i + 1];
        const double r   = xi * xi - xip;
        const double z   = 100.0 * r * r + (xi - 1.0) * (xi - 1.0);

        // d/dz (z/4000 − cos z + 1) = 1/4000 + sin(z)
        const double gp = 1.0/4000.0 + std::sin(z);

        // chain rule: dz/dxi, dz/dx_{i+1}
        const double dz_dxi  = 400.0 * xi * r + 2.0 * (xi - 1.0);
        const double dz_dxip = -200.0 * r;

        g[i]     += gp * dz_dxi;
        g[i + 1] += gp * dz_dxip;
    }
}

} // namespace optimsolution
