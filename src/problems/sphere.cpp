#include "sphere.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------
// Sphere metadata:
// - f(x) = sum_{i=1}^D x_i^2
// - Known global optimum: f* = 0 at x* = (0,...,0)
// - Modality:   unimodal
// - Separability: separable
// - Type/category: continuous benchmark
// - Typical bounds: [-100, 100]^D
// -----------------------------------------------------

Sphere::Sphere()
{
    setName("sphere");
    setFullName("Sphere benchmark function");
    setModality("unimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Sphere::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -100.0), u(dim, 100.0);
    setBounds(l, u);

    // Global minimizer x* = (0,...,0)
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Sphere::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double s = 0.0;
    for (int i = 0; i < D; ++i) {
        s += x[i] * x[i];
    }
    return s;
}

void Sphere::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);
    for (int i = 0; i < D; ++i) {
        g[i] = 2.0 * x[i];
    }
}

} // namespace optimsolution
