#include "rastrigin.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

// -----------------------------------------------------
// Rastrigin metadata:
// - Known global optimum: f* = 0 at x* = (0,...,0)
// - Modality:   multimodal
// - Separability: separable
// - Type/category: continuous benchmark
// - Bounds: [-5.12, 5.12]^D
// -----------------------------------------------------

Rastrigin::Rastrigin()
{

    setName("rastrigin");
    setFullName("Rastrigin benchmark function");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Rastrigin::init(int dim)
{
    Problem::init(dim);


    Vec l(dim, -5.12), u(dim, 5.12);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Rastrigin::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double s = 10.0 * D;
    for (int j = 0; j < D; ++j) {
        s += x[j]*x[j] - 10.0 * std::cos(2.0 * PI * x[j]);
    }
    return s;
}

// ∂/∂x_j [x_j^2 - 10 cos(2πx_j)] = 2x_j + 20π sin(2πx_j)
void Rastrigin::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);
    for (int j = 0; j < D; ++j) {
        g[j] = 2.0 * x[j] + 20.0 * PI * std::sin(2.0 * PI * x[j]);
    }
}

} // namespace optimsolution
