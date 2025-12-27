#include "test2n.h"
#include <cmath>

namespace optimsolution {

Test2n::Test2n()
{
    setName("test2n");
    setFullName("Separable quartic polynomial (Test2n)");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("synthetic polynomial benchmark");

}

void Test2n::init(int dim) {
    if (dim < 1) dim = 1;
    Problem::init(dim);

    Vec lo(dim, -5.0), hi(dim, 5.0);
    setBounds(lo, hi);
}

// f(x) = Σ 0.5 * (x_i^4 - 16 x_i^2 + 5 x_i)
double Test2n::evaluate_core(const Vec& x) {
    const int D = dimension();
    double f = 0.0;
    for (int i = 0; i < D; ++i) {
        const double xi = x[i];
        f += 0.5 * (xi*xi*xi*xi - 16.0*xi*xi + 5.0*xi);
    }
    return f;
}

// ∂f/∂x_i = 0.5 * (4 x_i^3 - 32 x_i + 5)
void Test2n::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();
    g.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        const double xi = x[i];
        g[i] = 0.5 * (4.0*xi*xi*xi - 32.0*xi + 5.0);
    }
}

} // namespace optimsolution
