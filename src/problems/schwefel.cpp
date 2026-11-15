#include "schwefel.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Schwefel::Schwefel()
{
    setName("schwefel");
    setFullName("Schwefel 2.26 function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

}

void Schwefel::init(int dim) {
    if (dim < 1) dim = 1;
    Problem::init(dim);

    Vec lo(dim, -500.0), hi(dim, 500.0);
    setBounds(lo, hi);

    // Global optimum x_i ≈ 420.968746..., f* ≈ 0
    Vec xopt(dim, 420.9687462275036);
    setKnownGlobalOptimum(0.0, xopt);
}

double Schwefel::evaluate_core(const Vec& x) {
    const int n = static_cast<int>(x.size());
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        sum += xi * std::sin(std::sqrt(std::fabs(xi)));
    }
    double f = 418.9829 * n - sum;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void Schwefel::gradient_core(const Vec& x, Vec& g) {
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
