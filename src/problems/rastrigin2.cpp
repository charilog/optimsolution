#include "rastrigin2.h"
#include <cmath>

namespace optimsolution {

Rastrigin2::Rastrigin2()
{
    setName("rastrigin2");
    setFullName("2D Rastrigin function (shifted, f* = -2)");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");

    // Global optimum: (0,0), f* = -2
    Vec xopt(2, 0.0);
    setKnownGlobalOptimum(-2.0, xopt);
}

void Rastrigin2::init(int /*dim*/) {
    // Always 2D
    Problem::init(2);
    Vec lo(2, -5.12), hi(2, 5.12);
    setBounds(lo, hi);
}

// f(x,y) = [x^2 - 10 cos(2πx) + 10] + [y^2 - 10 cos(2πy) + 10] - 2
double Rastrigin2::evaluate_core(const Vec& x) {
    const double pi = 3.1415926535897932384626433832795;
    const double X = x[0], Y = x[1];
    const double fx = X*X - 10.0 * std::cos(2.0 * pi * X) + 10.0;
    const double fy = Y*Y - 10.0 * std::cos(2.0 * pi * Y) + 10.0;
    return (fx + fy) - 2.0; // shift for f* = -2
}

// ∇f = (2x + 20π sin(2πx),  2y + 20π sin(2πy))
void Rastrigin2::gradient_core(const Vec& x, Vec& g) {
    g.assign(2, 0.0);
    const double pi = 3.1415926535897932384626433832795;
    const double X = x[0], Y = x[1];
    g[0] = 2.0 * X + 20.0 * pi * std::sin(2.0 * pi * X);
    g[1] = 2.0 * Y + 20.0 * pi * std::sin(2.0 * pi * Y);
}

} // namespace optimsolution
