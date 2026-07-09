#include "test30n.h"
#include <cmath>

namespace optimsolution {

Test30n::Test30n()
{
    setName("test30n");
    setFullName("Oscillatory non-separable benchmark (Test30n)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
  
}

void Test30n::init(int dim) {
    if (dim < 1) dim = 1;
    Problem::init(dim);

    // limits [-10, 10]^D
    Vec lo(dim, -10.0), hi(dim, 10.0);
    setBounds(lo, hi);

    // Global optimum: x* = (1,...,1), f* = 0
    Vec xopt(dim, 1.0);
    setKnownGlobalOptimum(0.0, xopt);
}

// f(x) = 0.1 * [ sin^2(3π x0) + Σ_{i=0}^{D-2} (x_i - 1)^2 (1 + sin^2(3π x_{i+1}))
//               + (x_{D-1} - 1)^2 (1 + sin^2(2π x_{D-1})) ]
//
// This is the standard "Generalized Penalized Function 2" (Yao, Liu & Lin,
// 1999). A previous version of this code multiplied sin^2(3π x0) by the
// sum instead of adding it, started the sum at i=1 instead of i=0 (silently
// dropping the (x0-1)^2(1+sin^2(3π x1)) term), and left the last term
// outside the 0.1 scaling — all three have been corrected here.
double Test30n::evaluate_core(const Vec& x) {
    const int D = dimension();
    const double pi = 3.1415926535897932384626433832795;

    double sum = 0.0;
    for (int i = 0; i < D - 1; ++i) {
        const double xi   = x[i];
        const double xip1 = x[i + 1];
        const double a = xi - 1.0;
        const double s = std::sin(3.0 * pi * xip1);
        sum += (a * a) * (1.0 + s * s);
    }

    const double s0 = std::sin(3.0 * pi * x[0]);

    const double xlast = x[D - 1];
    const double alast = xlast - 1.0;
    const double sl    = std::sin(2.0 * pi * xlast);

    return 0.1 * (s0 * s0 + sum + (alast * alast) * (1.0 + sl * sl));
}


// Gradient of f(x) = 0.1 * [ sin^2(3π x0) + Σ_{i=0}^{D-2} (x_i-1)^2 (1+sin^2(3π x_{i+1}))
//                            + (x_{D-1}-1)^2 (1+sin^2(2π x_{D-1})) ]
void Test30n::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();
    g.assign(D, 0.0);

    const double pi = 3.1415926535897932384626433832795;

    // d/dx0 [ sin^2(3*pi*x0) ] = 2*sin(3*pi*x0)*cos(3*pi*x0)*3*pi
    const double s0 = std::sin(3.0 * pi * x[0]);
    const double c0 = std::cos(3.0 * pi * x[0]);
    g[0] += 2.0 * s0 * c0 * (3.0 * pi);

    // Σ_{i=0}^{D-2} (x_i-1)^2 (1+sin^2(3*pi*x_{i+1}))
    for (int i = 0; i < D - 1; ++i) {
        const double a    = x[i] - 1.0;
        const double sip1 = std::sin(3.0 * pi * x[i + 1]);
        const double cip1 = std::cos(3.0 * pi * x[i + 1]);

        g[i]     += 2.0 * a * (1.0 + sip1 * sip1);
        g[i + 1] += (a * a) * 2.0 * sip1 * cip1 * (3.0 * pi);
    }

    // last term: (x_{D-1}-1)^2 (1+sin^2(2*pi*x_{D-1}))
    const double alast = x[D - 1] - 1.0;
    const double sl    = std::sin(2.0 * pi * x[D - 1]);
    const double cl    = std::cos(2.0 * pi * x[D - 1]);

    g[D - 1] += 2.0 * alast * (1.0 + sl * sl)
              + (alast * alast) * 2.0 * sl * cl * (2.0 * pi);

    // overall 0.1 scale applies to every term above
    for (int i = 0; i < D; ++i) g[i] *= 0.1;
}

} // namespace optimsolution
