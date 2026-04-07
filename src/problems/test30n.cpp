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

// f(x) = 0.1 * sin^2(3π x0) * Σ_{i=1..D-2} (x_i - 1)^2 (1 + sin^2(3π x_{i+1}))
//        + (x_{D-1} - 1)^2 (1 + sin^2(2π x_{D-1}))
double Test30n::evaluate_core(const Vec& x) {
    const int D = dimension();
    const double pi = 3.1415926535897932384626433832795;

    double sum = 0.0;
    if (D >= 3) {
        for (int i = 1; i < D - 1; ++i) {
            const double xi   = x[i];
            const double xip1 = x[i + 1];
            const double a = xi - 1.0;
            const double s = std::sin(3.0 * pi * xip1);
            sum += (a * a) * (1.0 + s * s);
        }
    }

    const double s0 = std::sin(3.0 * pi * x[0]);
    double f = 0.1 * (s0 * s0) * sum;

    
    const double xlast = x[D - 1];
    const double alast = xlast - 1.0;
    const double sl    = std::sin(2.0 * pi * xlast);
    f += (alast * alast) * (1.0 + sl * sl);

    return f;
}


// g0 = 0.1 * Σ (...) * 2 sin(3π x0) * (3π cos(3π x0))
// gi     += 2(xi-1)(1+sin^2(3π x_{i+1})) * m,  με m = 0.1 sin^2(3π x0)
// g_{i+1}+= (xi-1)^2 * 2*3π sin(3π x_{i+1}) cos(3π x_{i+1}) * m
// g_{D-1}+= 2(x_{D-1}-1)(1+sin^2(2π x_{D-1})) +
//           (x_{D-1}-1)^2 * 2*2π sin(2π x_{D-1}) cos(2π x_{D-1})
void Test30n::gradient_core(const Vec& x, Vec& g) {
    const int D = dimension();
    g.assign(D, 0.0);

    const double pi = 3.1415926535897932384626433832795;

    // sum για g0
    double sum = 0.0;
    if (D >= 3) {
        for (int i = 1; i < D - 1; ++i) {
            const double xi   = x[i];
            const double xip1 = x[i + 1];
            const double a = xi - 1.0;
            const double s = std::sin(3.0 * pi * xip1);
            sum += (a * a) * (1.0 + s * s);
        }
    }

    const double s0 = std::sin(3.0 * pi * x[0]);
    const double c0 = std::cos(3.0 * pi * x[0]);
    const double m  = 0.1 * (s0 * s0); // 0.1 * sin^2(3π x0)

    // g0
    g[0] = 0.1 * sum * 2.0 * s0 * (3.0 * pi * c0);

    // ζεύγη (i, i+1)
    if (D >= 3) {
        for (int i = 1; i < D - 1; ++i) {
            const double xi   = x[i];
            const double xip1 = x[i + 1];

            const double a    = xi - 1.0;
            const double sip1 = std::sin(3.0 * pi * xip1);
            const double cip1 = std::cos(3.0 * pi * xip1);

            g[i]     += 2.0 * a * (1.0 + sip1 * sip1) * m;
            g[i + 1] += (a * a) * (2.0 * 3.0 * pi * sip1 * cip1) * m;
        }
    }

    // τελευταίος δείκτης
    const double xlast = x[D - 1];
    const double alast = xlast - 1.0;
    const double sl    = std::sin(2.0 * pi * xlast);
    const double cl    = std::cos(2.0 * pi * xlast);

    g[D - 1] += 2.0 * alast * (1.0 + sl * sl)
              + (alast * alast) * (2.0 * 2.0 * pi * sl * cl);
}

} // namespace optimsolution
