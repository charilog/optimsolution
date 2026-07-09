#include "bucherastrigin.h"
#include <cmath>
#include <vector>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;

    // Same T_osz transform used by gallagher101.cpp / attractivesector.cpp.
    double tosz_scalar(double x)
    {
        if (x == 0.0)
            return 0.0;
        double sgn = (x > 0.0 ? 1.0 : -1.0);
        double h   = std::log(std::abs(x));
        double c1  = (x > 0.0 ? 10.0 : 5.5);
        double c2  = (x > 0.0 ? 7.9  : 3.1);
        return sgn * std::exp(h + 0.049 * (std::sin(c1 * h) + std::sin(c2 * h)));
    }

    // Standard BBOB boundary penalty, used by every other BBOB-style
    // function in this codebase (gallagher101.cpp, the corrected
    // attractivesector.cpp) but previously missing here.
    double f_pen(const std::vector<double>& x)
    {
        double s = 0.0;
        for (double xi : x) {
            double a = std::abs(xi) - 5.0;
            if (a > 0.0) s += a * a;
        }
        return s;
    }
}

// -----------------------------------------------------------------------------
// Buche-Rastrigin (BBOB F4), faithful implementation.
//
//   f(x) = 10*(D - sum_i cos(2*pi*z_i)) + sum_i z_i^2 + 100*f_pen(x)
//   z_i  = s_i * T_osz(x_i)                                  (x_opt = 0)
//   s_i  = 10 * 10^(0.5*i/(D-1))   if T_osz(x_i) > 0 AND i is ODD (1-indexed)
//        =      10^(0.5*i/(D-1))   otherwise
//
// FIX vs. the previous version of this file, which was missing THREE parts
// of the standard definition:
//   1) T_osz was never applied to x_i before scaling (z_i = x_i * s_i
//      directly) — T_osz is what gives every BBOB function its
//      characteristic small-scale irregularity/roughness.
//   2) The x10 stretch was applied to EVERY coordinate with x_i > 0, not
//      only to ODD-indexed (1-indexed) ones. This "checkerboard" asymmetry
//      across alternating dimensions is a defining, documented feature of
//      Buche-Rastrigin specifically (it's what distinguishes it from a
//      generically-asymmetric Rastrigin variant) — stretching every
//      positive coordinate instead changes the function's conditioning
//      pattern and difficulty substantially.
//   3) The 100*f_pen(x) boundary penalty was entirely absent, so a search
//      that stepped outside [-5,5]^D paid no explicit cost for it here
//      (unlike every other BBOB-style function in this codebase).
// The core algebraic identity 10*(D - sum cos(2*pi*z)) + sum z^2
//   == 10*D + sum(z^2 - 10*cos(2*pi*z))
// was already correct and is preserved.
// -----------------------------------------------------------------------------

BucheRastrigin::BucheRastrigin()
{
    setName("bucherastrigin");
    setFullName("Buche-Rastrigin function (BBOB-style variant)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("BBOB synthetic benchmark");

    setKnownGlobalOptimum(0.0);   // value
}

void BucheRastrigin::init(int dim)
{
    Problem::init(dim);

    // Standard domain for BBOB F4
    Vec l(dim, -5.0);
    Vec u(dim,  5.0);
    setBounds(l, u);

    // Global minimizer (deterministic version)
    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.0, xopt);

    // Precompute the base conditioning exponent 10^(0.5*i/(D-1)); the extra
    // x10 factor for odd (1-indexed) coordinates with T_osz(x_i) > 0 is
    // applied at evaluation time in evaluate_core(), since it depends on x.
    scale_.resize(dim);
    if (dim > 1) {
        for (int i = 0; i < dim; ++i)
            scale_[i] = std::pow(10.0, 0.5 * (double(i) / double(dim - 1)));
    } else if (dim == 1) {
        scale_[0] = 1.0;
    }
}

double BucheRastrigin::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;

    for (int i = 0; i < D; ++i) {
        const double xt = tosz_scalar(x[i]);

        // "i is odd" in the standard 1-indexed convention == "i is even"
        // in this 0-indexed loop (i=0 <-> 1-indexed i=1, etc.).
        const bool odd_1indexed = ((i % 2) == 0);

        double s = scale_[i];
        if (xt > 0.0 && odd_1indexed)
            s *= 10.0;

        const double z = s * xt;
        sum += z * z - 10.0 * std::cos(2.0 * PI * z);
    }

    const double fval = 10.0 * D + sum + 100.0 * f_pen(x);

    if (!(fval >= 0.0) || std::isnan(fval) || std::isinf(fval))
        return 1e12;

    return fval;
}

void BucheRastrigin::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences: T_osz's derivative is fiddly enough
    // (and this is a derivative-free benchmark suite anyway, per
    // gallagher101.cpp's same choice) that a finite-difference gradient is
    // the safer, less error-prone option here.
    const int D = dimension();
    g.assign(D, 0.0);

    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int k = 0; k < D; ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * 1e-6);
        if (x[k] + h > 5.0)
            h = std::min(h, 5.0 - x[k]);
        if (h <= 0.0) { g[k] = 0.0; continue; }

        xt[k] = x[k] + h;
        const double f1 = evaluate_core(xt);
        g[k] = (f1 - f0) / h;
        xt[k] = x[k];
    }
}

} // namespace optimsolution
