#include "langermann.h"
#include <cmath>

namespace optimsolution {

namespace { constexpr double LANG_PI = 3.141592653589793238462643383279502884; }

Langermann::Langermann()
{
    setName("langermann");
    setFullName("Langermann function (2D, m=5)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    A_ = { {3.0, 5.0}, {5.0, 2.0}, {2.0, 1.0}, {1.0, 4.0}, {7.0, 9.0} };
    c_ = { 1.0, 2.0, 5.0, 2.0, 3.0 };

    setKnownGlobalOptimum(-5.1621);
}

void Langermann::init(int /*dim*/)
{
    Problem::init(2);

    Vec lo(2, 0.0), hi(2, 10.0);
    setBounds(lo, hi);

    Vec xopt = {2.00299, 1.006};
    setKnownGlobalOptimum(-5.1621, xopt);
}

double Langermann::evaluate_core(const Vec& x)
{
    double f = 0.0;
    for (size_t i = 0; i < c_.size(); ++i) {
        double s = 0.0;
        for (int j = 0; j < 2; ++j) {
            const double d = x[j] - A_[i][j];
            s += d * d;
        }
        f -= c_[i] * std::exp(-s / LANG_PI) * std::cos(LANG_PI * s);
    }
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

// f = -Sum_i c_i*exp(-s_i/pi)*cos(pi*s_i), s_i = Sum_j (x_j-A_ij)^2
// df/dx_k = Sum_i c_i*exp(-s_i/pi)*2*(x_k-A_ik)*[cos(pi*s_i)/pi + pi*sin(pi*s_i)]
void Langermann::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    for (size_t i = 0; i < c_.size(); ++i) {
        double s = 0.0;
        double d[2];
        for (int j = 0; j < 2; ++j) {
            d[j] = x[j] - A_[i][j];
            s += d[j] * d[j];
        }
        const double e = std::exp(-s / LANG_PI);
        const double bracket = std::cos(LANG_PI * s) / LANG_PI + LANG_PI * std::sin(LANG_PI * s);
        const double coeff = c_[i] * e * bracket;

        g[0] += coeff * 2.0 * d[0];
        g[1] += coeff * 2.0 * d[1];
    }
}

} // namespace optimsolution
