#include "perm.h"
#include <cmath>
#include <vector>

namespace optimsolution {

Perm::Perm()
{
    setName("perm");
    setFullName("Perm function (d, beta)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");
}

void Perm::init(int dim)
{
    if (dim < 1) dim = 2;
    Problem::init(dim);

    const double b = static_cast<double>(dim);
    Vec lo(dim, -b), hi(dim, b);
    setBounds(lo, hi);

    // x*_i = i   (1-indexed)
    Vec xopt(dim, 0.0);
    for (int i = 1; i <= dim; ++i) xopt[i - 1] = static_cast<double>(i);
    setKnownGlobalOptimum(0.0, xopt);
}

// inner_i = Sum_{j=1}^{n} (j^i + beta) * ( (x_j/j)^i - 1 )   (1-indexed i,j)
// f = Sum_i inner_i^2
double Perm::evaluate_core(const Vec& x)
{
    const int n = dimension();
    double f = 0.0;

    for (int i = 1; i <= n; ++i) {
        double inner = 0.0;
        for (int j = 1; j <= n; ++j) {
            const double jp = std::pow(static_cast<double>(j), i);
            const double ratio = x[j - 1] / static_cast<double>(j);
            inner += (jp + beta_) * (std::pow(ratio, i) - 1.0);
        }
        f += inner * inner;
    }
    return f;
}

// d(inner_i)/dx_k = (k^i + beta) * i * x_k^(i-1) / k^i
// df/dx_k = Sum_i 2*inner_i * d(inner_i)/dx_k
void Perm::gradient_core(const Vec& x, Vec& g)
{
    const int n = dimension();
    g.assign(n, 0.0);

    // Precompute inner_i for all i (needed by every x_k derivative below).
    std::vector<double> inner(n + 1, 0.0); // 1-indexed, inner[0] unused
    for (int i = 1; i <= n; ++i) {
        double s = 0.0;
        for (int j = 1; j <= n; ++j) {
            const double jp = std::pow(static_cast<double>(j), i);
            const double ratio = x[j - 1] / static_cast<double>(j);
            s += (jp + beta_) * (std::pow(ratio, i) - 1.0);
        }
        inner[i] = s;
    }

    for (int k = 1; k <= n; ++k) {
        double gk = 0.0;
        const double kp_base = static_cast<double>(k);
        for (int i = 1; i <= n; ++i) {
            const double kp = std::pow(kp_base, i);
            const double dinner_dxk = (kp + beta_) * static_cast<double>(i)
                                     * std::pow(x[k - 1], i - 1) / kp;
            gk += 2.0 * inner[i] * dinner_dxk;
        }
        g[k - 1] = gk;
    }
}

} // namespace optimsolution
