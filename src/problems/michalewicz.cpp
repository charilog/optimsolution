#include "michalewicz.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Michalewicz::Michalewicz()
    : n_(0)
    , m_(10.0)
{
    setName("michalewicz");
    setFullName("Michalewicz function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");


}

void Michalewicz::init(int dim) {
    if (dim < 1) dim = 1;
    n_ = dim;
    Problem::init(n_);

    // Standard bounds [0, pi]^n
    const double PI = 3.141592653589793238462643383279502884;
    Vec lo(n_, 0.0), hi(n_, PI);
    setBounds(lo, hi);
}

double Michalewicz::evaluate_core(const Vec& x) {
    const double PI = 3.141592653589793238462643383279502884;

    double sum = 0.0;
    for (int i = 0; i < n_; ++i) {
        const double xi   = x[i];
        const double arg  = (static_cast<double>(i+1) * xi * xi) / PI;
        const double sxi  = std::sin(xi);
        const double sarg = std::sin(arg);
        // term_i = sin(xi) * sin(arg)^(2m)
        sum += sxi * std::pow(sarg, 2.0 * m_);
    }
    double f = -sum;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void Michalewicz::gradient_core(const Vec& x, Vec& g) {
    // Forward differences (όπως στο αρχικό)
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int i = 0; i < n_; ++i) {
        double h = std::max(1e-6, std::abs(x[i]) * 1e-6);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
