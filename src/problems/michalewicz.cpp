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
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
