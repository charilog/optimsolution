#include "diracproblem.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

DiracProblem::DiracProblem()
    : mu_()
    , sigma_(0.01)  // default: very narrow spike
{
    setName("dirac");
    setFullName("Dirac-like Gaussian spike function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous synthetic benchmark test function");

    // Location of the optimum will be set once dimension and μ are known in init()
    setKnownGlobalOptimum(0.0);
}

void DiracProblem::init(int dim)
{
    if (dim < 1)
        dim = 1;

    Problem::init(dim);

    // Default bounds
    Vec lo(dim, -1.0), hi(dim, 1.0);
    setBounds(lo, hi);

    // Default center μ = 0 with correct dimension
    mu_.assign(dim, 0.0);

    // Global minimum f* = 0 at x* = μ
    setKnownGlobalOptimum(0.0, mu_);
}

void DiracProblem::setCenter(const Vec& mu)
{
    const int D = dimension();
    mu_.assign(D, 0.0);
    for (int i = 0; i < D && i < static_cast<int>(mu.size()); ++i)
        mu_[i] = mu[i];

    // Update known optimum location (value remains 0)
    setKnownGlobalOptimum(0.0, mu_);
}

void DiracProblem::setSigma(double s)
{
    if (s > 0.0)
        sigma_ = s;
}

// f(x) = 1 - exp( -||x - μ||^2 / (2 σ^2) )
double DiracProblem::evaluate_core(const Vec& x)
{
    const int D = dimension();

    // Guard against invalid sigma
    if (!(sigma_ > 0.0) || std::isnan(sigma_) || std::isinf(sigma_))
        return 1e12;

    double r2 = 0.0;
    for (int i = 0; i < D; ++i) {
        const double d = x[i] - mu_[i];
        r2 += d * d;
    }

    const double denom = 2.0 * sigma_ * sigma_;
    const double e = std::exp(-r2 / denom);
    const double f = 1.0 - e;

    return std::isfinite(f) ? f : 1e12;
}

// ∇f(x) = exp(-r2 / (2 σ^2)) * (x - μ) / σ^2
void DiracProblem::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);

    if (!(sigma_ > 0.0) || std::isnan(sigma_) || std::isinf(sigma_))
        return;

    double r2 = 0.0;
    for (int i = 0; i < D; ++i) {
        const double d = x[i] - mu_[i];
        r2 += d * d;
    }

    const double denom  = 2.0 * sigma_ * sigma_;
    const double e      = std::exp(-r2 / denom);
    const double scale  = e / (sigma_ * sigma_);

    for (int i = 0; i < D; ++i)
        g[i] = scale * (x[i] - mu_[i]);
}

} // namespace optimsolution
