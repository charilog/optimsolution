#include "portfoliomv.h"

namespace optimsolution {

PortfolioMV::PortfolioMV()
{
 
    setName("portfoliomv");
    setFullName("Markowitz Mean–Variance Portfolio (long-only, soft sum-to-one)");
    setModality("unknown");
    setSeparability("non-separable");
    setCategory("financial optimization benchmark");

}

void PortfolioMV::init(int dim) {
    N_ = (dim >= 2) ? dim : 2;
    Problem::init(N_);

    buildProfiles();

    // Long-only bounds: 0 ≤ w_i ≤ 1
    Vec lo(N_, 0.0), hi(N_, 1.0);
    setBounds(lo, hi);
}

void PortfolioMV::buildProfiles() {
    mu_.assign(N_, 0.0);
    cov_.assign(N_ * N_, 0.0);

    // μ_i: linearly from 2% to 8% (if N_==1, set 5%)
    if (N_ == 1) {
        mu_[0] = 0.05;
    } else {
        for (int i = 0; i < N_; ++i) {
            mu_[i] = 0.02 + 0.06 * (double(i) / double(N_ - 1));
        }
    }

    // Σ_{ij} = σ_i σ_j ρ^{|i-j|}
    for (int i = 0; i < N_; ++i) {
        const double si = sigma_ * (N_ > 1 ? (1.0 + 0.2 * double(i) / double(N_-1)) : 1.0);
        for (int j = 0; j < N_; ++j) {
            const double sj = sigma_ * (N_ > 1 ? (1.0 + 0.2 * double(j) / double(N_-1)) : 1.0);
            const int   d   = std::abs(i - j);
            const double corr = std::pow(rho_, d);
            cov_[N_*i + j] = si * sj * corr;
        }
    }
    // μικρό ridge για SPD
    for (int i = 0; i < N_; ++i) cov_[N_*i + i] += ridge_;
}

double PortfolioMV::evaluate_core(const Vec& w) {
    // w^T Σ w
    double var = 0.0;
    for (int i = 0; i < N_; ++i)
        for (int j = 0; j < N_; ++j)
            var += w[i] * cov(i, j) * w[j];

    // μ^T w
    double ret = 0.0;
    for (int i = 0; i < N_; ++i) ret += mu_[i] * w[i];

    // (sum(w)-1)^2
    double sumw = 0.0;
    for (int i = 0; i < N_; ++i) sumw += w[i];
    const double sum_pen = (sumw - 1.0) * (sumw - 1.0);

    double f = w_risk_ * var - w_ret_ * ret + w_sum_ * sum_pen;
    if (!std::isfinite(f)) f = 1e12;
    return f;
}

Vec PortfolioMV::evaluateMultiCore(const Vec& w) {
    // Same underlying quantities as evaluate_core(), but reported as two
    // separate, un-weighted criteria instead of being combined into one
    // scalar. Both are "to minimize": risk (variance) and -(expected return).
    double var = 0.0;
    for (int i = 0; i < N_; ++i)
        for (int j = 0; j < N_; ++j)
            var += w[i] * cov(i, j) * w[j];

    double ret = 0.0;
    for (int i = 0; i < N_; ++i) ret += mu_[i] * w[i];

    double sumw = 0.0;
    for (int i = 0; i < N_; ++i) sumw += w[i];
    const double sum_pen = w_sum_ * (sumw - 1.0) * (sumw - 1.0);

    double f1 = var + sum_pen;
    double f2 = -ret + sum_pen;
    if (!std::isfinite(f1)) f1 = 1e12;
    if (!std::isfinite(f2)) f2 = 1e12;
    return { f1, f2 };
}

void PortfolioMV::gradient_core(const Vec& x, Vec& g) {
    // Numerical forward differences
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < (int)x.size(); ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
