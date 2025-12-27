#include "ofdmpower.h"

namespace optimsolution {

OFDMPower::OFDMPower()
{

    setName("ofdmpower");
    setFullName("OFDM Power Allocation (sum–rate with soft power constraint)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("communication systems benchmark");


}

void OFDMPower::init(int dim) {
    int N = (dim >= 2) ? dim : 2;
    Problem::init(N);

    // Deterministic, h_i = 1 / (1 + alpha * i)
    buildChannelProfile(N);

    // Bounds: 0 ≤ p_i ≤ Pmax_
    Vec lo(N, 0.0), hi(N, Pmax_);
    setBounds(lo, hi);
}

void OFDMPower::buildChannelProfile(int N) {
    h_.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        h_[i] = 1.0 / (1.0 + alpha_ * static_cast<double>(i));
    }

    if (N >= 3) {
        h_[N/3]   *= 1.05;
        h_[2*N/3] *= 0.95;
    }
}

double OFDMPower::evaluate_core(const Vec& p) {
    const int N = static_cast<int>(p.size());
    const double invln2 = 1.0 / std::log(2.0);

    // Negative sum-rate
    double rate = 0.0;
    for (int i = 0; i < N; ++i) {
        const double snr = std::max(0.0, h_[i] * p[i] / N0_);
        rate += std::log(1.0 + snr) * invln2; // log2(1+snr)
    }
    double f = -rate;

    // Soft penalty for exceeding total power
    double sumP = 0.0;
    for (int i = 0; i < N; ++i) sumP += p[i];
    const double overflow = std::max(0.0, sumP - Ptot_);
    f += wSum_ * overflow * overflow;

    if (!std::isfinite(f)) f = 1e12;
    return f;
}

void OFDMPower::gradient_core(const Vec& x, Vec& g) {
    // Numerical forward differences
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    const double rel = 1e-6, abs = 1e-6;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        double h = std::max(abs, std::abs(x[i]) * rel);
        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        g[i] = (fp - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
