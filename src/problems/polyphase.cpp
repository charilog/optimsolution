#include "polyphase.h"

namespace optimsolution {

Polyphase::Polyphase()
{
    setName("polyphase");
    setFullName("Polyphase sequence PSL minimization (aperiodic autocorrelation)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("sequence design / communication systems benchmark");

 
}

void Polyphase::init(int dim) {
    if (dim < 2) dim = 2;               
    Problem::init(dim);

    Vec lo(dim, 0.0), hi(dim, 2.0 * PI());
    setBounds(lo, hi);
}

void Polyphase::computePhis(const Vec& x, std::vector<double>& phis) const {
    const int n = static_cast<int>(x.size());
    phis.clear();
    phis.reserve(std::max(0, n - 1));

    for (int k = 1; k <= n - 1; ++k) {
        const int limit = n - k;  // aperiodic window
        double sum = 0.0;
        for (int j = 0; j < limit; ++j) {
            sum += std::cos(x[j] - x[j + k]);  // Re{ e^{ix_j} e^{-ix_{j+k}} }
        }
        phis.push_back(sum);
    }
}

double Polyphase::evaluate_core(const Vec& x) {
    // Compute aperiodic sidelobes και PSL
    std::vector<double> phis;
    computePhis(x, phis);

    double max_abs_phi = 0.0;
    for (double v : phis) {
        const double a = std::abs(v);
        if (a > max_abs_phi) max_abs_phi = a;
    }

    if (!std::isfinite(max_abs_phi)) return 1e12;
    return max_abs_phi;
}

void Polyphase::gradient_core(const Vec& x, Vec& g) {
    // Forward finite differences 
    g.assign(x.size(), 0.0);
    const double f0 = evaluate_core(x);

    Vec xt = x;
    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        const double h = std::max(1e-6, std::abs(x[i]) * 1e-6);
        xt[i] = x[i] + h;
        const double f1 = evaluate_core(xt);
        g[i] = (f1 - f0) / h;
        xt[i] = x[i];
    }
}

} // namespace optimsolution
