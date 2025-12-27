#include "fmsynth.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

FMSynth::FMSynth()
    : N_(100)
{
    setName("fmsynth");
    setFullName("FM Synth Parameter Estimation");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("real-world signal processing benchmark");

    // Not calling setKnownGlobalOptimum since true x* is unknown
}

void FMSynth::setup_target() {
    // Classic “CEC-like” target FM parameters
    const double p[6] = {1.0, 5.0, 1.5, 4.8, 2.0, 4.9};
    target_.resize(N_);
    for (int n = 0; n < N_; ++n) {
        target_[n] = p[0] * std::sin(p[1] * n +
                         p[2] * std::sin(p[3] * n +
                         p[4] * std::sin(p[5] * n)));
    }
}

void FMSynth::init(int /*dim*/) {
    // Force 6 variables
    Problem::init(6);

    lo_.assign(6, -6.4);
    hi_.assign(6,  6.35);
    setBounds(lo_, hi_);

    setup_target();
}

double FMSynth::evaluate_core(const Vec& x) {
    double mse = 0.0;

    for (int n = 0; n < N_; ++n) {
        const double y =
            x[0] * std::sin(x[1] * n +
            x[2] * std::sin(x[3] * n +
            x[4] * std::sin(x[5] * n)));

        const double diff = y - target_[n];
        mse += diff * diff;
    }

    return mse / static_cast<double>(N_);
}

void FMSynth::gradient_core(const Vec& x, Vec& g) {
    g.assign(x.size(), 0.0);

    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int i = 0; i < static_cast<int>(x.size()); ++i) {
        // forward difference step inside bounds
        double h = std::max(1e-6, std::abs(x[i]) * 1e-6);
        if (x[i] + h > hi_[i])
            h = std::min(h, hi_[i] - x[i]);
        if (h <= 0.0) {
            g[i] = 0.0;
            continue;
        }

        xt[i] = x[i] + h;
        const double fp = evaluate_core(xt);
        xt[i] = x[i];

        g[i] = (fp - f0) / h;
    }
}

} // namespace optimsolution
