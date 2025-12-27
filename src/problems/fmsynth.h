#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * FM Synthesizer Parameter Estimation Problem
 *
 * Variables x ∈ [-6.4, 6.35]^6
 * Objective: MSE between generated FM signal and fixed target signal (N samples).
 *
 * No known analytic optimum.
 */
class FMSynth : public Problem {
public:
    FMSynth();
    void init(int dim) override;                  // forces D = 6

protected:
    double evaluate_core(const Vec& x) override;  // MSE
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    int N_;                          // number of samples (default = 100)
    std::vector<double> target_;     // target FM waveform
    std::vector<double> lo_, hi_;    // bounds used also for FD gradient steps

    void setup_target();             // fills target_ using classic p-vector from literature
};

} // namespace optimsolution
