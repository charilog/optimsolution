#pragma once
#include "problem.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * OFDM Power Allocation (faithful surrogate).
 *
 * Decision: p[0..N-1] in [0, Pmax], N = dimension passed to init().
 * Objective (minimize):
 *   f(p) = - sum_{i=0}^{N-1} log2( 1 + h_i * p_i / N0 )
 *          + wSum * max(0, sum_i p_i - Ptot)^2
 *
 * Channel profile h_i is a deterministic, monotone-decreasing sequence
 * (no randomness) so that the problem is reproducible.
 */
class OFDMPower : public Problem {
public:
    OFDMPower();

    // dim = number of subcarriers (N). If dim<2 -> force N=2.
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& p) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // Fixed defaults (same semantics as your reference)
    double Ptot_   = 1.0;     // total power budget
    double Pmax_   = 1.0;     // per-tone upper bound
    double N0_     = 1e-3;    // noise PSD (normalized)
    double wSum_   = 100.0;   // weight for sum-power overflow penalty
    double alpha_  = 0.15;    // slope for deterministic h_i profile

    std::vector<double> h_;   // channel gains (size N)

    void buildChannelProfile(int N); // fills h_ deterministically from N
};

} // namespace optimsolution
