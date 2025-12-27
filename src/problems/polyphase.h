#pragma once
#include "problem.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * Polyphase PSL minimization.
 *
 * Variables: phases x_j in [0, 2π], j = 0..n-1 (n = dimension passed to init()).
 * Aperiodic autocorrelation sidelobes:
 *   φ_k = sum_{j=0}^{n-1-k} cos(x_j - x_{j+k}),   k = 1..n-1
 * Objective: PSL(x) = max_k |φ_k|   (to be MINIMIZED)
 */
class Polyphase : public Problem {
public:
    Polyphase();                              
    void init(int dim) override;              // sets dimension and [0,2π] bounds

protected:
    double evaluate_core(const Vec& x) override;  // returns PSL(x)
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward differences

private:
    static inline double PI() { return 3.141592653589793238462643383279502884; }

    // Compute φ_k for k=1..n-1 (aperiodic window)
    void computePhis(const Vec& x, std::vector<double>& phis) const;
};

} // namespace optimsolution
