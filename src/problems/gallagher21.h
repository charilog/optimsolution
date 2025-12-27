#pragma once
#include "problem.h"
#include <vector>

namespace optimsolution {

/**
 * Gallagher's Gaussian 21-me Peaks (BBOB f22), faithful to COCO/BBOB.
 *
 * - Continuous, multimodal, non-separable
 * - Bounds: [-5, 5]^D
 * - Deterministic instance given a seed (default: 42)
 * - One global rotation R, 21 peaks with randomized centers and shapes
 * - Scalar Tosz transform on outer expression + box penalty
 *
 * f(x) = tosz(10 - max_i w_i * exp(-0.5/D * (R (x - y_i))^T C_i (R (x - y_i))))^2
 *        + f_pen(x),
 *
 * where f_pen(x) = sum_j max(0, |x_j| - 5)^2 and f_opt = 0.
 */
class Gallagher21 : public Problem {
public:
    Gallagher21();

    // D >= 1 ( D>=2 at BBOB)
    void init(int dim) override;

    // Optionally change the seed before init() to get a different instance.
    void setSeed(unsigned seed) { seed_ = seed; }

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    // ---- instance params ----
    unsigned seed_ = 42;
    int D_ = 0;
    int K_ = 21;                // number of peaks

    // Peaks
    std::vector<Vec> Y_;        // centers y_i
    std::vector<double> w_;     // weights
    std::vector<Vec> diagCi_;   // diagonal entries of Ci after random permutation

    // Global rotation R (row-major D x D)
    std::vector<double> R_;

    // helpers
    void build_instance();
    void make_rotation(std::vector<double>& M);
    static double sqr(double v) { return v * v; }

    // BBOB utilities
    static double f_pen(const Vec& x);     // Σ max(0,|x|-5)^2
    static double tosz_scalar(double x);  // scalar Tosz (like COCO)
};

} // namespace optimsolution
