#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig;

// MMES -- Mixture Model-based Evolution Strategy.
// Reference: He, X.; Zheng, Z.; Zhou, Y. "MMES: Mixture Model-Based
//   Evolution Strategy for Large-Scale Optimization." IEEE Transactions
//   on Evolutionary Computation, 25(2), 320-333, 2021 (arXiv:2203.12675).
//
// This implementation is a direct, verified C++ port of the paper's full
// Algorithm 1 pseudocode and Table I parameter settings (retrieved and
// checked term-for-term, same standard applied to the other large-scale
// methods in this framework).
//
// Motivation for adding this alongside LM-CMA-ES/LM-MA-ES/Rm-ES/VkD-CMA-ES:
// those methods (the "first scheme" in the paper's own taxonomy) all use
// EVERY one of their m stored direction vectors on EVERY sample, which
// forces m to stay small (O(1) to O(log n)) since sampling cost scales
// with m -- capping how many variable correlations the model can ever
// represent (at most n*m degrees of freedom). MMES instead builds each
// sample from only l << m randomly-selected direction vectors (Fast
// Mixture Sampling, FMS), which lets m grow much larger (2*ceil(sqrt(n)),
// asymptotically more than the other methods' O(log n)) while keeping
// per-sample cost at O(l*n) -- independent of m. The reconstructed
// sampling distribution is proven (paper's Theorem 2) to have the EXACT
// SAME covariance as the "target" distribution that directly using all m
// vectors would produce, so this is not an approximation that discards
// information relative to the other methods' models -- it is a genuinely
// richer model (more captured correlations) sampled cheaply.
//
// Two components, following the paper's own section split:
//
//  (1) Fast Mixture Sampling (FMS): draw an isotropic z0 ~ N(0,I) and l
//      scalars z_1..z_l ~ N(0,1) (l="mixing strength", default 4); pick l
//      direction-vector indices via a geometric distribution favouring
//      more-recently-stored vectors (truncated to {0,...,m-1} and sampled
//      by inverse-CDF -- rejection sampling would be far too slow here,
//      since the untruncated geometric's mean 1/c_a is n/4, usually much
//      larger than m); then
//        z = sqrt(1-gamma)*z0 + sqrt(gamma/l) * sum_{k=1}^l z_k * q_{j_k}
//      The m direction vectors themselves are adapted/evicted using the
//      SAME "keep temporally well-spread slots" ring-buffer idea as
//      LM-CMA-ES's updateSet() (this framework's lmcmaes.cpp) and Rm-ES's
//      snapshot memory (rmes.cpp) -- MMES cites LM-CMA-ES directly for
//      this mechanism, so the same logical structure is reused here.
//
//  (2) Paired Test Adaptation (PTA): a step-size control rule that, unlike
//      CSA, needs no assumption about which coordinate system the
//      distribution lives in (relevant here since MMES's distribution
//      isn't a plain multivariate Gaussian with a fixed C, only
//      approximately so) -- it compares, rank-position by rank-position,
//      whether the current generation's i-th best individual improved on
//      the previous generation's i-th best, weighted by the same
//      recombination weights, smooths this into W via exponential
//      averaging, and treats Phi(W) as a live estimate of the current
//      "success probability" for a simple z-test-based step-size update.
class MMES : public Optimizer {
public:
    MMES() = default;
    ~MMES() override = default;

    std::string methodShortName() const override { return "mmes"; }
    std::string methodFullName()  const override {
        return "Mixture Model-based Evolution Strategy (MMES)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double safeEval(const Vec& x);
    void   ensureBounds(Vec& x) const;
    // Inverse-CDF sample of a geometric(c_a) distribution truncated to
    // {0,...,m_-1} (rejection sampling would be far too slow: the
    // untruncated mean 1/c_a is typically >> m_ -- see class-level comment).
    int    sampleTruncatedGeometric();

private:
    // --- configuration ---
    int    lambda_cfg_{-1};
    int    m_dirs_cfg_{-1};    // number of stored direction vectors (-1 = auto)
    int    l_mix_{4};          // mixing strength (paper's recommended default)
    double sigma0_{0.3};       // initial step, as a fraction of the average box range
    double c_a_cfg_{-1.0};
    double c_c_cfg_{-1.0};
    double c_sigma_{0.3};
    double d_sigma_{1.0};
    double alpha_z_{0.05};

    // --- derived, fixed once D is known ---
    int    lambda_{0};
    int    mu_{0};
    double mu_eff_{0.0};
    std::vector<double> w_;
    int    m_dirs_{0};
    double c_a_{0.0}, c_c_{0.0}, gamma_{0.0};
    int    T_{0}; // minimal generation-gap between consecutive stored vectors

    // --- algorithm state ---
    int    n_generations_{0};
    Vec    mean_;
    double sigma_{0.3};
    Vec    p_;                   // evolution path (feeds the direction-vector memory)
    double W_{0.0};               // PTA's smoothed success statistic

    std::vector<Vec>    q_;       // m_dirs_ stored direction vectors
    std::vector<int>    t_stamp_; // generation index each PHYSICAL slot was last written
    std::vector<int>    v_;       // logical-position -> physical-slot permutation (oldest..newest)

    // --- offspring buffers ---
    std::vector<Vec>    X_;
    std::vector<double> Y_;      // current generation's fitness, sorted ascending after each update
    std::vector<double> Y_bak_;  // previous generation's sorted-ascending fitness (for PTA)

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
