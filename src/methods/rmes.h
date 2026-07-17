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

// Rm-ES -- Rank-m Evolution Strategy.
// Reference: Li, Z.; Zhang, Q. "A Simple Yet Efficient Evolution Strategy
//   for Large-Scale Black-Box Optimization." IEEE Transactions on
//   Evolutionary Computation, 22(5), 637-646, 2018.
//
// This implementation is a direct, verified C++ port of the reference
// Python implementation in PyPop7 (Evolutionary-Intelligence/pypop,
// pypop7/optimizers/es/rmes.py and its base class r1es.py), retrieved and
// checked line-by-line rather than reconstructed from memory or a partial
// description -- given this framework's history of subtle algorithm bugs
// slipping in from incomplete/OCR'd sources (e.g. VkD-CMA-ES's sampling
// formula, the DG2 grouping cancellation), the two source files were
// fetched directly and translated term-for-term.
//
// Rm-ES departs from LM-CMA-ES's memory management in exactly the way the
// user asked for: LM-CMA-ES stores a ring buffer of m raw evolution-path
// SNAPSHOTS with an elaborate "keep temporally well-spread slots" eviction
// rule (Algorithm 5 in Loshchilov 2014, ported in this framework's
// lmcmaes.cpp). Rm-ES instead keeps growing a SINGLE principal search
// direction p via the classic rank-one CMA-style recursion
//   p <- (1-c) p + sqrt(c(2-c) mu_eff) (mean_new - mean_old)/sigma
// (which the paper's own framing describes as p acting like a momentum
// term that "cancels opposite update directions of the mean"), and only
// PERIODICALLY snapshots this single evolving p into one of m stored
// slots mp_1..mp_m -- using the same minimum-time-gap eviction idea as
// LM-CMA-ES/LM-MA-ES's multi-vector memory, but applied to snapshots of
// ONE accumulating momentum vector instead of m independently-adapted
// ones. Sampling then blends the current z with a weighted sum over the m
// stored snapshots:
//   x = mean + sigma*( a^m z + b * sum_{i=1}^m a^{m-i} r_i mp_i ),
//   a = sqrt(1-c_cov), b = sqrt(c_cov), r_i ~ N(0,1) drawn fresh per i.
// Step-size control uses a rank-based success rule (RSR) that compares the
// interleaved rankings of the previous and current generations' best-mu
// fitnesses, rather than CSA's evolution-path norm -- a further
// simplification specific to this R1ES/Rm-ES family.
class RmES : public Optimizer {
public:
    RmES() = default;
    ~RmES() override = default;

    std::string methodShortName() const override { return "rmes"; }
    std::string methodFullName()  const override {
        return "Rank-m Evolution Strategy (Rm-ES)";
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

private:
    // --- configuration ---
    int    lambda_cfg_{-1};
    int    m_paths_{2};          // number of stored evolution-path snapshots
    int    generation_gap_cfg_{-1}; // <=0 -> auto: ndim_problem
    double sigma0_{0.3};         // initial step, as a fraction of the average box range
    double c_cov_cfg_{-1.0};
    double c_cfg_{-1.0};
    double c_s_{0.3};
    double q_star_{0.3};
    double d_sigma_{1.0};

    // --- derived, fixed once D is known ---
    int    lambda_{0};
    int    mu_{0};
    double mu_eff_{0.0};
    std::vector<double> w_;
    double c_cov_{0.0}, c_{0.0};
    double a_{0.0}, a_m_{0.0}, b_{0.0}; // sampling coefficients (a=sqrt(1-c_cov), b=sqrt(c_cov))
    double p1_{0.0}, p2_{0.0};         // principal-direction update coefficients
    int    generation_gap_{0};
    std::vector<int> rr_full_;         // [1,2,...,2*mu_] for the rank-based success rule

    // --- algorithm state ---
    int    n_generations_{0};
    Vec    mean_;
    double sigma_{0.3};
    Vec    p_;                  // principal search direction
    double s_{0.0};             // cumulative rank rate
    std::vector<Vec>    mp_;    // m_paths_ stored snapshots of p
    std::vector<double> t_hat_; // generation-index "timestamp" of each snapshot

    // --- offspring buffers ---
    std::vector<Vec>    X_;
    std::vector<double> Y_;      // current generation's fitness (kept sorted ascending after each update)
    std::vector<double> Y_bak_;  // previous generation's sorted fitness (for RSR)

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
