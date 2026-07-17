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

// LM-MA-ES -- Limited-Memory Matrix Adaptation Evolution Strategy.
// Reference: Loshchilov, I.; Glasmachers, T.; Beyer, H.-G. "Limited-Memory
//   Matrix Adaptation for Large Scale Black-box Optimization." GECCO 2019
//   companion (extended version: arXiv:1705.06693). Builds on Beyer &
//   Sendhoff's Matrix Adaptation ES (MA-ES, IEEE TEC 2017), which itself
//   showed that CMA-ES's covariance matrix C and its unstable
//   eigendecomposition can be replaced by directly adapting a
//   transformation matrix M -- with no loss of search quality.
//
// LM-MA-ES goes one step further than MA-ES (and further than this
// framework's existing LM-CMA-ES): it never stores or updates ANY n x n
// matrix at all, not even implicitly via a Cholesky factor. Instead of one
// evolution path used on a single time scale (as in MA-ES/CMA-ES), it
// keeps m = O(log n) SEPARATE evolution-path-like vectors m_1..m_m, each
// with its OWN exponentially-decaying learning rate c_{c,i} -- so the
// stored vectors are "fading records of mean update steps on exponentially
// differing time scales" (the paper's own phrase), rather than a
// snapshot-based ring buffer of past directions requiring the
// temporal-distance slot-eviction bookkeeping LM-CMA-ES needs (see
// lmcmaes.cpp's updateSet()). This makes LM-MA-ES both simpler to
// implement correctly and, per the paper's own benchmarks, competitive
// with or faster than LM-CMA-ES at n up to 8192 -- at O(n log n) time and
// space per sample instead of LM-CMA-ES's O(mn) (the same asymptotic
// order, but with a smaller constant and no matrix square-root/Cholesky
// bookkeeping at all).
//
// Sampling a candidate direction d_i from z_i ~ N(0,I):
//   d_i <- z_i
//   for j = 1..min(t, m):  d_i <- (1-c_d,j)*d_i + c_d,j * m_j * (m_j^T d_i)
//   x_i <- y + sigma * d_i
// i.e. each stored vector m_j applies a cheap RANK-ONE correction to d_i in
// sequence (an O(n) inner product plus an O(n) axpy per vector, O(mn)
// total) -- this is mathematically the transformation matrix M applied to
// z_i, but M itself is never materialised.
//
// After evaluating all lambda offspring and selecting the best mu:
//   y      <- y + sigma * sum_i w_i * d_{i:lambda}          (mean update)
//   p_sigma<- (1-c_sigma)*p_sigma + sqrt(mu_w c_sigma(2-c_sigma)) * z_w
//   m_j    <- (1-c_c,j)*m_j + sqrt(mu_w c_c,j(2-c_c,j)) * z_w   for j=1..m
//   sigma  <- sigma * exp( (c_sigma/2) * (||p_sigma||^2/n - 1) )
// where z_w = sum_i w_i * z_{i:lambda} is the weighted-recombination of the
// selected offsprings' RAW (untransformed) random draws, computed once per
// generation and reused for both the p_sigma and every m_j update.
class LMMAES : public Optimizer {
public:
    LMMAES() = default;
    ~LMMAES() override = default;

    std::string methodShortName() const override { return "lmmaes"; }
    std::string methodFullName()  const override {
        return "Limited-Memory Matrix Adaptation Evolution Strategy (LM-MA-ES)";
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
    int    m_dirs_cfg_{-1};   // number of stored direction vectors (-1 = auto)
    double sigma0_{0.3};      // initial step size, as a FRACTION of the average box range
                               // (unlike lmcmaes.cpp's sigma0_ which is used as an
                               // absolute-range fraction directly -- see init() for
                               // the exact formula and delivery notes on why a very
                               // wide box needs care here)

    // --- derived, fixed for the whole run once D is known ---
    int    lambda_{0};
    int    mu_{0};
    int    m_dirs_{0};
    double mu_eff_{0.0};
    double c_sigma_{0.0};
    std::vector<double> c_d_;   // c_{d,j}, j=1..m_dirs_
    std::vector<double> c_c_;   // c_{c,j}, j=1..m_dirs_
    std::vector<double> w_;     // recombination weights, size mu_

    // --- algorithm state ---
    int    t_{0};              // generation counter (number of GENERATIONS completed)
    Vec    y_;                 // mean / incumbent estimate
    double sigma_{0.3};
    Vec    p_sigma_;
    std::vector<Vec> m_;       // m_dirs_ stored direction vectors, each size D

    // --- offspring buffers (reused each generation) ---
    std::vector<Vec>    Z_;    // raw z_i, size lambda_
    std::vector<Vec>    Dd_;   // transformed d_i, size lambda_
    std::vector<double> F_;    // fitness

    // --- in-run / final local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
