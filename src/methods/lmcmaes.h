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

// LM-CMA-ES: Limited Memory CMA-ES
// Reference: Loshchilov (2014/2017)
//   "A Computationally Efficient Limited Memory CMA-ES for Large Scale
//    Optimization" — GECCO 2014, arXiv:1404.5520
//   "LM-CMA: An Alternative to L-BFGS for Large-Scale Black-Box
//    Optimization" — Evolutionary Computation 2017
//
// Replaces the full n×n covariance matrix with m stored direction vectors
// (evolution paths p_c and their inverse-Cholesky images v).
// Sampling cost: O(m·n) per individual. Step-size: Population Success Rule.
class LMCMAES : public Optimizer {
public:
    LMCMAES() = default;
    ~LMCMAES() override = default;

    std::string methodShortName() const override { return "lmcmaes"; }
    std::string methodFullName()  const override {
        return "Limited Memory CMA-ES";
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

    // Safe evaluation: result is always finite (NaN/inf → +inf)
    // Updates best_f_ / best_x_ immediately so no update is ever missed.
    double safeEval(const Vec& x);

    void   ensureBounds(Vec& x);

    // Az():    x = A * z           (Algorithm 3, Loshchilov 2014)
    // Iterates from oldest stored vector to newest.
    Vec az(const Vec& z) const;

    // Ainvz(): x = A^{-1} * z     (Algorithm 4, Loshchilov 2014)
    // Same iteration order as Az (oldest→newest).
    Vec ainvz(const Vec& z) const;

    // UpdateSet(): direction vector slot selection  (Algorithm 5).
    // Returns slot index (jcur) to overwrite; updates j_[] and l_[].
    // Also increments n_stored_ when a new slot is first filled.
    int  updateSet(int t);

    // PSR sigma adaptation (Section 3.3, Loshchilov 2014).
    //   actual_sampled: how many individuals were actually evaluated this
    //   iteration (may be < lambda_ when budget is exhausted mid-loop).
    void updateSigmaPSR(int actual_sampled);

private:
    // --- configuration ---
    int    lambda_cfg_{-1};
    int    pop_cfg_{-1};
    double sigma0_{0.3};

    // PSR parameters (paper defaults: c_sigma=0.3, d_sigma computed in init, z*=0.25)
    double c_sigma_{0.3};
    double d_sigma_{1.0};
    bool   d_sigma_from_cfg_{false}; // true when user set d_sigma explicitly
    double zstar_{0.25};

    // number of direction vectors (default = lambda after init)
    int    m_vec_{0};
    int    nsteps_{0};     // max temporal gap between consecutive vectors (= m_vec_)

    // in-run / final local search
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // --- algorithm state ---
    int    iter_{0};         // iteration counter (incremented at end of one_iteration)
    int    n_stored_{0};     // number of valid slots currently in the ring buffer

    double sigma_{0.3};
    double sigma_init_{0.3}; // initial sigma (used as lower-bound reference)
    double c_c_{0.0};        // evolution-path learning rate = 1/m_vec_
    double c1_{0.0};         // rank-one learning rate = 1/(10*ln(n+1))
    double a_{0.0};          // sqrt(1-c1)  (Az/Ainvz scaling factor)
    double mu_eff_{0.0};     // effective selection mass

    // mean of the sampling distribution
    Vec    m_;
    // evolution path (cumulated mean-shift direction)
    Vec    pc_;
    // PSR cumulated success measurement
    double s_{0.0};

    // --- direction-vector ring buffer ---
    //   j_[i] = slot index, ordered oldest (i=0) → newest (i=n_stored_-1)
    //   l_[k] = iteration number when slot k was last written
    //   P_[k] = stored evolution path p_c^t
    //   V_[k] = stored v^t = A^{-1} * p_c^t
    //   b_[k] = Az update coefficient for slot k
    //   d_[k] = Ainvz update coefficient for slot k
    std::vector<int>    j_;
    std::vector<int>    l_;
    std::vector<Vec>    P_;
    std::vector<Vec>    V_;
    std::vector<double> b_;
    std::vector<double> d_;

    // --- sorted offspring population (best-first after each iteration) ---
    int                 lambda_{0};
    int                 mu_{0};
    std::vector<Vec>    X_;    // individuals, sorted best-first
    std::vector<double> FX_;   // corresponding fitness values, sorted best-first
    std::vector<double> FX_prev_; // previous generation fitnesses (for PSR)
    int                 prev_sampled_{0}; // actual_sampled from the previous iteration

    // selection weights (w_[0] ≥ w_[1] ≥ ... > 0, sum = 1)
    std::vector<double> w_;
};

} // namespace optimsolution
