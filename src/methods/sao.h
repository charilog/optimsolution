#pragma once
#include "optimizer.h"  // Comment translated from Greek.

#include <vector>
#include <random>
#include <limits>
#include <string>

namespace optimsolution {

// Simple (non-parallel) SAO: Sniffing -> Trailing -> (if no improve) Random -> optional local.
// Termination: if |best_f - last_best_f_| <= eps for consecutive "sim" iterations.
class SAO : public Optimizer {
public:
    SAO() = default;
    ~SAO() override = default;
	std::string methodShortName() const override { return "sao"; }
	std::string methodFullName()  const override { return "Smell Agent Optimization (SAO)"; }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // ----- Config / Params -----
    // Population override (per-method) for correct reporting.
    int  pop_cfg_{-1};
    int  final_population_{-1};

    // Comment translated from Greek.
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // SAO coefficients
    double sniff_w_{0.6};
    double sniff_a1_{1.4};
    double sniff_a2_{0.4};

    double trail_sigma0_{0.25};
    double trail_decay_{0.98};

    double rand_rate_{0.25};
    double rand_scale_{0.5};

    // ---- BSS-like simple stop: best_f unchanged for sim iterations within eps ----
    double bss_eps_{1e-12}; // optional from cfg key "eps"
    int    bss_sim_{12};    // optional from cfg key "sim"

    // ----- State -----
    std::vector<Vec>    X_;
    std::vector<Vec>    V_;
    std::vector<double> fX_;

    Vec    worst_x_;
    double worst_f_{-std::numeric_limits<double>::infinity()};

    double trail_sigma_k_{trail_sigma0_};
    int    K_{0}; // iteration counter

    // Comment translated from Greek.
    double last_best_f_{std::numeric_limits<double>::infinity()};
    int    same_best_iters_{0};
    bool   stopped_{false};

    // RNG (seeded from Optimizer::rng_).
    std::mt19937_64 rng_local_;
    std::uniform_real_distribution<double> U01_{0.0,1.0};
    std::normal_distribution<double>      N01_{0.0,1.0};

private:
    inline void ensureBounds(Vec& v);
    inline double eval(const Vec& v){ return prob_->evaluate(v); }

    // phases
    void sniffing_();
    bool trailing_(); // Comment translated from Greek.
    void random_();

    // helpers
    void evaluate_and_update_(int i);
    void recompute_worst_();

    // simple stop update
    void update_simple_bss_();
};

} // namespace optimsolution
