#pragma once

#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>
#include <deque>
#include <cctype>

namespace optimsolution {

// Simple (single-population) Sporulation-Inspired Optimization Algorithm
class SIOA : public Optimizer {
public:
    SIOA() = default;
    ~SIOA() override = default;
	std::string methodShortName() const override { return "sioa"; }
	std::string methodFullName()  const override { return "Sporulation-Inspired Optimization Algorithm (SIOA)"; }

    // Receives the global end-local hook from the runner.
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Reads settings from [sioa].
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // --- Settings ---
    int         pop_cfg_{-1};         // Population override from [sioa] (<=0: none).
    double      eps_stop_{1e-10};     // For NM-window BSS.
    int         NM_{12};              // Same default as bss(sim=12).
    int         plateau_iters_{-1};   // Off (optional extra criterion).

    // in-run local
    std::string local_method_{"none"};
    double      local_rate_{0.0};
    bool        inrun_on_improve_{false};

    // final local @ end
    bool        end_local_refine_{false};
    std::string end_local_method_{"none"};

    // SIOA parameters.
    double c1_{1.0};
    double c2_{1.0};
    double Rmin_{0.05};
    double Rmax_{0.5};
    double p_spor0_{0.5};
    double p_germ0_{0.5};
    double p_zero_{0.0};              // Comment translated from Greek.
    double adapt_R_kappa_{0.1};
    double adapt_prob_kappa_{0.05};
    std::string crowding_metric_{"bnorm"};

    // --- State ---
    std::vector<Vec>    X_;
    std::vector<double> fX_;
    Vec    gbest_x_;
    double gbest_f_{std::numeric_limits<double>::infinity()};

    // For manual NM-window BSS (compatible with stop_).
    std::deque<double> window_fmin_;
    bool stopping_hold_(double fmin_k);

    // self-adaptation
    std::mt19937_64 rng_local_;
    double R_{0.1};
    double p_spor_{0.5};
    double p_germ_{0.5};
    double last_avg_f_{std::numeric_limits<double>::infinity()};
    void adapt_controls_(double avg_f, double prev_avg);

    // helpers
    void ensureBounds(Vec& v);
    inline double eval(const Vec& v) { return prob_->evaluate(v); }
    int  most_similar_index_(const Vec& s) const;
    Vec  make_spore_(const Vec& xi);

    // bookkeeping
    int final_population_{-1};
};

} // namespace optimsolution
