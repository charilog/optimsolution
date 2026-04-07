#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

struct MethodConfig; // forward declaration

// Self-adaptive Differential Evolution (SaDE) with 4 strategies.
// Comment translated from Greek.
class SaDE : public Optimizer {
public:
    SaDE() = default;
    ~SaDE() override = default;
	std::string methodShortName() const override { return "sade"; }
	std::string methodFullName()  const override { return "Self-adaptive Differential Evolution (SaDE)"; }

    // Allows global [end_local_*] settings to be passed through the Optimizer.
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    // Called by the factory after construction.
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double eval(const Vec& x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec& x);

    // strat: 0..3
    bool generateTrial(int strat, int i, double Fi, double CRi,
                       int best_idx, Vec& ui,
                       std::uniform_real_distribution<double>& U01,
                       std::uniform_int_distribution<int>& Ui);

    int  selectStrategy(std::uniform_real_distribution<double>& U01);

private:
    // --- Config / population ---
    int    pop_cfg_{-1};      // per-method population (override global)
    int    pop_init_{100};    // fallback population

    // --- F & CR handling ---
    double F_min_{0.4};       // Uniform [F_min, F_max] per trial.
    double F_max_{0.9};

    int    Lp_{50};           // Learning period for strategy/CR adaptation.

    // Initial CR mean values for each strategy.
    static constexpr int NUM_STRAT_ = 4;
    double CR_init_[NUM_STRAT_] = {0.1, 0.9, 0.5, 0.9};

    // In-run local search
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final local @ end
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // --- State ---
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // Strategy adaptation (SaDE-style)
    std::vector<double> strat_prob_;  // Probability for each strategy.
    std::vector<int>    strat_success_;
    std::vector<int>    strat_attempt_;
    int                 gen_counter_{0};

    // CR means and pools per strategy.
    std::vector<double> CR_mean_;                       // Mean CR for each strategy.
    std::vector< std::vector<double> > CR_success_pool_; // CR of successful trials.
};

} // namespace optimsolution
