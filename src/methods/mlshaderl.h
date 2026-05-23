#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

// Multi-population LSHADE with simple reinforcement learning over mutation strategies.
class mLSHADE_RL : public Optimizer {
public:
    mLSHADE_RL() = default;
    ~mLSHADE_RL() override = default;
	std::string methodShortName() const override { return "mlshaderl"; }
	std::string methodFullName()  const override { return "Multi-operator L-SHADE with Reinforcement Learning"; }

    // Final local from [global], as in PSAO:
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    // Settings from [mlshaderl]
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // Helpers
    double eval(const Vec& x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec& x);
    int    selectStrategy(); // Mutation strategy selection via RL

private:
    // --- Configuration / population as in PSAO ---
    int pop_cfg_{-1};    // Per-method population (override of the global setting)
    int pop_init_{100};  // effective LSHADE initial pop
    int pop_min_{4};     // Minimum population after linear reduction

    // Population & archive
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;

    // Success-history based adaptation (LSHADE-style)
    int                 H_{10};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 mem_idx_{0};
    double              c_mem_{0.1};   // Learning rate for MF/MCR

    double pmin_{0.05};
    double pmax_{0.5};
    double arc_rate_{1.4};

    // Variances for sampling F/CR
    double cauchy_scale_F_{0.1};
    double normal_std_CR_{0.1};

    // Reinforcement-learning weights for strategies
    static constexpr int NUM_STRAT_ = 3;
    double epsilon_{0.1};      // exploration rate
    double rl_alpha_{0.2};     // Learning rate for the weights

    std::vector<double> strat_weight_;      // Weight/quality per strategy
    std::vector<double> strat_reward_acc_;  // Accumulated reward in the iteration
    std::vector<int>    strat_use_count_;   // Number of times it was used

    // In-run local search
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final local at the end (from [global])
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
