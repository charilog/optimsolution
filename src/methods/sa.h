#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// Parallel Simulated Annealing (one chain per agent)
class SA : public Optimizer {
public:
    SA() = default;
    ~SA() override = default;
	std::string methodShortName() const override { return "sa"; }
	std::string methodFullName()  const override { return "Simulated Annealing"; }

    // Final local search from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [sa] in the config
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;

        T0_        = mc.getDbl("T0",        T0_);
        Tmin_      = mc.getDbl("Tmin",      Tmin_);
        alpha_     = mc.getDbl("alpha",     alpha_);
        moves_     = mc.getInt("moves",     moves_);
        step_frac_ = mc.getDbl("step_frac", step_frac_);
        step_sigma_= mc.getDbl("step_sigma",step_sigma_);

        // in-run local
        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c : local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // helpers
    double eval(const std::vector<double>& x) { return prob_->evaluate(x); }
    void ensureBounds(std::vector<double>& x);

    
    void propose(int i, std::vector<double>& cand);

private:
    // State per agent
    std::vector<std::vector<double>> X_;   // current states
    std::vector<double>              FX_;  // current energies
    std::vector<double>              T_;   // temperatures

    // SA parameters
    double T0_{1.0};          // Initial temperature
    double Tmin_{1e-12};      // Temperature threshold
    double alpha_{0.95};      // Geometric cooling (T *= alpha)
    int    moves_{10};        // Proposals (Metropolis steps) per iteration & agent

    // Proposal: Gaussian around x, either scaled as a fraction of range, or fixed
    double step_frac_{0.05};  // Fraction of (ub-lb) per dimension
    double step_sigma_{0.0};  // If >0, it takes precedence as an absolute sigma (in search-space units)

    // in-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final polishing in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
