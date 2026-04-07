#pragma once
#include "optimizer.h"
#include <vector>
#include <string>

namespace optimsolution {

// Eel–Grouper Cooperative Optimizer (EGCO) – baseline
class EGCO : public Optimizer {
public:
    EGCO() = default;
    ~EGCO() override = default;
	std::string methodShortName() const override { return "egco"; }
	std::string methodFullName()  const override { return "Eel and Grouper Optimizer(EGCO)"; }

    // Method settings from [egco]
    void configure(const MethodConfig& mc) override {
        int po = mc.getInt("population", pop_);
        if (po > 0) pop_ = po;

        eel_frac_      = mc.getDbl("eel_frac", eel_frac_);
        if (eel_frac_ < 0.0) eel_frac_ = 0.0; if (eel_frac_ > 1.0) eel_frac_ = 1.0;
        eel_step_      = mc.getDbl("eel_step", eel_step_);
        eel_inertia_   = mc.getDbl("eel_inertia", eel_inertia_);
        coop_bias_     = mc.getDbl("coop_bias", coop_bias_);

        grp_beta_      = mc.getDbl("grouper_beta", grp_beta_);        // attraction
        grp_gamma_     = mc.getDbl("grouper_gamma", grp_gamma_);      // shrink
        grp_jitter_    = mc.getDbl("grouper_jitter", grp_jitter_);

        jitter_sigma_  = mc.getDbl("jitter_sigma", jitter_sigma_);

        local_method_  = mc.getStr("local_method", local_method_);
        for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

    // Enable final local refinement from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    double eval(const Vec& x) { return prob_->evaluate(x); }
    void   ensureBounds(Vec& x);
    void   elitismInject();
    double progress01() const;

    void   leadersABC(int& a, int& b, int& c) const; // Indices alpha, beta, delta
    Vec    topCentroid(int k) const;                 // Center of mass of the top-k individuals

private:
    // state
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    V_; // "Velocities" for the eel branch

    // params
    double eel_frac_{0.5};     // Population fraction treated as "eels"
    double eel_step_{0.2};     // Base step size
    double eel_inertia_{0.7};  // Momentum applied to V_
    double coop_bias_{0.15};   // Centroid mixing factor in the step

    double grp_beta_{1.5};     // Grouper attraction strength
    double grp_gamma_{0.7};    // Shrinkage factor
    double grp_jitter_{0.01};  // Small random perturbation

    double jitter_sigma_{0.0}; // Global micro-jitter strength

    // in-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // final local
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
