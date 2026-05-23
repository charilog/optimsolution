#pragma once
#include "optimizer.h"
#include <vector>
#include <string>

namespace optimsolution {

// Grey Wolf Optimizer (baseline)
class GWO : public Optimizer {
public:
    GWO() = default;
    ~GWO() override = default;
	std::string methodShortName() const override { return "gwo"; }
	std::string methodFullName()  const override { return "Grey Wolf Optimizer"; }

    // Settings from [gwo]
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;

        a_start_ = mc.getDbl("a_start", a_start_);
        a_end_   = mc.getDbl("a_end",   a_end_);
        if (a_start_ < 0.0) a_start_ = 0.0;
        if (a_end_   < 0.0) a_end_   = 0.0;

        jitter_sigma_ = mc.getDbl("jitter_sigma", jitter_sigma_);
        if (jitter_sigma_ < 0.0) jitter_sigma_ = 0.0;

        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

    // Enable/disable final local refinement from [global]
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
    void   elitismInject();              // Keeps the best solution inside FX_/X_ (replaces the worst).
    double progress01() const;           // calls/max_evals clamped

    void   computeLeaders(int& a, int& b, int& c) const; // indices alpha,beta,delta

private:
    // population
    std::vector<Vec> X_;
    std::vector<double> FX_;

    // params
    double a_start_{2.0};
    double a_end_{0.0};
    double jitter_sigma_{0.0};

    // in-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // final local
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
