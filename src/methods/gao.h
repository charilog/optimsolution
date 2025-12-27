#pragma once
#include "optimizer.h"
#include <vector>
#include <string>

namespace optimsolution {

// Giant Armadillo Optimizer (GAO) – baseline
class GAO : public Optimizer {
public:
    GAO() = default;
    ~GAO() override = default;
	std::string methodShortName() const override { return "gao"; }
	std::string methodFullName()  const override { return "Giant Armadillo Optimizer (GAO)"; }

    void configure(const MethodConfig& mc) override {
        int po = mc.getInt("population", pop_);
        if (po > 0) pop_ = po;

        burrow_frac_ = mc.getDbl("burrow_frac", burrow_frac_);
        roll_frac_   = mc.getDbl("roll_frac",   roll_frac_);
        forage_frac_ = mc.getDbl("forage_frac", forage_frac_);
        double s = burrow_frac_ + roll_frac_ + forage_frac_;
        if (s <= 0.0) { burrow_frac_ = 0.34; roll_frac_ = 0.33; forage_frac_ = 0.33; }
        else { burrow_frac_ /= s; roll_frac_ /= s; forage_frac_ /= s; }

        burrow_step_ = mc.getDbl("burrow_step", burrow_step_);
        roll_step_   = mc.getDbl("roll_step",   roll_step_);
        forage_beta_ = mc.getDbl("forage_beta", forage_beta_);
        forage_gamma_= mc.getDbl("forage_gamma",forage_gamma_);
        jitter_sigma_= mc.getDbl("jitter_sigma",jitter_sigma_);

        local_method_ = mc.getStr("local_method", local_method_);
        for (auto &c: local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;
    }

    // Final local refinement from [global]
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
    void   leadersABC(int& a, int& b, int& c) const; // Top 3 individuals
    double progress01() const;
    Vec    topCentroid(int k) const;

private:
    // Population state
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // Roles
    double burrow_frac_{0.34};
    double roll_frac_{0.33};
    double forage_frac_{0.33};

    // Move intensity
    double burrow_step_{0.2};
    double roll_step_{0.8};
    double forage_beta_{1.2};
    double forage_gamma_{0.7};

    double jitter_sigma_{0.0};

    // In-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final local (from [global])
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
