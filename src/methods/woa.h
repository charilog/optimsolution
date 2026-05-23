#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// Whale Optimization Algorithm (Mirjalili & Lewis, 2016)
class WOA : public Optimizer {
public:
    WOA() = default;
    ~WOA() override = default;
	std::string methodShortName() const override { return "woa"; }
	std::string methodFullName()  const override { return "Whale Optimization Algorithm"; }

    // Final local refinement from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from [woa] in the configuration file
    void configure(const MethodConfig& mc) override {
int pop_override = mc.getInt("population", pop_);
if (pop_override > 0) {
    pop_ = pop_override;
    this->setPopulation(pop_); // to allow StopController to read the correct value
}

        b_spiral_ = mc.getDbl("b", b_spiral_);
        a_start_  = mc.getDbl("a_start", a_start_);
        a_end_    = mc.getDbl("a_end",   a_end_);
        use_elite_explore_ = mc.getBool("use_elite_explore", use_elite_explore_);

        // in-run local
        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
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
    double progress01() const; // 0..1, derived from evals

private:
    // population
    std::vector<std::vector<double>> X_;
    std::vector<double>              FX_;

    // params
    double b_spiral_{1.0};     // spiral constant (b)
    double a_start_{2.0};      // a at progress=0
    double a_end_{0.0};        // a at progress=1
    bool   use_elite_explore_{false}; // if true, exploration can select from top individuals instead of a random one

    // in-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // final polishing in end()
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
