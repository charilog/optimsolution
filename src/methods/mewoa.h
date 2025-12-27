#pragma once
#include "optimizer.h"
#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>

namespace optimsolution {

// MEWOA: Multi-Elite Whale Optimization Algorithm
class MEWOA : public Optimizer {
public:
    MEWOA() = default;
    ~MEWOA() override = default;
	std::string methodShortName() const override { return "mewoa"; }
	std::string methodFullName()  const override { return "Modified Enhanced Whale Optimization Algorithm (MEWOA)"; }


    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }


    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;

        b_spiral_ = mc.getDbl("b", b_spiral_);
        a_start_  = mc.getDbl("a_start", a_start_);
        a_end_    = mc.getDbl("a_end",   a_end_);

        elite_frac_ = mc.getDbl("elite_frac", elite_frac_);
        if (elite_frac_ <= 0.0) elite_frac_ = 0.05;
        if (elite_frac_ > 1.0)  elite_frac_ = 1.0;

        use_ranked_elite_ = mc.getBool("use_ranked_elite", use_ranked_elite_);
        beta_             = mc.getDbl("beta", beta_);

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
    double progress01() const; // 0..1, from evals

    // elite set 
    void buildElite();
    int  sampleEliteIndex();   

private:
    // population
    std::vector<std::vector<double>> X_;
    std::vector<double>              FX_;

    // elite pool 
    std::vector<int>    elite_idx_;
    std::vector<double> elite_w_;    // normalized
    std::vector<double> elite_cdf_;  // cumulative

    // params
    double b_spiral_{1.0};     // spiral constant (b)
    double a_start_{2.0};      // a at progress=0
    double a_end_{0.0};        // a at progress=1

    double elite_frac_{0.1};   
    bool   use_ranked_elite_{true}; 
    double beta_{0.01};        


    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
