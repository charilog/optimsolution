#pragma once
#include "optimizer.h"
#include <vector>
#include <string>
#include <random>
#include <limits>

namespace optimsolution {

// Artificial Bee Colony (ABC) - baseline
class ABC : public Optimizer {
public:
    ABC() = default;
    ~ABC() override = default;
	std::string methodShortName() const override { return "abc"; }
	std::string methodFullName()  const override { return "Artificial Bee Colony (ABC)"; }
	
    // Final local refinement from [global]
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    // Settings from the [abc] section in the configuration
    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;

        onlooker_frac_ = mc.getDbl("onlooker_frac", onlooker_frac_);
        if (onlooker_frac_ < 0.0) onlooker_frac_ = 0.0;
        if (onlooker_frac_ > 1.0) onlooker_frac_ = 1.0;

        limit_ = mc.getInt("limit", limit_);
        if (limit_ < 1) limit_ = 50;

        neighbor_dims_ = mc.getInt("neighbor_dims", neighbor_dims_);
        if (neighbor_dims_ < 1) neighbor_dims_ = 1;

        // in-run local
        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
        double lr = mc.getDbl("local_rate", local_rate_);
        if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;

        // scout rate (optional “soft” limit)
        scout_rate_ = mc.getDbl("scout_rate", scout_rate_);
        if (scout_rate_ < 0.0) scout_rate_ = 0.0; if (scout_rate_ > 1.0) scout_rate_ = 1.0;
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    double eval(const std::vector<double>& x) { return prob_->evaluate(x); }
    void ensureBounds(std::vector<double>& x);

    // phases
    void employedPhase();
    void onlookerPhase();
    void scoutPhase();

    // helpers
    int  pickOtherIndex(int i);
    void neighborFrom(int i, int j, std::vector<double>& v);
    void elitismInject(); // Injects best_x_/best_f_ into the population (worst slot)

private:
    // population
    std::vector<std::vector<double>> X_;
    std::vector<double>              FX_;
    std::vector<int>                 trials_; // trial counters

    // params
    double onlooker_frac_{1.0}; // onlookers = round(pop_ * onlooker_frac_)
    int    limit_{50};          // max trials before scout reinit
    int    neighbor_dims_{1};   // # of dimensions to tweak per neighbor
    double scout_rate_{0.0};    // optional chance per iter to make a scout (0 = off)

    // in-run local
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // final local
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
