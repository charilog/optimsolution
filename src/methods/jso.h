#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

class JSO : public Optimizer {
public:
    JSO() = default;
    ~JSO() override = default;
    std::string methodShortName() const override { return "jso"; }
    std::string methodFullName()  const override { return "Single Objective Real-Parameter Optimization"; }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

    // Hook from the global layer for final local refinement.
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

private:
    // Population (kept configurable by the framework / user settings).
    int pop_init_{0};
    int pop_min_{4};

    // Success-history memory for F/CR.
    int H_{5};
    double c_mem_{0.1}; // kept for backward compatibility; jSO update uses equal weighting.
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int mem_idx_{0};

    // p-best range.
    double pmin_{0.05};
    double pmax_{0.25};

    // Archive settings.
    double arc_rate_{1.4};

    // Parameters for sampling F/CR.
    double cauchy_scale_F_{0.1};
    double normal_std_CR_{0.1};

    // Internal population and archive.
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;

    // In-run local search
    std::string local_method_{"none"};
    double      local_rate_{0.0};

    // Final local refinement at the end.
    bool        end_local_refine_{false};
    std::string end_local_method_;

    // Helpers.
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }
    void ensureInBounds(Vec& x);
    bool isInBounds(const Vec& x) const;
    void trimArchive(int max_size);
};

} // namespace optimsolution
