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

    // Opt-in to "[global] max_evals = <N>xD" (budget scales with the
    // problem's dimension) -- see options.h. Every other method is
    // unaffected by this override existing here.
    void setGeneralOptions(const GeneralOptions& g) override {
        Optimizer::setGeneralOptions(g);
        if (g.max_evals_per_dim > 0 && prob_) {
            max_evals_ = g.max_evals_per_dim * (long long)prob_->dimension();
        }
    }

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
    // jSO (Brest et al. 2017): p decreases linearly from pmax = 0.25 to
    // pmin = pmax/2 = 0.125 (was 0.05); archive size = NP, i.e. arc_rate = 1.0
    // (was 1.4, the L-SHADE code value). Same values as the reference code.
    double pmin_{0.125};
    double pmax_{0.25};

    // Archive settings.
    double arc_rate_{1.0};

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
