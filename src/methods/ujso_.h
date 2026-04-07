#pragma once
#include "optimizer.h"


#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

class UJSO : public Optimizer {
public:
    UJSO() = default;
    ~UJSO() override = default;

    std::string methodShortName() const override { return "ujso"; }
    std::string methodFullName()  const override { return "Updated Hybrid Differential Evolution JSO"; }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

private:
    // Population management.
    int pop_init_{0};
    int pop_min_{4};
    double np_exp_{1.0};
    double np_exp_min_{0.70};
    double np_exp_max_{2.50};

    // Success-history memories.
    int H_{6};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    std::vector<double> MP_;
    std::vector<double> MK_;

    // Adaptive sampling widths.
    std::vector<double> sigmaF_;
    std::vector<double> sigmaCR_;
    std::vector<double> sigmaP_;
    std::vector<double> sigmaK_;
    int mem_idx_{0};

    // Safe bounds for adaptive parameters.
    double p_floor_{0.03};
    double p_ceil_{0.40};
    double k_floor_{0.50};
    double k_ceil_{1.60};
    double sigma_floor_F_{0.03};
    double sigma_floor_CR_{0.03};
    double sigma_floor_P_{0.01};
    double sigma_floor_K_{0.03};

    // Archive adaptation.
    double archive_use_prob_{0.50};
    double arc_rate_init_{1.40};
    double arc_rate_eff_{1.40};
    double arc_rate_min_{0.50};
    double arc_rate_max_{3.00};

    // Internal population and archive.
    std::vector<Vec>   X_;
    std::vector<double> FX_;
    std::vector<Vec>   archive_;

    // In-run local search.
    std::string local_method_{"none"};
    double      local_rate_base_{0.0};
    double      local_rate_eff_{0.0};
    double      local_rate_max_{0.0};

    // Final local refinement.
    bool        end_local_refine_{false};
    std::string end_local_method_;

    // Search-state adaptation.
    int    stagnant_gens_{0};
    double prev_success_rate_{0.0};

    // Helpers.
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }

    void ensureInBounds(Vec& x, const Vec& parent);
    double meanNormalizedDistanceToBest() const;
    double normalizedDistanceToBest(const Vec& x) const;
    void updateAdaptiveState();
};

} // namespace optimsolution
