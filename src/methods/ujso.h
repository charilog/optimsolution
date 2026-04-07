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

    // Predictive evaluation gate.
    bool   eval_gate_enable_{true};
    int    eval_gate_retry_max_{2};
    double eval_gate_base_{0.12};
    double eval_gate_max_{0.55};
    double eval_gate_random_keep_{0.15};
    double eval_gate_close_radius_{0.015};
    double eval_gate_tiny_step_{0.0015};
    double eval_gate_margin_{0.015};
    long long gate_attempts_{0};
    long long gate_skips_{0};
    double success_mu_step_{0.050};
    double success_mu_bestgain_{0.010};
    double success_mu_novelty_{0.040};
    double success_mu_rank_{0.500};
    double failure_mu_step_{0.015};
    double failure_mu_bestgain_{-0.002};
    double failure_mu_novelty_{0.010};
    double failure_mu_rank_{0.350};
    int    gate_success_count_{0};
    int    gate_failure_count_{0};

    // Helpers.
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }

    void ensureInBounds(Vec& x, const Vec& parent);
    double meanNormalizedDistanceToBest() const;
    double normalizedDistanceToBest(const Vec& x) const;
    double normalizedDistance(const Vec& a, const Vec& b) const;
    double minSampledDistance(const Vec& x, int exclude_idx) const;
    double scoreTrialForEvaluation(const Vec& trial,
                                   const Vec& parent,
                                   int parent_idx,
                                   int parent_rank,
                                   int pop_size) const;
    bool shouldEvaluateTrial(const Vec& trial,
                             const Vec& parent,
                             int parent_idx,
                             int parent_rank,
                             int pop_size,
                             double fes_ratio,
                             double diversity,
                             double* out_score = nullptr);
    void updateEvaluationGateModel(double step,
                                   double best_gain,
                                   double novelty,
                                   double rank_badness,
                                   bool success);
    void updateAdaptiveState();
};

} // namespace optimsolution
