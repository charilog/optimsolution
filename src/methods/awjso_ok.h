#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

class AWJSO : public Optimizer {
public:
    AWJSO() = default;
    ~AWJSO() override = default;
    std::string methodShortName() const override { return "awjso"; }
    std::string methodFullName()  const override { return "Adaptive Weight jSO"; }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

private:
    int pop_init_{0};
    int pop_min_{4};

    int H_{5};
    double c_mem_{0.1};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int mem_idx_{0};

    double pmin_{0.05};
    double pmax_{0.25};

    double arc_rate_{1.4};
    double cauchy_scale_F_{0.1};
    double normal_std_CR_{0.1};

    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;

    std::string local_method_{"none"};
    double      local_rate_{0.0};

    bool        end_local_refine_{false};
    std::string end_local_method_;

    // ---------------------------------------------------------------
    // Mechanism A: Adaptive weighted current-to-pBest controller.
    //   adaptive_weight_enable_ = false  →  classic jSO Fw schedule
    //   adaptive_weight_enable_ = true   →  per-individual adaptive Fw
    // ---------------------------------------------------------------
    bool   adaptive_weight_enable_{true};
    double fw_min_mul_{0.55};
    double fw_max_mul_{1.65};
    double fw_abs_max_{1.35};
    double fw_target_success_{0.18};
    double success_ema_{0.0};
    int    no_best_improve_iters_{0};
    int    iteration_counter_{0};

    // ---------------------------------------------------------------
    // Mechanism B: Predictive prescreen.
    //   predictive_prescreen_enable_ = false  →  all candidates evaluated
    //   predictive_prescreen_enable_ = true   →  low-merit candidates skipped
    // ---------------------------------------------------------------
    bool   predictive_prescreen_enable_{false};
    double predictive_prescreen_start_{0.10};
    double predictive_prescreen_threshold_{0.16};
    double predictive_prescreen_step_floor_{0.015};
    double predictive_prescreen_explore_prob_{0.04};
    double predictive_prescreen_min_eval_frac_{0.25};
    double prescreen_skip_ema_{0.0};

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }
    void ensureInBounds(Vec& x);
    bool isInBounds(const Vec& x) const;
    void trimArchive(int max_size);

    static double clamp01(double v);
    double domainDiagonal() const;
    double normalizedDistance(const Vec& a, const Vec& b) const;
    double directionAlignment(const Vec& from, const Vec& to1, const Vec& to2) const;

    // Returns Fw multiplier (adaptive mode only; classic Fw is inlined in one_iteration).
    double adaptiveWeightMultiplier(const Vec& xi,
                                    const Vec& xp,
                                    int rank_pos,
                                    int N,
                                    double progress) const;

    // Returns true if this candidate should be skipped without evaluation.
    bool shouldSkipEvaluation(const Vec& xi,
                              const Vec& ui,
                              const Vec& xp,
                              const Vec& xr1,
                              const Vec& xr2,
                              int rank_pos,
                              int N,
                              double progress,
                              int evals_done_this_iter,
                              int max_skips_this_iter,
                              int skips_done_this_iter,
                              bool is_best_candidate);
};

} // namespace optimsolution
