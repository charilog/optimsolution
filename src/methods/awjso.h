#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <cmath>

namespace optimsolution {

// ---------------------------------------------------------------------------
// AWJSO – Adaptive Weight jSO with optional mechanisms
//
// Mode matrix (three independent flags):
//
//   adaptive_weight_enable_       (A) – per-individual adaptive Fw multiplier
//   predictive_prescreen_enable_  (B) – skip low-merit trial vectors
//   restart_enable_               (C) – stagnation-triggered partial restart
//
// All three default to false → pure classic jSO behaviour.
// Any combination can be enabled independently for A/B testing.
// ---------------------------------------------------------------------------
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
    // -----------------------------------------------------------------------
    // Core jSO parameters
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Mechanism A – Adaptive Fw multiplier
    //
    // When disabled: classic jSO three-phase Fw schedule (0.7/0.8/1.2 x Fi).
    // When enabled:  per-individual multiplier in [fw_min_mul_, fw_max_mul_]
    //                derived from progress, stagnation, rank, distances and
    //                recent-success pressure.
    //
    // BUG FIX: fw_max_mul_ was 1.65 and fw_abs_max_ was 1.35, both exceeding
    // classic jSO's maximum Fw of 1.2, making the adaptive mode up to 37%
    // more aggressive than classic.  Defaults corrected to 1.20 so the
    // adaptive range maps onto the same Fw envelope as classic jSO.
    //
    // BUG FIX: score normalization.  Raw score spans [-0.22, +1.00].
    // Original code used clamp01(0.50 + raw) which shifted the baseline to
    // 0.28, making fw_min_mul_ unreachable.  Now normalised correctly onto
    // [0, 1] via (raw - RAW_MIN) / RAW_SPAN.
    // -----------------------------------------------------------------------
    bool   adaptive_weight_enable_{false};
    double fw_min_mul_{0.55};   // minimum Fw multiplier
    double fw_max_mul_{1.20};   // maximum Fw multiplier – matches classic jSO max (was 1.65)
    double fw_abs_max_{1.20};   // hard cap on Fw itself               (was 1.35)
    double fw_target_success_{0.18};

    // State (updated every iteration when A is active)
    double success_ema_{0.0};
    int    no_best_improve_iters_{0};
    int    iteration_counter_{0};

    // -----------------------------------------------------------------------
    // Mechanism B – Predictive prescreen
    //
    // When disabled: every trial vector is evaluated (classic behaviour).
    // When enabled:  trial vectors whose merit score falls below an adaptive
    //                threshold are skipped without spending an evaluation.
    //
    // BUG FIX: original skip condition was
    //     (step < step_floor_ && merit < threshold)
    // The extra step gate made the mechanism inactive for large-step moves
    // with genuinely poor merit.  The merit formula already penalises small
    // steps via its first term.  Corrected to: merit < threshold only.
    // -----------------------------------------------------------------------
    bool   predictive_prescreen_enable_{false};
    double predictive_prescreen_start_{0.10};
    double predictive_prescreen_threshold_{0.16};
    double predictive_prescreen_step_floor_{0.015};
    double predictive_prescreen_explore_prob_{0.04};
    double predictive_prescreen_min_eval_frac_{0.25};

    // State
    double prescreen_skip_ema_{0.0};

    // -----------------------------------------------------------------------
    // Mechanism C – Stagnation-triggered partial restart
    //
    // When disabled: no restarts (classic behaviour).
    // When enabled:  if the global best has not improved for
    //                restart_stagnation_window_ consecutive iterations AND
    //                minimum progress has been reached AND the restart budget
    //                is not exhausted, a partial restart is performed:
    //
    //   1. Top (restart_elite_frac_ * N) individuals are kept unchanged.
    //   2. Remaining individuals are re-initialised:
    //        restart_local_frac_      → Gaussian around best_x_ with
    //                                   per-dimension sigma that shrinks
    //                                   as progress advances.
    //        1 - restart_local_frac_  → uniform draw from full domain.
    //   3. A fraction restart_mem_reset_frac_ of writable MF/MCR slots is
    //      reset to initial values so the memory can re-learn.
    //   4. The archive is cleared (positions are stale after restart).
    //   5. Stagnation counters for A and C are reset.
    //
    // This mechanism directly addresses inter-run inconsistency (large gap
    // between best value and mean) caused by premature convergence to local
    // optima.
    // -----------------------------------------------------------------------
    bool   restart_enable_{false};
    int    restart_stagnation_window_{20};  // no-improvement iters before restart
    double restart_elite_frac_{0.15};       // top fraction kept unchanged
    double restart_local_frac_{0.50};       // fraction of new indiv. placed near best
    double restart_radius_init_{0.25};      // initial per-dim sigma as frac of dim range
    double restart_mem_reset_frac_{0.60};   // fraction of writable MF/MCR slots to reset
    int    restart_max_count_{5};           // maximum restarts per run
    double restart_min_progress_{0.10};     // do not restart before this budget fraction

    // State
    int restart_count_{0};
    int restart_stagnation_ctr_{0};

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }

    void   ensureInBounds(Vec& x);
    bool   isInBounds(const Vec& x) const;
    void   trimArchive(int max_size);
    void   doRestart(double progress);   // Mechanism C implementation

    static double clamp01(double v);
    double domainDiagonal() const;
    double normalizedDistance(const Vec& a, const Vec& b) const;
    double directionAlignment(const Vec& from, const Vec& to1, const Vec& to2) const;

    double adaptiveWeightMultiplier(const Vec& xi,
                                    const Vec& xp,
                                    int rank_pos,
                                    int N,
                                    double progress) const;

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
