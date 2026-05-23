#pragma once
#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace optimsolution {

/**
 * PolyphaseDE
 *
 * Multi-strategy, phase-aware DE specifically targeted at hard continuous
 * problems of the polyphase / fmsynth type.
 *
 *  - 3 strategies:
 *      0: rand/1/bin (exploration-heavy)
 *      1: current-to-pbest/1/bin (balanced)
 *      2: pbest/2/bin (aggressive exploitation)
 *  -
 */
class PolyphaseDE : public Optimizer {
public:
    PolyphaseDE()  = default;
    ~PolyphaseDE() override = default;

    std::string methodShortName() const override { return "polyde"; }

    std::string methodFullName() const override {
        return "Polyphase Expert Multi-Strategy DE";
    }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // --- Basic control ---
    int    pop_{100};          // base population size
    int    dim_{0};
    int    iter_{0};

    // Multi-strategy (3 strategies)
    static constexpr int NST_ = 3;
    std::vector<double> p_strat_;      // probabilities per strategy (size NST_)
    std::vector<double> succ_strat_;   // successes per strategy (window)
    std::vector<double> trial_strat_;  // trials per strategy (window)

    // Strategy-specific parameter means (muF, muCR)
    double muF_[NST_]   {0.7, 0.6, 0.4};
    double muCR_[NST_]  {0.5, 0.8, 0.9};
    double c_adapt_{0.1};      // learning rate for muF/muCR + probs

    // Parameter bounds per strategy
    double F_lo_[NST_]  {0.4, 0.3, 0.2};
    double F_hi_[NST_]  {1.0, 0.9, 0.8};
    double CR_lo_[NST_] {0.0, 0.1, 0.4};
    double CR_hi_[NST_] {0.9, 1.0, 1.0};

    // Exploration ↔ exploitation schedule
    double phase_explore_end_{0.40};  // 0–0.4: exploration
    double phase_balance_end_{0.80};  // 0.4–0.8: balance
    // 0.8–1.0: aggressive exploitation

    // Micro-restart
    int    stagn_iters_{0};
    int    stagn_trigger_soft_{25};   // mild restart
    int    stagn_trigger_strong_{70}; // strong restart
    double restart_frac_soft_{0.10};
    double restart_frac_strong_{0.25};
    double restart_sigma_{0.20};      // radius (fraction of box size)

    // Diversity-based info (simple)
    double div_low_{0.05};
    double div_high_{0.40};

    // Local search
    bool        end_local_enable_{true};   // final local search
    std::string end_local_method_{"lbfgs"};
    double      inrun_local_base_{0.0};    // base prob, scaled by phase

    // Population & archive
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    archive_;
    double archive_rate_{1.5};             // |A| <= archive_rate_ * N

    // Note: best_x_/best_f_ are NOT defined here.
    // Uses those from Optimizer (protected best_x_, best_f_).

    // RNG
    std::mt19937_64 rng_{};
    std::uniform_real_distribution<double> U01_{0.0, 1.0};

    // --- Helpers ---

    inline double clamp_(double v, double lo, double hi) const {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    double eval(const Vec& x) {
        return prob_->evaluate(x);
    }

    void   ensureBounds(Vec& x);
    double bnDistance_(const Vec& a, const Vec& b) const;

    double progress01_() const;        // 0..1 based on calls/max_evals_
    double diversityBestNorm_() const; // normalized diversity around best

    // Archive
    void   archivePush_(const Vec& x);
    int    archivePickIndex_();

    // Multi-strategy param helpers
    int    sampleStrategy_();
    void   sampleFCR_(int strat, double& F, double& CR);
    void   updateStrategyStats_(int strat, double parent_f, double child_f);

    // Strategy operators
    void   strat0_rand1_bin_(int i, Vec& trial, double F, double CR);             // exploration
    void   strat1_current_to_pbest1_bin_(int i, Vec& trial, double F, double CR); // balanced
    void   strat2_pbest2_bin_(int i, Vec& trial, double F, double CR);            // aggressive

    int    pickPbestIndex_(double pfrac);

    // Can exclude up to 5 indices (a..e) in order to select a distinct r.
    int    pickDistinct_(int n,
                         int a = -1,
                         int b = -1,
                         int c = -1,
                         int d = -1,
                         int e = -1);

    // Stagnation & restarts
    void   microRestart_(bool strong);

    // Local search
    void   finalLocalSearch_();
};

} // namespace optimsolution
