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

// Extremely aggressive Archive-based Quarantine DE (simplified & always-attacking)
class AARQ : public Optimizer {
public:
    AARQ() = default;
    ~AARQ() override = default;

    std::string methodShortName() const override { return "aarq"; }

    std::string methodFullName() const override {
        return "Aggressive Archive-based Quarantine Differential Evolution (AARQ)";
    }

    // from [global] settings (end-of-run local search)
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // ---- Core DE / SHADE-style parameters ----
    int    pop_{200};                 // default population (overridden by framework/config)
    double agent_fraction_{1.0};      // update whole population each iteration

    // Success-history (SHADE-like)
    double muF_{0.9};                 // very aggressive F setting
    double muCR_{0.9};                // high CR setting
    double sh_c_{0.10};               // learning rate for muF_/muCR_
    double F_lo_{0.05}, F_hi_{1.8};
    double CR_lo_{0.0},  CR_hi_{1.0};

    // p-best & archive (JADE-style)
    double pbest_frac_{0.20};         // larger p-best set for strong but not overly narrow exploitation
    double archive_rate_{1.5};        // |A| <= archive_rate_ * N

    // Restricted Tournament Replacement (RTR)
    int    rtr_pool_{12};
    double rtr_min_replace_gain_{0.0}; // relative gain threshold

    // Stagnation & restart (very aggressive)
    int    stagnation_trigger_{15};   // number of iterations without improvement that defines "max" stagnation
    double restart_frac_{0.45};       // maximum fraction of the worst individuals that may be restarted
    double restart_sigma_{0.30};      // radius around the best (in box units) for near-best restarts

    // In-run local search (optional)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};     // probability per improved offspring

    // End-of-run local search (global settings)
    bool        end_local_refine_{false};
    std::string end_local_method_{};

    // ---- State ----
    std::vector<Vec>    X_;           // population
    std::vector<double> FX_;          // fitness
    std::vector<Vec>    archive_;     // external archive

    int start_agent_{0};              // kept for compatibility (not critical here)

    // RNG and seeding
    std::mt19937_64 rng_{};
    std::uniform_real_distribution<double> U01_{0.0, 1.0};

    uint64_t user_seed_{0};
    int      run_id_hint_{-1};
    uint64_t seed_used_{0};
    uint64_t runs_started_{0};

    // stagnation counter (used only to scale restart intensity)
    int stagn_iters_{0};

    // explicit population override (from config)
    int pop_override_{-1};

private:
    inline double clamp_(double v, double lo, double hi) const {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    void   ensureBounds(Vec& v);
    double eval(const Vec& v) { return prob_->evaluate(v); }

    int    eliteIndexFinite() const;                  // index of best finite fitness
    int    pickDistinct(int n, int a=-1, int b=-1, int c=-1);
    void   pushArchive_(const Vec& x);
    int    pickArchiveIndex_();
    int    pickPbestIndex_();

    double bnDistance_(const Vec& a, const Vec& b) const;
    int    pickRTRNeighbor_(const Vec& trial, const std::vector<int>& pool) const;

    // Always uses aggressive micro-restart, with a dynamic rate based on stagn_iters_
    void   microRestart_();

    // DE operator (p-best/1 + archive, binomial crossover)
    void   trial_pbest1A_bin_(int i, const Vec& xi, Vec& tr, double F, double CR);
};

} // namespace optimsolution
