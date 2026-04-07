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

namespace optimsolution {

/**
 * RARQ: Roulette Adaptive Robust Quarantine
 *
 * Hybrid DE-based population method with:
 *  - pbest/1 with archive (ARQ kernel)
 *  - quarantine of outliers
 *  - micro-restarts
 *  - roulette selection among 3 high-level strategies:
 *      0: stepCore()       → baseline ARQ kernel
 *      1: stepIntensify()  → aggressive exploitation near p-best
 *      2: stepExplore()    → aggressive exploration + micro-restarts
 */
class RARQ : public Optimizer {
public:
    RARQ()  = default;
    ~RARQ() override = default;

    std::string methodShortName() const override { return "rarq"; }
    std::string methodFullName()  const override {
        return "Roulette Adaptive Robust Quarantine (RARQ)";
    }

    void configure(const MethodConfig &mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override {}   // Final local search is already handled by Optimizer

private:
    using Vec = std::vector<double>;

    // --- population ---
    int pop_init_{50};
    int pop_min_{4};
    int N_{0};

    // --- roulette ---
    static constexpr int H_ = 3;
    double ni_[H_]{};      // adaptive weights
    double success_[H_]{}; // counters
    double n0_{2.0};       // Initial weight (double to eliminate C4244)
    double delta_{0.05};
    int    nrst_{0};

    // --- population storage ---
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // --- ARQ history ---
    double muF_{0.5};
    double muCR_{0.9};
    double F_lo_{0.1};
    double F_hi_{1.0};
    double CR_lo_{0.0};
    double CR_hi_{1.0};
    double sh_c_{0.1};

    // --- archive ---
    std::vector<Vec> archive_;
    std::size_t      archive_max_{200};

    // --- misc ---
    int iter_{0};
    int no_improv_iters_{0};
    int micro_restart_period_{50};

    // in-run local search (per-method)
    std::string local_method_;
    double      local_rate_{0.0};

    // Roulette helpers
    std::pair<int, double> rouletteSelect();
    void resetRoulette();

    // Strategies
    void stepCore();
    void stepIntensify();
    void stepExplore();

    // ARQ primitives
    void ensureBounds(Vec &x) const;
    void pushArchive_(const Vec &x);
    int  pickArchiveIndex_();              // No longer const (uses rng_)
    int  pickPbestIndex_(double pfrac);    // No longer const (uses rng_)
    double bnDistance_(const Vec &a, const Vec &b) const;
    int pickRTRNeighbor_(const Vec &trial,
                         const std::vector<int> &pool) const;

    void quarantineOutliers_();
    void microRestart_();

    void trial_pbest1A_bin_(int i,
                            const Vec &xi,
                            Vec &tr,
                            double F,
                            double CR,
                            double pfrac,
                            bool useArchive);
};

} // namespace optimsolution
