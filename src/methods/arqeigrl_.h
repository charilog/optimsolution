#pragma once
#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace optimsolution {

/**
 * Hybrid ARQ–LSHADE–RL optimizer (ARQEigRL)
 *
 *  - LSHADE-style population size reduction + success-history MF/MCR.
 *  - ARQ-style diversity: bnDistance, quarantine outliers, micro-restart.
 *  - RL (-greedy + weighted roulette) over four mutation strategies.
 *
 * Strategies (arms):
 *   S0: current-to-pbest/1 + archive (ARQ/JSO-like, exploitation).
 *   S1: rand/1/bin (exploration).
 *   S2: best/1/bin (strong exploitation).
 *   S3: Gaussian search around best_x_ (CMA-like local exploration).
 */
class ARQEigRL : public Optimizer {
public:
    ARQEigRL()  = default;
    ~ARQEigRL() override = default;

    std::string methodShortName() const override { return "arqeigrl"; }
    std::string methodFullName()  const override { return "Hybrid ARQ–LSHADE–RL (ARQEigRL)"; }

    void configure(const MethodConfig& mc) override;
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // --- Population & dimension ---
    int pop_init_{0};
    int pop_min_{4};
    int pop_cfg_{-1};   // if >0 overrides global population

    int D_{0};          // dimension cache

    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // --- LSHADE success-history memory for F/CR ---
    int                 H_{10};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 mem_idx_{0};
    double              c_mem_{0.1};

    double              pmin_{0.05};
    double              pmax_{0.25};

    // --- Archive (JADE/LSHADE style) ---
    double              archive_rate_{1.0};
    std::vector<Vec>    archive_;

    // --- RL over mutation strategies ---
    static constexpr int NUM_STRAT_ = 4;
    double              epsilon_{0.1};   // exploration rate (fixed via settings)
    double              rl_alpha_{0.2};  // learning rate for weights
    std::vector<double> strat_weight_;
    std::vector<double> strat_reward_acc_;
    std::vector<int>    strat_use_count_;

    // --- ARQ-style diversity / restart machinery ---
    double              rtr_frac_{0.15};       // fraction of the population in the RTR pool
    int                 rtr_pool_{5};          // pool size
    double              outlier_alpha_{1.5};   // threshold multiplier for the IQR
    double              quarantine_rate_{0.5};
    double              restart_frac_{0.25};
    double              restart_sigma_{0.3};

    int                 stagnation_window_{20};
    int                 stagnation_counter_{0};
    double              last_best_{std::numeric_limits<double>::infinity()};

    // --- In-run & final local search ---
    std::string         local_method_{"lbfgs"};
    double              local_rate_{0.0};
    bool                end_local_refine_{false};
    std::string         end_local_method_;

    // --- RNG helpers ---
    std::uniform_real_distribution<double> U01_{0.0, 1.0};

    // ===== helper methods =====
    double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    }

    void ensureInBounds(Vec& x) {
        if (!prob_) return;
        const auto& L = prob_->lb();
        const auto& U = prob_->ub();
        for (size_t j = 0; j < x.size(); ++j) {
            if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
            if (x[j] < L[j]) x[j] = L[j];
            if (x[j] > U[j]) x[j] = U[j];
        }
    }

    static double clamp(double v, double lo, double hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    // LSHADE-style sampling of F / CR from MF_/MCR_ (random memory index)
    void sample_F_CR(double& F, double& CR);

    // RL strategy selection (as in mLSHADE_RL)
    int  selectStrategy();

    // diversity distance (ARQ bnDistance_)
    double bnDistance(const Vec& a, const Vec& b) const;

    // RTR neighbor as in ARQ (does not use RNG -> can be const)
    int pickRTRNeighbor(const Vec& trial, const std::vector<int>& pool) const;

    // Archive handling (uses RNG -> is not const)
    void pushArchive(const Vec& x);
    int  pickArchiveIndex();

    // p-best index (top fraction of population) (uses RNG -> is not const)
    int  pickPBestIndex(double pbest_frac);

    // strategies implementations
    void make_trial_S0(int i, Vec& trial, double F, double CR, double pbest_frac);
    void make_trial_S1(int i, Vec& trial, double F, double CR);
    void make_trial_S2(int i, Vec& trial, double F, double CR);
    void make_trial_S3_gaussian(Vec& trial, double sigma);

    // quarantine & micro-restart (copied from ARQ)
    void quarantineOutliers();
    void microRestart();

    // utility: random index != i (uses RNG -> is not const)
    int  distinctIndex(int n, int i);

    void shrinkPopulationLPSR();
};

} // namespace optimsolution
