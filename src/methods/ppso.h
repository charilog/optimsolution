#pragma once
#include "optimizer.h"
#include "init.h"

#include <vector>
#include <random>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstddef>
#include <deque>
#include <numeric>   // iota

namespace optimsolution {

// Parallel PSO with islands using a unified naming convention consistent with GA.
// - Uses key: population (in the method block). If missing, falls back to [global].population.
// - GA-style in-run local: local_rate / local_method (+ inrun_on_improve).
// - Propagates the top Np individuals every NR steps (1to1, 2toN, nto1, NtoN).
// - Random inertia per iteration: w_iter = 0.5 + U[0, 0.5].
// - Per-island stagnation criterion (NM consecutive |delta| <= eps).
// - stop_after_islands: when >= K islands have completed, freezes everything and
//   feeds a stable snapshot to the global stop so it terminates immediately.
class PPSO : public Optimizer {
public:
    PPSO() = default;
    ~PPSO() override = default;
	std::string methodShortName() const override { return "ppso"; }
	std::string methodFullName()  const override { return "Parallel Particle Swarm Optimization"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    struct Island {
        std::vector<Vec> X;      // Positions.
        std::vector<Vec> V;      // Velocities.
        std::vector<Vec> Pbest;  // Personal bests.
        std::vector<double> fX;      // Current value.
        std::vector<double> fPbest;  // pbest value.
        Vec    gbest_x;
        double gbest_f{std::numeric_limits<double>::infinity()};
        std::deque<double> delta_hist; // front = last fmin, rest = |delta|.
        std::mt19937_64 rng;
    };

    // --- Parameters (consistent with GA) ---
    int         pop_cfg_{-1};            // Population from the method block (if provided).
    int         islands_{1};
    int         NR_{15};
    int         Np_{15};
    double      c1_{1.0};
    double      c2_{1.0};
    double      vmax_frac_{0.2};
    std::string propagation_{"NtoN"};    // 1to1 | 2toN | nto1 | NtoN
    double      eps_stop_{1e-6};
    int         NM_{15};

    bool        use_openmp_{true};
    int         threads_{0};             // 0 => auto

    // In-run local search (GA-style for logging).
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        inrun_on_improve_{true};

    // Final local search in end().
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // Global termination based on the number of finished islands.
    int         stop_after_islands_{-1}; // if <= 0 => default = islands_.

    // --- State ---
    std::vector<Island>        isl_;
    std::vector<unsigned char> island_done_; // 1 => island finished.
    int    K_{0};
    int    particles_per_island_{0};
    int    final_population_{-1};
    Vec    global_best_x_;
    double global_best_f_{std::numeric_limits<double>::infinity()};
    bool   hard_stop_now_{false};       // When true, freezes everything and feeds a stable snapshot.

private:
    inline double clamp(double v, double lo, double hi) const {
        return v<lo?lo:(v>hi?hi:v);
    }
    void   ensureBounds(Vec& v);
    double eval(const Vec& v){ return prob_->evaluate(v); }

    double inertia_omega_(std::mt19937_64& rng) const; // w_iter = 0.5 + U[0, 0.5].
    void   pso_single_iteration_(Island& S);
    void   propagate_();
    bool   stopping_hold_(Island& S, double fmin_k);

    void   send_best_Np_(int src, const std::vector<int>& dst_ids);
    std::vector<int> best_indices_(const Island& S, int N) const;
    int    worst_index_(const Island& S) const;
};

} // namespace optimsolution
