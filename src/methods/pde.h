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

// Parallel Differential Evolution (DE) with islands, migration, and GA-style local search.
// Strategies: rand1bin, best1bin, current2best1bin, pbest1bin.
// Keys (as in GA/ppso): population, islands, NR, Np, propagation, eps_stop, NM,
//                      local_rate, local_method, inrun_on_improve,
//                      end_local_refine, end_local_method,
//                      stop_after_islands,
// DE-keys: F, CR, strategy, p_top, jitter.
// Additional per-island stop: island_plateau_iters, island_target_f.
class PDE : public Optimizer {
public:
    PDE() = default;
    ~PDE() override = default;
	std::string methodShortName() const override { return "pde"; }
	std::string methodFullName()  const override { return "Parallel Differential Evolution"; }

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
        std::vector<Vec> X;      // population
        std::vector<Vec> Trial;  // trial vectors
        std::vector<double> fX;  // fitness
        Vec    gbest_x;
        double gbest_f{std::numeric_limits<double>::infinity()};
        std::deque<double> delta_hist; // front=last fmin, rest=|delta|
        std::mt19937_64 rng;
    };

    // --- Parameters (consistent with GA/ppso) ---
    int         pop_cfg_{-1};            // per-method population (if provided)
    int         islands_{1};
    int         NR_{15};
    int         Np_{15};
    std::string propagation_{"NtoN"};    // 1to1 | 2toN | nto1 | NtoN
    double      eps_stop_{1e-6};
    int         NM_{15};

    bool        use_openmp_{true};
    int         threads_{0};             // 0 => auto

    // In-run local (logging as in GA/ppso)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        inrun_on_improve_{true};

    // Final local search
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // Global stop: stop when that many islands have completed
    int         stop_after_islands_{-1}; // if <=0 => islands_

    // --- DE params ---
    double      F_{0.5};
    double      CR_{0.9};
    std::string strategy_{"rand1bin"};   // rand1bin|best1bin|current2best1bin|pbest1bin
    double      p_top_{0.2};             // for pbest (0,1]
    double      jitter_{0.0};            // F*(1 + U[-jitter, +jitter])

    // --- NEW: per-island completion policies (to activate stop_after_islands) ---
    int         island_plateau_iters_{-1};                         // <=0: off (e.g. 150-300)
    double      island_target_f_{std::numeric_limits<double>::quiet_NaN()}; // NaN: off

    // --- State ---
    std::vector<Island>        isl_;
    std::vector<unsigned char> island_done_; // 1 => island completed
    std::vector<int>           last_improve_iter_; // last gbest improvement iteration per island
    int    K_{0};
    int    particles_per_island_{0};
    int    final_population_{-1};
    Vec    global_best_x_;
    double global_best_f_{std::numeric_limits<double>::infinity()};
    bool   hard_stop_now_{false};

private:
    inline double clamp(double v, double lo, double hi) const {
        return v<lo?lo:(v>hi?hi:v);
    }
    void   ensureBounds(Vec& v);
    double eval(const Vec& v){ return prob_->evaluate(v); }

    // DE helpers
    void   de_iteration_island_(Island& S);
    void   mutate_rand1_(Island& S, int i, Vec& donor);
    void   mutate_best1_(Island& S, int i, Vec& donor);
    void   mutate_current2best1_(Island& S, int i, Vec& donor);
    void   mutate_pbest1_(Island& S, int i, Vec& donor);

    void   crossover_bin_(Island& S, int i, const Vec& donor, Vec& trial);

    // migration & stopping
    void   propagate_();
    bool   stopping_hold_(Island& S, double fmin_k);

    void   send_best_Np_(int src, const std::vector<int>& dst_ids);
    std::vector<int> best_indices_(const Island& S, int N) const;
    int    worst_index_(const Island& S) const;

    // utilities
    int    pick_random_excluding_(int n, std::mt19937_64& rng, int ex) const;
    void   pick_3_distinct_(int n, std::mt19937_64& rng, int i, int& r1, int& r2, int& r3) const;
    double F_tuned_(std::mt19937_64& rng) const; // F*(1 + U[-jitter, +jitter])
};

} // namespace optimsolution
