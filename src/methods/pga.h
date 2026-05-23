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

// Parallel Genetic Algorithm (steady-state, elitist) with islands/migration and GA-style local search.
// Keys (as in GA/ppso/pde): population, islands, NR, Np, propagation, eps_stop, NM,
//                          local_rate, local_method, inrun_on_improve,
//                          end_local_refine, end_local_method,
//                          stop_after_islands,
// GA-keys: tk (tournament size), uox_p (uniform crossover prob per gene), pm, mut_sigma,
//          optionally use_openmp, threads.
// Additional per-island stop: island_plateau_iters, island_target_f.
class PGA : public Optimizer {
public:
    PGA() = default;
    ~PGA() override = default;
	std::string methodShortName() const override { return "pga"; }
	std::string methodFullName()  const override { return "Parallel Genetic Algorithm"; }

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
        std::vector<Vec>    X;    // population
        std::vector<double> fX;   // fitness
        Vec    gbest_x;
        double gbest_f{std::numeric_limits<double>::infinity()};
        std::deque<double> delta_hist; // front = last fmin, rest = |delta|.
        std::mt19937_64 rng;
    };

    // --- Parameters (consistent with GA/ppso/pde) ---
    int         pop_cfg_{-1};            // Per-method population (if provided).
    int         islands_{1};
    int         NR_{15};
    int         Np_{15};
    std::string propagation_{"NtoN"};    // 1to1 | 2toN | NtoN | nto1
    double      eps_stop_{1e-6};
    int         NM_{15};

    bool        use_openmp_{true};
    int         threads_{0};             // 0 => auto

    // In-run local (GA-style)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};
    bool        inrun_on_improve_{true};

    // Final local search.
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // Global stop: stops when this many islands have completed.
    int         stop_after_islands_{-1}; // if <= 0 => islands_.

    // --- GA params ---
    int         tk_{3};          // tournament size
    double      uox_p_{0.5};     // Uniform crossover probability (preference for parent 1 per gene).
    double      pm_{0.1};        // mutation prob per gene
    double      mut_sigma_{0.1}; // Mutation sigma as a fraction of the search range.

    // --- NEW: per-island completion policies ---
    int         island_plateau_iters_{-1};                         // <=0: off
    double      island_target_f_{std::numeric_limits<double>::quiet_NaN()}; // NaN: off

    // --- State ---
    std::vector<Island>        isl_;
    std::vector<unsigned char> island_done_; // 1 => island finished.
    std::vector<int>           last_improve_iter_; // Last iteration of gbest improvement per island.
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

    // GA helpers
    int    tournamentSelect_(Island& S);
    void   crossover_uox_(Island& S, const Vec& p1, const Vec& p2, Vec& child);
    void   mutate_gauss_(Island& S, Vec& child);

    // migration & stopping
    void   propagate_();
    bool   stopping_hold_(Island& S, double fmin_k);

    void   send_best_Np_(int src, const std::vector<int>& dst_ids);
    std::vector<int> best_indices_(const Island& S, int N) const;
    int    worst_index_(const Island& S) const;
};

} // namespace optimsolution
