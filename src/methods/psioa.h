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
#include <cctype>

namespace optimsolution {

// PSIOA — Parallel Sporulation-Inspired Optimization Algorithm (parallel islands)
//
// Config keys (same philosophy as ppso/pde/pga):
//   population, islands, NR, Np, propagation, eps_stop, NM,
//   local_rate, local_method, inrun_on_improve,
//   end_local_refine, end_local_method,
//   stop_after_islands, use_openmp, threads
//
// PSIOA-specific:
//   c1, c2: coefficients of the sporulation movement
//   Rmin, Rmax: lower/upper dispersion radius
//   pspor0, pgerm0: initial sporulation/germination probabilities
//   pzero: probability of zero-reset per dimension
//   adapt_R_kappa: adaptation rate for R (0..1)
//   adapt_prob_kappa: adaptation rate for probabilities (0..1)
//   crowding_metric: "bnorm" (bounds-normalized L2) or "l2"
//   island_plateau_iters: per-island plateau termination
//   island_target_f: per-island target value
class PSIOA : public Optimizer {
public:
    PSIOA() = default;
    ~PSIOA() override = default;
	std::string methodShortName() const override { return "psioa"; }
	std::string methodFullName()  const override { return "Parallel Sporulation-Inspired Optimization Algorithm (PSIOA)"; }

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
        std::vector<Vec>    X;    // population (microorganisms)
        std::vector<double> fX;   // fitness
        Vec    gbest_x;
        double gbest_f{std::numeric_limits<double>::infinity()};
        std::deque<double> delta_hist; // For the NM window: front holds last fmin, rest |delta|
        std::mt19937_64 rng;

        // self-adaptive controls
        double R;             // dispersal radius
        double p_spor;        // sporulation probability
        double p_germ;        // germination probability
        // trend trackers
        double last_avg_f{std::numeric_limits<double>::infinity()};
    };

    // --- Main parameters (uniform naming) ---
    int         pop_cfg_{-1};
    int         islands_{1};
    int         NR_{10};
    int         Np_{10};
    std::string propagation_{"NtoN"};
    double      eps_stop_{1e-6};
    int         NM_{15};

    bool        use_openmp_{true};
    int         threads_{0};

    // locals (in-run & end)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};      // p_loc (0.005 ~ 0.5%, consistent with common setups)
    bool        inrun_on_improve_{true};
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    int         stop_after_islands_{-1};

    // --- PSIOA coefficients & adaptation ---
    double c1_{1.0};
    double c2_{1.0};
    double Rmin_{0.05};
    double Rmax_{0.5};
    double p_spor0_{0.5};
    double p_germ0_{0.5};
    double p_zero_{0.02};          // per-dimension zero-reset prob
    double adapt_R_kappa_{0.1};    // R adaptation rate
    double adapt_prob_kappa_{0.05};
    std::string crowding_metric_{"bnorm"};

    // per-island policies
    int         island_plateau_iters_{-1};                         // <=0 off
    double      island_target_f_{std::numeric_limits<double>::quiet_NaN()};

    // --- State ---
    std::vector<Island>        isl_;
    std::vector<unsigned char> island_done_;
    std::vector<int>           last_improve_iter_;
    int    K_{0};
    int    final_population_{-1};
    int    per_island_{0};
    Vec    global_best_x_;
    double global_best_f_{std::numeric_limits<double>::infinity()};
    bool   hard_stop_now_{false};

private:
    inline double clamp(double v, double lo, double hi) const {
        return v<lo?lo:(v>hi?hi:v);
    }
    void   ensureBounds(Vec& v);
    double eval(const Vec& v){ return prob_->evaluate(v); }

    // PSIOA primitives
    Vec    make_spore_(Island& S, const Vec& xi) const;
    int    most_similar_index_(const Island& S, const Vec& s) const;

    // adaptation & stopping
    void   adapt_controls_(Island& S, double avg_f, double prev_avg);
    bool   stopping_hold_(Island& S, double fmin_k);

    // migration
    void   propagate_();
    void   send_best_Np_(int src, const std::vector<int>& dst_ids);
    std::vector<int> best_indices_(const Island& S, int N) const;
    int    worst_index_(const Island& S) const;
};

} // namespace optimsolution
