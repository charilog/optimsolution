#pragma once
#include "optimizer.h"

#include <vector>
#include <random>
#include <limits>
#include <string>
#include <atomic>
#include <functional>

namespace optimsolution {

// Parallel SAO with islands (PSAO).
// Sniffing -> Trailing -> Random per island, with in-run local search
// and final local optimization in end(), as in GA.
class PSAO : public Optimizer {
public:
    PSAO() = default;
    ~PSAO() override = default;
	std::string methodShortName() const override { return "psao"; }
	std::string methodFullName()  const override { return "Parallel Smell Agent Optimization (PSAO)"; }
    // Final local search from [global], as in GA:
    // Read global end_local_* and store them locally.
    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        // Also keep the base flags for the summary
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = finalLocalMethod();
    }

    // Reads per-method options from the [psao] block of optimsolution.cfg
    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    // ---------- Config ----------
    // Per-method population (override of global for correct reporting)
    int  pop_cfg_{-1};
    int  final_population_{-1};

    // Islands / threads / migration
    int  islands_{1};
    int  threads_{0};                // 0 => hardware_concurrency()
    bool use_openmp_{false};         // Parallel across islands using std::thread

    int  NR_{10};                    // migration period (iterations)
    int  Np_{1};                     // migrants per migration (top-Np)
    int  stop_after_islands_{-1};    // How many islands must terminate (<=0 => islands_)

    // In-run local search (per island, on best_x)
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // Final local @ end (method-level, as in GA/SAO)
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // SAO–like coefficients
    double sniff_w_{0.6};
    double sniff_a1_{1.4};
    double sniff_a2_{0.4};

    double trail_sigma0_{0.25};
    double trail_decay_{0.98};

    double rand_rate_{0.25};
    double rand_scale_{0.5};

    // BSS per island (same naming as stop rule)
    double bss_eps_{1e-12};      // cfg key: eps
    int    bss_sim_{12};         // cfg key: sim

    // Propagation strategy
    enum class Propagation { Ring1to1, OneToN, AllToAll };
    Propagation propagation_{Propagation::Ring1to1};

    // ---------- Island state ----------
    struct Island {
        std::vector<Vec>    X;     // Positions
        std::vector<Vec>    V;     // Velocities
        std::vector<double> fX;    // Cost values

        Vec    best_x;
        double best_f{std::numeric_limits<double>::infinity()};

        Vec    worst_x;
        double worst_f{-std::numeric_limits<double>::infinity()};

        double trail_sigma_k{0.25};

        double last_best_f{std::numeric_limits<double>::infinity()};
        int    same_best_iters{0};
        bool   stopped{false};

        std::mt19937_64 rng_local;
        std::uniform_real_distribution<double> U01{0.0,1.0};
        std::normal_distribution<double>       N01{0.0,1.0};

        int size{0};
    };

    std::vector<Island> isl_;
    std::atomic<int>    islands_stopped_{0};
    int                 K_{0};
    bool                globally_stopped_{false};

private:
    // helpers
    inline void ensureBounds(Vec& v);
    inline double eval(const Vec& v){ return prob_->evaluate(v); }

    void init_island_(Island& I, int subpop);
    void island_sniffing_(Island& I);
    bool island_trailing_(Island& I);    // True if the island best improved
    void island_random_(Island& I);
    void island_eval_and_update_(Island& I, int i);
    void island_recompute_worst_(Island& I);
    void island_update_bss_(Island& I);

    void parallel_for_islands_(const std::function<void(int)>& fn);

    // migration variants
    void migrate_ring_1to1_topN_();
    void migrate_one_to_N_topN_();
    void migrate_all_to_all_topN_();

    // helpers for migration
    static void top_indices_(const std::vector<double>& fX, int take, std::vector<int>& out);
    void replace_worst_with_group_(Island& D, const std::vector<Vec>& srcX, const std::vector<double>& srcF);

    void collect_snapshot_(std::vector<double>& allf);
};

} // namespace optimsolution
