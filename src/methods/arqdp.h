#pragma once

#include "optimizer.h"

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cctype>
#include <deque>

namespace optimsolution {

struct MethodConfig;

/**
 * ARQDP — v10 (Far-Horizon Prediction + Best-First Attack)
 *
 * Key upgrades vs v7:
 *  - Directional Prediction is now TRUST-GATED: applied only when it proves useful.
 *  - Adds "Best Shots": multiple aggressive candidates around current best per iteration.
 *  - Adds "Escape Burst": opposition + heavy-tail jumps when stagnation is detected.
 *  - Keeps hard runtime safety for ded/eld (dim-resize + finite + bounds sanitization).
 */
class ARQDP : public Optimizer {
public:
    ARQDP() = default;
    ~ARQDP() override = default;

    std::string methodShortName() const override { return "arqedp"; }
    std::string methodFullName()  const override { return "ARQ Directional Prediction + Far-Horizon Lookahead + Best-First Attack (ARQDP-v10)"; }

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
    using Vec = std::vector<double>;

    // ------------ bounds cache ------------
    Vec  Lc_;
    Vec  Uc_;
    bool bounds_ready_{false};

    void prepareBoundsCache();
    static double clamp01(double x);

    // dimension safety / sanitization
    void ensureDim(Vec& x, int D) const;
    void sanitizeInPlace(Vec& x);
    Vec  sanitizedCopy(const Vec& x);

    // recompute best after forced changes (restart/resize)
    void recomputeBest();

    // ------------ population ------------
    int pop_init_{100};
    int N_{0};

    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // ------------ Adaptive Population Leaps (APL) ------------
    bool   adaptive_population_{true};
    int    pop_min_{20};
    int    pop_max_{200};
    int    pop_warmup_iters_{5};
    int    pop_check_interval_{3};
    int    pop_window_{12};

    double pop_success_thr_{0.08};
    double pop_impr_thr_{1e-4};
    double pop_div_low_{0.03};
    double pop_div_high_{0.15};

    double pop_shrink_factor_{0.25};
    double pop_expand_factor_{2.0};
    double pop_elite_frac_{0.25};

    int    pop_cooldown_{6};
    int    pop_cooldown_left_{0};

    int    iter_{0};
    std::deque<double> best_hist_;

    // ------------ ARQ / JADE-like controls ------------
    int    H_{20};
    double pbest_{0.11};
    double Fmin_{0.10};
    double Fmax_{1.00};
    double archiverate_{1.0};

    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 k_mem_{0};

    // RTR / restart / robust quarantine
    int    rtr_k_{7};
    double outlier_alpha_{1.5};
    double outlier_rho_{0.20};
    double qsigma_{0.10};
    double worst_frac_{0.10};
    double rsigma_{0.20};
    int    stagnationtrigger_{30};

    bool   quarantine_enabled_{true};

    int    no_improve_{0};
    double best_prev_{std::numeric_limits<double>::infinity()};

    // ------------ Directional Prediction (DP) ------------
    bool   dir_enabled_{true};
        int  dir_sign_{+1}; // +1 or -1 (sign-probed direction)

    double pdir_{0.55};

    double dir_c_{0.35};            // learning rate (dir_learn)
    double dir_beta_{0.60};         // direction scale (dirF)
    double dir_beta_min_{0.0};
    double dir_beta_max_{1.75};

    double dir_clip_{0.30};
    double dir_decay_{0.02};
    int    dir_warmup_{2};
    double dir_step_clip_{0.50};
    double dir_eps_{1e-12};

    // Trust gating
    double dir_trust_{0.50};        // [0,1]
    double dir_trust_inc_{0.06};
    double dir_trust_dec_{0.10};
    double dir_trust_min_{0.05};
    double dir_trust_max_{0.95};
    int    dir_disable_after_{4};   // consecutive bad iters
    int    dir_disabled_for_{0};
    int    dir_disabled_period_{6}; // iterations

    int    dir_bad_iters_{0};
    int    dir_good_iters_{0};

    Vec    dir_path_bn_;
    Vec    dir_step_sum_bn_;
    double dir_step_wsum_{0.0}; // sum weights for directional steps

    // Learn direction also from successful steps even when direction was not used
    double dir_passive_factor_{0.25};

    // Far-horizon prediction (look-ahead) from best-trajectory in BN space
    int    dir_hist_len_{18};        // number of best positions stored (>=3)
    double dir_far_mix_{0.55};       // blend: velocity vs (best-centroid) pull [0..1]
    double dir_far_clip_{0.65};      // per-dimension BN clip for far moves
    double dir_lookahead_{4.0};      // dynamic multiplier for far steps (bigger => "look further")
    double dir_lookahead_min_{0.7};
    double dir_lookahead_max_{7.0};
    double dir_lookahead_grow_{1.12};
    double dir_lookahead_shrink_{0.92};

    std::deque<Vec> best_bn_hist_;   // BN best positions history
    Vec             dir_far_bn_;     // normalized far direction in BN space
    // Endgame caps (avoid slow tail due to heavy inner loops)
    int    endgame_shots_cap_{6};
    int    endgame_refine_cap_{200};

    // ------------ Horizon Scan (Far-looking prediction that can jump basins) ------------
    bool   horizon_enabled_{true};
    int    horizon_trigger_{12};        // activate when no_improve_ >= trigger
    int    horizon_shots_{10};          // evals per scan
    double horizon_scale0_{0.9};        // base step in BN, multiplied by dir_lookahead_
    double horizon_scale_growth_{1.55}; // geometric growth per shot
    double horizon_scale_max_{8.0};     // max step multiplier in BN
    double horizon_orth_sigma_{0.06};   // orthogonal jitter in BN
    double horizon_accept_q_{0.70};     // allow injection if candidate beats this quantile of current pop
    bool   horizon_bridge_enabled_{true};
    double horizon_bridge_dist_{0.35};  // normalized BN distance (0..1) required for barrier-crossing acceptance
    double horizon_bridge_alpha_{1.0};  // temperature multiplier for barrier-crossing acceptance
    double horizon_div_thr_{0.10};      // or trigger earlier if diversity below this
    int    horizon_cooldown_{4};
    int    horizon_cooldown_left_{0};

    // ------------ Hard Restart (forced basin escape when long stagnation) ------------
    bool   hard_restart_enabled_{true};
    int    hard_restart_trigger_{55};   // no_improve_ threshold
    double hard_restart_frac_{0.35};    // fraction of population to restart
    int    hard_restart_cooldown_{12};
    int    hard_restart_cooldown_left_{0};
    int    hard_restart_max_{3};        // max hard restarts per run
    int    hard_restart_count_{0};
    int    basin_memory_{10};           // how many best basins to remember (BN)
// ------------ Best-First Attack (BFA) ------------
    bool   best_shots_enabled_{true};
    int    best_shots_{10};          // candidates per iter
    int    best_shot_every_{1};      // every k iterations
    double best_shot_sigma_{0.08};   // BN scale
    double best_shot_cauchy_prob_{0.35};
    double best_shot_dir_mix_{0.55};
    double best_shot_elite_mix_{0.45};
    double best_shot_clip_{0.40};

    // ------------ Escape Burst (anti-local) ------------
    bool   escape_enabled_{true};
    int    escape_trigger_{22};      // stagnation iters
    double escape_frac_{0.20};       // fraction of pop
    double escape_sigma_{0.22};      // BN jump
    double escape_opposition_prob_{0.55};
    double escape_cauchy_prob_{0.45};

    // ------------ Precision Burst Local Exploiter (PBLE) ------------
    bool   refine_enabled_{true};

    int    refine_trigger_{18};
    int    refine_every_{2};
    double refine_div_thr_{0.07};
    double refine_success_thr_{0.07};

    int    refine_budget_{1200};
    double refine_subspace_frac_{0.18};
    int    refine_subspace_min_{12};
    double refine_cauchy_prob_{0.18};
    double refine_coord_prob_{0.30};

    double refine_sigma0_{0.10};
    double refine_sigma_{0.10};
    double refine_sigma_min_{1e-6};
    double refine_sigma_max_{0.30};
    double refine_shrink_{0.70};
    double refine_grow_{1.10};

    double refine_clip_{0.35};
    double refine_dir_mix_{0.35};

    // ------------ in-run local search (optional) ------------
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // ------------ final local refinement ------------
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // ------------ helpers ------------
    int    randInt(int lo, int hi);
    double randU();
    double randN01();
    double cauchy(double loc, double scale);

        int pickIndexExcluding(const std::vector<int>& exclude);

void sampleDistinctExcluding(int N, int k,
                                 const std::vector<int>& exclude,
                                 std::vector<int>& out);

    // safe evaluate
    double evalX(const Vec& v);

    // bounds helpers
    static double reflectInto(double x, double L, double U);
    void ensureBounds(Vec& x);

    // archive
    std::vector<Vec> A_;
    void addToArchive(const Vec& x);
    void trimArchive();

    // BN transform (uses cached bounds)
    void toBN(const Vec& x, Vec& y) const;
    void fromBN(const Vec& y, Vec& x) const;
    double distBN(const Vec& a, const Vec& b) const;

    // trial generation
    void sample_F_CR(double& F, double& CR, double muF, double muCR);
    void makeTrialBase(int i, const std::vector<int>& sorted_idx, double F, double CR, double pbestFrac, Vec& u, bool& used_dir);

    // direction helpers
    void resetDirection();
    void resetRefine();
    void accumulateDirectionFromSuccess(int replaced_idx, const Vec& u, double gain, bool used_dir);
    void updateDirectionTrust(double success_rate, double rel_impr,
                          double dir_use_rate,
                          double dir_succ_rate);
    void applyDirectionBias(Vec& v, bool& used_dir);
    void updateFarPrediction(double rel_impr, double diversity_bn);
    void buildBlendedDirectionBN(Vec& out, double far_weight) const;
    double cosineSim(const Vec& a, const Vec& b) const;
    void clipVec(Vec& v, double clipAbs) const;
    void normalizeVec(Vec& v, double eps) const;

    // selection
    bool selectionRTR(int parent, const Vec& u, double fu,
                      bool used_dir,
                      double F, double CR,
                      std::vector<double>& SF, std::vector<double>& SCR,
                      std::vector<double>& gains,
                      int& succ_total,
                      int& succ_dir);

    // adaptive update of memories
    void updateMemories(const std::vector<double>& SF,
                        const std::vector<double>& SCR,
                        const std::vector<double>& gains);

    // quarantine + restart
    void quarantineAndRestart();

    // APL
    double estimateDiversityBN(int sampleCount);
    double relImprovementFromHistory() const;
    bool maybeAdaptivePopulationLeap(const std::vector<int>& sorted_idx,
                                       double success_rate,
                                       double diversity_bn,
                                       double rel_impr);
    void   resizePopulation(int newN, const std::vector<int>& sorted_idx);
    void   injectNewIndividuals(int addCount);

    // PBLE
    bool   shouldRefine(double success_rate, double diversity_bn) const;
    bool   precisionBurst(double diversity_bn);

    // Best-First Attack / Escape
    bool   bestShots(const std::vector<int>& sorted_idx, double diversity_bn);
    void   escapeBurst(const std::vector<int>& sorted_idx);

    bool   horizonScan(const std::vector<int>& sorted_idx, double diversity_bn);
    void   hardRestart(const std::vector<int>& sorted_idx);

    // stats
    void sortByFitness(std::vector<int>& idx) const;
    static double quantile(std::vector<double> v, double q01);
};

} // namespace optimsolution
