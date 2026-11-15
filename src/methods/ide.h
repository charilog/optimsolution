#pragma once
// =============================================================================
//  iDE -- improved Differential Evolution
//
//  L-SRTDE core (Stanovov & Semenkin, IEEE CEC 2024 winner; see lsrtde.h)
//  extended with the mechanisms of SPARQ that were measured to improve the
//  final solutions. Every module can be switched off in [ide]; with all
//  modules off iDE reproduces L-SRTDE exactly (same random stream, same
//  results), which is how the port is verified.
//
//  Modules (each one is independent; defaults chosen by an ablation study on
//  CEC 2017, D = 10 and 30, 15 seeds, Mann-Whitney p < 0.05 vs L-SRTDE):
//   (1) EIG  -- eigen-coordinate binomial crossover (SPARQ / EA4eig idea):
//               the crossover is done in the eigen-basis of the covariance of
//               the best eig_frac part of the main population, i.e.
//               rotation-invariantly (all CEC 2017/2022 functions are
//               rotated). Its share is ADAPTIVE (eig_adapt = 1): p = SR_eig /
//               (SR_eig + SR_bin) from the success rates of the two crossovers
//               (forgetting factor 0.9 per generation), clamped to
//               [eig_pmin, eig_pmax] -- separable problems drift to the
//               coordinate-wise crossover, rotated ones to the eigen one. The
//               basis is refreshed every eig_period generations while the
//               population has >= D+2 members and is kept afterwards.
//                                                            default: ON
//   (2) ARCH -- external archive (SPARQ / JADE): front members overwritten by
//               successful trials enter an archive of size archive_rate*N; the
//               second member r2 of the difference vector is drawn from
//               main + archive (more diversity; Schwefel F10, D=30: 427 -> 129).
//                                                            default: ON
//   (3) POLISH -- end-phase elite refinement (SPARQ "elite polish"), here a
//               (1+1)-CMA-ES (Igel, Suttorp & Hansen, GECCO 2006) started from
//               the best solution after polish_start of the budget, with the
//               population's eigen-basis as the initial shape, a hard cap of
//               polish_budget * max_evals evaluations and SPARQ's efficiency
//               arbitration (switched off after polish_patience bursts that
//               gain less per evaluation than the DE). No significant gain on
//               CEC 2017 and a loss on Weierstrass-60 (1e-5 -> 2e-3).
//                                                            default: OFF
//   (4) REJUV -- stagnation rejuvenation (SPARQ): partial restart of the
//               front after rejuv_gens generations without improvement.
//               Measured HARMFUL (8 of 29 functions worse at D=30).
//                                                            default: OFF
//   (5) FS   -- funnel scouting & adoption (iDE-FS). Diagnosis on CEC 2022:
//               on multi-funnel functions (composition F10/F12, Rosenbrock
//               D=20) the iDE dynamics fall DETERMINISTICALLY into the same
//               funnel (F10: always the o3 component, 2500; best-of-8
//               independent restarts changes nothing), whereas search
//               dynamics of a different kind land elsewhere (CLPSO enters the
//               global o1 funnel of F10, an unrotated Schwefel component).
//               When the DE has STAGNATED -- best improved by less than
//               scout_tol (relative) over scout_window of the budget, by less
//               than scout_gain_frac of the front's fitness spread (median -
//               best), that spread is below scout_fspread*|best|, and the
//               front has collapsed (spread < scout_spread of the diagonal) --
//               a scouting phase starts: a CLPSO swarm (Liang et al. 2006) and
//               an IPOP-CMA-ES, both started from fresh random samples and
//               never fed by the DE, receive scout_ratio evaluations per DE
//               evaluation, up to min(scout_max, scout_max_remain*remaining).
//               A scout best that beats the DE is ADOPTED: far from the DE's
//               best (another funnel) it replaces the worst scout_adopt_frac
//               of both populations together with a small cloud of samples
//               and the phase ends; the DE then descends to the bottom of the
//               new funnel. Because the trigger needs a stalled, collapsed
//               DE, functions on which the DE is still descending (or sits on
//               a plateau with a spread-out population, Rastrigin-type) never
//               pay for the scouts.                         default: ON
//               CEC 2022 (15 seeds, vs iDE without FS): 3 wins, 21 ties,
//               0 losses; F10 D10 mean 100 -> 73 (best 100 -> 6.8), F10 D20
//               mean 107 -> 94 (best 100 -> 18.6), F12 D10 164.9 -> 164.6.
//
//  Ablation (wins / ties / losses vs L-SRTDE, CEC 2017, 15 seeds):
//      EIG (fixed p=0.4) alone      D10 1/26/2   D30 3/24/2
//      ARCH alone                   D10 2/27/0   D30 2/25/2
//      POLISH alone                 D10 0/29/0   D30 1/28/0
//      REJUV alone                  D10 0/24/5   D30 0/21/8
//      EIG(fixed)+ARCH+POLISH       D10 0/28/1   D30 4/24/1
//      EIG(adaptive)+ARCH           D10 3/25/1   D30 3/26/0
//      EIG(adaptive)+ARCH+FS (default): see (5)
//
//  Mechanisms of SPARQ that were NOT taken over, and why:
//   * OBL (opposition-based) basin escape -- the opposite point lb+ub-x is
//     biased towards the centre of the box (it helps only when the optimum is
//     at the centre, e.g. unshifted test functions);
//   * Thompson bandit ARQ/IDE switching, Levy quarantine, RTR replacement,
//     SHADE F memory -- L-SRTDE's own success-rate F and two-population
//     scheme were better in the CEC 2017/2022 comparison (L-SRTDE won 65,
//     SPARQ 5 of 111 function/dimension pairs).
// =============================================================================

#include "optimizer.h"

#include <vector>
#include <string>
#include <random>
#include <cmath>

namespace optimsolution {

class IDE : public Optimizer {
public:
    IDE() = default;
    ~IDE() override = default;

    std::string methodShortName() const override { return "iDE"; }
    std::string methodFullName()  const override {
        return "iDE (L-SRTDE + adaptive eigen crossover + archive + funnel scouting CLPSO/CMA-ES)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);
        end_local_refine_ = finalLocalEnabled();
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;
    using Mat = std::vector<std::vector<double>>;

    enum class BoundMode { Random, Midpoint, Clip };

    // ---- L-SRTDE core ------------------------------------------------------
    double eval(const Vec& x);
    int    randInt(int n);
    double randU01();
    double sampleF(double meanF);
    double sampleCR(double mcr);
    double repairComponent(int j, double v, double parent_j);
    double weightedLehmerMean(const std::vector<double>& values,
                              const std::vector<double>& weights) const;
    int    targetPopulationSize() const;
    void   removeWorstFromFront(int new_size);
    void   sortAndTruncateMain(int new_size);
    void   sortedOrder(const std::vector<double>& f, int n, std::vector<int>& idx) const;
    void   noteBest(const Vec& x, double f);

    // ---- (1) eigen crossover -------------------------------------------------
    void   refreshEigenBasis();
    static void jacobiEigen(const Mat& A, Mat& V, Vec& w);

    // ---- (2) archive ---------------------------------------------------------
    void   archivePush(const Vec& x);

    // ---- (3) polish: (1+1)-CMA-ES ----------------------------------------------
    void   polishInit();
    void   polishBurst();
    bool   choleskyLower(const Mat& C, Mat& A) const;
    void   injectIntoPopulations(const Vec& x, double f);

    // ---- (4) rejuvenation ------------------------------------------------------
    void   rejuvenate();

    // ---- (5) funnel scouting & adoption (iDE-FS) ---------------------------------
    struct ClpsoScout {                 // comprehensive learning PSO (Liang et al. 2006)
        bool active = false, started = false;
        long long quota = 0, used = 0;
        int  N = 0;
        std::vector<Vec> X, V, P;       // positions, velocities, pbest
        Vec  FP, Pc, vmax;
        std::vector<std::vector<int>> ex; // exemplar per particle and dimension
        std::vector<int> noimp;
        Vec  bx; double bf = 0.0;       // scout best
    };
    struct CmaScout {                   // (mu/mu_w, lambda)-CMA-ES with IPOP restarts
        bool active = false, started = false;
        long long quota = 0, used = 0;
        int  lambda0 = 0, lambda = 0, mu = 0, restarts = 0;
        Vec  w; double mueff = 0, cc = 0, cs = 0, c1 = 0, cmu = 0, damps = 0, chiN = 0;
        Vec  m, pc, ps, d;              // mean, paths, sqrt of eigenvalues
        Mat  C, B;                      // covariance, eigenvectors (columns)
        double sigma = 0.0;
        int  gen = 0, eigen_gen = 0, stall = 0;
        double run_best = 0.0;
        Vec  bx; double bf = 0.0;       // scout best over all restarts
    };
    void   scoutsStart(long long total);           // start a scouting phase with 'total' evaluations
    bool   scoutsRun(long long budget);            // spend up to 'budget' evaluations; true = funnel adopted
    double mainBest(Vec* x = nullptr) const;       // best member of the DE populations
    double frontSpread() const;                    // front diversity relative to the box diagonal
    void   clpsoStart(ClpsoScout& s);
    void   clpsoExemplars(ClpsoScout& s, int i);
    void   clpsoStep(ClpsoScout& s);
    void   cmaRestart(CmaScout& s);
    void   cmaEigen(CmaScout& s);
    void   cmaStep(CmaScout& s);
    double scoutEval(const Vec& x);
    bool   scoutAdopt(const Vec& sx, double sf, const Vec& spread);   // true = another funnel

    // ---- populations -------------------------------------------------------
    std::vector<Vec>    main_X_;
    std::vector<double> main_F_;
    std::vector<Vec>    front_X_;
    std::vector<double> front_F_;
    std::vector<Vec>    archive_;

    // ---- adaptation state --------------------------------------------------
    std::vector<double> MCR_;
    int    memory_iter_    = 0;
    int    pf_index_       = 0;
    double success_rate_   = 0.5;
    int    n_init_         = 0;
    int    generation_     = 0;

    // eigen state
    Mat    B_;                       // columns = eigenvectors (row-major B_[i][j])
    Vec    eig_val_;                 // eigenvalues of the last basis
    bool   eig_valid_      = false;
    int    gens_since_eig_ = 0;
    double eig_p_cur_      = 0.4;    // current probability of the eigen crossover
    double eig_ts_ = 0.0, eig_tn_ = 0.0, bin_ts_ = 0.0, bin_tn_ = 0.0;   // decayed success counts

    // polish state
    bool   pol_started_    = false;
    bool   pol_dead_       = false;
    long long pol_used_    = 0;
    Vec    pol_x_;
    double pol_f_          = 0.0;
    double pol_sigma_      = 0.0;
    double pol_psucc_      = 0.0;
    Vec    pol_pc_;
    Mat    pol_C_, pol_A_;
    int    pol_fail_       = 0;
    int    pol_bad_        = 0;      // consecutive bursts less efficient than the DE

    // scouting state
    ClpsoScout clpso_;
    CmaScout   cma_;
    long long  scout_total_used_ = 0;
    int        scout_adoptions_  = 0;
    bool       scout_phase_      = false;   // a scouting phase is running (the DE is paused)
    int        scout_phases_     = 0;       // scouting phases started in this run
    std::vector<std::pair<long long, double>> best_hist_;   // (evaluations, DE best) per generation

    // rejuvenation state
    double last_best_      = 0.0;
    int    stall_gens_     = 0;
    int    rejuv_count_    = 0;

    // ---- configuration: L-SRTDE core ([ide] section) -----------------------
    int    pop_override_       = 0;
    double np_init_mult_       = 20.0;
    int    np_min_             = 4;
    int    memory_size_        = 5;
    double mcr_init_           = 1.0;
    double memory_c_           = 0.5;
    double sigma_cr_           = 0.05;
    double f_base_             = 0.4;
    double f_amp_              = 0.25;
    double f_sr_coef_          = 5.0;
    double sigma_f_            = 0.02;
    double pbest_rate_         = 0.7;
    double pbest_sr_coef_      = 7.0;
    int    pbest_min_          = 2;
    double rank_pressure_      = 3.0;
    double sr_init_            = 0.5;
    bool   delta_from_parent_  = false;
    BoundMode bound_mode_      = BoundMode::Random;

    // ---- configuration: modules --------------------------------------------
    bool   eig_enable_     = true;
    double eig_prob_       = 0.4;
    double eig_frac_       = 0.5;
    int    eig_period_     = 5;
    int    eig_max_dim_    = 100;
    bool   eig_adapt_      = true;     // adapt eig_prob from the success rates of both crossovers
    double eig_pmin_       = 0.1;
    double eig_pmax_       = 0.9;

    bool   arch_enable_    = true;
    double arch_rate_      = 1.0;

    bool   polish_enable_  = false;
    double polish_start_   = 0.85;
    double polish_budget_  = 0.05;
    int    polish_patience_ = 3;

    bool   rejuv_enable_   = false;
    int    rejuv_gens_     = 0;        // 0 = automatic (max(50, 5*D/... see init))
    double rejuv_keep_     = 0.25;
    double rejuv_cutoff_   = 0.7;

    bool   scout_enable_      = true;
    double scout_after_       = 0.15;  // earliest start of a scouting phase (fraction of the budget)
    double scout_window_      = 0.10;  // stagnation window (fraction of the budget)
    double scout_tol_         = 1e-4;  // stagnation: relative improvement of the DE best over the window below this
    double scout_max_         = 0.35;  // budget of a scouting phase (fraction of max_evals) ...
    double scout_max_remain_  = 0.60;  // ... but at most this fraction of the remaining budget
    int    scout_max_phases_  = 1;     // scouting phases per run
    double scout_clpso_share_ = 0.7;   // share of a phase for CLPSO (rest: CMA-ES)
    double scout_adopt_frac_  = 0.20;  // fraction of the population replaced by an adopted funnel
    double scout_min_dist_    = 0.05;  // "another funnel" = farther than this fraction of the box diagonal
    double scout_spread_      = 0.03;  // stagnation also requires the front spread below this (fraction of the diagonal)
    double scout_ratio_       = 1.0;   // scout evaluations per DE evaluation during a scouting phase
    double scout_gain_frac_   = 0.1;   // stagnation: window gain <= this * (median - best) of the front
    double scout_fspread_     = 2e-3;  // stagnation: (median - best) of the front below this * |best|
    int    scout_cma_max_dim_ = 100;   // no CMA scout above this dimension (O(D^3) eigendecomposition)

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int debug_ide_ = 0;
};

} // namespace optimsolution
