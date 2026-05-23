#pragma once
#include "optimizer.h"
#include "init.h"
#include <vector>
#include <deque>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <string>
#include <cmath>
#include <utility>

namespace optimsolution {

// ============================================================================
// HADES -- Hybrid Adaptive Differential Evolution Strategy
//
// Core mechanisms:
//   * Adaptive dual crossover (Thompson bandit over {axis, eigen})
//   * Tiered stagnation response (cascade: severe -> OBL -> soft)
//         Level 1 (soft)   : exploration spike + gentle reseed
//         Level 2 (OBL)    : opposition-based basin escape (relaxed trigger)
//         Level 3 (severe) : partial uniform restart
//   * Elite-ever pool used in pbest mutation + occasional wild F-spikes
//   * NLPSR non-linear population reduction
//   * SHADE circular memory of size H with jSO terminal slot
//   * LSHADE-RSP rank-biased r1/r2 with time-varying pressure
//   * jSO schedules (pbest, CR floor, F cap, K(F))
//   * Eigen-coordinate basis via inline Jacobi eigendecomposition
//   * Thompson sampling bandit over strategies {ARQ, IDE}
//   * Levy-flight Mantegna quarantine
//   * Restricted Tournament Replacement
//   * LATE-PHASE SPIKE SUPPRESSOR (NEW): the wild F Cauchy(1.2, 0.3) spikes
//         get disabled only when progress > late_spike_progress_ AND the
//         recent success rate is healthy (SR > health_spike_rate_).  This is
//         a narrow, late-phase-only guard against disruption of fine-tuning
//         on separable axis-aligned problems such as non-continuous
//         Rastrigin.  Early/mid phase behavior is identical to vanilla HADES.
// ============================================================================

class HADES : public Optimizer {
public:
    HADES() = default;
    ~HADES() override = default;

    std::string methodShortName() const override { return "hades"; }
    std::string methodFullName()  const override {
        return "HADES: Hybrid Adaptive DE with dual-crossover bandit, "
               "ensemble strategies, tiered stagnation restarts";
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override {}

private:
    using Vec = std::vector<double>;
    using Mat = std::vector<std::vector<double>>;

    // ------------------------------------------------------------
    // Population
    // ------------------------------------------------------------
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    A_;          // external archive (JADE-style, FIFO)

    // Per-individual IDE params (self-adaptive)
    std::vector<double> CBF_;
    std::vector<double> CBCR_;

    // ------------------------------------------------------------
    // NLPSR  (N(t) = Ninit + (Nmin-Ninit) * progress(t)^alpha)
    //   alpha = 1   : linear LSHADE-style reduction
    //   alpha < 1   : faster shrink early, slower late  (NL-SHADE-RSP style)
    //   alpha > 1   : slower shrink early, faster late
    // ------------------------------------------------------------
    int    Ninit_{0};
    int    Nmin_{4};
    double nlpsr_alpha_{0.5};
    int    pop_scale_{18};

    // ------------------------------------------------------------
    // SHADE circular memory
    // ------------------------------------------------------------
    int    H_{6};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int    memK_{0};
    double MF_terminal_{0.9};
    double MCR_terminal_{0.9};

    double Flo_{0.05};
    double Fhi_{1.40};

    // ------------------------------------------------------------
    // jSO schedules
    // ------------------------------------------------------------
    double pbest_max_{0.25};
    double pbest_min_{0.10};

    // ------------------------------------------------------------
    // Rank-biased selection
    // ------------------------------------------------------------
    double kr_init_{2.0};
    double kr_final_{3.0};

    // ------------------------------------------------------------
    // Eigen basis
    // ------------------------------------------------------------
    int    eig_period_{10};
    double eig_frac_{0.50};
    Mat    B_rot_;
    bool   eig_valid_{false};
    int    iters_since_eig_{0};
    int    eig_min_D_{2};

    // ------------------------------------------------------------
    // Adaptive crossover bandit {0 = axis-aligned, 1 = eigen}
    // p_eig_ used ONLY as initial Beta prior on the eigen arm.
    // ------------------------------------------------------------
    double p_eig_{0.40};
    double cx_a_[2]{1.0, 1.0};
    double cx_b_[2]{1.0, 1.0};
    double cx_decay_{0.97};
    int    cx_adaptive_{1};       // 1 = Thompson, 0 = fixed p_eig_

    // ------------------------------------------------------------
    // RTR
    // ------------------------------------------------------------
    int    rtr_pool_{14};
    double rtr_pool_frac_{0.10};

    // ------------------------------------------------------------
    // Archive
    // ------------------------------------------------------------
    double archive_rate_{1.5};

    // ------------------------------------------------------------
    // Strategy bandit
    // ------------------------------------------------------------
    int                 h_{2};       // [0]=ARQ, [1]=IDE
    std::vector<double> bandit_a_;
    std::vector<double> bandit_b_;
    double              bandit_decay_{0.97};
    int                 bootstrap_arq_iters_{2};
    int                 bootstrap_left_{0};

    // ------------------------------------------------------------
    // Lévy quarantine
    // ------------------------------------------------------------
    double outlier_alpha_{1.0};
    double outlier_rho_{0.08};
    double levy_beta_{1.5};
    double qscale_{0.10};

    // ------------------------------------------------------------
    // Tiered stagnation response
    // ------------------------------------------------------------
    int    stag_soft_{15};
    int    stag_trigger_{30};
    int    stag_severe_{80};

    int    soft_boost_length_{6};
    int    soft_boost_left_{0};
    double soft_F_extra_{0.3};
    double soft_reseed_frac_{0.05};
    double exploration_spike_p_{0.03};

    double var_collapse_ratio_{1e-3};
    int    obl_patience_mult_{2};
    int    obl_cooldown_{0};
    int    obl_cooldown_init_{80};
    int    obl_skip_cooldown_{4};   // brief cooldown when OBL early-returns
    double obl_frac_{0.30};

    double severe_restart_frac_{0.30};
    int    severe_cooldown_{0};
    int    severe_cooldown_init_{150};

    double best_prev_{std::numeric_limits<double>::infinity()};
    int    no_improve_{0};

    // ------------------------------------------------------------
    // LATE-PHASE SPIKE SUPPRESSOR (NEW)
    // Rolling tracker of (successes, attempts) from stepARQ and stepIDE.
    // The wild-F Cauchy(1.2, 0.3) exploration spikes get suppressed ONLY
    // when BOTH conditions hold:
    //     progress(t) > late_spike_progress_
    //     recent SR  > health_spike_rate_
    // Early/mid phase behavior is identical to vanilla HADES.  This is a
    // minimally invasive, late-phase-only guard targeted at the
    // non-continuous-Rastrigin-style fine-tuning regression.
    // ------------------------------------------------------------
    std::deque<std::pair<int,int>> success_history_;
    int    success_window_{10};
    double late_spike_progress_{0.70};
    double health_spike_rate_{0.15};

    // ------------------------------------------------------------
    // Elite-ever pool
    // ------------------------------------------------------------
    std::vector<Vec>    elite_X_;
    std::vector<double> elite_F_;
    int                 elite_cap_{5};
    double              p_use_elite_{0.15};

    // ------------------------------------------------------------
    // ARQ sweep size
    // ------------------------------------------------------------
    double agent_fraction_{1.0};

    // ------------------------------------------------------------
    // IDE scheduling
    // ------------------------------------------------------------
    int    gmax_{0};
    double T_{0.0};
    int    g_{0};
    int    gt_{0};
    int    Tcurr_{0};
    int    ide_progress_sync_{1};
    int    ide_strict_improve_{1};

    int    debug_{0};

    // ------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------
    inline double eval(const Vec& v) {
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    void   ensureBounds(Vec& v);
    int    randInt(int lo, int hi);
    double randU();
    double cauchy(double loc, double scale);
    double gaussN(double mu, double sig);
    double sampleLevy();
    double progress01() const;

    void   sortByFitness();
    int    targetPopulationSize() const;
    void   shrinkTo(int N);

    double currentKR() const;
    double currentPbest() const;
    int    rankBasedPick(const std::vector<int>& ord, int forbid) const;

    void   initMemory();
    void   sampleFCR(double& F, double& CR);
    void   updateMemoryFromSuccess(const std::vector<double>& SF,
                                   const std::vector<double>& SCR,
                                   const std::vector<double>& SG);

    void   archivePush(const Vec& x);
    void   archiveTrim(int N);

    void   jacobiEigen(const Mat& Ain, Mat& V, std::vector<double>& w) const;
    void   recomputeEigenBasis();
    void   applyBt(const Mat& B, const Vec& x, Vec& out) const;
    void   applyB (const Mat& B, const Vec& x, Vec& out) const;
    void   eigenBinomialCrossover(int D, const Vec& base, const Vec& v,
                                  double CR, Vec& u);

    int    thompsonPick();
    void   banditDecay();
    void   banditRecord(int k, int successes, int attempts);

    int    pickCrossoverMode();
    void   recordCrossoverOutcome(int mode, bool success);
    void   cxBanditDecay();

    void   updateElitePool(const Vec& x, double f);

    // Rolling success tracker (NEW)
    void   pushSuccessHistory(int successes, int attempts);
    double recentSuccessRate() const;
    int    recentAttempts() const;

    double distBN(const Vec& a, const Vec& b) const;
    bool   selectionRTR(int parentIndex, const Vec& u, double fu,
                        double F, double CR, int cx_mode,
                        std::vector<double>& SF,
                        std::vector<double>& SCR,
                        std::vector<double>& SG);

    double computeK(double F) const;
    void   makeTrialARQ(int i, const std::vector<int>& ord,
                        double F, double CR, int& cx_mode, Vec& u);
    void   stepARQ();
    void   stepIDE();

    static double quantile(std::vector<double> v, double q01);
    void   quarantineLevy();
    void   softBoostReseed();
    void   oblBasinEscape(bool allow_without_variance_collapse);
    void   severeRestart();

    void   sampleIDEParamsAt(int idx);
    void   inheritIDEParams(int dst, int src);

    double normalizedPopSpread() const;
};

} // namespace optimsolution
