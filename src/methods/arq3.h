#pragma once
#include "optimizer.h"
#include "init.h"
#include <vector>
#include <random>
#include <limits>
#include <numeric>
#include <algorithm>
#include <string>
#include <cmath>
#include <utility>

namespace optimsolution {

// ============================================================================
// ARQ3 -- successor of ARQ2 designed to be competitive on CEC benchmarks.
//
// Core additions over ARQ2:
//  (1) NLPSR : non-linear population size reduction (LSHADE/NL-SHADE style)
//  (2) SHADE : circular memory of size H for (muF, muCR) with jSO terminal slot
//  (3) RSP   : rank-based selection of r1/r2 with time-varying pressure
//  (4) jSO   : pbest schedule, CR floor, F cap by progress; K(F) time-varying
//  (5) EIG   : eigen-coordinate binomial crossover (rotation invariance)
//  (6) TS    : Thompson sampling bandit for strategy selection (ARQ vs IDE)
//  (7) LEVY  : Levy-flight (Mantegna) perturbations in quarantine
//  (8) OBL   : on-demand opposition-based basin escape when variance collapses
//  (9) Adaptive rtr_pool / agent_fraction scaling with current N
// ============================================================================

class ARQ3 : public Optimizer {
public:
    ARQ3() = default;
    ~ARQ3() override = default;

    std::string methodShortName() const override { return "arq3"; }
    std::string methodFullName()  const override {
        return "ARQ3: NLPSR + SHADE-H + LSHADE-RSP + jSO + Eigen-crossover + "
               "Thompson bandit + Levy quarantine + OBL basin-escape";
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
    // Population state
    // ------------------------------------------------------------
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    A_;          // external archive (JADE-style)

    // per-individual F/CR for IDE strategy (self-adaptive)
    std::vector<double> CBF_;
    std::vector<double> CBCR_;

    // ------------------------------------------------------------
    // NLPSR: non-linear population size reduction
    // ------------------------------------------------------------
    int    Ninit_{0};
    int    Nmin_{4};
    double nlpsr_alpha_{0.5};        // 1.0 = linear, <1 = slower shrink early
    int    pop_scale_{18};           // Ninit = pop_scale * D if not given

    // ------------------------------------------------------------
    // SHADE: circular memory for F/CR adaptation
    // ------------------------------------------------------------
    int    H_{6};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int    memK_{0};
    double MF_terminal_{0.9};        // jSO fixed terminal slot value
    double MCR_terminal_{0.9};

    // ARQ sampling bounds
    double Flo_{0.05};
    double Fhi_{1.40};

    // ------------------------------------------------------------
    // jSO-style schedules
    // ------------------------------------------------------------
    double pbest_max_{0.25};
    double pbest_min_{0.10};

    // ------------------------------------------------------------
    // LSHADE-RSP: rank-based selective pressure
    // ------------------------------------------------------------
    double kr_init_{2.0};            // selective pressure at start
    double kr_final_{3.0};           // and at end of budget

    // ------------------------------------------------------------
    // Eigen-coordinate crossover (EA4Eig / CoBiDE style)
    // ------------------------------------------------------------
    double p_eig_{0.40};
    int    eig_period_{10};
    double eig_frac_{0.50};
    Mat    B_rot_;                   // D x D eigenvectors (columns)
    bool   eig_valid_{false};
    int    iters_since_eig_{0};
    int    eig_min_D_{2};

    // ------------------------------------------------------------
    // RTR (Restricted Tournament Replacement)
    // ------------------------------------------------------------
    int    rtr_pool_{14};
    double rtr_pool_frac_{0.10};     // dynamic scaling cap

    // ------------------------------------------------------------
    // Archive
    // ------------------------------------------------------------
    double archive_rate_{1.5};

    // ------------------------------------------------------------
    // Thompson sampling bandit (replaces EA4Eig roulette)
    // ------------------------------------------------------------
    int                 h_{2};       // two heuristics: [0]=ARQ, [1]=IDE
    std::vector<double> bandit_a_;
    std::vector<double> bandit_b_;
    double              bandit_decay_{0.97};  // non-stationary forgetting
    int                 bootstrap_arq_iters_{2};
    int                 bootstrap_left_{0};

    // ------------------------------------------------------------
    // Quarantine (Levy flight Mantegna)
    // ------------------------------------------------------------
    double outlier_alpha_{1.0};
    double outlier_rho_{0.08};
    double levy_beta_{1.5};
    double qscale_{0.10};

    // ------------------------------------------------------------
    // On-demand OBL basin escape
    // ------------------------------------------------------------
    int    stag_trigger_{30};
    double var_collapse_ratio_{1e-3};
    int    obl_cooldown_{0};
    int    obl_cooldown_init_{80};
    double obl_frac_{0.30};
    double best_prev_{std::numeric_limits<double>::infinity()};
    int    no_improve_{0};

    // ------------------------------------------------------------
    // agent_fraction (how many parents to try per ARQ pass)
    // ------------------------------------------------------------
    double agent_fraction_{1.0};     // default: evaluate all parents

    // ------------------------------------------------------------
    // IDE scheduling (from EA4Eig IDE strategy)
    // ------------------------------------------------------------
    int    gmax_{0};
    double T_{0.0};
    int    g_{0};
    int    gt_{0};
    int    Tcurr_{0};
    int    ide_progress_sync_{1};
    int    ide_strict_improve_{1};

    // debug
    int    debug_{0};

    // ------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------
    inline double eval(const Vec& v) {
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    // utilities
    void   ensureBounds(Vec& v);
    int    pickDistinct(int n, int a = -1, int b = -1, int c = -1);
    int    randInt(int lo, int hi);
    double randU();
    double cauchy(double loc, double scale);
    double gaussN(double mu, double sig);
    double sampleLevy();                  // Mantegna 1994
    double progress01() const;

    // sorting / population management
    void   sortByFitness();
    int    targetPopulationSize() const;
    void   shrinkTo(int N);

    // rank-based r-index picking (ord is sorted-by-fitness order)
    double currentKR() const;
    double currentPbest() const;
    int    rankBasedPick(const std::vector<int>& ord, int forbid) const;

    // SHADE memory
    void   initMemory();
    void   sampleFCR(double& F, double& CR);
    void   updateMemoryFromSuccess(const std::vector<double>& SF,
                                   const std::vector<double>& SCR,
                                   const std::vector<double>& SG);

    // archive
    void   archivePush(const Vec& x);
    void   archiveTrim(int N);

    // eigen machinery
    void   jacobiEigen(const Mat& Ain, Mat& V, std::vector<double>& w) const;
    void   recomputeEigenBasis();
    void   applyBt(const Mat& B, const Vec& x, Vec& out) const;   // out = B^T x
    void   applyB (const Mat& B, const Vec& x, Vec& out) const;   // out = B   x
    void   eigenBinomialCrossover(int D, const Vec& base, const Vec& v,
                                  double CR, Vec& u);

    // Thompson bandit
    int    thompsonPick();
    void   banditDecay();
    void   banditRecord(int k, int successes, int attempts);

    // RTR selection
    double distBN(const Vec& a, const Vec& b) const;
    bool   selectionRTR(int parentIndex, const Vec& u, double fu,
                        double F, double CR,
                        std::vector<double>& SF,
                        std::vector<double>& SCR,
                        std::vector<double>& SG);

    // ARQ/IDE trial construction
    double computeK(double F) const;
    void   makeTrialARQ(int i, const std::vector<int>& ord,
                        double F, double CR, Vec& u);
    void   stepARQ();
    void   stepIDE();

    // quarantine and restart
    static double quantile(std::vector<double> v, double q01);
    void   quarantineLevy();
    void   oblBasinEscape();

    // IDE parameter re-seed / inherit
    void   sampleIDEParamsAt(int idx);
    void   inheritIDEParams(int dst, int src);

    // Population std-dev relative to box size (for OBL trigger)
    double normalizedPopSpread() const;
};

} // namespace optimsolution
