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

class ARQ2 : public Optimizer {
public:
    ARQ2() = default;
    ~ARQ2() override = default;

    std::string methodShortName() const override { return "arq2"; }
    std::string methodFullName()  const override { return "ARQ2: ARQ/IDE roulette with Quarantine + ARQ-only micro-restart + jSO-style K"; }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override {}

private:
    using Vec = std::vector<double>;

    // --- core state ---
    std::vector<Vec>    X_;
    std::vector<double> FX_;
    std::vector<Vec>    A_;   // archive

    // ARQ-only stagnation tracking for micro-restart
    double best_prev_{std::numeric_limits<double>::infinity()};
    int    no_improve_{0};

    // --- ARQ parameters ---
    int    pop_override_{-1};

    double pbest_{0.12};
    double agent_fraction_{0.60};

    double muF_{0.60};
    double muCR_{0.85};

    // sampling bounds for ARQ
    double Flo_{0.05};
    double Fhi_{1.40};

    // quarantine
    double outlier_alpha_{1.0};
    double outlier_rho_{0.08};
    double qsigma_{0.10};

    // ARQ-only micro-restart
    double worst_frac_{0.08};
    double rsigma_{0.18};
    int    stagnation_trigger_{24};

    // success-history smoothing (ARQ only)
    double shc_{0.10};

    // RTR
    int rtr_pool_{14};

    // archive capacity = archive_rate * N
    double archive_rate_{1.5};

    // --- roulette (EA4Eig-like, two heuristics: ARQ and IDE) ---
    int    h_{2};
    int    n0_{2};
    double delta_{0.0};
    std::vector<double> ni_;
    std::vector<double> cni_;
    std::vector<int>    success_;
    int    nrst_{0};
    int    bootstrap_arq_iters_{2};
    int    bootstrap_left_{0};
    int    roulette_normalize_{1};
    int    ide_progress_sync_{1};
    int    ide_strict_improve_{1};

    // --- IDE scheduling / parameters copied from EA4Eig ---
    int    gmax_{0};
    double T_{0.0};
    int    g_{0};
    int    gt_{0};
    int    Tcurr_{0};
    std::vector<double> CBF_;
    std::vector<double> CBCR_;

    // debug
    int debug_arq_{0};

    // ─── safety constants ───────────────────────────────────────────────────
    // Maximum reflection bounces in ensureBounds before falling back to clamp.
    // Prevents infinite loops caused by very large overshoots.  (FIX #1)
    static constexpr int kMaxBounce    = 10;

    // Maximum Cauchy-retry attempts when sampling F for IDE parameters.
    // In practice the loop terminates in 1-2 tries; the cap is a safety net.
    static constexpr int kMaxCauchyTries = 20;

    // Maximum rejection-sampling attempts in pickDistinct.  (FIX #6)
    static constexpr int kMaxPickTries  = 100;

private:
    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    void ensureBounds(Vec& v);
    int  pickDistinct(int n, int a=-1, int b=-1, int c=-1);
    int  randInt(int lo, int hi);
    double randU();
    double cauchy(double loc, double scale);

    void sampleDistinctExcluding(int N, int k,
                                 const std::vector<int>& exclude,
                                 std::vector<int>& out);
    void sortByFitness();

    double distBN(const Vec& a, const Vec& b) const;
    static double quantile(std::vector<double> v, double q01);

    void archivePush(const Vec& x);
    void archiveTrim(int N);

    void sample_F_CR(double& F, double& CR,
                     std::cauchy_distribution<double>& cauchyF,
                     std::normal_distribution<double>& normCR);

    void update_mu_from_success(const std::vector<double>& SF,
                                const std::vector<double>& SCR,
                                const std::vector<double>& SG);

    std::pair<int,double> rouletteSelect() const;

    double progress01() const;
    int    ideGenerationFromProgress() const;
    void   resetIDEParamsAt(int idx);
    void   inheritIDEParams(int dst, int src);

    double computeK(double F) const;

    void makeTrialARQ(int i, const std::vector<int>& ord, double F, double CR, Vec& u);

    bool selectionRTR(int parentIndex, const Vec& u, double fu,
                      double F, double CR,
                      std::vector<double>& SF,
                      std::vector<double>& SCR,
                      std::vector<double>& SG);

    void stepARQ();
    void stepIDE();

    void quarantine();
    void microRestartARQ();
};

} // namespace optimsolution
