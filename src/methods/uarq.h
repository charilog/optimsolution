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

namespace optimsolution {

class UARQ : public Optimizer {
public:
    UARQ() = default;
    ~UARQ() override = default;

    std::string methodShortName() const override { return "uarq"; }
    std::string methodFullName()  const override { return "uARQ: Adaptive RTR with Quarantine + LPSR"; }

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
    double best_prev_{std::numeric_limits<double>::infinity()};
    int    no_improve_{0};

    // --- population management (LPSR) ---
    int pop_override_{-1};     // initial population override from config
    int pop_init_{-1};         // actual initial population used (set in init)
    int pop_min_{4};           // minimum population for linear reduction

    // --- uARQ parameters (paper-like defaults) ---
    double pbest_{0.12};
    double agent_fraction_{0.60};

    double muF_{0.60};
    double muCR_{0.85};

    // sampling bounds
    double Flo_{0.05};
    double Fhi_{1.40};

    // quarantine
    double outlier_alpha_{1.0};
    double outlier_rho_{0.08};
    double qsigma_{0.10};

    // micro-restart
    double worst_frac_{0.08};
    double rsigma_{0.18};
    int    stagnation_trigger_{24};

    // SH smoothing
    double shc_{0.10};

    // RTR
    int rtr_pool_{14};

    // archive capacity = archive_rate * N
    double archive_rate_{1.5};

    // debug
    int debug_arq_{0};

private:
    // --- helpers ---
    inline double eval(const Vec& v){
        double f = prob_->evaluate(v);
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    void ensureBounds(Vec& v);
    int  pickDistinct(int n, int a=-1, int b=-1, int c=-1);

    // bounds-normalized distance
    double distBN(const Vec& a, const Vec& b) const;

    // robust quantiles
    static double quantile(std::vector<double> v, double q01);

    // archive
    void archivePush(const Vec& x);
    void archiveTrim(int N);

    // parameter policy
    void sample_F_CR(double& F, double& CR,
                     std::cauchy_distribution<double>& cauchyF,
                     std::normal_distribution<double>& normCR);

    void update_mu_from_success(const std::vector<double>& SF,
                                const std::vector<double>& SCR,
                                const std::vector<double>& SG);

    // trial generation: pbest/1/bin with archive support
    void makeTrial(int i, const std::vector<int>& ord, double F, double CR, Vec& u);

    // selection RTR (parent first, else nearest from pool by BN-distance)
    bool selectionRTR(int parentIndex, const Vec& u, double fu,
                      double F, double CR,
                      std::vector<double>& SF,
                      std::vector<double>& SCR,
                      std::vector<double>& SG);

    // maintenance
    void quarantine_and_restart();

    // LPSR
    void applyLPSR();
    void reducePopulationTo(int newN);
};

} // namespace optimsolution
