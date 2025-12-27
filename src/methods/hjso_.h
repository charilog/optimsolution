#pragma once

#include "optimizer.h"

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>

namespace optimsolution {

// Matches the definition in options.h.
struct MethodConfig;

class HJSO : public Optimizer {
public:
    HJSO() = default;
    ~HJSO() override = default;
	std::string methodShortName() const override { return "hjso"; }
	std::string methodFullName()  const override { return "HJSO: EA4eig hybrid shell with ARQ as default core"; }

    std::string name() const { return "hjso"; }

    // Reads global end_local_refine / end_local_method settings (as in PPSO).
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

    // --- population ---
    int pop_init_{100};
    int pop_min_{10};
    int N_{0};
    int N_init_run_{0};

    // --- roulette ---
    int    h_{4};
    int    n0_{2};
    double delta_{0.0};
    std::vector<double> ni_;
    std::vector<double> cni_;
    std::vector<int>    success_;
    int    nrst_{0};

    // --- population & fitness ---
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // --- cobide / jSO eigen-like controls ---
    double CBps_{0.5};
    double peig_{0.4};
    std::vector<double> CBF_;
    std::vector<double> CBCR_;
    bool   ceig_{false};

    // --- IDE scheduling ---
    int    gmax_{0};
    double T_{0.0};
    int    GT_{0};
    int    g_{0};
    int    gt_{0};
    int    Tcurr_{0};
    double SRT_{0.0};

    // --- CMA-ES parameters (simplified) ---
    double              sigma_{0.0};
    double              myeps_{1e-6};
    int                 mu_{0};
    std::vector<double> weights_;
    double              mueff_{0.0};
    double              cc_{0.0}, cs_{0.0}, c1_{0.0}, cmu_{0.0}, damps_{0.0};
    std::vector<double> pc_;
    std::vector<double> ps_;
    std::vector<double> B_;         // D x D, identity
    std::vector<double> diagD_;     // D, all 1
    std::vector<double> C_;         // D x D, identity
    std::vector<double> invsqrtC_;  // D x D, identity
    int                 eigeneval_{0};
    double              chiN_{0.0};
    std::vector<Vec>    oldPop_;

    // --- jSO / archive ---
    int                 Asize_max_{0};
    int                 Asize_{0};
    std::vector<Vec>    A_;
    int                 H_jso_{5};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int                 k_mem_{0};
    double              pmax_{0.25};
    double              pmin_{0.125};

    // --- in–run local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // --- final local refinement ---
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // === helpers ===
    inline double eval(const Vec& v) {
        double f = prob_ ? prob_->evaluate(v) : std::numeric_limits<double>::infinity();
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    void ensureBounds(Vec& x);
    int  randInt(int lo, int hi);
    double randU();
    double randN01();
    double cauchy(double loc, double scale);

    void sampleDistinct(int N, int k, std::vector<int>& out);
    void sampleDistinctExcluding(int N, int k,
                                 const std::vector<int>& exclude,
                                 std::vector<int>& out);

    std::pair<int,double> rouletteSelect() const;
    void sortByFitness();
    void shrinkPopulation(int newN);
    void addToArchive(const Vec& x);


    // === ARQ core (executed in default branch instead of JSO) ===
    // State
    std::vector<Vec>    ARQA_;   // ARQ archive (separate from jSO archive A_)
    double arq_best_prev_{std::numeric_limits<double>::infinity()};
    int    arq_no_improve_{0};

    // Parameters (ARQ: Adaptive RTR with Quarantine)
    double arq_pbest_{0.12};
    double arq_agent_fraction_{1.00};

    double arq_muF_{0.60};
    double arq_muCR_{0.85};

    double arq_Flo_{0.05};
    double arq_Fhi_{1.40};

    // quarantine / restart
    double arq_outlier_alpha_{1.0};
    double arq_outlier_rho_{0.08};
    double arq_qsigma_{0.10};

    double arq_worst_frac_{0.08};
    double arq_rsigma_{0.18};
    int    arq_stagnation_trigger_{24};

    // success-history smoothing
    double arq_shc_{0.10};

    // RTR pool size
    int    arq_rtr_pool_{14};

    // archive capacity = arq_archive_rate_ * N
    double arq_archive_rate_{1.5};

    int    debug_arq_{0};

    // ARQ helpers
    void   arq_archivePush(const Vec& x);
    void   arq_archiveTrim(int N);
    double arq_distBN(const Vec& a, const Vec& b) const;

    void   arq_sample_F_CR(double& F, double& CR,
                          std::cauchy_distribution<double>& cauchyF,
                          std::normal_distribution<double>& normCR);

    int    arq_pickDistinct(int n, int a=-1, int b=-1, int c=-1);
    static double arq_quantile(std::vector<double> v, double q01);
    void   arq_update_mu_from_success(const std::vector<double>& SF,
                                      const std::vector<double>& SCR,
                                      const std::vector<double>& SG);

    void   arq_makeTrial(int i, const std::vector<int>& ord, double F, double CR, Vec& u);

    bool   arq_selectionRTR(int parentIndex, const Vec& u, double fu,
                            double F, double CR,
                            std::vector<double>& SF,
                            std::vector<double>& SCR,
                            std::vector<double>& SG);

    void   arq_quarantine_and_restart();

    // sub-algorithms
    void stepCobide();
    void stepIDE();
    void stepCMAES();
    void stepJSO();
    void stepARQ();
};

} // namespace optimsolution
