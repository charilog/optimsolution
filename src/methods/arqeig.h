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

namespace optimsolution  {

// Matches the definition in options.h
struct MethodConfig;

class ARQEig : public Optimizer {
public:
    ARQEig() = default;
    ~ARQEig() override = default;

    std::string methodShortName() const override { return "arqeig"; }
    std::string methodFullName()  const override { return "ARQ with Eigen-like coordinate learning (ARQEig)"; }

    // Retrieves global end_local_refine / end_local_method (as in EA4Eig)
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

    // ---------------- population ----------------
    int pop_init_{100};   // initial N from settings
    int N_{0};            // current, may change dynamically

    std::vector<Vec>    X_;
    std::vector<double> FX_;

    // --- Adaptive Population Leaps (APL) ---
    // Goal: allow *large* multiplicative/level jumps in N based on online signals
    // (progress, success rate, diversity, stagnation), without requiring problem labels.
    bool   adaptive_population_{true};
    int    pop_min_{20};          // hard lower bound for N (>=4 enforced)
    int    pop_max_{200};         // hard upper bound for N
    int    pop_warmup_iters_{5};  // allow early "probe" decisions
    int    pop_check_interval_{3};
    int    pop_window_{12};       // best-history window length for progress signal

    double pop_success_thr_{0.08}; // if success rate below -> considered "ineffective N"
    double pop_impr_thr_{1e-4};    // relative improvement threshold over window
    double pop_div_low_{0.03};     // low diversity (BN-space RMS) -> may expand
    double pop_div_high_{0.15};    // very high diversity + no progress -> may shrink (optional)

    double pop_shrink_factor_{0.25}; // big jumps: e.g. 100 -> 25 (then clamped to pop_min=20)
    double pop_expand_factor_{2.0};  // big jumps: e.g. 20 -> 40 -> 80
    double pop_elite_frac_{0.25};    // when shrinking: keep elite fraction, rest sampled from best-half

    int    pop_cooldown_{6};      // iterations to wait after a resize (avoid oscillations)
    int    pop_cooldown_left_{0};

    int    iter_{0};
    std::deque<double> best_hist_;
    int    last_resize_iter_{-999999};

    // ---------------- ARQ / JADE-like controls ----------------
    int    H_{20};             // memory size
    double pbest_{0.11};       // fraction for pbest selection
    double Fmin_{0.10};
    double Fmax_{1.00};
    double archiverate_{1.0};  // archive size factor (cap = archiverate * N)

    std::vector<double> MF_;   // size H_
    std::vector<double> MCR_;  // size H_
    int                 k_mem_{0};

    // RTR / restart / robust quarantine
    int    rtr_k_{7};              // pool size for restricted tournament replacement
    double outlier_alpha_{1.5};    // IQR fence multiplier
    double outlier_rho_{0.20};     // fraction of outliers to attempt fix
    double qsigma_{0.10};          // quarantine gaussian scale (in normalized space)
    double worst_frac_{0.10};      // fraction of worst to micro-restart
    double rsigma_{0.20};          // restart gaussian scale (in normalized space)
    int    stagnationtrigger_{30}; // iterations without improvement

    int    no_improve_{0};
    double best_prev_{std::numeric_limits<double>::infinity()};

    // ---------------- Eig controls ----------------
    int    eiginterval_{5};    // recompute basis every k iterations
    double peig_{0.40};        // prob to use eig coordinates for trial generation
    double eig_eps_{1e-12};    // regularization for covariance

    int eig_age_{0};
    bool eig_ready_{false};

    // mean in normalized space (BN)
    Vec mean_bn_;

    // eigenbasis matrix B_ (row-major)
    // B_[i*D + j] = B(i,j)
    std::vector<double> B_;

    // ---------------- in-run local search (optional) ----------------
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // ---------------- final local refinement ----------------
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

private:
    // ---------------- helpers: rng & sampling ----------------
    int    randInt(int lo, int hi);
    double randU();
    double randN01();
    double cauchy(double loc, double scale);

    void sampleDistinctExcluding(int N, int k,
                                 const std::vector<int>& exclude,
                                 std::vector<int>& out);

    // ---------------- bounds + evaluation ----------------
    inline double eval(const Vec& v) {
        double f = prob_ ? prob_->evaluate(v) : std::numeric_limits<double>::infinity();
        if (!std::isfinite(f)) f = 1e100;
        return f;
    }

    void ensureBounds(Vec& x);

    // ---------------- archive ----------------
    std::vector<Vec> A_; // JADE-style external archive
    void addToArchive(const Vec& x);
    void trimArchive();

    // ---------------- stats ----------------
    void sortByFitness(std::vector<int>& idx) const;

    static double quantile(std::vector<double> v, double q01);

    // normalized (BN) transform using lb/ub
    void toBN(const Vec& x, Vec& y) const;
    void fromBN(const Vec& y, Vec& x) const;

    // ---------------- Eigen-basis (no Eigen library) ----------------
    void recomputeEigenBasis(const std::vector<int>& sorted_idx);
    void jacobiEigenSymmetric(std::vector<double>& A, int D,
                              std::vector<double>& V, std::vector<double>& evals,
                              int max_sweeps, double tol);

    // matrix ops with B (row-major)
    void matT_vec(const std::vector<double>& M, int D, const Vec& x, Vec& y) const; // y = M^T x
    void mat_vec(const std::vector<double>& M, int D, const Vec& x, Vec& y) const;  // y = M x

    // ---------------- trial generation ----------------
    void sample_F_CR(double& F, double& CR, double muF, double muCR);
    void makeTrialBase(int i, const std::vector<int>& sorted_idx, double F, double CR, Vec& u);
    void makeTrialEig (int i, const std::vector<int>& sorted_idx, double F, double CR, Vec& u);

    // selection
    bool selectionRTR(int parent, const Vec& u, double fu,
                      double F, double CR,
                      std::vector<double>& SF, std::vector<double>& SCR,
                      std::vector<double>& gains);

    double distBN(const Vec& a, const Vec& b) const;

    // adaptive update of memories
    void updateMemories(const std::vector<double>& SF,
                        const std::vector<double>& SCR,
                        const std::vector<double>& gains);

    // quarantine + restart
    void quarantineAndRestart();

    // ---------------- Adaptive Population Leaps (APL) ----------------
    double estimateDiversityBN(int sampleCount);
    double relImprovementFromHistory() const;
    void   maybeAdaptivePopulationLeap(const std::vector<int>& sorted_idx,
                                       double success_rate,
                                       double diversity_bn,
                                       double rel_impr);
    void   resizePopulation(int newN, const std::vector<int>& sorted_idx);
    void   injectNewIndividuals(int addCount);
};

} // namespace optimsolution
