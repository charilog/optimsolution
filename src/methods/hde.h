#pragma once

#include "optimizer.h"

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>

namespace optimsolution  {

// Matches the definition in options.h
struct MethodConfig;

class HDE : public Optimizer {
public:
    HDE() = default;
    ~HDE() override = default;

    std::string methodShortName() const override { return "hde"; }
    std::string methodFullName()  const override { return "Hybrid Differential Evolution (HDE)"; }

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
    int    h_{6};
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
    std::vector<Vec>    oldPop_;    // Previous CMA sample batch (index-aligned to CMA sampling, not individuals)

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

    // --- ARQ mechanisms (case operator only) ---
    std::vector<Vec>    A_arq_;
    double              arq_pbest_{0.12};
    double              arq_agent_fraction_{0.60};
    double              arq_muF_{0.60};
    double              arq_muCR_{0.85};
    double              arq_Flo_{0.05};
    double              arq_Fhi_{1.40};
    int                 arq_rtr_pool_{14};
    double              arq_archive_rate_{1.5};
    double              arq_shc_{0.10};

    // --- BHO mechanisms (case operator only) ---
    int                 bho_iters_{0};
    int                 bho_sinceBest_{0};
    double              bho_heal_prob_{0.65};
    double              bho_heal_rate_{0.30};
    double              bho_wound_strength_init_{0.40};
    int                 bho_stagnation_kick_{20};
    int                 bho_stagnation_restart_{300};
    double              bho_elite_kick_sigma_{0.01};
    double              bho_restart_frac_{0.25};

    // --- in-run local search ---
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0};

    // --- final local refinement ---
    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // === helpers ===
    inline double eval(const Vec& v) {
        if (!prob_) return std::numeric_limits<double>::infinity();
        double f = prob_->evaluate(v);
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

    // Sorting and LPSR shrink (keeps per-individual parameters aligned).
    // Note: oldPop_ is not sorted because it is CMA-sample-index aligned.
    void sortByFitness();
    void shrinkPopulation(int newN);
    void addToArchive(const Vec& x);
// sub-algorithms / operators
    void stepCobide(); // case 0
    void stepIDE();    // case 1
    void stepCMAES();  // case 2
    void stepJSO();    // case 3 and default
};

} // namespace optimsolution
