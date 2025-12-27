#pragma once

#include "optimizer.h"

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>

namespace optimsolution {

// Matches the definition in options.h
struct MethodConfig;

class EA4Eig : public Optimizer {
public:
    EA4Eig() = default;
    ~EA4Eig() override = default;
	std::string methodShortName() const override { return "ea4eig"; }
	std::string methodFullName()  const override { return "Evolutionary Algorithms with Eigen crossover (EA4eig)"; }

    std::string name() const { return "ea4eig"; }

    // Reads the global end_local_refine / end_local_method (as in PPSO)
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
        return prob_ ? prob_->evaluate(v) : std::numeric_limits<double>::infinity();
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

    // sub-algorithms
    void stepCobide();
    void stepIDE();
    void stepCMAES();
    void stepJSO();
};

} // namespace optimsolution
