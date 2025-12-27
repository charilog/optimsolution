#pragma once

#include "optimizer.h"
#include "options.h"

#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <numeric>
#include <cmath>

// Χρησιμοποιούμε Eigen για eig(cov(...)) & CMA-ES
#include <Eigen/Dense>

namespace optimsolution {

struct MethodConfig;

class EA4Eig : public Optimizer {
public:
    EA4Eig() = default;
    ~EA4Eig() override = default;

    std::string methodShortName() const override { return "ea4eig"; }
    std::string methodFullName() const override {
        return "Evolutionary Algorithms with Eigen Crossover (EA4Eig)";
    }

    // από το global config (όπως στα άλλα methods)
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

    // ===== population & bounds =====
    int  N_{0};          // current population size
    int  N_init_{100};   // initial population
    int  N_min_{10};     // minimum population
    Vec  lb_;
    Vec  ub_;
    std::vector<Vec>    X_;   // population
    std::vector<double> FX_;  // fitness

    // ===== roulette (4 heuristics) =====
    static constexpr int H_ = 4;
    int    n0_{2};
    double delta_{0.0};
    std::vector<double> ni_;      // "credits"
    std::vector<double> cni_;
    std::vector<int>    success_;
    int nrst_{0};

    // ===== CoBiDE / Eigen crossover =====
    double CBps_{0.5};   // ποσοστό elite για eig
    double peig_{0.4};   // πιθανότητα να χρησιμοποιηθεί eig
    std::vector<double> CBF_;
    std::vector<double> CBCR_;

    // ===== IDE scheduling =====
    int    gmax_{0};
    double T_{0.0};
    int    GT_{0};
    int    gt_{0};
    int    g_{0};
    int    Tcurr_{0};

    // ===== CMA-ES state =====
    double sigma_{0.0};
    double myeps_{1e-15};
    int    mu_{0};
    std::vector<double> weights_;
    double mueff_{0.0};
    double cc_{0.0}, cs_{0.0}, c1_{0.0}, cmu_{0.0}, damps_{0.0};
    Eigen::VectorXd pc_;
    Eigen::VectorXd ps_;
    Eigen::MatrixXd B_;
    Eigen::VectorXd D_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd invsqrtC_;
    long long eigeneval_{0};
    double chiN_{0.0};
    Eigen::MatrixXd oldPop_;

    // ===== jSO / archive =====
    int              Asize_max_{0};
    int              Asize_{0};
    std::vector<Vec> A_;
    int              Hjso_{5};
    std::vector<double> MF_;
    std::vector<double> MCR_;
    int              k_mem_{0};
    double           pmax_{0.25};
    double           pmin_{0.125};

    // ===== local search (σου, default OFF) =====
    std::string local_method_{"lbfgs"};
    double      local_rate_{0.0}; // 0 => disabled

    bool        end_local_refine_{false};
    std::string end_local_method_{"lbfgs"};

    // ===== helpers =====
    inline double eval(const Vec& x) {
        return prob_ ? prob_->evaluate(x)
                     : std::numeric_limits<double>::infinity();
    }

    void ensureBounds(Vec& x);
    double randU();
    double randN01();
    int    randInt(int lo, int hi);
    double cauchyRnd(double x0, double gamma);

    void sampleDistinct(int N, int k, std::vector<int>& out);
    void sampleDistinctExcluding(int N, int k,
                                 const std::vector<int>& exclude,
                                 std::vector<int>& out);

    std::pair<int, double> rouletteSelect() const;

    void sortByFitness();
    void shrinkPopulationTo(int newN);
    void addToArchive(const Vec& x);

    bool computeEigenVectorsFromElite(const std::vector<int>& eliteIdx,
                                      Eigen::MatrixXd& eigVecs);

    // 4 heuristics
    void stepCoBiDE();
    void stepIDE();
    void stepCMAES();
    void stepJSO();
};

} // namespace optimsolution
