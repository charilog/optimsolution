#pragma once
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "problem.h"

namespace optimsolution {

// Result of a multi-objective run: the final non-dominated (Pareto) set.
// paretoX[i] is the decision vector for the i-th Pareto point, paretoF[i] is
// its objective vector (size == problem->numObjectives()).
struct MOOResult {
    std::vector<Vec> paretoX;
    std::vector<Vec> paretoF;
    long long evals = 0;
    long long generations = 0;
};

// Minimal, self-contained base class for multi-objective (Pareto-front)
// optimizers. Deliberately NOT derived from the single-objective Optimizer
// class in optimizer.h -- the two have different result shapes (one best
// point vs. a Pareto set) and different stopping/bookkeeping needs, so
// keeping them separate avoids any risk of disturbing the existing
// single-objective run() machinery used by Single/Batch/Sensitivity modes.
class MOOOptimizer {
public:
    virtual ~MOOOptimizer() = default;

    void setProblem(Problem* p) { prob_ = p; }
    void setSeed(unsigned long long seed) { rng_.seed(seed); }
    void setPopulationSize(int n) { population_ = (n > 1 ? n : 2); }
    void setGenerations(int g) { generations_ = (g > 0 ? g : 1); }

    // Method-specific parameters (e.g. "crossover_prob", "mutation_prob"),
    // set from the CLI/config the same way single-objective methods read
    // their parameters -- kept as a simple string map to avoid a parallel
    // options/config subsystem just for this.
    void setParam(const std::string& name, double value) { params_[name] = value; }
    double param(const std::string& name, double def) const {
        auto it = params_.find(name);
        return (it != params_.end()) ? it->second : def;
    }

    int populationSize() const { return population_; }
    int generations() const { return generations_; }

    virtual std::string name() const = 0;
    virtual MOOResult run() = 0;

protected:
    Problem* prob_ = nullptr;
    std::mt19937_64 rng_{std::random_device{}()};
    int population_ = 100;
    int generations_ = 200;
    std::unordered_map<std::string, double> params_;
};

} // namespace optimsolution
