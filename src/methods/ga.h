#pragma once
#include "optimizer.h"
#include <vector>
#include <string>
#include <random>
#include <limits>

namespace optimsolution {

// Genetic Algorithm (GA) - ported from Genetic
//
// Adaptive extensions (each independently enable/disable):
//
//   Adaptive Mutation  (adaptive_mutation_enable):
//     mutation_rate decreases linearly with FE-progress from
//     adaptive_mutation_max  →  adaptive_mutation_min
//     (high exploration early, full exploitation at the end)
//
//   Adaptive Crossover (adaptive_crossover_enable):
//     crossover_rate increases linearly with FE-progress from
//     adaptive_crossover_min  →  adaptive_crossover_max
//     (gentle mixing early, maximum recombination pressure at the end)
//
//   Together they follow the JSO-style schedule:
//     early  : high mutation  + low  crossover  (exploration)
//     middle : medium both
//     late   : low  mutation  + high crossover  (full exploitation attack)
//
//   When both are disabled the algorithm is identical to the base GA.

class GA : public Optimizer {
public:
    GA() = default;
    ~GA() override = default;

    std::string methodShortName() const override { return "ga"; }
    std::string methodFullName()  const override { return "Genetic Algorithm (GA)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override {
        int pop_override = mc.getInt("population", pop_);
        if (pop_override > 0) pop_ = pop_override;
        chromosome_count_ = pop_;

        max_generations_ = mc.getInt("max_generations", max_generations_);
        if (max_generations_ < 1) max_generations_ = 200;

        selection_rate_ = mc.getDbl("selection_rate", selection_rate_);
        if (selection_rate_ < 0.0) selection_rate_ = 0.0;
        if (selection_rate_ > 1.0) selection_rate_ = 1.0;

        mutation_rate_ = mc.getDbl("mutation_rate", mutation_rate_);
        if (mutation_rate_ < 0.0) mutation_rate_ = 0.0;
        if (mutation_rate_ > 1.0) mutation_rate_ = 1.0;

        crossover_rate_ = mc.getDbl("crossover_rate", crossover_rate_);
        if (crossover_rate_ < 0.0) crossover_rate_ = 0.0;
        if (crossover_rate_ > 1.0) crossover_rate_ = 1.0;

        local_rate_ = mc.getDbl("local_rate", local_rate_);
        if (local_rate_ < 0.0) local_rate_ = 0.0;
        if (local_rate_ > 1.0) local_rate_ = 1.0;

        tournament_size_ = mc.getInt("tournament_size", tournament_size_);
        if (tournament_size_ < 2) tournament_size_ = 2;

        lsearch_items_ = mc.getInt("lsearch_items", lsearch_items_);
        if (lsearch_items_ < 0) lsearch_items_ = 0;

        lsearch_gens_ = mc.getInt("lsearch_gens", lsearch_gens_);
        if (lsearch_gens_ < 1) lsearch_gens_ = 20;

        selection_method_  = mc.getStr("selection_method",  selection_method_);
        crossover_method_  = mc.getStr("crossover_method",  crossover_method_);
        mutation_method_   = mc.getStr("mutation_method",   mutation_method_);
        lsearch_method_    = mc.getStr("lsearch_method",    lsearch_method_);
        local_method_      = mc.getStr("local_method",      local_method_);

        for (auto& c : selection_method_) c = (char)std::tolower((unsigned char)c);
        for (auto& c : crossover_method_) c = (char)std::tolower((unsigned char)c);
        for (auto& c : mutation_method_)  c = (char)std::tolower((unsigned char)c);
        for (auto& c : lsearch_method_)   c = (char)std::tolower((unsigned char)c);
        for (auto& c : local_method_)     c = (char)std::tolower((unsigned char)c);

        // ── Adaptive mutation ──────────────────────────────────────────────────
        // rate decreases linearly: max_mut → min_mut  as  FEs: 0 → max_evals
        adaptive_mutation_enable_ = (mc.getInt("adaptive_mutation_enable", 0) != 0);
        adaptive_mutation_min_    = mc.getDbl("adaptive_mutation_min", adaptive_mutation_min_);
        adaptive_mutation_max_    = mc.getDbl("adaptive_mutation_max", adaptive_mutation_max_);
        if (adaptive_mutation_min_ < 0.0) adaptive_mutation_min_ = 0.0;
        if (adaptive_mutation_max_ > 1.0) adaptive_mutation_max_ = 1.0;
        if (adaptive_mutation_min_ > adaptive_mutation_max_)
            std::swap(adaptive_mutation_min_, adaptive_mutation_max_);

        // ── Adaptive crossover ─────────────────────────────────────────────────
        // rate increases linearly: min_cr → max_cr  as  FEs: 0 → max_evals
        adaptive_crossover_enable_ = (mc.getInt("adaptive_crossover_enable", 0) != 0);
        adaptive_crossover_min_    = mc.getDbl("adaptive_crossover_min", adaptive_crossover_min_);
        adaptive_crossover_max_    = mc.getDbl("adaptive_crossover_max", adaptive_crossover_max_);
        if (adaptive_crossover_min_ < 0.0) adaptive_crossover_min_ = 0.0;
        if (adaptive_crossover_max_ > 1.0) adaptive_crossover_max_ = 1.0;
        if (adaptive_crossover_min_ > adaptive_crossover_max_)
            std::swap(adaptive_crossover_min_, adaptive_crossover_max_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    double eval(const std::vector<double>& x) { return prob_->evaluate(x); }
    bool   inBounds(const std::vector<double>& x);

    // GA phases
    void calcFitnessArray();
    void selectionSort();
    void crossoverPhase();
    void mutatePhase();

    // FE-based adaptive parameter update — called once per iteration
    void updateAdaptive();

    // Internal local search
    void localSearchAt(int pos);
    void localCrossover(int pos);
    void localMutate(int pos);
    void localSiman(int pos);
    void localDE(int pos);

    // Roulette
    struct RouletteEntry {
        std::vector<double> x;
        double weight{0.0};
    };
    std::vector<RouletteEntry> makeChromosomesForRoulette();
    int selectWithRoulette(const std::vector<RouletteEntry>& roulette);
    int selectWithTournament();

    // Crossover operators
    void makeChildrenUniform(
        const std::vector<double>& p1, const std::vector<double>& p2,
        std::vector<double>& c1, std::vector<double>& c2);
    void makeChildrenOnePoint(
        const std::vector<double>& p1, const std::vector<double>& p2,
        std::vector<double>& c1, std::vector<double>& c2);
    void makeChildrenDouble(
        const std::vector<double>& p1, const std::vector<double>& p2,
        std::vector<double>& c1, std::vector<double>& c2);

    double deltaIter(double y);

private:
    std::vector<std::vector<double>> population_;
    std::vector<double>              fitness_;
    std::vector<std::vector<double>> children_array_;

    // ── Base params ────────────────────────────────────────────────────────────
    int         chromosome_count_{200};
    int         max_generations_{200};    // used in deltaIter only
    double      selection_rate_{0.10};
    double      mutation_rate_{0.05};     // live value (updated each iteration when adaptive on)
    double      crossover_rate_{1.0};     // live value (updated each iteration when adaptive on)
    double      local_rate_{0.005};
    int         tournament_size_{8};
    int         lsearch_items_{20};
    int         lsearch_gens_{20};
    std::string selection_method_{"roulette"};
    std::string crossover_method_{"double"};
    std::string mutation_method_{"double"};
    std::string lsearch_method_{"none"};
    std::string local_method_{"lbfgs"};

    int generation_{0};

    // ── Adaptive mutation ──────────────────────────────────────────────────────
    bool   adaptive_mutation_enable_{false};
    double adaptive_mutation_min_{0.06};
    double adaptive_mutation_max_{0.24};

    // ── Adaptive crossover ─────────────────────────────────────────────────────
    bool   adaptive_crossover_enable_{false};
    double adaptive_crossover_min_{0.68};
    double adaptive_crossover_max_{0.98};

    // ── Final local refinement ─────────────────────────────────────────────────
    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
