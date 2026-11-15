#pragma once
#include "optimizer.h"
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>

namespace optimsolution {

/**
 * PEAO -- ported from a standalone OPTIMUS-framework implementation to
 * optimsolution's Optimizer interface.
 *
 * Population-based method with K equal-sized (last one absorbs any
 * remainder) subpopulations. Each individual generates three trial
 * candidates per iteration -- a "sine/best-gap" candidate (Eq. 2), and two
 * Levy-flight-driven candidates using a random difference vector (Eq. 4
 * vector-Levy and Eq. 7 scalar-Levy) -- and keeps the best of the three if
 * it improves on itself (a per-individual greedy hill-climb, NOT a
 * population-wide selection). An "adaptive factor" AF(t) = sqrt(t/T) grows
 * from 0 to 1 over the run and scales how strongly candidates are pulled
 * toward the current global best. Periodically, subpopulations optionally
 * exchange individuals via "propagation" (elite-for-worst swaps between
 * two subpopulations, several topologies) and/or "communication" (the
 * worst individual of every subpopulation is replaced, via one of three
 * strategies: the global best, the mean of all subpopulations' bests, or a
 * circular hand-off of best positions between neighbouring subpopulations).
 *
 * *** Adaptations made for optimsolution compatibility ***
 * (the original source is written against a different framework, OPTIMUS,
 * with a different Optimizer base class and no CLI/GUI of its own)
 *
 *   - Optimizer lifecycle: OPTIMUS's init()/step()/terminated()/done() maps
 *     onto optimsolution's init()/one_iteration()/end(); termination is now
 *     governed entirely by optimsolution's own StopController/max_evals_
 *     convention (via updateStop()), NOT by the original per-subcluster
 *     "doublebox"/"similarity" termination objects (OPTIMUS-specific types
 *     this codebase does not have) or by PEAO's own maxIterations counter
 *     directly stopping the run. maxIterations_ is kept purely as the
 *     denominator of the adaptive factor AF(t) = sqrt(t/T) -- set it to
 *     roughly max_evals/(3*population) if you want AF(t) to reach 1.0 at
 *     the point your evaluation budget is actually exhausted (each
 *     iteration spends 3 evaluations per individual).
 *   - RNG: all randomness (candidate generation, Levy sampling, strategy
 *     selection, propagation/communication partner choice) now goes
 *     through the base class's seeded rng_, instead of unseeded rand() and
 *     a fresh std::random_device-seeded generator per call -- needed so a
 *     PEAO run is reproducible given the same --seed, like every other
 *     method in this codebase.
 *   - Bound handling: the original algorithm never clamps a generated
 *     candidate back into [lb, ub]. optimsolution's Problem implementations
 *     generally assume in-bounds input, so every candidate is clamped
 *     (component-wise, with a midpoint fallback for non-finite values)
 *     before evaluation -- an added safety step, not present upstream.
 *   - No internal OpenMP: the original parallelizes across subpopulations
 *     with #pragma omp parallel for and mutates shared state (including
 *     inside propagate(), OUTSIDE any #pragma omp critical section) from
 *     multiple threads at once. optimsolution already offers its own
 *     independent-runs OpenMP parallelism (see [global] parallel_runs);
 *     giving every method its own additional internal parallelism on top
 *     of that would oversubscribe threads and isn't safe for every
 *     Problem implementation in this codebase (some have non-thread-safe
 *     global state). This port is fully sequential.
 *   - Local search: the original's per-individual local search only
 *     overwrites fitnessArray[i], never samples[i] itself -- leaving the
 *     recorded position and fitness inconsistent. This port instead uses
 *     optimsolution's own local_rate/local_method hook (the same
 *     convention used by every other method here), which updates both the
 *     position and the fitness together.
 *   - subPopEnable is read from configuration for interface fidelity but,
 *     exactly as in the given source, is not used by any other logic --
 *     it is a currently-inert parameter there too, not something dropped
 *     during porting.
 *   - Two genuine bugs found and fixed relative to the given source (see
 *     the comments at createCandidate1() and the propagate() call site in
 *     peao.cpp for the full reasoning): candidate1 (Eq. 2) was missing its
 *     `current[j] +` base term, making it a raw difference vector rather
 *     than a position (almost always clamped to a near-random corner,
 *     wasting roughly a third of the evaluation budget every iteration);
 *     and propagate() was fired once PER SUBPOPULATION whenever triggered
 *     instead of once per triggering iteration, so e.g. propagationMethod
 *     = NtoN (already an all-pairs exchange per call) would repeat that
 *     full exchange subPopulation_ times back to back and rapidly
 *     homogenize the population.
 *   - end_local_refine/end_local_method can now also be set directly in
 *     this method's own [peao] section (in addition to the existing
 *     global mechanism, setEndLocalFromGlobal()), since configure() is
 *     called after that global setup and would otherwise silently ignore
 *     a per-method override placed in [peao].
 */
class PEAO : public Optimizer {
public:
    PEAO() = default;
    ~PEAO() override = default;
    std::string methodShortName() const override { return "peao"; }
    std::string methodFullName()  const override { return "PEAO (Propagation-Enzyme Adaptive Optimizer)"; }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        end_local_refine_ = enable;
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override {
        // population=0 (or unset): internal method, no external published
        // literature default; N=100 is this codebase's general convention.
        // Previously population=0 silently fell back to [global]
        // population (default 100) -- fixed here.
        int p = mc.getInt("population", 0);
        pop_ = (p > 3) ? p : 100;
        this->setPopulation(pop_);

        maxIterations_ = mc.getInt("maxiters", maxIterations_);
        if (maxIterations_ < 1) maxIterations_ = 1;

        subPopulation_ = mc.getInt("subPopulation", subPopulation_);
        if (subPopulation_ < 1) subPopulation_ = 1;
        subPopEnable_ = mc.getInt("subPopEnable", subPopEnable_); // see class comment: inert, kept for fidelity

        // Local search: original PEAO parameter name is "localsearchRate";
        // "local_rate" is accepted as an alias for consistency with every
        // other method's config key in this codebase. Only one rate is
        // actually used (local_rate_) -- there is no separate, unused
        // "localsearchRate_" member.
        local_method_ = mc.getStr("local_method", local_method_);
        for (auto& c : local_method_) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        double lr = mc.getDbl("localsearchRate", local_rate_);
        lr = mc.getDbl("local_rate", lr);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;
        local_rate_ = lr;

        propEnable_ = mc.getInt("prop", propEnable_ ? 1 : 0) != 0;
        propagationRate_ = mc.getInt("propagationRate", propagationRate_);
        if (propagationRate_ < 1) propagationRate_ = 1;
        propagationNumber_ = mc.getInt("propagationNumber", propagationNumber_);
        if (propagationNumber_ < 0) propagationNumber_ = 0;
        propagationMethod_ = mc.getStr("propagationMethod", propagationMethod_);

        communicationThreshold_ = mc.getInt("communicationThreshold", communicationThreshold_);
        if (communicationThreshold_ < 0) communicationThreshold_ = 0;
        enzymeConcentration_ = mc.getDbl("enzymeConcentration", enzymeConcentration_);
        levyBetaTimes10_ = mc.getInt("levyBetaTimes10", levyBetaTimes10_);
        communicationStrategyMode_ = mc.getInt("communicationStrategyMode", communicationStrategyMode_);

        // Per-method override for the end-of-run local refinement, in
        // addition to the existing global mechanism (setEndLocalFromGlobal()).
        // configure() runs after that global setup in the normal flow, so
        // without reading these here, a value placed directly in [peao]
        // would be silently ignored.
        const int endRefine = mc.getInt("end_local_refine", end_local_refine_ ? 1 : 0);
        end_local_refine_ = (endRefine != 0);
        end_local_method_ = mc.getStr("end_local_method", end_local_method_);
    }

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    double eval(const Vec& x) { return prob_->evaluate(x); }
    void clampToBounds(Vec& x) const;

    double adaptiveFactor() const;
    double levyBeta() const { return double(levyBetaTimes10_) / 10.0; }
    double levySigma(double beta) const;
    double levyScalar(double beta);
    Vec    levyVector(int dim, double beta);

    void updateLeadersForSubpop(int subPopIndex);
    void updateAllLeaders();

    Vec createCandidate1(const Vec& current, const Vec& gbest, double AFt);
    Vec buildDifferenceVector();
    Vec createCandidate2A(const Vec& current, const Vec& gbest, const Vec& diff, double AFt);
    Vec createCandidate2B(const Vec& current, const Vec& gbest, const Vec& diff, double AFt);

    void communicate();
    void communicationStrategy1();
    void communicationStrategy2();
    void communicationStrategy3();

    void propagate();
    void propagateBetween(int sender, int receiver);

private:
    std::vector<Vec>    X_;
    std::vector<double> FX_;

    int dimension_{0};
    int subSize_{0};

    std::vector<double> bestF_, worstF_, bestFOld_;
    std::vector<int>    bestIdx_, worstIdx_;
    std::vector<Vec>    bestX_, worstX_;

    Vec    gBestX_;
    double gBestF_{0.0};

    // ---- Parameters (defaults match the original OPTIMUS PEAO source) ----
    int    maxIterations_{100};
    int    subPopulation_{10};
    int    subPopEnable_{1};        // inert, see class comment
    bool   propEnable_{false};
    std::string propagationMethod_{"1to1"};
    int    propagationNumber_{0};
    int    propagationRate_{10};
    int    communicationThreshold_{10};
    double enzymeConcentration_{0.1};
    int    levyBetaTimes10_{15};
    int    communicationStrategyMode_{0}; // 0=random, 1/2/3=fixed strategy

    std::string local_method_{"none"};
    double      local_rate_{0.01}; // original PEAO default for "localsearchRate"

    bool        end_local_refine_{false};
    std::string end_local_method_{};
};

} // namespace optimsolution
