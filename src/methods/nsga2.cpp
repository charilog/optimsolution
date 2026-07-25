#include "nsga2.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace optimsolution {

namespace {

struct Individual {
    Vec x;
    Vec f;
    int rank = 0;
    double crowd = 0.0;
};

// Pareto dominance under minimization: a dominates b iff a is no worse than b
// in every objective and strictly better in at least one.
bool dominates(const Vec& a, const Vec& b) {
    bool anyBetter = false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] > b[i]) return false;
        if (a[i] < b[i]) anyBetter = true;
    }
    return anyBetter;
}

std::vector<std::vector<int>> fastNonDominatedSort(std::vector<Individual>& pop) {
    const int n = static_cast<int>(pop.size());
    std::vector<std::vector<int>> dominatedBy(n);
    std::vector<int> dominationCount(n, 0);
    std::vector<std::vector<int>> fronts;

    std::vector<int> front0;
    for (int p = 0; p < n; ++p) {
        for (int q = 0; q < n; ++q) {
            if (p == q) continue;
            if (dominates(pop[p].f, pop[q].f)) {
                dominatedBy[p].push_back(q);
            } else if (dominates(pop[q].f, pop[p].f)) {
                ++dominationCount[p];
            }
        }
        if (dominationCount[p] == 0) {
            pop[p].rank = 0;
            front0.push_back(p);
        }
    }
    fronts.push_back(front0);

    int fi = 0;
    while (fi < static_cast<int>(fronts.size()) && !fronts[fi].empty()) {
        std::vector<int> next;
        for (int p : fronts[fi]) {
            for (int q : dominatedBy[p]) {
                if (--dominationCount[q] == 0) {
                    pop[q].rank = fi + 1;
                    next.push_back(q);
                }
            }
        }
        ++fi;
        if (!next.empty()) fronts.push_back(next);
    }
    return fronts;
}

void assignCrowdingDistance(std::vector<Individual>& pop, const std::vector<int>& front) {
    for (int idx : front) pop[idx].crowd = 0.0;
    const int n = static_cast<int>(front.size());
    if (n == 0) return;
    const int m = static_cast<int>(pop[front[0]].f.size());

    for (int k = 0; k < m; ++k) {
        std::vector<int> sorted = front;
        std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
            return pop[a].f[k] < pop[b].f[k];
        });
        pop[sorted.front()].crowd = std::numeric_limits<double>::infinity();
        pop[sorted.back()].crowd  = std::numeric_limits<double>::infinity();

        const double fmin = pop[sorted.front()].f[k];
        const double fmax = pop[sorted.back()].f[k];
        const double range = fmax - fmin;
        if (range <= 1e-300) continue;

        for (int i = 1; i < n - 1; ++i) {
            if (!std::isfinite(pop[sorted[i]].crowd)) continue;
            pop[sorted[i]].crowd += (pop[sorted[i + 1]].f[k] - pop[sorted[i - 1]].f[k]) / range;
        }
    }
}

// NSGA-II selection order: lower rank wins; ties broken by higher crowding distance.
bool crowdedBetter(const Individual& a, const Individual& b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    return a.crowd > b.crowd;
}

} // namespace

MOOResult NSGA2::run() {
    MOOResult result;
    if (!prob_) return result;

    const int dim = prob_->dimension();
    const Vec& lb = prob_->lb();
    const Vec& ub = prob_->ub();
    const int N = population_;
    const int G = generations_;

    const double pc   = param("crossover_prob", 0.9);
    const double pm   = param("mutation_prob", dim > 0 ? (1.0 / double(dim)) : 0.1);
    const double etaC = param("eta_c", 15.0);
    const double etaM = param("eta_m", 20.0);

    std::uniform_real_distribution<double> u01(0.0, 1.0);

    auto randomIndividual = [&]() {
        Individual ind;
        ind.x.resize(dim);
        for (int i = 0; i < dim; ++i) {
            std::uniform_real_distribution<double> ud(lb[i], ub[i]);
            ind.x[i] = ud(rng_);
        }
        ind.f = prob_->evaluateMulti(ind.x);
        return ind;
    };

    auto clampToBounds = [&](Vec& x) {
        for (int i = 0; i < dim; ++i) {
            if (x[i] < lb[i]) x[i] = lb[i];
            if (x[i] > ub[i]) x[i] = ub[i];
        }
    };

    // Simulated Binary Crossover (SBX).
    auto sbxCrossover = [&](const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
        c1 = p1;
        c2 = p2;
        if (u01(rng_) > pc) return;

        for (int i = 0; i < dim; ++i) {
            if (u01(rng_) > 0.5) continue; // per-variable crossover switch
            const double x1 = p1[i], x2 = p2[i];
            if (std::abs(x1 - x2) < 1e-14) continue;

            const double lo = lb[i], hi = ub[i];
            const double xl = std::min(x1, x2);
            const double xu = std::max(x1, x2);
            const double rnd = u01(rng_);

            auto betaqFor = [&](double beta) {
                const double alpha = 2.0 - std::pow(beta, -(etaC + 1.0));
                if (rnd <= 1.0 / alpha) {
                    return std::pow(rnd * alpha, 1.0 / (etaC + 1.0));
                }
                return std::pow(1.0 / (2.0 - rnd * alpha), 1.0 / (etaC + 1.0));
            };

            const double beta1  = 1.0 + (2.0 * (xl - lo) / std::max(1e-300, xu - xl));
            const double beta2  = 1.0 + (2.0 * (hi - xu) / std::max(1e-300, xu - xl));
            const double betaq1 = betaqFor(beta1);
            const double betaq2 = betaqFor(beta2);

            double child1 = 0.5 * ((x1 + x2) - betaq1 * (xu - xl));
            double child2 = 0.5 * ((x1 + x2) + betaq2 * (xu - xl));
            child1 = std::min(std::max(child1, lo), hi);
            child2 = std::min(std::max(child2, lo), hi);

            if (u01(rng_) <= 0.5) { c1[i] = child2; c2[i] = child1; }
            else                  { c1[i] = child1; c2[i] = child2; }
        }
    };

    // Polynomial mutation, in place.
    auto polyMutate = [&](Vec& x) {
        for (int i = 0; i < dim; ++i) {
            if (u01(rng_) > pm) continue;
            const double lo = lb[i], hi = ub[i];
            if (hi <= lo) continue;

            const double xi = x[i];
            const double delta1 = (xi - lo) / (hi - lo);
            const double delta2 = (hi - xi) / (hi - lo);
            const double rnd = u01(rng_);
            const double mutPow = 1.0 / (etaM + 1.0);
            double deltaq;
            if (rnd < 0.5) {
                const double xy = 1.0 - delta1;
                const double val = 2.0 * rnd + (1.0 - 2.0 * rnd) * std::pow(xy, etaM + 1.0);
                deltaq = std::pow(val, mutPow) - 1.0;
            } else {
                const double xy = 1.0 - delta2;
                const double val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * std::pow(xy, etaM + 1.0);
                deltaq = 1.0 - std::pow(val, mutPow);
            }
            x[i] = std::min(std::max(xi + deltaq * (hi - lo), lo), hi);
        }
    };

    auto tournamentSelect = [&](const std::vector<Individual>& pop) -> const Individual& {
        std::uniform_int_distribution<int> ui(0, static_cast<int>(pop.size()) - 1);
        const Individual& a = pop[ui(rng_)];
        const Individual& b = pop[ui(rng_)];
        return crowdedBetter(a, b) ? a : b;
    };

    // ---- Initial population ----
    std::vector<Individual> pop;
    pop.reserve(N);
    for (int i = 0; i < N; ++i) pop.push_back(randomIndividual());

    {
        auto fronts0 = fastNonDominatedSort(pop);
        for (const auto& fr : fronts0) assignCrowdingDistance(pop, fr);
    }

    long long evals = static_cast<long long>(N);

    for (int gen = 0; gen < G; ++gen) {
        std::vector<Individual> offspring;
        offspring.reserve(N);
        while (static_cast<int>(offspring.size()) < N) {
            const Individual& p1 = tournamentSelect(pop);
            const Individual& p2 = tournamentSelect(pop);
            Vec c1, c2;
            sbxCrossover(p1.x, p2.x, c1, c2);
            polyMutate(c1);
            polyMutate(c2);
            clampToBounds(c1);
            clampToBounds(c2);

            Individual i1;
            i1.x = std::move(c1);
            i1.f = prob_->evaluateMulti(i1.x);
            offspring.push_back(std::move(i1));
            ++evals;

            if (static_cast<int>(offspring.size()) < N) {
                Individual i2;
                i2.x = std::move(c2);
                i2.f = prob_->evaluateMulti(i2.x);
                offspring.push_back(std::move(i2));
                ++evals;
            }
        }

        std::vector<Individual> combined;
        combined.reserve(pop.size() + offspring.size());
        for (auto& ind : pop) combined.push_back(ind);
        for (auto& ind : offspring) combined.push_back(std::move(ind));

        auto fronts = fastNonDominatedSort(combined);
        for (const auto& fr : fronts) assignCrowdingDistance(combined, fr);

        std::vector<Individual> nextPop;
        nextPop.reserve(N);
        for (const auto& fr : fronts) {
            if (static_cast<int>(nextPop.size() + fr.size()) <= N) {
                for (int idx : fr) nextPop.push_back(combined[idx]);
            } else {
                std::vector<int> remaining = fr;
                std::sort(remaining.begin(), remaining.end(), [&](int a, int b) {
                    return combined[a].crowd > combined[b].crowd;
                });
                const int need = N - static_cast<int>(nextPop.size());
                for (int k = 0; k < need; ++k) nextPop.push_back(combined[remaining[k]]);
                break;
            }
            if (static_cast<int>(nextPop.size()) >= N) break;
        }
        pop = std::move(nextPop);
    }

    auto finalFronts = fastNonDominatedSort(pop);
    const std::vector<int> front0 = finalFronts.empty() ? std::vector<int>{} : finalFronts[0];

    result.paretoX.reserve(front0.size());
    result.paretoF.reserve(front0.size());
    for (int idx : front0) {
        result.paretoX.push_back(pop[idx].x);
        result.paretoF.push_back(pop[idx].f);
    }
    result.evals = evals;
    result.generations = G;
    return result;
}

} // namespace optimsolution
