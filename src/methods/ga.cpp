#include "ga.h"
#include "init.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace optimsolution {

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

bool GA::inBounds(const std::vector<double>& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j)
        if (x[j] < L[j] || x[j] > U[j]) return false;
    return true;
}

// deltaIter — faithful to original Genetic::deltaIter:
//   y * (1 - r^(1 - generation/maxGenerations))
double GA::deltaIter(double y) {
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_);
    return y * (1.0 - std::pow(r, 1.0 - generation_ * 1.0 / max_generations_));
}

// ─────────────────────────────────────────────────────────────────────────────
//  updateAdaptive — FE-based linear schedule (JSO-style)
//
//  progress p = calls / max_evals  ∈ [0, 1]
//
//  Adaptive Mutation  (when enabled):
//    mutation_rate = max_mut - (max_mut - min_mut) * p
//    p=0 → max_mut  (maximum exploration, large perturbations)
//    p=1 → min_mut  (minimum perturbation, full exploitation)
//
//  Adaptive Crossover (when enabled):
//    crossover_rate = min_cr + (max_cr - min_cr) * p
//    p=0 → min_cr   (gentle recombination, preserve structure)
//    p=1 → max_cr   (maximum recombination pressure, full attack)
//
//  Middle (p≈0.5): both are at their midpoints — balanced exploration/exploitation.
//
//  When a mechanism is disabled its parameter is never touched, remaining at
//  the value set in configure() — identical behaviour to the base GA.
// ─────────────────────────────────────────────────────────────────────────────

void GA::updateAdaptive() {
    if (!adaptive_mutation_enable_ && !adaptive_crossover_enable_) return;

    // Clamp p to [0, 1]
    double p = 0.0;
    if (max_evals_ > 0)
        p = std::min(1.0, static_cast<double>(prob_->calls()) /
                              static_cast<double>(max_evals_));

    if (adaptive_mutation_enable_) {
        mutation_rate_ = adaptive_mutation_max_ -
                         (adaptive_mutation_max_ - adaptive_mutation_min_) * p;
    }

    if (adaptive_crossover_enable_) {
        crossover_rate_ = adaptive_crossover_min_ +
                          (adaptive_crossover_max_ - adaptive_crossover_min_) * p;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  init  (faithful to Genetic::init)
// ─────────────────────────────────────────────────────────────────────────────

void GA::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    chromosome_count_ = pop_;
    generation_       = 0;

    // When adaptive is disabled keep the configured fixed values.
    // When adaptive is enabled initialise to the starting-point values
    // (p=0 → exploration end of the schedule).
    if (adaptive_mutation_enable_)
        mutation_rate_  = adaptive_mutation_max_;
    if (adaptive_crossover_enable_)
        crossover_rate_ = adaptive_crossover_min_;
    else if (!adaptive_crossover_enable_)
        crossover_rate_ = 1.0;   // base GA: always crossover

    Initializer initSampler;
    initSampler.configure(initopt_);
    population_ = initSampler.samplePopulation(*prob_, rng_, chromosome_count_);

    fitness_.assign((size_t)chromosome_count_, std::numeric_limits<double>::infinity());
    children_array_.assign((size_t)chromosome_count_, std::vector<double>(D, 0.0));

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < chromosome_count_; ++i) {
        double f = eval(population_[i]);
        fitness_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = population_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    updateStop(fitness_);
    printBest();
}

// ─────────────────────────────────────────────────────────────────────────────
//  CalcFitnessArray  (faithful to Genetic::CalcFitnessArray)
// ─────────────────────────────────────────────────────────────────────────────

void GA::calcFitnessArray() {
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i = 0; i < chromosome_count_; ++i) {
        if (prob_->calls() >= max_evals_) break;

        fitness_[i] = eval(population_[i]);

        if (local_rate_ > 0.0 && U01(rng_) < local_rate_) {
            auto [xl, fl] = localSearch(local_method_, population_[i]);
            population_[i] = std::move(xl);
            fitness_[i]    = fl;
        }

        if (fitness_[i] < best_f_) { best_f_ = fitness_[i]; best_x_ = population_[i]; }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Selection  (faithful to Genetic::Selection — bubble sort ascending)
// ─────────────────────────────────────────────────────────────────────────────

void GA::selectionSort() {
    for (int i = 0; i < chromosome_count_; ++i) {
        for (int j = 0; j < chromosome_count_ - 1; ++j) {
            if (fitness_[j + 1] < fitness_[j]) {
                std::swap(population_[j],  population_[j + 1]);
                std::swap(fitness_[j],     fitness_[j + 1]);
            }
        }
    }
    if (fitness_[0] < best_f_) { best_f_ = fitness_[0]; best_x_ = population_[0]; }
}

// ─────────────────────────────────────────────────────────────────────────────
//  makeChromosomesForRoulette / selectWithRoulette / selectWithTournament
//  (faithful to Genetic originals)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<GA::RouletteEntry> GA::makeChromosomesForRoulette() {
    double maxValue = -1e+100;
    for (int i = 0; i < chromosome_count_; ++i) {
        double a = std::fabs(fitness_[i]);
        if (i == 0 || a >= maxValue) maxValue = a;
    }
    std::vector<RouletteEntry> copy(chromosome_count_);
    for (int i = 0; i < chromosome_count_; ++i) {
        copy[i].x     = population_[i];
        double y      = fitness_[i];
        y = -y;
        y += maxValue;
        copy[i].weight = y;
    }
    std::sort(copy.begin(), copy.end(),
              [](const RouletteEntry& a, const RouletteEntry& b) {
                  return a.weight < b.weight;
              });
    return copy;
}

int GA::selectWithRoulette(const std::vector<RouletteEntry>& roulette) {
    double sumFitness = 0.0;
    for (int i = 0; i < chromosome_count_; ++i)
        sumFitness += roulette[i].weight;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double randomValue = U01(rng_) * sumFitness;
    double totalSum    = 0.0;
    for (int i = 0; i < chromosome_count_; ++i) {
        totalSum += roulette[i].weight;
        if (totalSum >= randomValue) return i;
    }
    return chromosome_count_ - 1;
}

int GA::selectWithTournament() {
    std::uniform_int_distribution<int> Uid(0, chromosome_count_ - 1);
    int    minPos   = -1;
    double minValue = 1e+100;
    for (int i = 0; i < tournament_size_; ++i) {
        int    randPos = Uid(rng_);
        double y       = fitness_[randPos];
        if (y < minValue || i == 0) { minValue = y; minPos = randPos; }
    }
    return minPos;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Crossover operators  (faithful to Genetic originals)
// ─────────────────────────────────────────────────────────────────────────────

void GA::makeChildrenUniform(
    const std::vector<double>& p1, const std::vector<double>& p2,
    std::vector<double>& c1, std::vector<double>& c2) {
    std::uniform_int_distribution<int> Ubin(0, 1);
    for (int i = 0; i < (int)c1.size(); ++i) {
        int r = Ubin(rng_);
        if (r == 0) { c1[i] = p1[i]; c2[i] = p2[i]; }
        else        { c1[i] = p2[i]; c2[i] = p1[i]; }
    }
}

void GA::makeChildrenOnePoint(
    const std::vector<double>& p1, const std::vector<double>& p2,
    std::vector<double>& c1, std::vector<double>& c2) {
    std::uniform_int_distribution<int> Ucut(0, (int)c1.size() - 1);
    int randPos = Ucut(rng_);
    for (int i = 0; i < (int)c1.size(); ++i)
        c1[i] = (i < randPos) ? p1[i] : p2[i];
    for (int i = 0; i < (int)c2.size(); ++i)
        c2[i] = (i < randPos) ? p2[i] : p1[i];
}

void GA::makeChildrenDouble(
    const std::vector<double>& p1, const std::vector<double>& p2,
    std::vector<double>& c1, std::vector<double>& c2) {
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    for (int i = 0; i < (int)c1.size(); ++i) {
        double alpha = -0.5 + 2.0 * U01(rng_);   // [-0.5, 1.5]
        c1[i] = alpha * p1[i] + (1.0 - alpha) * p2[i];
        c2[i] = alpha * p2[i] + (1.0 - alpha) * p1[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Crossover phase  (faithful to Genetic::Crossover + adaptive gate)
//
//  Adaptive crossover gate:
//    Each child pair is generated only if U[0,1] < crossover_rate_.
//    When adaptive is off, crossover_rate_ = 1.0 → always crossover (= base GA).
//    When adaptive is on, crossover_rate_ grows from min_cr to max_cr with FEs:
//      early runs (low p): pairs often copied unchanged → diversity preserved
//      late runs  (high p): almost every pair is recombined → exploitation pressure
// ─────────────────────────────────────────────────────────────────────────────

void GA::crossoverPhase() {
    std::vector<RouletteEntry> roulette;
    if (selection_method_ == "roulette")
        roulette = makeChromosomesForRoulette();

    int countChildren = (int)((1.0 - selection_rate_) * chromosome_count_);
    if (countChildren % 2 == 1) countChildren++;

    const int D = prob_->dimension();
    int generatedChildren = 0;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    while (true) {
        std::vector<double> parent1, parent2;

        if (selection_method_ == "roulette") {
            int pos1 = selectWithRoulette(roulette);
            int pos2 = selectWithRoulette(roulette);
            parent1  = roulette[pos1].x;
            parent2  = roulette[pos2].x;
        } else {
            int pos1 = selectWithTournament();
            int pos2 = selectWithTournament();
            parent1  = population_[pos1];
            parent2  = population_[pos2];
        }

        std::vector<double> child1(D), child2(D);

        if (U01(rng_) < crossover_rate_) {
            // ── Recombination ──────────────────────────────────────────────
            if (crossover_method_ == "uniform")
                makeChildrenUniform(parent1, parent2, child1, child2);
            else if (crossover_method_ == "onepoint")
                makeChildrenOnePoint(parent1, parent2, child1, child2);
            else  // "double" (default)
                makeChildrenDouble(parent1, parent2, child1, child2);

            if (!inBounds(child1)) child1 = parent1;
            if (!inBounds(child2)) child2 = parent2;
        } else {
            // ── No crossover: children inherit parents (diversity preserved) ─
            child1 = parent1;
            child2 = parent2;
        }

        children_array_[generatedChildren]     = child1;
        children_array_[generatedChildren + 1] = child2;
        generatedChildren += 2;
        if (generatedChildren >= countChildren) break;
    }

    // Replace worst slots (tail of sorted population) with generated children
    for (int i = 0; i < countChildren; ++i) {
        int pos = chromosome_count_ - i - 1;
        population_[pos] = children_array_[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mutate  (faithful to Genetic::Mutate + adaptive mutation_rate_)
//  Elitism: i starts at 1, population_[0] (best) is never mutated.
//
//  Adaptive mutation:
//    mutation_rate_ decreases from adaptive_mutation_max to adaptive_mutation_min
//    as FEs progress from 0 to max_evals.
//    early  (high rate) → large, frequent perturbations → exploration
//    late   (low rate)  → rare, small perturbations     → exploitation
//    When adaptive is off, mutation_rate_ = configured constant (= base GA).
// ─────────────────────────────────────────────────────────────────────────────

void GA::mutatePhase() {
    const auto& ll = prob_->lb();
    const auto& rr = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ubin(0, 1);

    for (int i = 1; i < chromosome_count_; ++i) {
        for (int j = 0; j < prob_->dimension(); ++j) {
            if (U01(rng_) >= mutation_rate_) continue;

            std::vector<double> x = population_[i];
            double oldValue = x[j];

            if (mutation_method_ == "random") {
                double delta     = 0.05 * x[j];
                int    direction = (Ubin(rng_) == 1) ? 1 : -1;
                x[j] = x[j] + direction * delta;
            } else {
                // "double" non-uniform mutation
                int    t     = Ubin(rng_);
                double right = rr[j];
                double left  = ll[j];
                if (t == 0)
                    x[j] = x[j] + deltaIter(right - x[j]);
                else
                    x[j] = x[j] + deltaIter(x[j] - left);
            }

            if (!inBounds(x))
                x[j] = oldValue;

            population_[i] = x;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal local search: localCrossover  (faithful to Genetic::localCrossover)
// ─────────────────────────────────────────────────────────────────────────────

void GA::localCrossover(int pos) {
    const int D = prob_->dimension();
    std::uniform_int_distribution<int>     Upos(0, chromosome_count_ - 1);
    std::uniform_int_distribution<int>     Udim(0, D - 1);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::vector<double> g(D);

    for (int iters = 1; iters <= 100; ++iters) {
        if (prob_->calls() >= max_evals_) break;

        int gpos     = Upos(rng_);
        int cutpoint = Udim(rng_);

        for (int j = 0; j < D; ++j) g[j] = population_[pos][j];

        double alpha = -0.5 + 2.0 * U01(rng_);
        g[cutpoint]  = alpha       * population_[pos][cutpoint]  +
                       (1.0-alpha) * population_[gpos][cutpoint];

        if (!inBounds(g)) continue;

        double f = eval(g);
        if (f < fitness_[pos]) {
            population_[pos][cutpoint] = g[cutpoint];
            fitness_[pos]              = f;
            if (f < best_f_) { best_f_ = f; best_x_ = population_[pos]; }
        } else {
            g[cutpoint] = alpha       * population_[gpos][cutpoint]  +
                          (1.0-alpha) * population_[pos][cutpoint];
            if (!inBounds(g)) continue;

            double f2 = eval(g);
            if (f2 < fitness_[pos]) {
                population_[pos][cutpoint] = g[cutpoint];
                fitness_[pos]              = f2;
                if (f2 < best_f_) { best_f_ = f2; best_x_ = population_[pos]; }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  localMutate  (faithful to Genetic::localMutate)
// ─────────────────────────────────────────────────────────────────────────────

void GA::localMutate(int pos) {
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ubin(0, 1);

    for (int i = 0; i < D; ++i) {
        if (prob_->calls() >= max_evals_) break;

        double gold      = population_[pos][i];
        double delta     = 0.05 * U01(rng_) * gold;
        double direction = (Ubin(rng_) == 1) ? 1.0 : -1.0;

        population_[pos][i] = gold + direction * delta;

        if (!inBounds(population_[pos])) {
            population_[pos][i] = gold;
            continue;
        }

        double f = eval(population_[pos]);
        if (f < fitness_[pos]) {
            fitness_[pos] = f;
            if (f < best_f_) { best_f_ = f; best_x_ = population_[pos]; }
        } else {
            population_[pos][i] = gold;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  localSiman  (faithful to Genetic::localSiman)
// ─────────────────────────────────────────────────────────────────────────────

void GA::localSiman(int pos) {
    double T0 = 1e+8;

    std::vector<double> bestx_local = population_[pos];
    double              besty_local = fitness_[pos];

    const int D = prob_->dimension();
    std::vector<double> y(D);
    std::vector<double> xpoint = population_[pos];
    double              ypoint = fitness_[pos];

    int       k    = 1;
    const int neps = 100;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Udim(0, D - 1);
    std::uniform_int_distribution<int>     Ubin(0, 1);

    while (true) {
        for (int i = 1; i <= neps; ++i) {
            if (prob_->calls() >= max_evals_) goto siman_done;

            for (int j = 0; j < D; ++j) y[j] = xpoint[j];

            for (int j = 0; j < 30; ++j) {
                int    randPos   = Udim(rng_);
                double range     = 0.1;
                double old       = y[randPos];
                int    direction = (Ubin(rng_) == 1) ? 1 : -1;
                double newValue  = y[randPos] + direction * U01(rng_) * range * y[randPos];
                y[randPos] = newValue;
                if (!inBounds(y)) y[randPos] = old;
            }

            double fy = eval(y);
            if (std::isnan(fy) || std::isinf(fy)) continue;

            if (fy < ypoint) {
                xpoint = y;
                ypoint = fy;
                if (ypoint < besty_local) { bestx_local = xpoint; besty_local = ypoint; }
            } else {
                double r     = std::fabs(U01(rng_));
                double ratio = std::exp(-(fy - ypoint) / T0);
                double xmin  = (ratio < 1.0) ? ratio : 1.0;
                if (r < xmin) {
                    xpoint = y;
                    ypoint = fy;
                    if (ypoint < besty_local) { bestx_local = xpoint; besty_local = ypoint; }
                }
            }
        }

        const double alpha = 0.8;
        T0 = T0 * std::pow(alpha, k);
        ++k;
        if (T0 <= 1e-6) break;
    }

siman_done:
    population_[pos] = bestx_local;
    fitness_[pos]    = besty_local;
    if (besty_local < best_f_) { best_f_ = besty_local; best_x_ = bestx_local; }
}

// ─────────────────────────────────────────────────────────────────────────────
//  localDE  (faithful to the "de" branch inside Genetic::LocalSearch)
// ─────────────────────────────────────────────────────────────────────────────

void GA::localDE(int pos) {
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Udim(0, D - 1);
    std::vector<double> g(D);

    int randomA, randomB, randomC;
    do {
        randomA = selectWithTournament();
        randomB = selectWithTournament();
        randomC = selectWithTournament();
    } while (randomA == randomB || randomB == randomC || randomC == randomA);

    double CR          = 0.9;
    double F           = 0.8;
    int    n           = D;
    int    randomIndex = Udim(rng_);

    for (int i = 0; i < n; ++i) {
        if (prob_->calls() >= max_evals_) break;

        if (i == randomIndex || U01(rng_) <= CR) {
            double old_value = population_[pos][i];
            F = -0.5 + 2.0 * U01(rng_);
            population_[pos][i] = population_[randomA][i] +
                                   std::abs(F * (population_[randomB][i] -
                                                 population_[randomC][i]));

            if (!inBounds(population_[pos])) {
                population_[pos][i] = old_value;
                continue;
            }

            for (int j = 0; j < n; ++j) g[j] = population_[pos][j];
            double trial_fitness = eval(g);

            if (std::fabs(trial_fitness) < std::fabs(fitness_[pos])) {
                std::printf("NEW DE VALUE[%d] = %lf=>%lf\n", pos, fitness_[pos], trial_fitness);
                fitness_[pos] = trial_fitness;
                if (trial_fitness < best_f_) { best_f_ = trial_fitness; best_x_ = population_[pos]; }
            } else {
                population_[pos][i] = old_value;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  LocalSearch dispatcher  (faithful to Genetic::LocalSearch)
// ─────────────────────────────────────────────────────────────────────────────

void GA::localSearchAt(int pos) {
    if      (lsearch_method_ == "crossover") localCrossover(pos);
    else if (lsearch_method_ == "mutate")    localMutate(pos);
    else if (lsearch_method_ == "siman")     localSiman(pos);
    else if (lsearch_method_ == "de")        localDE(pos);
}

// ─────────────────────────────────────────────────────────────────────────────
//  one_iteration  (faithful to Genetic::step + FE-based adaptive update)
// ─────────────────────────────────────────────────────────────────────────────

void GA::one_iteration() {
    if (!prob_) return;
    ++generation_;

    // Update adaptive parameters at the START of the iteration so they reflect
    // the FE-progress BEFORE we spend the current iteration's evaluations.
    updateAdaptive();

    calcFitnessArray();
    selectionSort();
    crossoverPhase();
    mutatePhase();

    if (lsearch_gens_ > 0 &&
        generation_ % lsearch_gens_ == 0 &&
        lsearch_method_ != "none") {
        localSearchAt(0);
        std::uniform_int_distribution<int> Upos(0, chromosome_count_ - 1);
        for (int i = 0; i < lsearch_items_; ++i) {
            if (prob_->calls() >= max_evals_) break;
            localSearchAt(Upos(rng_));
        }
        selectionSort();
    }

    updateStop(fitness_);
    printBest();
}

// ─────────────────────────────────────────────────────────────────────────────
//  end  (faithful to Genetic::done)
// ─────────────────────────────────────────────────────────────────────────────

void GA::end() {
    if (!prob_) return;
    if (!end_local_refine_ || end_local_method_.empty()) return;

    auto [xl, fl] = localSearch(end_local_method_, best_x_);
    if (fl < best_f_) {
        best_f_ = fl;
        best_x_ = std::move(xl);
    }
    fitness_[0]    = best_f_;
    population_[0] = best_x_;

    updateStop(fitness_);
    printBest();
}

} // namespace optimsolution
