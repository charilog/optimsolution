#include "gahs.h"
#include "init.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace optimsolution {

bool GAHS::inBounds(const Vec& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (x[j] < L[j] || x[j] > U[j]) return false;
    }
    return true;
}

double GAHS::deltaIter(double y) {
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_);
    return y * (1.0 - std::pow(r, 1.0 - generation_ * 1.0 / max_generations_));
}

void GAHS::init() {
    if (!prob_) return;

    const int D = prob_->dimension();
    chromosome_count_ = std::max(3, pop_);
    generation_ = 0;
    this->setPopulation(chromosome_count_);

    Initializer initSampler;
    initSampler.configure(initopt_);
    population_ = initSampler.samplePopulation(*prob_, rng_, chromosome_count_);

    fitness_.assign((size_t)chromosome_count_, std::numeric_limits<double>::infinity());
    children_array_.assign((size_t)chromosome_count_, Vec(D, 0.0));

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < chromosome_count_; ++i) {
        double f = eval(population_[i]);
        fitness_[i] = f;
        if (f < best_f_) {
            best_f_ = f;
            best_x_ = population_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    if (debug_gahs_) {
        std::fprintf(stdout,
            "[gahs] cfg -> pop=%d, sel=%.4f, mut=%.4f, HMCR=%.6f, PAR=%.6f, bw_scale=%.6f, adaptive_bw=%s, improvisations=%d\n",
            chromosome_count_, selection_rate_, mutation_rate_, HMCR_, PAR_, bw_scale_,
            adaptive_bw_ ? "on" : "off", improvisations_);
        std::fflush(stdout);
    }

    updateStop(fitness_);
    printBest();
}

void GAHS::calcFitnessArray() {
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i = 0; i < chromosome_count_; ++i) {
        if (prob_->calls() >= max_evals_) break;

        fitness_[i] = eval(population_[i]);

        if (local_rate_ > 0.0 && !local_method_.empty()) {
            double r = U01(rng_);
            if (r < local_rate_) {
                auto [xl, fl] = localSearch(local_method_, population_[i]);
                if (std::isfinite(fl) && fl < fitness_[i]) {
                    population_[i] = std::move(xl);
                    fitness_[i] = fl;
                }
            }
        }

        if (fitness_[i] < best_f_) {
            best_f_ = fitness_[i];
            best_x_ = population_[i];
        }
    }
}

void GAHS::selectionSort() {
    for (int i = 0; i < chromosome_count_; ++i) {
        for (int j = 0; j < chromosome_count_ - 1; ++j) {
            if (fitness_[j + 1] < fitness_[j]) {
                std::swap(population_[j], population_[j + 1]);
                std::swap(fitness_[j], fitness_[j + 1]);
            }
        }
    }
    if (!fitness_.empty() && fitness_[0] < best_f_) {
        best_f_ = fitness_[0];
        best_x_ = population_[0];
    }
}

std::vector<GAHS::RouletteEntry> GAHS::makeChromosomesForRoulette() {
    double maxValue = -1e100;
    for (int i = 0; i < chromosome_count_; ++i) {
        double a = std::fabs(fitness_[i]);
        if (i == 0 || a >= maxValue) maxValue = a;
    }

    std::vector<RouletteEntry> copy((size_t)chromosome_count_);
    for (int i = 0; i < chromosome_count_; ++i) {
        copy[(size_t)i].x = population_[(size_t)i];
        double y = fitness_[(size_t)i];
        y = -y;
        y += maxValue;
        if (!std::isfinite(y) || y < 0.0) y = 0.0;
        copy[(size_t)i].weight = y;
    }

    std::sort(copy.begin(), copy.end(),
              [](const RouletteEntry& a, const RouletteEntry& b) {
                  return a.weight < b.weight;
              });
    return copy;
}

int GAHS::selectWithRoulette(const std::vector<RouletteEntry>& roulette) {
    double sumFitness = 0.0;
    for (int i = 0; i < chromosome_count_; ++i) sumFitness += roulette[(size_t)i].weight;
    if (!(sumFitness > 0.0)) {
        std::uniform_int_distribution<int> Uid(0, chromosome_count_ - 1);
        return Uid(rng_);
    }

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double randomValue = U01(rng_) * sumFitness;
    double totalSum = 0.0;
    for (int i = 0; i < chromosome_count_; ++i) {
        totalSum += roulette[(size_t)i].weight;
        if (totalSum >= randomValue) return i;
    }
    return chromosome_count_ - 1;
}

int GAHS::selectWithTournament() {
    std::uniform_int_distribution<int> Uid(0, chromosome_count_ - 1);
    int minPos = -1;
    double minValue = 1e100;
    for (int i = 0; i < tournament_size_; ++i) {
        int randPos = Uid(rng_);
        double y = fitness_[(size_t)randPos];
        if (y < minValue || i == 0) {
            minValue = y;
            minPos = randPos;
        }
    }
    return minPos;
}

void GAHS::makeChildrenUniform(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
    std::uniform_int_distribution<int> Ubin(0, 1);
    for (int i = 0; i < (int)c1.size(); ++i) {
        int r = Ubin(rng_);
        if (r == 0) {
            c1[(size_t)i] = p1[(size_t)i];
            c2[(size_t)i] = p2[(size_t)i];
        } else {
            c1[(size_t)i] = p2[(size_t)i];
            c2[(size_t)i] = p1[(size_t)i];
        }
    }
}

void GAHS::makeChildrenOnePoint(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
    std::uniform_int_distribution<int> Ucut(0, (int)c1.size() - 1);
    int randPos = Ucut(rng_);
    for (int i = 0; i < (int)c1.size(); ++i) c1[(size_t)i] = (i < randPos) ? p1[(size_t)i] : p2[(size_t)i];
    for (int i = 0; i < (int)c2.size(); ++i) c2[(size_t)i] = (i < randPos) ? p2[(size_t)i] : p1[(size_t)i];
}

void GAHS::makeChildrenDouble(const Vec& p1, const Vec& p2, Vec& c1, Vec& c2) {
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    for (int i = 0; i < (int)c1.size(); ++i) {
        double alpha = -0.5 + 2.0 * U01(rng_);
        c1[(size_t)i] = alpha * p1[(size_t)i] + (1.0 - alpha) * p2[(size_t)i];
        c2[(size_t)i] = alpha * p2[(size_t)i] + (1.0 - alpha) * p1[(size_t)i];
    }
}

void GAHS::crossoverPhase() {
    std::vector<RouletteEntry> roulette;
    if (selection_method_ == "roulette") roulette = makeChromosomesForRoulette();

    int countChildren = (int)((1.0 - selection_rate_) * chromosome_count_);
    if (countChildren % 2 == 1) countChildren++;
    if (countChildren <= 0) return;
    if (countChildren > chromosome_count_) countChildren = chromosome_count_ - (chromosome_count_ % 2);

    const int D = prob_->dimension();
    int generatedChildren = 0;

    while (generatedChildren < countChildren) {
        Vec parent1, parent2;

        if (selection_method_ == "roulette") {
            int pos1 = selectWithRoulette(roulette);
            int pos2 = selectWithRoulette(roulette);
            parent1 = roulette[(size_t)pos1].x;
            parent2 = roulette[(size_t)pos2].x;
        } else {
            int pos1 = selectWithTournament();
            int pos2 = selectWithTournament();
            parent1 = population_[(size_t)pos1];
            parent2 = population_[(size_t)pos2];
        }

        Vec child1((size_t)D), child2((size_t)D);

        if (crossover_method_ == "uniform")
            makeChildrenUniform(parent1, parent2, child1, child2);
        else if (crossover_method_ == "onepoint")
            makeChildrenOnePoint(parent1, parent2, child1, child2);
        else
            makeChildrenDouble(parent1, parent2, child1, child2);

        if (!inBounds(child1)) child1 = parent1;
        if (!inBounds(child2)) child2 = parent2;

        children_array_[(size_t)generatedChildren] = child1;
        if (generatedChildren + 1 < countChildren)
            children_array_[(size_t)(generatedChildren + 1)] = child2;
        generatedChildren += 2;
    }

    for (int i = 0; i < countChildren; ++i) {
        int pos = chromosome_count_ - i - 1;
        population_[(size_t)pos] = children_array_[(size_t)i];
    }
}

void GAHS::mutatePhase() {
    const auto& ll = prob_->lb();
    const auto& rr = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Ubin(0, 1);

    for (int i = 1; i < chromosome_count_; ++i) {
        for (int j = 0; j < prob_->dimension(); ++j) {
            double r = U01(rng_);
            if (r < mutation_rate_) {
                Vec x = population_[(size_t)i];
                double oldValue = x[(size_t)j];

                if (mutation_method_ == "random") {
                    double delta = 0.05 * x[(size_t)j];
                    int direction = (Ubin(rng_) == 1) ? 1 : -1;
                    x[(size_t)j] = x[(size_t)j] + direction * delta;
                } else {
                    int t = Ubin(rng_);
                    double right = rr[(size_t)j];
                    double left  = ll[(size_t)j];
                    if (t == 0)
                        x[(size_t)j] = x[(size_t)j] + deltaIter(right - x[(size_t)j]);
                    else
                        x[(size_t)j] = x[(size_t)j] + deltaIter(x[(size_t)j] - left);
                }

                if (!inBounds(x)) x[(size_t)j] = oldValue;
                population_[(size_t)i] = x;
            }
        }
    }
}

void GAHS::ensureBounds(Vec& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] : 1.0);
        if (lo > hi) std::swap(lo, hi);
        if (!std::isfinite(x[j])) x[j] = 0.5 * (lo + hi);
        if (x[j] < lo) x[j] = lo;
        if (x[j] > hi) x[j] = hi;
    }
}

size_t GAHS::worstIndex() const {
    if (fitness_.empty()) return 0;
    size_t wi = 0;
    double wv = fitness_[0];
    for (size_t i = 1; i < fitness_.size(); ++i) {
        if (fitness_[i] > wv) {
            wv = fitness_[i];
            wi = i;
        }
    }
    return wi;
}

double GAHS::bandwidthForDim(int j) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double lo = (j < (int)L.size() ? L[(size_t)j] : -1.0);
    double hi = (j < (int)U.size() ? U[(size_t)j] : 1.0);
    if (lo > hi) std::swap(lo, hi);
    double range = hi - lo;
    if (!std::isfinite(range) || range <= 0.0) range = 1.0;

    double bw0 = std::isfinite(bw_abs_) ? bw_abs_ : bw_scale_ * range;
    double bw1 = std::isfinite(bw_min_abs_) ? bw_min_abs_ : bw_min_scale_ * range;
    if (bw0 < 0.0) bw0 = 0.0;
    if (bw1 < 0.0) bw1 = 0.0;

    if (!adaptive_bw_) {
        return (bw0 > 0.0 ? bw0 : 1e-12 * range);
    }

    double t = 0.0;
    if (max_evals_ > 0) {
        t = (double)prob_->calls() / (double)max_evals_;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
    }

    double bw = bw0 * (1.0 - t) + bw1 * t;
    if (!std::isfinite(bw) || bw <= 0.0) bw = 1e-12 * range;
    return bw;
}

void GAHS::harmonyPhase() {
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)population_.size();
    if (N <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> I(0, std::max(0, N - 1));

    const int K = std::max(1, improvisations_ > 0 ? improvisations_ : N);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int it = 0; it < K; ++it) {
        if (prob_->calls() >= max_evals_) break;

        Vec u((size_t)D, 0.0);

        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[(size_t)j] : -1.0);
            double hi = (j < (int)U.size() ? U[(size_t)j] : 1.0);
            if (lo > hi) std::swap(lo, hi);

            if (U01(rng_) < HMCR_ && N > 0) {
                int k = I(rng_);
                u[(size_t)j] = population_[(size_t)k][(size_t)j];
                if (U01(rng_) < PAR_) {
                    double bw = bandwidthForDim(j);
                    u[(size_t)j] += (U01(rng_) * 2.0 - 1.0) * bw;
                }
            } else {
                double r = U01(rng_);
                u[(size_t)j] = lo + r * (hi - lo);
            }
        }

        ensureBounds(u);
        double fu = eval(u);

        if (local_rate_ > 0.0 && !local_method_.empty()) {
            if (U01(rng_) < local_rate_) {
                auto [xl, fl] = localSearch(local_method_, u);
                if (std::isfinite(fl) && fl < fu) {
                    u = std::move(xl);
                    fu = fl;
                }
            }
        }

        size_t w = worstIndex();
        if (w < fitness_.size() && fu < fitness_[w]) {
            population_[w] = u;
            fitness_[w] = fu;
            if (fu < best_f_) {
                best_f_ = fu;
                best_x_ = u;
            }
        }
    }

    if (!population_.empty() && !fitness_.empty()) {
        double minv = fitness_[0];
        for (size_t i = 1; i < fitness_.size(); ++i) minv = std::min(minv, fitness_[i]);
        if (std::isfinite(best_f_) && best_f_ < minv) {
            size_t w = worstIndex();
            population_[w] = best_x_;
            fitness_[w] = best_f_;
        }
    }
}

void GAHS::localCrossover(int pos) {
    const int D = prob_->dimension();
    std::uniform_int_distribution<int> Upos(0, chromosome_count_ - 1);
    std::uniform_int_distribution<int> Udim(0, D - 1);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    Vec g((size_t)D);

    for (int iters = 1; iters <= 100; ++iters) {
        if (prob_->calls() >= max_evals_) break;

        int gpos = Upos(rng_);
        int cutpoint = Udim(rng_);

        for (int j = 0; j < D; ++j) g[(size_t)j] = population_[(size_t)pos][(size_t)j];

        double alpha = -0.5 + 2.0 * U01(rng_);
        g[(size_t)cutpoint] = alpha * population_[(size_t)pos][(size_t)cutpoint] +
                              (1.0 - alpha) * population_[(size_t)gpos][(size_t)cutpoint];

        if (!inBounds(g)) continue;

        double f = eval(g);
        if (f < fitness_[(size_t)pos]) {
            population_[(size_t)pos][(size_t)cutpoint] = g[(size_t)cutpoint];
            fitness_[(size_t)pos] = f;
            if (f < best_f_) {
                best_f_ = f;
                best_x_ = population_[(size_t)pos];
            }
        } else {
            g[(size_t)cutpoint] = alpha * population_[(size_t)gpos][(size_t)cutpoint] +
                                  (1.0 - alpha) * population_[(size_t)pos][(size_t)cutpoint];
            if (!inBounds(g)) continue;

            double f2 = eval(g);
            if (f2 < fitness_[(size_t)pos]) {
                population_[(size_t)pos][(size_t)cutpoint] = g[(size_t)cutpoint];
                fitness_[(size_t)pos] = f2;
                if (f2 < best_f_) {
                    best_f_ = f2;
                    best_x_ = population_[(size_t)pos];
                }
            }
        }
    }
}

void GAHS::localMutate(int pos) {
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Ubin(0, 1);

    for (int i = 0; i < D; ++i) {
        if (prob_->calls() >= max_evals_) break;

        double gold = population_[(size_t)pos][(size_t)i];
        double delta = 0.05 * U01(rng_) * gold;
        double direction = (Ubin(rng_) == 1) ? 1.0 : -1.0;
        double gnew = gold + direction * delta;

        population_[(size_t)pos][(size_t)i] = gnew;

        if (!inBounds(population_[(size_t)pos])) {
            population_[(size_t)pos][(size_t)i] = gold;
            continue;
        }

        double f = eval(population_[(size_t)pos]);
        if (f < fitness_[(size_t)pos]) {
            fitness_[(size_t)pos] = f;
            if (f < best_f_) {
                best_f_ = f;
                best_x_ = population_[(size_t)pos];
            }
        } else {
            population_[(size_t)pos][(size_t)i] = gold;
        }
    }
}

void GAHS::localSiman(int pos) {
    double T0 = 1e8;
    Vec bestx_local = population_[(size_t)pos];
    double besty_local = fitness_[(size_t)pos];

    const int D = prob_->dimension();
    Vec y((size_t)D);

    Vec xpoint = population_[(size_t)pos];
    double ypoint = fitness_[(size_t)pos];

    int k = 1;
    const int neps = 100;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Udim(0, D - 1);
    std::uniform_int_distribution<int> Ubin(0, 1);

    while (true) {
        for (int i = 1; i <= neps; ++i) {
            if (prob_->calls() >= max_evals_) goto siman_done;

            for (int j = 0; j < D; ++j) y[(size_t)j] = xpoint[(size_t)j];

            for (int j = 0; j < 30; ++j) {
                int randPos = Udim(rng_);
                double range = 0.1;
                double old = y[(size_t)randPos];
                int direction = (Ubin(rng_) == 1) ? 1 : -1;
                double newValue = y[(size_t)randPos] + direction * U01(rng_) * range * y[(size_t)randPos];
                y[(size_t)randPos] = newValue;
                if (!inBounds(y)) y[(size_t)randPos] = old;
            }

            double fy = eval(y);
            if (std::isnan(fy) || std::isinf(fy)) continue;

            if (fy < ypoint) {
                xpoint = y;
                ypoint = fy;
                if (ypoint < besty_local) {
                    bestx_local = xpoint;
                    besty_local = ypoint;
                }
            } else {
                double r = std::fabs(U01(rng_));
                double ratio = std::exp(-(fy - ypoint) / T0);
                double xmin = (ratio < 1.0) ? ratio : 1.0;
                if (r < xmin) {
                    xpoint = y;
                    ypoint = fy;
                    if (ypoint < besty_local) {
                        bestx_local = xpoint;
                        besty_local = ypoint;
                    }
                }
            }
        }

        const double alpha = 0.8;
        T0 = T0 * std::pow(alpha, k);
        k = k + 1;
        if (T0 <= 1e-6) break;
    }

siman_done:
    population_[(size_t)pos] = bestx_local;
    fitness_[(size_t)pos] = besty_local;
    if (besty_local < best_f_) {
        best_f_ = besty_local;
        best_x_ = bestx_local;
    }
}

void GAHS::localDE(int pos) {
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Udim(0, D - 1);

    Vec g((size_t)D);

    int randomA, randomB, randomC;
    do {
        randomA = selectWithTournament();
        randomB = selectWithTournament();
        randomC = selectWithTournament();
    } while (randomA == randomB || randomB == randomC || randomC == randomA);

    double CR = 0.9;
    double F = 0.8;
    int n = D;
    int randomIndex = Udim(rng_);

    for (int i = 0; i < n; ++i) {
        if (prob_->calls() >= max_evals_) break;

        if (i == randomIndex || U01(rng_) <= CR) {
            double old_value = population_[(size_t)pos][(size_t)i];
            F = -0.5 + 2.0 * U01(rng_);
            population_[(size_t)pos][(size_t)i] = population_[(size_t)randomA][(size_t)i] +
                std::abs(F * (population_[(size_t)randomB][(size_t)i] - population_[(size_t)randomC][(size_t)i]));

            if (!inBounds(population_[(size_t)pos])) {
                population_[(size_t)pos][(size_t)i] = old_value;
                continue;
            }

            for (int j = 0; j < n; ++j) g[(size_t)j] = population_[(size_t)pos][(size_t)j];
            double trial_fitness = eval(g);

            if (std::fabs(trial_fitness) < std::fabs(fitness_[(size_t)pos])) {
                fitness_[(size_t)pos] = trial_fitness;
                if (trial_fitness < best_f_) {
                    best_f_ = trial_fitness;
                    best_x_ = population_[(size_t)pos];
                }
            } else {
                population_[(size_t)pos][(size_t)i] = old_value;
            }
        }
    }
}

void GAHS::localSearchAt(int pos) {
    if      (lsearch_method_ == "crossover") localCrossover(pos);
    else if (lsearch_method_ == "mutate")    localMutate(pos);
    else if (lsearch_method_ == "siman")     localSiman(pos);
    else if (lsearch_method_ == "de")        localDE(pos);
}

void GAHS::one_iteration() {
    if (!prob_) return;
    ++generation_;

    // Evaluate current GA population
    calcFitnessArray();
    selectionSort();

    // Standard GA search
    crossoverPhase();
    mutatePhase();

    // Re-evaluate GA offspring before HS operates on the memory
    calcFitnessArray();
    selectionSort();

    // HS memory consideration + pitch adjustment + worst replacement
    harmonyPhase();
    selectionSort();

    if (lsearch_gens_ > 0 && generation_ % lsearch_gens_ == 0 && lsearch_method_ != "none") {
        localSearchAt(0);
        std::uniform_int_distribution<int> Upos(0, chromosome_count_ - 1);
        for (int i = 0; i < lsearch_items_; ++i) {
            if (prob_->calls() >= max_evals_) break;
            int pos = Upos(rng_);
            localSearchAt(pos);
        }
        selectionSort();
    }

    updateStop(fitness_);
    printBest();
}

void GAHS::end() {
    if (!prob_) return;
    if (!end_local_refine_ || end_local_method_.empty()) return;
    if (population_.empty() || fitness_.empty()) return;

    auto [xl, fl] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(fl) && fl < best_f_) {
        best_f_ = fl;
        best_x_ = std::move(xl);
    }

    fitness_[0] = best_f_;
    population_[0] = best_x_;

    updateStop(fitness_);
    printBest();
}

} // namespace optimsolution
