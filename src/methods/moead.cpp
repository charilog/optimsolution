#include "moead.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace optimsolution {

namespace {

double euclid2(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// Tchebycheff scalarizing function: g(F | lambda, z) = max_j lambda_j*|F_j - z_j|
// (with a small epsilon on lambda_j to avoid a zero weight collapsing a term).
double tchebycheff(const Vec& F, const Vec& lambda, const Vec& z) {
    double g = 0.0;
    for (size_t j = 0; j < F.size(); ++j) {
        const double lj = std::max(lambda[j], 1e-6);
        g = std::max(g, lj * std::fabs(F[j] - z[j]));
    }
    return g;
}

} // namespace

MOOResult MOEAD::run() {
    MOOResult result;
    if (!prob_) return result;

    const int D = prob_->dimension();
    const Vec& lb = prob_->lb();
    const Vec& ub = prob_->ub();
    const int N = std::max(4, population_);
    const int G = std::max(1, generations_);
    const int M = prob_->numObjectives();

    const int T          = static_cast<int>(param("neighbor_size", 20.0));
    const double deF      = param("de_F", 0.5);
    const double pc        = param("crossover_prob", 1.0);
    const double pm        = param("mutation_prob", D > 0 ? (1.0 / double(D)) : 0.1);
    const double etaM      = param("eta_m", 20.0);
    const int nrMax        = static_cast<int>(param("max_replace", 2.0));
    const int neighSize    = std::min(std::max(2, T), N);

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // ---- Weight vectors (2 objectives: a uniform sweep of the simplex). ----
    // NOTE: only 2-objective problems are supported by this weight generator
    // (see the class comment in moead.h).
    std::vector<Vec> lambda(N, Vec(M, 1.0 / double(std::max(1, M))));
    if (M == 2) {
        for (int i = 0; i < N; ++i) {
            const double t = (N > 1) ? double(i) / double(N - 1) : 0.5;
            lambda[i] = { t, 1.0 - t };
        }
    }

    // ---- Neighborhoods: T closest weight vectors to each i (by index). ----
    std::vector<std::vector<int>> neighbors(N);
    {
        std::vector<int> order(N);
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) order[k] = k;
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                return euclid2(lambda[i], lambda[a]) < euclid2(lambda[i], lambda[b]);
            });
            neighbors[i].assign(order.begin(), order.begin() + neighSize);
        }
    }

    // ---- Initial population ----
    std::vector<Vec> X(N, Vec(D));
    std::vector<Vec> Fx(N, Vec(M));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            std::uniform_real_distribution<double> ud(lb[j], ub[j]);
            X[i][j] = ud(rng_);
        }
        Fx[i] = prob_->evaluateMulti(X[i]);
    }

    Vec z(M, std::numeric_limits<double>::infinity());
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) z[j] = std::min(z[j], Fx[i][j]);
    }

    auto clampToBounds = [&](Vec& x) {
        for (int j = 0; j < D; ++j) {
            if (x[j] < lb[j]) x[j] = lb[j];
            if (x[j] > ub[j]) x[j] = ub[j];
        }
    };

    for (int gen = 0; gen < G; ++gen) {
        for (int i = 0; i < N; ++i) {
            // Pick two distinct random neighbors (DE-style donor pair) from B(i).
            const auto& B = neighbors[i];
            std::uniform_int_distribution<int> Ub(0, static_cast<int>(B.size()) - 1);
            int r1 = B[Ub(rng_)], r2 = B[Ub(rng_)];
            int guard = 0;
            while (r2 == r1 && guard++ < 20) r2 = B[Ub(rng_)];

            // DE/rand-like variation: y = x_i + F*(x_r1 - x_r2), then binomial crossover with x_i.
            Vec y = X[i];
            const int jrand = std::uniform_int_distribution<int>(0, D - 1)(rng_);
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < pc || j == jrand) {
                    y[j] = X[i][j] + deF * (X[r1][j] - X[r2][j]);
                }
            }
            // Polynomial mutation.
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) > pm) continue;
                const double lo = lb[j], hi = ub[j];
                if (hi <= lo) continue;
                const double yi = y[j];
                const double delta1 = (yi - lo) / (hi - lo);
                const double delta2 = (hi - yi) / (hi - lo);
                const double rnd = U01(rng_);
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
                y[j] = std::min(std::max(yi + deltaq * (hi - lo), lo), hi);
            }
            clampToBounds(y);

            const Vec Fy = prob_->evaluateMulti(y);
            for (int j = 0; j < M; ++j) z[j] = std::min(z[j], Fy[j]);

            // Update up to nrMax neighboring sub-problems whose Tchebycheff
            // value improves with y, in a shuffled order of B(i).
            std::vector<int> order = B;
            std::shuffle(order.begin(), order.end(), rng_);
            int replaced = 0;
            for (int idx : order) {
                if (replaced >= nrMax) break;
                const double gOld = tchebycheff(Fx[idx], lambda[idx], z);
                const double gNew = tchebycheff(Fy, lambda[idx], z);
                if (gNew <= gOld) {
                    X[idx] = y;
                    Fx[idx] = Fy;
                    ++replaced;
                }
            }
        }
    }

    // ---- Extract the non-dominated subset of the final population. ----
    auto dominates = [](const Vec& a, const Vec& b) {
        bool better = false;
        for (size_t k = 0; k < a.size(); ++k) {
            if (a[k] > b[k]) return false;
            if (a[k] < b[k]) better = true;
        }
        return better;
    };
    std::vector<bool> dominated(N, false);
    for (int i = 0; i < N; ++i) {
        if (dominated[i]) continue;
        for (int j = 0; j < N; ++j) {
            if (i == j || dominated[j]) continue;
            if (dominates(Fx[j], Fx[i])) { dominated[i] = true; break; }
        }
    }
    for (int i = 0; i < N; ++i) {
        if (!dominated[i]) {
            result.paretoX.push_back(X[i]);
            result.paretoF.push_back(Fx[i]);
        }
    }
    result.evals = static_cast<long long>(N) * (1 + G);
    result.generations = G;
    return result;
}

} // namespace optimsolution
