#include "mopso.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace optimsolution {

namespace {

struct ArchiveMember {
    Vec x;
    Vec f;
    double crowd = 0.0;
};

bool dominates(const Vec& a, const Vec& b) {
    bool better = false;
    for (size_t k = 0; k < a.size(); ++k) {
        if (a[k] > b[k]) return false;
        if (a[k] < b[k]) better = true;
    }
    return better;
}

// Crowding distance over an archive (assumes all members are mutually
// non-dominated already), same convention as NSGA-II's crowding operator.
void assignCrowding(std::vector<ArchiveMember>& arc) {
    const int n = static_cast<int>(arc.size());
    for (auto& m : arc) m.crowd = 0.0;
    if (n <= 2) {
        for (auto& m : arc) m.crowd = std::numeric_limits<double>::infinity();
        return;
    }
    const int M = static_cast<int>(arc[0].f.size());
    std::vector<int> order(n);
    for (int k = 0; k < M; ++k) {
        for (int i = 0; i < n; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return arc[a].f[k] < arc[b].f[k]; });
        arc[order.front()].crowd = std::numeric_limits<double>::infinity();
        arc[order.back()].crowd = std::numeric_limits<double>::infinity();
        const double fmin = arc[order.front()].f[k];
        const double fmax = arc[order.back()].f[k];
        const double range = fmax - fmin;
        if (range <= 1e-300) continue;
        for (int i = 1; i < n - 1; ++i) {
            if (!std::isfinite(arc[order[i]].crowd)) continue;
            arc[order[i]].crowd += (arc[order[i + 1]].f[k] - arc[order[i - 1]].f[k]) / range;
        }
    }
}

// Insert a candidate into the archive if it is not dominated by any current
// member; remove any archive members the candidate dominates.
void insertIntoArchive(std::vector<ArchiveMember>& arc, const Vec& x, const Vec& f) {
    for (const auto& m : arc) {
        if (dominates(m.f, f)) return; // candidate is dominated; discard
    }
    arc.erase(std::remove_if(arc.begin(), arc.end(),
                              [&](const ArchiveMember& m) { return dominates(f, m.f); }),
              arc.end());
    // Avoid inserting an exact duplicate of an existing member.
    for (const auto& m : arc) {
        if (m.f == f) return;
    }
    ArchiveMember nm;
    nm.x = x;
    nm.f = f;
    arc.push_back(std::move(nm));
}

void pruneArchive(std::vector<ArchiveMember>& arc, int maxSize, std::mt19937_64& rng) {
    while (static_cast<int>(arc.size()) > maxSize) {
        assignCrowding(arc);
        // Remove the most crowded (smallest crowding distance) member;
        // break ties randomly among the most-crowded candidates.
        double worst = std::numeric_limits<double>::infinity();
        std::vector<int> worstIdx;
        for (int i = 0; i < static_cast<int>(arc.size()); ++i) {
            if (arc[i].crowd < worst) { worst = arc[i].crowd; worstIdx = {i}; }
            else if (arc[i].crowd == worst) { worstIdx.push_back(i); }
        }
        std::uniform_int_distribution<int> ui(0, static_cast<int>(worstIdx.size()) - 1);
        const int rem = worstIdx[ui(rng)];
        arc.erase(arc.begin() + rem);
    }
}

} // namespace

MOOResult MOPSO::run() {
    MOOResult result;
    if (!prob_) return result;

    const int D = prob_->dimension();
    const Vec& lb = prob_->lb();
    const Vec& ub = prob_->ub();
    const int N = std::max(4, population_);
    const int G = std::max(1, generations_);

    const double w    = param("inertia_w", 0.4);
    const double c1    = param("c1", 1.5);
    const double c2    = param("c2", 1.5);
    const int archiveCap = static_cast<int>(param("archive_size", double(N)));
    const double pMut   = param("mutation_prob", 0.05);

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // ---- Swarm initialization ----
    std::vector<Vec> X(N, Vec(D)), V(N, Vec(D, 0.0));
    std::vector<Vec> Fx(N);
    std::vector<Vec> pbestX(N);
    std::vector<Vec> pbestF(N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            std::uniform_real_distribution<double> ud(lb[j], ub[j]);
            X[i][j] = ud(rng_);
            const double range = ub[j] - lb[j];
            std::uniform_real_distribution<double> vd(-std::fabs(range) * 0.1, std::fabs(range) * 0.1);
            V[i][j] = vd(rng_);
        }
        Fx[i] = prob_->evaluateMulti(X[i]);
        pbestX[i] = X[i];
        pbestF[i] = Fx[i];
    }

    std::vector<ArchiveMember> archive;
    for (int i = 0; i < N; ++i) insertIntoArchive(archive, X[i], Fx[i]);
    pruneArchive(archive, archiveCap, rng_);
    assignCrowding(archive);

    auto pickLeader = [&]() -> const Vec& {
        if (archive.empty()) return X[std::uniform_int_distribution<int>(0, N - 1)(rng_)];
        // Binary tournament favoring higher crowding distance (more diverse regions).
        std::uniform_int_distribution<int> ua(0, static_cast<int>(archive.size()) - 1);
        const int i1 = ua(rng_), i2 = ua(rng_);
        return (archive[i1].crowd >= archive[i2].crowd) ? archive[i1].x : archive[i2].x;
    };

    auto clampToBounds = [&](Vec& x) {
        for (int j = 0; j < D; ++j) {
            if (x[j] < lb[j]) x[j] = lb[j];
            if (x[j] > ub[j]) x[j] = ub[j];
        }
    };

    for (int gen = 0; gen < G; ++gen) {
        for (int i = 0; i < N; ++i) {
            const Vec& leader = pickLeader();
            for (int j = 0; j < D; ++j) {
                const double r1 = U01(rng_), r2 = U01(rng_);
                V[i][j] = w * V[i][j] + c1 * r1 * (pbestX[i][j] - X[i][j]) + c2 * r2 * (leader[j] - X[i][j]);
                X[i][j] += V[i][j];
            }
            // Light re-randomization mutation (diversity preservation, common
            // in MOPSO variants to counter premature convergence).
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < pMut) {
                    std::uniform_real_distribution<double> ud(lb[j], ub[j]);
                    X[i][j] = ud(rng_);
                    V[i][j] = 0.0;
                }
            }
            clampToBounds(X[i]);

            Fx[i] = prob_->evaluateMulti(X[i]);

            if (dominates(Fx[i], pbestF[i])) {
                pbestX[i] = X[i];
                pbestF[i] = Fx[i];
            } else if (!dominates(pbestF[i], Fx[i]) && U01(rng_) < 0.5) {
                // Mutually non-dominated: keep the newer one about half the time.
                pbestX[i] = X[i];
                pbestF[i] = Fx[i];
            }

            insertIntoArchive(archive, X[i], Fx[i]);
        }
        pruneArchive(archive, archiveCap, rng_);
        assignCrowding(archive);
    }

    for (const auto& m : archive) {
        result.paretoX.push_back(m.x);
        result.paretoF.push_back(m.f);
    }
    result.evals = static_cast<long long>(N) * (1 + G);
    result.generations = G;
    return result;
}

} // namespace optimsolution
