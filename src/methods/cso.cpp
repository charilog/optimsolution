#include "cso.h"
#include "init.h"
#include <numeric>
#include <limits>

namespace optimsolution {

void CSO::configure(const MethodConfig& mc) {
    int pop_override = mc.getInt("population", pop_);
    if (pop_override > 3) pop_ = pop_override;

    phi_cfg_      = mc.getDbl("phi", phi_cfg_);
    phi_default_  = mc.getDbl("phi_default", phi_default_);
    v_init_frac_  = mc.getDbl("v_init_frac", v_init_frac_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

double CSO::safeEval(const Vec& x) {
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

void CSO::ensureBounds(Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void CSO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (pop_ < 4) pop_ = 4;
    // Pairing needs an even swarm size; bump by one if necessary (a single
    // leftover particle would otherwise sit out every generation).
    if (pop_ % 2 != 0) ++pop_;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    X_.assign(pop_, Vec(D, 0.0));
    V_.assign(pop_, Vec(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i = 0; i < pop_; ++i) {
        for (int j = 0; j < D; ++j) {
            const double lo = L[j], hi = U[j];
            const double range = (std::isfinite(lo) && std::isfinite(hi)) ? (hi - lo) : 20.0;
            const double base_lo = std::isfinite(lo) ? lo : -10.0;
            X_[i][j] = base_lo + U01(rng_) * range;
            // Small initial velocity (fraction of the box range), matching
            // the paper's guidance that CSO does not rely on large initial
            // exploratory velocities the way some PSO variants do -- the
            // competitive updates themselves generate the exploration.
            V_[i][j] = (U01(rng_) * 2.0 - 1.0) * v_init_frac_ * range;
        }
        FX_[i] = safeEval(X_[i]);
    }

    phi_ = (phi_cfg_ > 0.0) ? phi_cfg_ : phi_default_;

    Vec fx(FX_);
    updateStop(fx);
    printBest();
}

void CSO::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const int N = pop_;

    // Swarm mean, computed once per generation from the CURRENT population
    // (before any of this generation's updates).
    Vec xbar(D, 0.0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < D; ++j)
            xbar[j] += X_[i][j];
    for (int j = 0; j < D; ++j) xbar[j] /= (double)N;

    // Random pairing: shuffle particle indices, pair consecutive entries.
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng_);

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int p = 0; p + 1 < N; p += 2) {
        const int a = order[p];
        const int b = order[p + 1];

        // Winner (lower fitness, since minimizing) carries over UNCHANGED;
        // only the loser is updated and re-evaluated.
        const int winner = (FX_[a] <= FX_[b]) ? a : b;
        const int loser  = (winner == a) ? b : a;

        const Vec& xw = X_[winner];
        Vec& xl = X_[loser];
        Vec& vl = V_[loser];

        for (int j = 0; j < D; ++j) {
            const double r1 = U01(rng_), r2 = U01(rng_), r3 = U01(rng_);
            vl[j] = r1 * vl[j]
                  + r2 * (xw[j] - xl[j])
                  + phi_ * r3 * (xbar[j] - xl[j]);
            xl[j] = xl[j] + vl[j];
        }

        ensureBounds(xl);
        FX_[loser] = safeEval(xl);
        // Winner's X_[winner]/V_[winner] are left exactly as they were --
        // no update, no re-evaluation (this is what keeps a CSO
        // "generation" at N/2 evaluations regardless of D).
    }

    // Optional in-run local search after a successful global-best improvement.
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;

                // Inject the refinement back into the swarm by replacing
                // the current worst individual (same convention used by
                // the DE/PSO-family methods in this framework).
                int worst = 0;
                for (int i = 1; i < N; ++i) if (FX_[i] > FX_[worst]) worst = i;
                X_[worst] = best_x_;
                FX_[worst] = best_f_;
            }
        }
    }

    Vec fx(FX_);
    printBest();
    updateStop(fx);
}

void CSO::end() {
    if (!end_local_refine_)        return;
    if (!prob_)                    return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    double floc      = refinement.second;

    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }
    printBest();
}

} // namespace optimsolution
