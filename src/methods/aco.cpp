#include "aco.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>

namespace optimsolution {

// ---------------------------------------------------------------------------
// ensureBounds
// ---------------------------------------------------------------------------
void ACO::ensureBounds(std::vector<double>& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// ---------------------------------------------------------------------------
// valueAtLevel  –  linearly spaced grid on [lb, ub]
// ---------------------------------------------------------------------------
double ACO::valueAtLevel(int j, int l) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double lb = L[j], ub = U[j];
    if (!std::isfinite(lb)) lb = -1.0;
    if (!std::isfinite(ub)) ub =  1.0;
    if (levels_ <= 1) return 0.5 * (lb + ub);
    double t = (double)l / (double)(levels_ - 1);
    return lb + t * (ub - lb);
}

// ---------------------------------------------------------------------------
// sampleLevel
//
// FIX #2: Removed gbest-biased heuristic (eta = 1/(1+|v-gbest_j|)).
//         That heuristic collapsed exploration progressively toward gbest,
//         making pheromone updates redundant and destroying diversity on
//         multimodal landscapes.
//
//         Replacement: eta = 1  (uniform heuristic).
//         Pheromone alone now drives selection, which is the canonical ACO
//         behaviour for discrete/grid-based construction.  If a problem-
//         specific heuristic is available it can be re-introduced here
//         without coupling to gbest.
// ---------------------------------------------------------------------------
int ACO::sampleLevel(int j) {
    std::vector<double> w(levels_, 0.0);
    double sumw = 0.0;
    for (int l = 0; l < levels_; ++l) {
        // eta = 1  →  weight = tau^alpha  (heuristic exponent beta not applied)
        double ww = std::pow(std::max(tau_[j][l], 0.0), alpha_);
        w[l]  = ww;
        sumw += ww;
    }
    if (sumw <= 0.0) {
        // fallback: uniform
        std::uniform_int_distribution<int> Ui(0, levels_ - 1);
        return Ui(rng_);
    }
    std::uniform_real_distribution<double> U01(0.0, sumw);
    double r   = U01(rng_);
    double acc = 0.0;
    for (int l = 0; l < levels_; ++l) {
        acc += w[l];
        if (r <= acc) return l;
    }
    return levels_ - 1;
}

// ---------------------------------------------------------------------------
// evaporate
// ---------------------------------------------------------------------------
void ACO::evaporate() {
    for (auto& row : tau_) {
        for (double& t : row) {
            t *= (1.0 - rho_);
            if (t < tau_min_) t = tau_min_;
        }
    }
}

// ---------------------------------------------------------------------------
// deposit
//
// FIX #1: Replaced  amount = Q / (f - fmin + eps)  with rank-based deposit.
//
//         The original formula caused a catastrophic overflow for the best
//         ant (rank 0), where  f - fmin == 0  →  amount = Q/eps = Q/1e-32.
//         Even clamped by tau_max this created extreme imbalance:  the best
//         ant saturated its pheromone cells in one step while all others
//         contributed nearly nothing.
//
//         Rank-based rule:  amount_r = Q / (r + 1)
//           r=0 (best)  →  Q / 1
//           r=1         →  Q / 2
//           r=2         →  Q / 3  ...
//         This gives a smooth, stable, and well-known weighting (MMAS-like).
//
// FIX #5 (implicit): Rank-based scaling automatically differentiates between
//         ants even when their f-values are very close.
// ---------------------------------------------------------------------------
void ACO::deposit(const std::vector<int>& order) {
    int k = std::min(deposit_top_, (int)order.size());
    const int D = prob_->dimension();

    for (int r = 0; r < k; ++r) {
        int idx    = order[r];
        const auto& x = X_[idx];
        double amount  = Q_ / (double)(r + 1);   // rank-based, always finite & positive

        for (int j = 0; j < D; ++j) {
            // find nearest grid level to x[j]
            int    best_l = 0;
            double best_d = std::numeric_limits<double>::infinity();
            for (int l = 0; l < levels_; ++l) {
                double d = std::fabs(valueAtLevel(j, l) - x[j]);
                if (d < best_d) { best_d = d; best_l = l; }
            }
            tau_[j][best_l] += amount;
            if (tau_[j][best_l] > tau_max_) tau_[j][best_l] = tau_max_;
        }
    }
}

// ---------------------------------------------------------------------------
// init
//
// FIX #4: Removed the wasteful double-initialisation.
//         Original code ran initSampler.samplePopulation() (spending up to
//         max(pop_,8) evaluations) purely to seed best_x_, then discarded
//         all those solutions and ran another full constructive pass for
//         pop_ ants.  The first batch was never stored in X_/FX_.
//
//         Fix: a single pass that:
//           (a) samples pop_ solutions via the configured Initializer,
//           (b) stores them directly in X_/FX_,
//           (c) seeds best_x_ from the best of that batch.
//         No evaluation budget is wasted.
// ---------------------------------------------------------------------------
void ACO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    // initialise pheromone matrix
    tau_.assign(D, std::vector<double>(std::max(1, levels_), tau0_));

    X_.assign (pop_, std::vector<double>(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    // Single initialisation pass: use Initializer to get pop_ diverse seeds,
    // store them in X_/FX_, and record the global best.
    {
        Initializer initSampler;
        initSampler.configure(initopt_);
        auto X0 = initSampler.samplePopulation(*prob_, rng_, pop_);

        for (int i = 0; i < pop_; ++i) {
            auto& x = X0[i];
            ensureBounds(x);
            double f = eval(x);
            X_[i]  = std::move(x);
            FX_[i] = f;
            if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
            if (prob_->calls() >= max_evals_) break;
        }
    }

    // Deposit pheromone from the initial population (seeds the pheromone matrix
    // with meaningful signal before the first proper iteration).
    {
        std::vector<int> order(pop_);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b){ return FX_[a] < FX_[b]; });
        deposit(order);
    }

    printBest();
    updateStop(FX_);
}

// ---------------------------------------------------------------------------
// one_iteration
//
// FIX #3: Replaced elitist per-slot replacement with full replacement.
//
//         Original code kept old solutions in X_/FX_ whenever a new ant was
//         worse, then deposited pheromone based on those stale entries.  This
//         caused:
//           - Pheromone to reinforce positions that were never actually
//             constructed in the current iteration.
//           - Permanent "ghost" ants that block fresh exploration of the grid.
//
//         Fix: X_/FX_ are always fully overwritten with the new ants each
//         iteration.  The global best (best_x_, best_f_) is maintained
//         separately via elitism — it is never lost.
// ---------------------------------------------------------------------------
void ACO::one_iteration() {
    if (!prob_) return;
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Construct a completely new population each iteration (full replacement).
    for (int i = 0; i < pop_; ++i) {
        std::vector<double> x(D, 0.0);
        for (int j = 0; j < D; ++j) {
            int l  = sampleLevel(j);   // pheromone-only roulette (no gbest bias)
            x[j]   = valueAtLevel(j, l);
        }
        ensureBounds(x);
        double f = eval(x);

        // optional in-run local search
        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_) {
            auto [xl, fl] = localSearch(local_method_, x);
            if (fl < f) { x = std::move(xl); f = fl; }
        }

        // Always overwrite slot i (full replacement)
        X_[i]  = std::move(x);
        FX_[i] = f;

        // Update global best
        if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }

        if (prob_->calls() >= max_evals_) break;
    }

    // Pheromone update
    evaporate();

    std::vector<int> order(pop_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return FX_[a] < FX_[b]; });
    deposit(order);

    printBest();
    updateStop(FX_);
}

// ---------------------------------------------------------------------------
// end
//
// FIX #6: Added updateStop() after the local refinement so that final
//         statistics are consistent with the rest of the methods (DE, GA, …).
// ---------------------------------------------------------------------------
void ACO::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // Replace worst slot with the polished best (consistency with DE/GA)
    if (!X_.empty() && !FX_.empty()) {
        size_t worst = 0;
        double fw    = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > fw) { fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    printBest();
    updateStop(FX_);   // FIX #6: was missing in original
}

} // namespace optimsolution
