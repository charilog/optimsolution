#include "sa.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>

namespace optimsolution {

void SA::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// FIX: not const, so RNG distributions can use non-const rng_
void SA::propose(int i, std::vector<double>& cand) {
    // Gaussian proposal around X_[i], per dimension
    const int D = (int)cand.size();
    std::normal_distribution<double> N01(0.0, 1.0);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int j=0; j<D; ++j){
        double span = (std::isfinite(L[j]) && std::isfinite(U[j])) ? (U[j]-L[j]) : 1.0;
        double sigma = (step_sigma_ > 0.0) ? step_sigma_ : (step_frac_ * span);
        if (i < (int)stepMul_.size()) sigma *= stepMul_[i];
        cand[j] = X_[i][j] + sigma * N01(rng_);
    }
}

void SA::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    // Initial sampling via Initializer
    Initializer initSampler;
    initSampler.configure(initopt_);

    X_  = initSampler.samplePopulation(*prob_, rng_, std::max(pop_, 1));
    FX_.assign((size_t)pop_, std::numeric_limits<double>::infinity());
    T_.assign((size_t)pop_, std::max(T0_, Tmin_));
    stepMul_.assign((size_t)pop_, 1.0);

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    // Evaluate initial agents
    for (int i=0; i<pop_; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        // A NaN objective value (e.g. the problem hits a singularity at a
        // legitimately in-bounds point) must never enter the Metropolis
        // arithmetic below: since NaN compares false to everything
        // (including itself), an agent whose CURRENT energy is NaN would
        // have dE = fnew - NaN = NaN forever afterwards, i.e. `dE <= 0.0`
        // is always false and the Metropolis acceptance probability
        // exp(-dE/T) is also NaN -- that agent would never accept another
        // candidate for the rest of the run, silently wasting its share of
        // the evaluation budget. Treating NaN as +Infinity keeps the
        // pre-existing (and already-correct) handling of genuine infeasible
        // penalties, and lets the agent recover on its very next proposal.
        if (std::isnan(FX_[i])) FX_[i] = std::numeric_limits<double>::infinity();
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) return;
    }

    // T0 <= 0: automatic initial temperature, relative to the objective's
    // scale (a fixed T0 = 1 is meaningless when |f| ~ 1e4): T0 is set so that
    // an average uphill move is accepted with probability 0.8, i.e.
    // T0 = -mean(dE+) / ln(0.8), from up to 20 trial proposals of agent 0.
    if (T0_ <= 0.0) {
        double sum = 0.0; int cnt = 0;
        for (int t = 0; t < 20 && prob_->calls() < max_evals_; ++t) {
            std::vector<double> cand = X_[0];
            propose(0, cand);
            ensureBounds(cand);
            const double fc = eval(cand);
            if (std::isfinite(fc) && fc < best_f_) { best_f_ = fc; best_x_ = cand; }
            const double dE = fc - FX_[0];
            if (std::isfinite(dE) && dE > 0.0) { sum += dE; ++cnt; }
        }
        const double T0 = (cnt > 0) ? -(sum / cnt) / std::log(0.8) : 1.0;
        T_.assign((size_t)pop_, std::max(T0, Tmin_));
    }

    printBest();
    updateStop(FX_);
}

void SA::one_iteration(){
    if (!prob_) return;
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // For each agent, perform "moves_" Metropolis steps
    for (int i=0; i<pop_; ++i){
        double Ti = T_[i];
        int accepted = 0, tried = 0;

        for (int m=0; m<moves_; ++m){
            std::vector<double> cand = X_[i];
            propose(i, cand);
            ensureBounds(cand);

            double fnew = eval(cand);
            // Same NaN-to-+Infinity normalization as in init() -- a NaN
            // fnew must never reach the dE/exp() arithmetic below, or it
            // (correctly) gets rejected via `dE <= 0.0` being false, but
            // ALSO poisons FX_[i] with NaN going forward if ever accepted,
            // which would then permanently disable acceptance for this
            // agent for the rest of the run (NaN compares false to
            // everything, so every future dE would be NaN too). Treated as
            // +Infinity, a NaN result is just "very bad" -- correctly
            // rejected here, and if FX_[i] itself is ever +Infinity, a
            // later finite proposal still correctly gets accepted.
            if (std::isnan(fnew)) fnew = std::numeric_limits<double>::infinity();

            // Metropolis criterion
            // Note: if both fnew and FX_[i] are +Infinity (current state
            // AND the new candidate both undefined/infeasible), a plain
            // `fnew - FX_[i]` is the indeterminate form Infinity-Infinity,
            // which is itself NaN in IEEE754 -- guarded explicitly here so
            // dE can never be NaN regardless of how fnew/FX_[i] end up.
            double dE = std::isinf(fnew) && std::isinf(FX_[i]) ? 0.0 : (fnew - FX_[i]);
            bool accept = false;
            if (dE <= 0.0) {
                accept = true;
            } else {
                double Tuse = std::max(Ti, 1e-300);
                double p = std::exp(-dE / Tuse);
                if (U01(rng_) < p) accept = true;
            }

            ++tried;
            if (accept) {
                ++accepted;
                X_[i] = std::move(cand);
                FX_[i] = fnew;

                // Optional in-run local immediately after acceptance
                if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
                    auto [xl, fl] = localSearch(local_method_, X_[i]);
                    if (fl < FX_[i]) { X_[i] = std::move(xl); FX_[i] = fl; }
                }

                if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
            }

            if (prob_->calls() >= max_evals_) break;
        }

        // FIX: step-size adaptation (Corana, Marchesi, Martini & Ridella 1987).
        // The proposal step used to stay at step_frac*(ub-lb) for the whole
        // run, so SA could never refine below that resolution (sphere-10D:
        // f ~ 1e2). The multiplier grows/shrinks to keep the acceptance
        // ratio of each block of 'moves' proposals within [0.4, 0.6].
        if (adapt_step_ && tried > 0 && i < (int)stepMul_.size()) {
            const double a = (double)accepted / (double)tried;
            double& s = stepMul_[i];
            if (a > 0.6)      s *= 1.0 + 2.0 * (a - 0.6) / 0.4;
            else if (a < 0.4) s /= 1.0 + 2.0 * (0.4 - a) / 0.4;
            const double smax = (step_frac_ > 0.0) ? 1.0 / step_frac_ : 1e6;
            if (s < 1e-15) s = 1e-15;
            if (s > smax)  s = smax;
        }

        // Geometric cooling
        Ti *= alpha_;
        if (Ti < Tmin_) Ti = Tmin_;
        T_[i] = Ti;

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void SA::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    printBest();
}

} // namespace optimsolution
