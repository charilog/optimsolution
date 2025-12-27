#include "bho.h"
#include "init.h"   // Same pattern as GA for population initialization
#include <numeric>

namespace optimsolution {

// ---------------- configure([bho]) ----------------
void BHO::configure(const MethodConfig& mc)
{
    // population override (as in GA)
    int pop_override = mc.getInt("population", pop_);
    if (pop_override > 3) pop_ = pop_override;

    // native keys
    heal_prob           = mc.getDbl("heal_prob",           heal_prob);
    heal_rate           = mc.getDbl("heal_rate",           heal_rate);
    wound_strength_init = mc.getDbl("wound_strength_init", wound_strength_init);
    stagnation_kick     = mc.getInt("stagnation_kick",     stagnation_kick);
    stagnation_restart  = mc.getInt("stagnation_restart",  stagnation_restart);
    elite_kick_sigma    = mc.getDbl("elite_kick_sigma",    elite_kick_sigma);
    restart_frac        = mc.getDbl("restart_frac",        restart_frac);
    print_stride        = mc.getInt("print_stride",        print_stride);

    // in-run local
    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c: local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0; if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    // ---- Aliases used in the .cfg ----
    // soft stagnation threshold
    stagnation_kick     = mc.getInt("stagn_thr_soft", stagnation_kick);
    // catastrophic reset trigger
    stagnation_restart  = mc.getInt("cat_reset_thr",  stagnation_restart);
    // restart fraction
    restart_frac        = mc.getDbl("cat_reset_frac", restart_frac);
    // sigma for kick/reset
    elite_kick_sigma    = mc.getDbl("cat_sigma",       elite_kick_sigma);
    // initial intensity for wounding/strike
    wound_strength_init = mc.getDbl("alpha_strike_scale", wound_strength_init);

    // heal reinforcement
    double stagn_boost  = mc.getDbl("stagn_boost", 1.0);
    if (std::isfinite(stagn_boost) && stagn_boost > 0.0) {
        heal_rate = std::max(0.0, heal_rate * stagn_boost);
    }
    // enable/disable catastrophic reset
    std::string cat_reset = mc.getStr("cat_reset", "");
    for (auto& c: cat_reset) c = (char)std::tolower((unsigned char)c);
    if (!cat_reset.empty()) {
        if (cat_reset=="no" || cat_reset=="false" || cat_reset=="0")
            stagnation_restart = std::numeric_limits<int>::max()/4;
    }
}

// --- Safe evaluation: never returns inf/NaN ---
double BHO::eval_safe(const Vec& x)
{
    double v = prob_->evaluate(x);
    if (!std::isfinite(v)) v = 1e100; // large finite penalty
    return v;
}

BHO::Vec BHO::clamp_to_bounds(const Vec& x) const
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    Vec y = x;
    for (int j = 0; j < D_; ++j) {
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (y[j] < lo) y[j] = lo;
        if (y[j] > hi) y[j] = hi;
    }
    return y;
}

void BHO::seed_midpoint(Vec& out) const
{
    out.assign(D_, 0.0);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int j = 0; j < D_; ++j) {
        const double lo = (j < (int)L.size() ? L[j] : -1.0);
        const double hi = (j < (int)U.size() ? U[j] :  1.0);
        out[j] = 0.5 * (lo + hi);
    }
}

void BHO::seed_uniform()
{
    // Same logic as GA: Initializer + initopt_ from [global]
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N_);

    FX_.assign(N_, std::numeric_limits<double>::infinity());
    for (int i = 0; i < N_; ++i) {
        FX_[i] = eval_safe(X_[i]);
    }
}

void BHO::ensure_finite_best()
{
    if (std::isfinite(best_f_)) return;
    Vec mid;
    seed_midpoint(mid);
    const double v = eval_safe(mid);
    best_f_ = v;
    best_x_ = mid;
}

void BHO::elite_gaussian_kick()
{
    for (int j = 0; j < D_; ++j) best_x_[j] += elite_kick_sigma * N01_(rng_);
    best_x_ = clamp_to_bounds(best_x_);
    const double fv = eval_safe(best_x_);
    if (fv < best_f_) best_f_ = fv;
}

void BHO::soft_kick_population()
{
    int elite = 0;
    double fbest = FX_[0];
    for (int i = 1; i < N_; ++i) if (FX_[i] < fbest) { fbest = FX_[i]; elite = i; }

    for (int i = 0; i < N_; ++i) {
        if (i == elite) continue;
        Vec y = X_[i];
        for (int j = 0; j < D_; ++j) y[j] += elite_kick_sigma * N01_(rng_);
        y = clamp_to_bounds(y);
        const double fy = eval_safe(y);
        if (fy < FX_[i]) {
            archive_.push_back(X_[i]);
            X_[i]  = y;
            FX_[i] = fy;
            if (fy < best_f_) {
                best_f_ = fy;
                best_x_ = y;
                sinceBest_ = 0;
            }
        }
        if (prob_->calls() >= max_evals_) return;
    }
    sinceBest_ = 0;
}

void BHO::restart_partial()
{
    int n_resample = std::max(1, int(std::round(restart_frac * N_)));
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    int elite = 0;
    double fbest = FX_[0];
    for (int i = 1; i < N_; ++i) if (FX_[i] < fbest) { fbest = FX_[i]; elite = i; }

    for (int k = 0; k < n_resample; ++k) {
        int i = int(U01_(rng_) * N_);
        if (i == elite) i = (i + 1) % N_;
        for (int j = 0; j < D_; ++j) {
            const double lo = (j < (int)L.size() ? L[j] : -1.0);
            const double hi = (j < (int)U.size() ? U[j] :  1.0);
            X_[i][j] = lo + U01_(rng_) * (hi - lo);
        }
        const double fy = eval_safe(X_[i]);
        archive_.push_back(X_[i]);
        FX_[i] = fy;
        if (fy < best_f_) {
            best_f_ = fy;
            best_x_ = X_[i];
            sinceBest_ = 0;
        }
        if (prob_->calls() >= max_evals_) return;
    }
}

// ---------------- core lifecycle ----------------
void BHO::init()
{
    if (!prob_) return;

    D_ = prob_->dimension();

    // N_ from the base pop_ (pop_ can be overridden from [bho] in configure)
    N_ = std::max(4, population());

    iters_ = 0;
    sinceBest_ = 0;
    archive_.clear();

    seed_uniform();

    int elite = 0;
    double fbest = FX_[0];
    for (int i = 1; i < N_; ++i) if (FX_[i] < fbest) { fbest = FX_[i]; elite = i; }
    best_f_ = fbest;
    best_x_ = X_[elite];

    ensure_finite_best();

    printBest();
}

void BHO::one_iteration()
{
    if (!prob_) return;
    ++iters_;

    int elite = 0;
    double fbest = FX_[0];
    for (int i = 1; i < N_; ++i) if (FX_[i] < fbest) { fbest = FX_[i]; elite = i; }

    bool improved = false;
    if (fbest < best_f_) {
        best_f_ = fbest;
        best_x_ = X_[elite];
        sinceBest_ = 0;
        improved = true;
    } else {
        ++sinceBest_;
    }

    double tfrac = (max_iters_ > 0) ? std::min(1.0, double(iters_) / double(max_iters_)) : 0.0;
    double wound = std::max(0.05 * wound_strength_init, wound_strength_init * (1.0 - tfrac));

    std::uniform_real_distribution<double> U01(0.0,1.0);

    for (int i = 0; i < N_; ++i) {
        if (i == elite) continue;

        Vec y = X_[i];
        bool do_heal = (U01(rng_) < heal_prob);
        if (do_heal) {
            for (int j = 0; j < D_; ++j) {
                double step = heal_rate * (best_x_[j] - X_[i][j]) + elite_kick_sigma * N01_(rng_);
                y[j] = X_[i][j] + step;
            }
        } else {
            for (int j = 0; j < D_; ++j)
                y[j] = best_x_[j] + wound * N01_(rng_);
        }

        y = clamp_to_bounds(y);
        double fy = eval_safe(y);
        if (fy < FX_[i]) {
            // In-run local search, as in GA (after acceptance)
            if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, y);
                if (floc < fy) { y = std::move(xloc); fy = floc; }
            }

            archive_.push_back(X_[i]);
            X_[i]  = y;
            FX_[i] = fy;
            if (fy < best_f_) {
                best_f_ = fy;
                best_x_ = y;
                sinceBest_ = 0;
                improved = true;
            }
        }
        if (prob_->calls() >= max_evals_) return;
    }

    if (sinceBest_ > stagnation_kick) {
        elite_gaussian_kick();
        soft_kick_population();
    }
    if (sinceBest_ > stagnation_restart) {
        restart_partial();
        sinceBest_ = 0;
    }

    if (improved || (print_stride > 0 && (iters_ % std::max(1, print_stride) == 0))) {
        printBest();
    }

    // update the stop rules with the current fitness distribution (as in GA)
    updateStop(FX_);
}

void BHO::end()
{
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    if (!X_.empty() && !FX_.empty()){
        size_t worst = 0; double fw = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > fw){ fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
