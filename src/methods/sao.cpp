#include "sao.h"
#include "init.h"   // Only in the .cpp to avoid include cycles.

#include <algorithm>
#include <numeric>
#include <cmath>

namespace optimsolution {

void SAO::configure(const MethodConfig& mc){
    // Per-method population (overrides global for correct reporting).
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) Optimizer::setPopulation(pop_cfg_); // Ensures correct reporting before the run.

    // locals
    local_method_      = mc.getStr("local_method",  local_method_);
    local_rate_        = mc.getDbl("local_rate",    local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    end_local_refine_  = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_  = mc.getStr("end_local_method",  end_local_method_);

    // SAO params
    sniff_w_           = mc.getDbl("sniff_w",   sniff_w_);
    sniff_a1_          = mc.getDbl("sniff_a1",  sniff_a1_);
    sniff_a2_          = mc.getDbl("sniff_a2",  sniff_a2_);

    trail_sigma0_      = mc.getDbl("trail_sigma0", trail_sigma0_);
    trail_decay_       = mc.getDbl("trail_decay",  trail_decay_);

    rand_rate_         = mc.getDbl("rand_rate",  rand_rate_);
    rand_scale_        = mc.getDbl("rand_scale", rand_scale_);

    // Optional: same names as the stop rule for compatibility (not required in cfg).
    bss_eps_           = mc.getDbl("eps", bss_eps_);
    bss_sim_           = mc.getInt("sim", bss_sim_);
}

void SAO::init(){
    if (!prob_) return;

    // Applies the per-method population override again (safety).
    if (pop_cfg_ > 0) Optimizer::setPopulation(pop_cfg_);
    final_population_ = std::max(4, Optimizer::population());

    const int D = prob_->dimension();

    // Initializes via Initializer.
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, final_population_);

    V_.assign(final_population_, Vec(D, 0.0));
    fX_.assign(final_population_, std::numeric_limits<double>::infinity());

    rng_local_.seed( 0xB5297A4DUL ^ (uint64_t)rng_() );

    best_f_  = std::numeric_limits<double>::infinity();
    worst_f_ = -std::numeric_limits<double>::infinity();

    for (int i=0;i<final_population_; ++i){
        ensureBounds(X_[i]);
        fX_[i] = eval(X_[i]);
        if (fX_[i] < best_f_) { best_f_ = fX_[i]; best_x_ = X_[i]; }
        if (fX_[i] > worst_f_){ worst_f_ = fX_[i]; worst_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    trail_sigma_k_    = trail_sigma0_;
    K_                = 0;
    stopped_          = false;
    last_best_f_      = best_f_;
    same_best_iters_  = 0;

    updateStop(fX_);    // For the general report.
    printBest();
}

inline void SAO::ensureBounds(Vec& v){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0;j<v.size(); ++j){
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j]+U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

// ---------------- Sniffing ----------------
void SAO::sniffing_(){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i=0;i<final_population_; ++i){
        for (int j=0;j<D; ++j){
            const double r1 = U01_(rng_local_);
            const double r2 = U01_(rng_local_);
            V_[i][j] = sniff_w_ * V_[i][j]
                     + sniff_a1_ * r1 * (best_x_[j]  - X_[i][j])
                     - sniff_a2_ * r2 * (X_[i][j]    - worst_x_[j]);
            X_[i][j] += V_[i][j];
            if (X_[i][j] < L[j]) X_[i][j] = L[j];
            if (X_[i][j] > U[j]) X_[i][j] = U[j];
        }
        evaluate_and_update_(i);
        if (prob_->calls() >= max_evals_) return;
    }
}

// --------------- Trailing -----------------
bool SAO::trailing_(){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    bool improved = false;
    for (int i=0;i<final_population_; ++i){
        Vec xi = X_[i];
        for (int j=0;j<D; ++j){
            const double scale = trail_sigma_k_ * (std::abs(best_x_[j] - xi[j]) + 1e-16);
            const double step  = scale * N01_(rng_local_);
            xi[j] = xi[j] + step;
            if (xi[j] < L[j]) xi[j] = L[j];
            if (xi[j] > U[j]) xi[j] = U[j];
        }
        const double f = eval(xi);
        if (f <= fX_[i]){
            X_[i]  = std::move(xi);
            fX_[i] = f;
            if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; improved = true; }
        }
        if (prob_->calls() >= max_evals_) break;
    }
    trail_sigma_k_ = std::max(1e-12, trail_sigma_k_ * trail_decay_);
    recompute_worst_();
    return improved;
}

// ---------------- Random ------------------
void SAO::random_(){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i=0;i<final_population_; ++i){
        Vec xi = X_[i];
        for (int j=0;j<D; ++j){
            if (U01_(rng_local_) < rand_rate_){
                const double range = (U[j]-L[j]);
                const double step  = (U01_(rng_local_) - 0.5) * 2.0 * rand_scale_ * range;
                xi[j] = xi[j] + step;
                if (xi[j] < L[j]) xi[j] = L[j];
                if (xi[j] > U[j]) xi[j] = U[j];
            }
        }
        const double f = eval(xi);
        if (f <= fX_[i]){
            X_[i]  = std::move(xi);
            fX_[i] = f;
            if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
        }
        if (prob_->calls() >= max_evals_) break;
    }
    recompute_worst_();
}

void SAO::evaluate_and_update_(int i){
    const double f = eval(X_[i]);
    fX_[i] = f;
    if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
    if (f > worst_f_){ worst_f_ = f; worst_x_ = X_[i]; }
}

void SAO::recompute_worst_(){
    worst_f_ = -std::numeric_limits<double>::infinity();
    for (int i=0;i<final_population_; ++i){
        if (fX_[i] > worst_f_){ worst_f_ = fX_[i]; worst_x_ = X_[i]; }
    }
}

// ---------- simple BSS update ----------
void SAO::update_simple_bss_(){
    if (!std::isfinite(last_best_f_)) {
        last_best_f_ = best_f_;
        same_best_iters_ = 1;
        return;
    }
    const double diff = std::fabs(best_f_ - last_best_f_);
    if (diff <= bss_eps_){
        ++same_best_iters_;
    } else {
        last_best_f_ = best_f_;
        same_best_iters_ = 1;
    }
    if (same_best_iters_ >= bss_sim_){
        stopped_ = true;
        // Comment translated from Greek.
        // Comment translated from Greek.
        if (prob_) {
            // Comment translated from Greek.
            max_evals_ = prob_->calls();
        }
    }
}

// --------------- Driver iteration ----------
void SAO::one_iteration(){
    if (!prob_) return;

    // Comment translated from Greek.
    if (stopped_) {
        return;
    }

    // 1) Sniffing
    sniffing_();
    if (stopped_ || prob_->calls() >= max_evals_) { return; }

    // 2) Trailing
    bool improved = trailing_();
    if (stopped_ || prob_->calls() >= max_evals_) { return; }

    // 3) Random (only as a third stage).
    if (!improved){
        random_();
        if (stopped_ || prob_->calls() >= max_evals_) { return; }
    }

    // 4) Optional in-run local search.
    if (!stopped_ && local_rate_ > 0.0 && !local_method_.empty() && U01_(rng_local_) < local_rate_){
        auto [xloc, floc] = localSearch(local_method_, best_x_);
        if (!xloc.empty() && floc < best_f_){
            best_x_ = std::move(xloc);
            best_f_ = floc;
        }
    }

    // Comment translated from Greek.
    update_simple_bss_();

    // If not stopped, performs the standard reporting for this iteration.
    if (!stopped_){
        updateStop(fX_);
        printBest();
        ++K_;
    }
}

// --------------- Finalize ------------------
void SAO::end(){
    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()){
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (!xloc.empty() && floc < best_f_){
            best_x_ = std::move(xloc);
            best_f_ = floc;
        }
        // Replaces the worst with the best for consistency.
        if (!X_.empty()){
            int worst = 0; double fw = fX_[0];
            for (int i=1;i<final_population_; ++i){ if (fX_[i] > fw){ fw = fX_[i]; worst = i; } }
            X_[worst]  = best_x_;
            fX_[worst] = best_f_;
        }
        // Comment translated from Greek.
        printBest();
    }
    updateStop(fX_);
}

} // namespace optimsolution
