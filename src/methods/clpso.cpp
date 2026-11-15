#include "clpso.h"
#include <cctype>
#include "init.h"
#include <cmath>
#include <limits>

namespace optimsolution {

void CLPSO::configure(const MethodConfig& mc)
{
    // population = 0 (or unset): CLPSO paper default, resolved in init()
    pop_cfg_ = mc.getInt("population", 0);
    if (pop_cfg_ > 3) setPopulation(pop_cfg_);

    w_max_       = mc.getDbl("w_max", w_max_);
    w_min_       = mc.getDbl("w_min", w_min_);
    c_           = mc.getDbl("c", c_);
    refresh_gap_ = mc.getInt("refresh_gap", refresh_gap_);
    pc_min_      = mc.getDbl("pc_min", pc_min_);
    pc_max_      = mc.getDbl("pc_max", pc_max_);
    pc_exp_      = mc.getDbl("pc_exp", pc_exp_);
    vmax_frac_   = mc.getDbl("vmax_frac", vmax_frac_);
    clamp_pos_   = mc.getBool("clamp_pos", clamp_pos_);
    gbest_coef_  = mc.getDbl("gbest_coef", gbest_coef_);
    if (gbest_coef_ < 0.0) gbest_coef_ = 0.0;
    if (refresh_gap_ < 0) refresh_gap_ = 0;
    if (pc_min_ < 0.0) pc_min_ = 0.0;
    if (pc_max_ > 1.0) pc_max_ = 1.0;
    if (pc_max_ < pc_min_) pc_max_ = pc_min_;
    if (vmax_frac_ <= 0.0) vmax_frac_ = 0.2;

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

double CLPSO::evalAndUpdateBest(const VecD& x)
{
    const double f = prob_ ? prob_->evaluate(x) : std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

void CLPSO::clampVelocity(VecD& v) const
{
    const int D = (int)v.size();
    for (int j = 0; j < D; ++j) {
        if (std::isfinite(Vmax_[j])) {
            const double vmax = Vmax_[j];
            if (v[j] >  vmax) v[j] =  vmax;
            if (v[j] < -vmax) v[j] = -vmax;
        }
    }
}

void CLPSO::ensureBounds(VecD& x) const
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void CLPSO::handleBounds(VecD& x, VecD& v) const
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j])) {
            x[j] = 0.5 * (L[j] + U[j]);
            if (j < (int)v.size()) v[j] = 0.0;
            continue;
        }

        const double lj = L[j];
        const double uj = U[j];

        // Reflective boundary handling with damping to preserve swarm dynamics.
        if (x[j] < lj) {
            x[j] = lj + (lj - x[j]);
            if (j < (int)v.size()) v[j] = -0.5 * v[j];
        } else if (x[j] > uj) {
            x[j] = uj - (x[j] - uj);
            if (j < (int)v.size()) v[j] = -0.5 * v[j];
        }

        // Final clamp (handles pathological overshoots and degenerate bounds).
        if (x[j] < lj) x[j] = lj;
        if (x[j] > uj) x[j] = uj;
    }
}

void CLPSO::computePcSchedule()
{
    Pc_.assign(pop_, pc_min_);
    if (pop_ <= 1) {
        if (pop_ == 1) Pc_[0] = pc_min_;
        return;
    }

    // Standard CLPSO schedule (Liang et al., 2006) uses an exponential shaping across particle indices.
    // Particles are indexed 0..pop_-1; higher index => higher Pc.
    const double denom = std::exp(pc_exp_) - 1.0;
    for (int i = 0; i < pop_; ++i) {
        const double t = (pop_ == 1) ? 0.0 : (double)i / (double)(pop_ - 1);
        const double shaped = (std::exp(pc_exp_ * t) - 1.0) / (denom > 0.0 ? denom : 1.0);
        Pc_[i] = pc_min_ + (pc_max_ - pc_min_) * shaped;
        if (Pc_[i] < 0.0) Pc_[i] = 0.0;
        if (Pc_[i] > 1.0) Pc_[i] = 1.0;
    }
}

void CLPSO::selectExemplarsForParticle(int i, bool force_nonself)
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if ((int)exemplar_.size() != pop_) exemplar_.assign(pop_, std::vector<int>(D, 0));
    if ((int)exemplar_[i].size() != D) exemplar_[i].assign(D, 0);

    if (pop_ <= 1) {
        std::fill(exemplar_[i].begin(), exemplar_[i].end(), 0);
        return;
    }

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Ui(0, pop_ - 1);

    int nonself_dims = 0;

    for (int j = 0; j < D; ++j) {
        // With probability (1 - Pc[i]) learn from itself (i.e., pbest of i).
        if (U01(rng_) > Pc_[i]) {
            exemplar_[i][j] = i;
            continue;
        }

        // Tournament selection between two candidates (excluding i) using pbest fitness.
        int a = Ui(rng_);
        int b = Ui(rng_);
        int guard = 0;
        while (a == i && guard++ < 10) a = Ui(rng_);
        guard = 0;
        while ((b == i || b == a) && guard++ < 10) b = Ui(rng_);
        if (a == i) a = (i + 1) % pop_;
        if (b == i) b = (i + 2) % pop_;

        int winner = a;
        if (Fpbest_[b] < Fpbest_[a]) winner = b;
        exemplar_[i][j] = winner;
        if (winner != i) ++nonself_dims;
    }

    // A well-known pitfall: if all dimensions select self, the particle can stagnate.
    // Enforce at least one non-self exemplar dimension when requested and when learning is enabled.
    if (force_nonself && D > 0 && Pc_[i] > 0.0 && nonself_dims == 0) {
        std::uniform_int_distribution<int> Udim(0, D - 1);
        const int j = Udim(rng_);

        std::uniform_int_distribution<int> Uother(0, pop_ - 2);
        int other = Uother(rng_);
        if (other >= i) ++other; // skip i
        exemplar_[i][j] = other;
    }
}

void CLPSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    // population = 0: Liang et al. (2006) use 10 particles for 10-D and 40 for
    // 30-D problems (it used to fall back to the [global] population, 100,
    // which leaves CLPSO only ~200 generations at 10,000*D evaluations).
    if (pop_cfg_ <= 3) setPopulation(D <= 10 ? 10 : 40);

    X_.clear();
    V_.clear();
    FX_.clear();
    Pbest_.clear();
    Fpbest_.clear();
    Vmax_.clear();
    Pc_.clear();
    no_improve_.clear();
    exemplar_.clear();

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    // Initialize population (same pattern as PSO)
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);

    V_.assign(pop_, VecD(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    Pbest_.assign(pop_, VecD(D, 0.0));
    Fpbest_.assign(pop_, std::numeric_limits<double>::infinity());

    // Velocity bounds per dimension
    Vmax_.assign(D, 0.0);
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (int j = 0; j < D; ++j) {
        const double span = U[j] - L[j];
        Vmax_[j] = std::isfinite(span) ? std::fabs(span) * vmax_frac_ : std::numeric_limits<double>::infinity();
        if (Vmax_[j] <= 0.0) Vmax_[j] = 1.0;
    }

    // Evaluate initial population, set pbest, update global best
    for (int i = 0; i < pop_; ++i) {
        // BUGFIX: respect the evaluation budget during initialisation
        FX_[i] = (prob_->calls() < max_evals_) ? evalAndUpdateBest(X_[i])
                                               : std::numeric_limits<double>::infinity();
        Pbest_[i]  = X_[i];
        Fpbest_[i] = FX_[i];
    }

    // CLPSO schedules and exemplars
    computePcSchedule();

    no_improve_.assign(pop_, 0);
    exemplar_.assign(pop_, std::vector<int>(D, 0));
    for (int i = 0; i < pop_; ++i) {
        selectExemplarsForParticle(i, /*force_nonself=*/true);
    }

    printBest();
}

void CLPSO::one_iteration()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0 || pop_ <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Inertia schedule across iterations
    // BUGFIX: the inertia schedule used iters_/max_iters_; max_iters is only a
    // 5e6 safety cap (the real budget is max_evals), so w stayed at w_max = 0.9
    // for the whole run and the swarm never converged (sphere-10D: f ~ 100
    // after 20000 evaluations; reference CLPSO: ~1e-9).
    const double t = progressFraction();
    const double w = w_max_ - (w_max_ - w_min_) * t;

    // Optional global-best term. It is NOT part of CLPSO (Liang et al. 2006,
    // v = w v + c r (pbest_fi(d) - x)); it is off by default (gbest_coef = 0)
    // and can be enabled from [clpso] (it ramps up linearly over the run).
    const double alpha = t;
    const double c_gb  = gbest_coef_ * c_;

    for (int i = 0; i < pop_; ++i) {

        // Refresh exemplars if stagnated (no pbest improvement for refresh_gap_ iterations)
        if (refresh_gap_ > 0 && no_improve_[i] >= refresh_gap_) {
            selectExemplarsForParticle(i, /*force_nonself=*/true);
            std::fill(V_[i].begin(), V_[i].end(), 0.0);
            no_improve_[i] = 0;
        }

        VecD vnew = V_[i];
        VecD xnew = X_[i];

        for (int j = 0; j < D; ++j) {
            const int e = exemplar_[i][j];
            const double target = Pbest_[e][j];

            const double r1 = U01(rng_);
            const double r2 = U01(rng_);
            // Hybrid CLPSO update: comprehensive learning plus a weak global-best term (activated progressively).
            vnew[j] = w * vnew[j] + c_ * r1 * (target - xnew[j]) + (alpha * c_gb) * r2 * (best_x_[j] - xnew[j]);
        }

        clampVelocity(vnew);

        for (int j = 0; j < D; ++j) {
            xnew[j] += vnew[j];
        }

        if (clamp_pos_) handleBounds(xnew, vnew);

        // Evaluate new position (positions always move; only pbest updates are greedy)
        const double fnew = evalAndUpdateBest(xnew);

        X_[i]  = std::move(xnew);
        V_[i]  = std::move(vnew);
        FX_[i] = fnew;

        bool pbest_improved = false;
        if (fnew < Fpbest_[i]) {
            Pbest_[i]  = X_[i];
            Fpbest_[i] = fnew;
            pbest_improved = true;
            no_improve_[i] = 0;
        } else {
            ++no_improve_[i];
        }

        // In-run local search only after pbest improvement (framework-aligned)
        if (pbest_improved && local_rate_ > 0.0 && !local_method_.empty()) {
            if (U01(rng_) < local_rate_) {
                auto refined = localSearch(local_method_, Pbest_[i]);
                const auto& xloc = refined.first;
                const double floc = refined.second;

                if (floc < Fpbest_[i]) {
                    Pbest_[i]  = xloc;
                    Fpbest_[i] = floc;

                    // Also place refined solution into the current position to exploit it immediately
                    X_[i]  = xloc;
                    FX_[i] = floc;

                    if (floc < best_f_) {
                        best_f_ = floc;
                        best_x_ = xloc;
                    }
                    no_improve_[i] = 0;
                }
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void CLPSO::end()
{
    // End polishing controlled only by [global] (same pattern as PSO/DE).
    if (!end_local_refine_) return;
    if (!prob_) return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    const double floc = refinement.second;

    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // Optional: inject refined best into the worst current individual
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (worst_idx < X_.size()) {
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    printBest();
}

} // namespace optimsolution
