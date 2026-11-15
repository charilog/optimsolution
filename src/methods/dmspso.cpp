#include "dmspso.h"
#include "init.h"
#include "pso_family_common.h"
#include <cmath>
#include <cctype>
#include <numeric>
#include <algorithm>

namespace optimsolution {

void DMSPSO::configure(const MethodConfig& mc)
{
    pop_cfg_      = mc.getInt("population", 0);
    sub_size_     = mc.getInt("sub_size", sub_size_);
    n_swarms_cfg_ = mc.getInt("n_swarms", n_swarms_cfg_);
    R_            = mc.getInt("R", R_);
    w_            = mc.getDbl("w", w_);
    w_end_        = mc.getDbl("w_end", w_end_);
    c1_           = mc.getDbl("c1", c1_);
    c2_           = mc.getDbl("c2", c2_);
    vmax_frac_    = mc.getDbl("vmax_frac", vmax_frac_);
    global_phase_ = mc.getDbl("global_phase", global_phase_);
    ls_period_    = mc.getInt("ls_period", ls_period_);
    ls_count_     = mc.getInt("ls_count", ls_count_);
    bound_mode_ = psofamily::parseBoundMode(mc.getStr("bound_mode", psofamily::boundModeName(bound_mode_)), bound_mode_);

    if (sub_size_ < 2) sub_size_ = 2;
    if (R_ < 1) R_ = 1;
    if (vmax_frac_ <= 0.0) vmax_frac_ = 0.2;
    global_phase_ = std::max(0.0, std::min(1.0, global_phase_));
    if (ls_period_ < 0) ls_period_ = 0;
    if (ls_count_ < 1) ls_count_ = 1;

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

bool DMSPSO::tryEval(const Vec& x, double& f)
{
    if (!prob_ || prob_->calls() >= max_evals_) return false;
    f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return true;
}

void DMSPSO::regroup()
{
    // Random permutation -> consecutive blocks of sub_size_ particles
    std::vector<int> perm(pop_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng_);
    group_.assign(pop_, 0);
    for (int k = 0; k < pop_; ++k) group_[perm[k]] = std::min(k / sub_size_, n_swarms_ - 1);
}

void DMSPSO::maybeLocalSearch(int i)
{
    if (local_rate_ <= 0.0 || local_method_.empty() || local_method_ == "none") return;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    if (U01(rng_) >= local_rate_) return;

    auto refined = localSearch(local_method_, P_[i]);
    if (refined.second < FP_[i]) {
        P_[i] = refined.first;  FP_[i] = refined.second;
        X_[i] = refined.first;  FX_[i] = refined.second;
        if (FP_[i] < FP_[g_]) g_ = i;
        if (refined.second < best_f_) { best_f_ = refined.second; best_x_ = refined.first; }
    }
}

void DMSPSO::periodicLocalSearch()
{
    // DMS-L-PSO: refine the pbests of ls_count_ random particles; each result
    // replaces the nearest pbest if it is better.
    if (local_method_.empty() || local_method_ == "none") return;
    const int D = prob_->dimension();
    std::vector<int> perm(pop_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng_);
    const int cnt = std::min(ls_count_, pop_);

    for (int k = 0; k < cnt; ++k) {
        if (prob_->calls() >= max_evals_) return;
        const int i = perm[k];
        auto refined = localSearch(local_method_, P_[i]);
        const Vec& xr = refined.first;
        const double fr = refined.second;
        if (!std::isfinite(fr)) continue;
        if (fr < best_f_) { best_f_ = fr; best_x_ = xr; }

        int nearest = 0;
        double dbest = std::numeric_limits<double>::infinity();
        for (int j = 0; j < pop_; ++j) {
            double s = 0.0;
            for (int d = 0; d < D; ++d) { const double t = P_[j][d] - xr[d]; s += t * t; }
            if (s < dbest) { dbest = s; nearest = j; }
        }
        if (fr < FP_[nearest]) {
            P_[nearest] = xr;
            FP_[nearest] = fr;
            if (fr < FP_[g_]) g_ = nearest;
        }
    }
}

void DMSPSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    if (pop_cfg_ > 3) {
        n_swarms_ = std::max(1, (pop_cfg_ + sub_size_ - 1) / sub_size_);
    } else if (n_swarms_cfg_ > 0) {
        n_swarms_ = n_swarms_cfg_;
    } else {
        n_swarms_ = std::max(10, (int)std::lround(0.3 * D));
    }
    pop_ = n_swarms_ * sub_size_;
    gen_ = 0;

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    Vmax_ = psofamily::velocityLimits(L, U, vmax_frac_);
    no_eval_gens_ = 0;

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    V_.assign(pop_, Vec(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    for (int i = 0; i < pop_; ++i) {
        for (int d = 0; d < D; ++d)
            V_[i][d] = std::isfinite(Vmax_[d]) ? -Vmax_[d] + 2.0 * Vmax_[d] * U01(rng_) : 0.0;
        double f;
        if (tryEval(X_[i], f)) FX_[i] = f;
    }
    if (best_x_.empty() && !X_.empty()) best_x_ = X_[0];

    P_ = X_;
    FP_ = FX_;
    g_ = 0;
    for (int i = 1; i < pop_; ++i) if (FP_[i] < FP_[g_]) g_ = i;

    regroup();
    lbest_.assign(n_swarms_, 0);
    printBest();
}

void DMSPSO::one_iteration()
{
    if (!prob_ || prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const double prog = progressFraction();
    const bool global_phase = (prog >= global_phase_);
    // Inertia: constant w (CEC 2008 paper) or, with w_end >= 0, linear w -> w_end
    const double w = (w_end_ >= 0.0) ? w_ + (w_end_ - w_) * prog : w_;

    // Sub-swarm bests from the current pbests
    if (!global_phase) {
        std::fill(lbest_.begin(), lbest_.end(), -1);
        for (int i = 0; i < pop_; ++i) {
            const int s = group_[i];
            if (lbest_[s] < 0 || FP_[i] < FP_[lbest_[s]]) lbest_[s] = i;
        }
    }

    const psofamily::BoundMode mode =
        (bound_mode_ == psofamily::BoundMode::Skip && no_eval_gens_ >= psofamily::kMaxGenerationsWithoutEval)
            ? psofamily::BoundMode::Clamp : bound_mode_;
    int evaluated = 0;
    for (int i = 0; i < pop_; ++i) {
        Vec& x = X_[i];
        Vec& v = V_[i];
        const Vec& p = P_[i];
        const Vec& l = global_phase ? P_[g_] : P_[lbest_[group_[i]]];
        for (int d = 0; d < D; ++d) {
            const double r1 = U01(rng_), r2 = U01(rng_);
            v[d] = w * v[d] + c1_ * r1 * (p[d] - x[d]) + c2_ * r2 * (l[d] - x[d]);
        }
        psofamily::clampVelocity(v, Vmax_);
        for (int d = 0; d < D; ++d) x[d] += v[d];
        if (!psofamily::applyBounds(x, v, L, U, mode)) continue;   // out of range: no evaluation, no pbest update
        double f;
        if (!tryEval(x, f)) break;
        ++evaluated;
        FX_[i] = f;
        if (f < FP_[i]) {
            P_[i] = x;
            FP_[i] = f;
            if (f < FP_[g_]) g_ = i;
            maybeLocalSearch(i);
        }
        if (prob_->calls() >= max_evals_) break;
    }

    no_eval_gens_ = evaluated ? 0 : no_eval_gens_ + 1;
    ++gen_;
    if (!global_phase && gen_ % R_ == 0) regroup();
    if (ls_period_ > 0 && gen_ % ls_period_ == 0) periodicLocalSearch();

    printBest();
    updateStop(FX_);
}

void DMSPSO::end()
{
    if (!end_local_refine_ || !prob_ || end_local_method_.empty()) return;
    auto refinement = localSearch(end_local_method_, best_x_);
    if (refinement.second < best_f_) {
        best_f_ = refinement.second;
        best_x_ = refinement.first;
    }
    printBest();
}

} // namespace optimsolution
