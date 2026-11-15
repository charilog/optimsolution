#include "hclpso.h"
#include "init.h"
#include <cmath>
#include <cctype>
#include <algorithm>

namespace optimsolution {

void HCLPSO::configure(const MethodConfig& mc)
{
    pop_cfg_ = mc.getInt("population", 0);
    pop_ = (pop_cfg_ > 3) ? pop_cfg_ : 40;

    g1_cfg_      = mc.getInt("g1", g1_cfg_);
    expl_ratio_  = mc.getDbl("expl_ratio", expl_ratio_);
    w_start_     = mc.getDbl("w_start", w_start_);
    w_end_       = mc.getDbl("w_end", w_end_);
    k_start_     = mc.getDbl("k_start", k_start_);
    k_end_       = mc.getDbl("k_end", k_end_);
    c1_start_    = mc.getDbl("c1_start", c1_start_);
    c1_end_      = mc.getDbl("c1_end", c1_end_);
    c2_start_    = mc.getDbl("c2_start", c2_start_);
    c2_end_      = mc.getDbl("c2_end", c2_end_);
    pc_min_      = mc.getDbl("pc_min", pc_min_);
    pc_max_      = mc.getDbl("pc_max", pc_max_);
    pc_exp_      = mc.getDbl("pc_exp", pc_exp_);
    refresh_gap_ = mc.getInt("refresh_gap", refresh_gap_);
    vmax_frac_   = mc.getDbl("vmax_frac", vmax_frac_);
    bound_mode_  = psofamily::parseBoundMode(mc.getStr("bound_mode", psofamily::boundModeName(bound_mode_)), bound_mode_);

    if (expl_ratio_ < 0.0) expl_ratio_ = 0.0;
    if (expl_ratio_ > 1.0) expl_ratio_ = 1.0;
    pc_min_ = std::max(0.0, std::min(1.0, pc_min_));
    pc_max_ = std::max(pc_min_, std::min(1.0, pc_max_));
    if (refresh_gap_ < 0) refresh_gap_ = 0;
    if (vmax_frac_ <= 0.0) vmax_frac_ = 0.2;

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

bool HCLPSO::tryEval(const Vec& x, double& f)
{
    if (!prob_ || prob_->calls() >= max_evals_) return false;
    f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return true;
}

void HCLPSO::buildExemplar(int i)
{
    // CL exemplar exactly as in the authors' code: per dimension, the better
    // (pbest fitness) of two random "friends" from the allowed pool -- the
    // exploration group for exploration particles, the whole swarm for
    // exploitation ones -- is used with probability Pc_i, otherwise the
    // particle's own pbest; at least one dimension is forced to learn from a
    // friend. The exemplar stores the pbest COORDINATES at build time
    // (snapshot); it is rebuilt only after a stagnation of refresh_gap.
    const int D = prob_->dimension();
    const int pool = (i < g1_) ? g1_ : pop_;
    std::uniform_int_distribution<int> Upool(0, pool - 1);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    Vec& ex = E_[i];
    ex = P_[i];
    bool any = false;
    for (int d = 0; d < D; ++d) {
        const int a = Upool(rng_), b = Upool(rng_);
        const int friend_idx = (FP_[a] < FP_[b]) ? a : b;
        if (U01(rng_) < Pc_[i]) {
            ex[d] = P_[friend_idx][d];
            any = true;
        }
    }
    if (!any && D > 0) {
        std::uniform_int_distribution<int> Ud(0, D - 1);
        const int d = Ud(rng_);
        const int a = Upool(rng_), b = Upool(rng_);
        ex[d] = P_[(FP_[a] < FP_[b]) ? a : b][d];
    }
}

void HCLPSO::maybeLocalSearch(int i)
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

void HCLPSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (pop_ < 4) pop_ = 4;

    // Exploration-group size (reference code: 15 of 40)
    g1_ = (g1_cfg_ > 0) ? g1_cfg_ : (int)std::floor(expl_ratio_ * pop_ + 0.5);
    g1_ = std::max(2, std::min(pop_ - 1, g1_));

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();
    no_eval_gens_ = 0;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    Vmax_ = psofamily::velocityLimits(L, U, vmax_frac_);

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

    // Learning probabilities over the whole swarm (reference code:
    // 0.25 * (exp(t) - 1) / (exp(10) - 1), t = linspace(0, 10, N))
    Pc_.assign(pop_, pc_min_);
    const double denom = std::exp(pc_exp_) - 1.0;
    for (int i = 0; i < pop_; ++i) {
        const double t = (pop_ > 1) ? (double)i / (double)(pop_ - 1) : 0.0;
        const double shaped = (denom > 0.0) ? (std::exp(pc_exp_ * t) - 1.0) / denom : t;
        Pc_[i] = pc_min_ + (pc_max_ - pc_min_) * shaped;
    }

    E_.assign(pop_, Vec(D, 0.0));
    stay_.assign(pop_, 0);
    for (int i = 0; i < pop_; ++i) buildExemplar(i);

    printBest();
}

void HCLPSO::one_iteration()
{
    if (!prob_ || prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const double t  = progressFraction();
    const double w  = w_start_  + (w_end_  - w_start_)  * t;
    const double K  = k_start_  + (k_end_  - k_start_)  * t;
    const double c1 = c1_start_ + (c1_end_ - c1_start_) * t;
    const double c2 = c2_start_ + (c2_end_ - c2_start_) * t;

    // Synchronous generation (reference code): every particle moves with the
    // gbest of the previous generation, then pbests/gbest/exemplars update.
    const Vec gb = P_[g_];
    const psofamily::BoundMode mode =
        (bound_mode_ == psofamily::BoundMode::Skip && no_eval_gens_ >= psofamily::kMaxGenerationsWithoutEval)
            ? psofamily::BoundMode::Clamp : bound_mode_;

    int evaluated = 0;
    std::vector<char> improved(pop_, 0);
    for (int i = 0; i < pop_; ++i) {
        Vec& x = X_[i];
        Vec& v = V_[i];
        const Vec& ex = E_[i];
        if (i < g1_) {
            for (int d = 0; d < D; ++d)
                v[d] = w * v[d] + K * U01(rng_) * (ex[d] - x[d]);
        } else {
            for (int d = 0; d < D; ++d) {
                const double r1 = U01(rng_), r2 = U01(rng_);
                v[d] = w * v[d] + c1 * r1 * (ex[d] - x[d]) + c2 * r2 * (gb[d] - x[d]);
            }
        }
        psofamily::clampVelocity(v, Vmax_);
        for (int d = 0; d < D; ++d) x[d] += v[d];
        if (!psofamily::applyBounds(x, v, L, U, mode)) continue;  // out of range: not evaluated

        double f;
        if (!tryEval(x, f)) break;
        ++evaluated;
        FX_[i] = f;
        if (f < FP_[i]) {
            P_[i] = x;
            FP_[i] = f;
            improved[i] = 1;
        }
    }
    no_eval_gens_ = evaluated ? 0 : no_eval_gens_ + 1;

    // pbest stagnation counters (not evaluated counts as not improved)
    for (int i = 0; i < pop_; ++i) stay_[i] = improved[i] ? 0 : stay_[i] + 1;
    for (int i = 0; i < pop_; ++i) if (FP_[i] < FP_[g_]) g_ = i;

    for (int i = 0; i < pop_; ++i) if (improved[i]) maybeLocalSearch(i);

    // Exemplar refresh: counter EXCEEDS refresh_gap (reference: "> 5")
    for (int i = 0; i < pop_; ++i) {
        if (stay_[i] > refresh_gap_) {
            buildExemplar(i);
            stay_[i] = 0;
        }
    }

    printBest();
    updateStop(FX_);
}

void HCLPSO::end()
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
