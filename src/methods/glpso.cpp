#include "glpso.h"
#include "init.h"
#include "pso_family_common.h"
#include <cmath>
#include <cctype>
#include <numeric>
#include <algorithm>

namespace optimsolution {

void GLPSO::configure(const MethodConfig& mc)
{
    pop_cfg_ = mc.getInt("population", 0);
    pop_ = (pop_cfg_ > 3) ? pop_cfg_ : 50;

    w_         = mc.getDbl("w", w_);
    c_         = mc.getDbl("c", c_);
    pm_        = mc.getDbl("pm", pm_);
    sg_        = mc.getInt("sg", sg_);
    tour_frac_ = mc.getDbl("tour_frac", tour_frac_);
    vmax_frac_ = mc.getDbl("vmax_frac", vmax_frac_);
    init_exemplar_offspring_ = mc.getBool("init_exemplar_offspring", init_exemplar_offspring_);
    bound_mode_ = psofamily::parseBoundMode(mc.getStr("bound_mode", psofamily::boundModeName(bound_mode_)), bound_mode_);
    pm_        = std::max(0.0, std::min(1.0, pm_));
    if (sg_ < 1) sg_ = 1;
    tour_frac_ = std::max(0.0, std::min(1.0, tour_frac_));
    if (vmax_frac_ <= 0.0) vmax_frac_ = 0.2;

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

bool GLPSO::tryEval(const Vec& x, double& f)
{
    if (!prob_ || prob_->calls() >= max_evals_) return false;
    f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return true;
}

void GLPSO::maybeLocalSearch(int i)
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

void GLPSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (pop_ < 2) pop_ = 2;

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

    P_ = X_;  FP_ = FX_;
    E_ = P_;  FE_ = FP_;
    stag_.assign(pop_, 0);
    g_ = 0;
    for (int i = 1; i < pop_; ++i) if (FP_[i] < FP_[g_]) g_ = i;

    // Initial exemplars (reference implementation): one bred and evaluated
    // offspring per particle, accepted unconditionally. With
    // init_exemplar_offspring = 0 the exemplars start as the pbests.
    if (init_exemplar_offspring_) {
        Vec o(D);
        for (int i = 0; i < pop_; ++i) {
            breed(i, o);
            double fo;
            if (!tryEval(o, fo)) break;
            E_[i] = o;
            FE_[i] = fo;
        }
    }

    printBest();
}

void GLPSO::one_iteration()
{
    if (!prob_ || prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const int N = pop_;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const int tour = std::max(1, std::min(N, (int)std::lround(tour_frac_ * N)));
    const psofamily::BoundMode mode =
        (bound_mode_ == psofamily::BoundMode::Skip && no_eval_gens_ >= psofamily::kMaxGenerationsWithoutEval)
            ? psofamily::BoundMode::Clamp : bound_mode_;
    int evaluated = 0;
    std::vector<int> idx(N);
    Vec o(D);

    for (int i = 0; i < N; ++i) {
        // ---- Genetic layer: breed an offspring exemplar -------------------
        breed(i, o);

        double fo;
        if (!tryEval(o, fo)) break;
        if (fo < FE_[i]) {
            E_[i] = o;
            FE_[i] = fo;
            stag_[i] = 0;
        } else {
            ++stag_[i];
        }

        // Tournament once the exemplar has failed to improve MORE than sg
        // times in a row (reference implementation: "stag > sg")
        if (stag_[i] > sg_) {
            // 20%N-tournament among exemplars
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng_);
            int win = idx[0];
            for (int t = 1; t < tour; ++t) if (FE_[idx[t]] < FE_[win]) win = idx[t];
            if (win != i) { E_[i] = E_[win]; FE_[i] = FE_[win]; }
            stag_[i] = 0;
        }

        // ---- PSO layer: learn from the exemplar ---------------------------
        Vec& x = X_[i];
        Vec& v = V_[i];
        const Vec& e = E_[i];
        for (int d = 0; d < D; ++d) v[d] = w_ * v[d] + c_ * U01(rng_) * (e[d] - x[d]);
        psofamily::clampVelocity(v, Vmax_);
        for (int d = 0; d < D; ++d) x[d] += v[d];
        if (!psofamily::applyBounds(x, v, L, U, mode)) continue;

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
    printBest();
    updateStop(FX_);
}

void GLPSO::breed(int i, Vec& o)
{
    // Crossover (Eq. 5) per dimension with a random particle k_d, then
    // mutation (Eq. 6) with probability pm per dimension.
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Ui(0, pop_ - 1);
    const Vec& Pi = P_[i];
    const Vec& Gb = P_[g_];
    o.resize(D);
    for (int d = 0; d < D; ++d) {
        const int k = Ui(rng_);
        if (FP_[i] < FP_[k]) {
            const double r = U01(rng_);
            o[d] = r * Pi[d] + (1.0 - r) * Gb[d];
        } else {
            o[d] = P_[k][d];
        }
        if (U01(rng_) < pm_) o[d] = L[d] + (U[d] - L[d]) * U01(rng_);
    }
}

void GLPSO::end()
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
