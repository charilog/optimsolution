#include "apso.h"
#include "init.h"
#include "pso_family_common.h"
#include <cmath>
#include <cctype>
#include <algorithm>

namespace optimsolution {

namespace {

// Fuzzy membership functions of the four evolutionary states (paper, Eq. 9)
double muS1(double f) {            // exploration
    if (f <= 0.4) return 0.0;
    if (f <= 0.6) return 5.0 * f - 2.0;
    if (f <= 0.7) return 1.0;
    if (f <= 0.8) return -10.0 * f + 8.0;
    return 0.0;
}
double muS2(double f) {            // exploitation
    if (f <= 0.2) return 0.0;
    if (f <= 0.3) return 10.0 * f - 2.0;
    if (f <= 0.4) return 1.0;
    if (f <= 0.6) return -5.0 * f + 3.0;
    return 0.0;
}
double muS3(double f) {            // convergence
    if (f <= 0.1) return 1.0;
    if (f <= 0.3) return -5.0 * f + 1.5;
    return 0.0;
}
double muS4(double f) {            // jumping out
    if (f <= 0.7) return 0.0;
    if (f <= 0.9) return 5.0 * f - 3.5;
    return 1.0;
}

} // namespace

int APSO::classifyState(double f, int prev)
{
    if (!std::isfinite(f)) f = 0.0;
    f = std::max(0.0, std::min(1.0, f));
    if (prev < 1 || prev > 4) prev = 1;

    const double mu[5] = { 0.0, muS1(f), muS2(f), muS3(f), muS4(f) };
    const double eps = 1e-12;
    int cand[4], nc = 0;
    for (int s = 1; s <= 4; ++s) if (mu[s] > eps) cand[nc++] = s;
    if (nc == 0) {                    // cannot happen with Eq. 9, guard only
        int best = 1;
        for (int s = 2; s <= 4; ++s) if (mu[s] > mu[best]) best = s;
        return best;
    }
    if (nc == 1) return cand[0];

    auto isCand = [&](int s) { for (int k = 0; k < nc; ++k) if (cand[k] == s) return true; return false; };
    const int succ = (prev % 4) + 1;          // S1->S2->S3->S4->S1
    const int pred = ((prev + 2) % 4) + 1;    // S1<-S2<-S3<-S4<-S1
    if (isCand(prev)) return prev;
    if (isCand(succ)) return succ;
    if (isCand(pred)) return pred;
    int best = cand[0];
    for (int k = 1; k < nc; ++k) if (mu[cand[k]] > mu[best]) best = cand[k];
    return best;
}

void APSO::configure(const MethodConfig& mc)
{
    pop_cfg_ = mc.getInt("population", 0);
    pop_ = (pop_cfg_ > 3) ? pop_cfg_ : 20;

    c_init_    = mc.getDbl("c_init", c_init_);
    c_min_     = mc.getDbl("c_min", c_min_);
    c_max_     = mc.getDbl("c_max", c_max_);
    c_sum_max_ = mc.getDbl("c_sum_max", c_sum_max_);
    delta_min_ = mc.getDbl("delta_min", delta_min_);
    delta_max_ = mc.getDbl("delta_max", delta_max_);
    sigma_max_ = mc.getDbl("sigma_max", sigma_max_);
    sigma_min_ = mc.getDbl("sigma_min", sigma_min_);
    vmax_frac_ = mc.getDbl("vmax_frac", vmax_frac_);
    use_els_   = mc.getBool("els", use_els_);
    bound_mode_ = psofamily::parseBoundMode(mc.getStr("bound_mode", psofamily::boundModeName(bound_mode_)), bound_mode_);

    if (c_max_ < c_min_) std::swap(c_min_, c_max_);
    if (delta_max_ < delta_min_) std::swap(delta_min_, delta_max_);
    if (sigma_max_ < sigma_min_) std::swap(sigma_min_, sigma_max_);
    if (vmax_frac_ <= 0.0) vmax_frac_ = 0.2;

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

bool APSO::tryEval(const Vec& x, double& f)
{
    if (!prob_ || prob_->calls() >= max_evals_) return false;
    f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return true;
}

double APSO::evolutionaryFactor() const
{
    const int N = pop_;
    const int D = prob_->dimension();
    if (N < 2) return 0.0;
    std::vector<double> sum(N, 0.0);
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double s = 0.0;
            for (int d = 0; d < D; ++d) {
                const double t = X_[i][d] - X_[j][d];
                s += t * t;
            }
            const double dist = std::sqrt(s);
            sum[i] += dist;
            sum[j] += dist;
        }
    }
    double dmin = std::numeric_limits<double>::infinity(), dmax = 0.0;
    for (int i = 0; i < N; ++i) {
        sum[i] /= (double)(N - 1);
        dmin = std::min(dmin, sum[i]);
        dmax = std::max(dmax, sum[i]);
    }
    const double dg = sum[g_];
    if (!(dmax - dmin > 0.0)) return 0.0;   // all particles equidistant/collapsed
    return std::max(0.0, std::min(1.0, (dg - dmin) / (dmax - dmin)));
}

void APSO::adaptParameters(int state, double f)
{
    w_ = 1.0 / (1.0 + 1.5 * std::exp(-2.6 * f));

    std::uniform_real_distribution<double> Ud(delta_min_, delta_max_);
    const double d1 = Ud(rng_), d2 = Ud(rng_);
    switch (state) {
        case 1: c1_ += d1;       c2_ -= d2;       break;  // exploration
        case 2: c1_ += 0.5 * d1; c2_ -= 0.5 * d2; break;  // exploitation
        case 3: c1_ += 0.5 * d1; c2_ += 0.5 * d2; break;  // convergence
        case 4: c1_ -= d1;       c2_ += d2;       break;  // jumping out
        default: break;
    }
    c1_ = std::max(c_min_, std::min(c_max_, c1_));
    c2_ = std::max(c_min_, std::min(c_max_, c2_));
    const double s = c1_ + c2_;
    if (s > c_sum_max_) {
        c1_ = c1_ / s * c_sum_max_;
        c2_ = c2_ / s * c_sum_max_;
    }
}

void APSO::elitistLearning()
{
    const int D = prob_->dimension();
    if (D <= 0) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    const double prog  = progressFraction();
    const double sigma = sigma_max_ - (sigma_max_ - sigma_min_) * prog;
    std::uniform_int_distribution<int> Ud(0, D - 1);
    std::normal_distribution<double> Ng(0.0, sigma);

    Vec P = P_[g_];
    const int d = Ud(rng_);
    P[d] += (U[d] - L[d]) * Ng(rng_);
    if (P[d] < L[d]) P[d] = L[d];
    if (P[d] > U[d]) P[d] = U[d];

    double f;
    if (!tryEval(P, f)) return;

    if (f < FP_[g_]) {
        // Accepted as the new global best (stored as that particle's pbest)
        P_[g_] = P;
        FP_[g_] = f;
    } else {
        // Replace the worst particle
        int worst = 0;
        for (int i = 1; i < pop_; ++i) if (FX_[i] > FX_[worst]) worst = i;
        X_[worst] = P;
        FX_[worst] = f;
        if (f < FP_[worst]) {
            P_[worst] = P;
            FP_[worst] = f;
            if (f < FP_[g_]) g_ = worst;
        }
    }
}

void APSO::maybeLocalSearch(int i)
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

void APSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (pop_ < 3) pop_ = 3;

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();
    state_ = 1;
    w_ = 0.9;
    c1_ = c2_ = c_init_;

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

    printBest();
}

void APSO::one_iteration()
{
    if (!prob_ || prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // ---- Evolutionary state estimation and parameter adaptation ---------
    const double f = evolutionaryFactor();
    state_ = classifyState(f, state_);
    adaptParameters(state_, f);

    // ---- Elitist learning in the convergence state -----------------------
    if (use_els_ && state_ == 3) elitistLearning();

    // ---- Standard global-best PSO move -----------------------------------
    const psofamily::BoundMode mode =
        (bound_mode_ == psofamily::BoundMode::Skip && no_eval_gens_ >= psofamily::kMaxGenerationsWithoutEval)
            ? psofamily::BoundMode::Clamp : bound_mode_;
    int evaluated = 0;
    for (int i = 0; i < pop_; ++i) {
        if (prob_->calls() >= max_evals_) break;
        Vec& x = X_[i];
        Vec& v = V_[i];
        const Vec& p = P_[i];
        const Vec& g = P_[g_];
        for (int d = 0; d < D; ++d) {
            const double r1 = U01(rng_), r2 = U01(rng_);
            v[d] = w_ * v[d] + c1_ * r1 * (p[d] - x[d]) + c2_ * r2 * (g[d] - x[d]);
        }
        psofamily::clampVelocity(v, Vmax_);
        for (int d = 0; d < D; ++d) x[d] += v[d];
        if (!psofamily::applyBounds(x, v, L, U, mode)) continue;
        double fx;
        if (!tryEval(x, fx)) break;
        ++evaluated;
        FX_[i] = fx;
        if (fx < FP_[i]) {
            P_[i] = x;
            FP_[i] = fx;
            if (fx < FP_[g_]) g_ = i;
            maybeLocalSearch(i);
        }
    }
    no_eval_gens_ = evaluated ? 0 : no_eval_gens_ + 1;

    printBest();
    updateStop(FX_);
}

void APSO::end()
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
