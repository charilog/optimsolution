#include "spso2011.h"
#include "init.h"
#include "pso_family_common.h"
#include <cmath>
#include <cctype>
#include <numeric>
#include <algorithm>

namespace optimsolution {

void SPSO2011::configure(const MethodConfig& mc)
{
    pop_cfg_ = mc.getInt("population", 0);
    pop_ = (pop_cfg_ > 3) ? pop_cfg_ : 40;   // SPSO-2011 suggested swarm size

    K_          = std::max(1, mc.getInt("K", K_));
    w_          = mc.getDbl("w", w_);
    c_          = mc.getDbl("c", c_);
    rand_order_ = mc.getBool("rand_order", rand_order_);
    volume_uniform_ = mc.getBool("sphere_volume_uniform", volume_uniform_);
    bound_mode_ = psofamily::parseBoundMode(mc.getStr("bound_mode", psofamily::boundModeName(bound_mode_)), bound_mode_);

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

bool SPSO2011::tryEval(const Vec& x, double& f)
{
    if (!prob_ || prob_->calls() >= max_evals_) return false;
    f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return true;
}

void SPSO2011::buildTopology()
{
    // Each particle j informs itself and K particles drawn at random (with
    // replacement, so a particle may inform fewer than K distinct others).
    // We store the reverse relation: informants_[i] = { j : j informs i }.
    const int N = pop_;
    std::vector<std::vector<char>> link(N, std::vector<char>(N, 0));
    std::uniform_int_distribution<int> Ui(0, N - 1);
    for (int j = 0; j < N; ++j) {
        link[j][j] = 1;
        for (int k = 0; k < K_; ++k) link[j][Ui(rng_)] = 1;
    }
    informants_.assign(N, {});
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (link[j][i]) informants_[i].push_back(j);
}

int SPSO2011::bestInformant(int i) const
{
    int best = i;
    for (int j : informants_[i])
        if (FP_[j] < FP_[best]) best = j;
    return best;
}

void SPSO2011::randomInSphere(const Vec& center, double radius, Vec& out)
{
    // Clerc (2012), Sec. 3.4: isotropic direction (normalised Gaussian
    // vector) and, by default, a UNIFORM RADIUS r = U(0, R) -- the resulting
    // point is denser near the centre G (not uniform in volume). With
    // sphere_volume_uniform = 1 the radius is R * U^(1/D) instead (uniform in
    // the volume of the ball).
    const int D = (int)center.size();
    out = center;
    if (!(radius > 0.0) || !std::isfinite(radius)) return;

    std::normal_distribution<double> N01(0.0, 1.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    Vec dir(D);
    double norm2 = 0.0;
    int guard = 0;
    do {
        norm2 = 0.0;
        for (int d = 0; d < D; ++d) { dir[d] = N01(rng_); norm2 += dir[d] * dir[d]; }
    } while (norm2 <= 0.0 && ++guard < 10);
    if (norm2 <= 0.0) return;
    const double u = U01(rng_);
    const double rad = volume_uniform_ ? radius * std::pow(u, 1.0 / (double)D) : radius * u;
    const double r = rad / std::sqrt(norm2);
    for (int d = 0; d < D; ++d) out[d] = center[d] + r * dir[d];
}

void SPSO2011::maybeLocalSearch(int i)
{
    if (local_rate_ <= 0.0 || local_method_.empty() || local_method_ == "none") return;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    if (U01(rng_) >= local_rate_) return;

    auto refined = localSearch(local_method_, P_[i]);
    if (refined.second < FP_[i]) {
        P_[i]  = refined.first;  FP_[i] = refined.second;
        X_[i]  = refined.first;  FX_[i] = refined.second;
        if (FP_[i] < FP_[g_]) g_ = i;
        if (refined.second < best_f_) { best_f_ = refined.second; best_x_ = refined.first; }
    }
}

void SPSO2011::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (pop_ < 2) pop_ = 2;
    if (w_ <= 0.0) w_ = 1.0 / (2.0 * std::log(2.0));
    if (c_ <= 0.0) c_ = 0.5 + std::log(2.0);

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();
    no_eval_gens_ = 0;

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    V_.assign(pop_, Vec(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    for (int i = 0; i < pop_; ++i) {
        for (int d = 0; d < D; ++d) {
            // v_d = U(lb_d - x_d, ub_d - x_d)
            const double lo = L[d] - X_[i][d], hi = U[d] - X_[i][d];
            V_[i][d] = (std::isfinite(lo) && std::isfinite(hi)) ? lo + (hi - lo) * U01(rng_) : 0.0;
        }
        double f;
        if (tryEval(X_[i], f)) FX_[i] = f;
    }
    if (best_x_.empty() && !X_.empty()) best_x_ = X_[0];

    P_ = X_;
    FP_ = FX_;
    g_ = 0;
    for (int i = 1; i < pop_; ++i) if (FP_[i] < FP_[g_]) g_ = i;

    buildTopology();
    printBest();
}

void SPSO2011::one_iteration()
{
    if (!prob_ || prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const int N = pop_;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    const double prev_best = FP_[g_];

    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    if (rand_order_) std::shuffle(order.begin(), order.end(), rng_);

    const psofamily::BoundMode mode =
        (bound_mode_ == psofamily::BoundMode::Skip && no_eval_gens_ >= psofamily::kMaxGenerationsWithoutEval)
            ? psofamily::BoundMode::Clamp : bound_mode_;
    int evaluated = 0;
    Vec G(D), xs(D);
    for (int idx = 0; idx < N; ++idx) {
        const int i = order[idx];
        const int l = bestInformant(i);   // asynchronous: uses the latest pbests

        Vec& x = X_[i];
        Vec& v = V_[i];
        const Vec& p = P_[i];

        double r2 = 0.0;
        if (l != i) {
            const Vec& pl = P_[l];
            for (int d = 0; d < D; ++d) {
                G[d] = x[d] + c_ * (p[d] + pl[d] - 2.0 * x[d]) / 3.0;
                r2 += (G[d] - x[d]) * (G[d] - x[d]);
            }
        } else {
            for (int d = 0; d < D; ++d) {
                G[d] = x[d] + c_ * (p[d] - x[d]) / 2.0;
                r2 += (G[d] - x[d]) * (G[d] - x[d]);
            }
        }
        randomInSphere(G, std::sqrt(r2), xs);

        for (int d = 0; d < D; ++d) {
            v[d] = w_ * v[d] + xs[d] - x[d];
            x[d] = x[d] + v[d];
        }
        // Confinement (SPSO-2011, default bound_mode = reflect): clamp the
        // position, reverse and halve the offending velocity components
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

    // Adaptive random topology: new links if no improvement this iteration
    if (!(FP_[g_] < prev_best)) buildTopology();

    printBest();
    updateStop(FX_);
}

void SPSO2011::end()
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
