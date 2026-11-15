#include "olpso.h"
#include "init.h"
#include "pso_family_common.h"
#include <cmath>
#include <cctype>
#include <algorithm>

namespace optimsolution {

void OLPSO::configure(const MethodConfig& mc)
{
    pop_cfg_ = mc.getInt("population", 0);
    pop_ = (pop_cfg_ > 3) ? pop_cfg_ : 40;

    w_start_   = mc.getDbl("w_start", w_start_);
    w_end_     = mc.getDbl("w_end", w_end_);
    c_         = mc.getDbl("c", c_);
    vmax_frac_ = mc.getDbl("vmax_frac", vmax_frac_);
    G_         = mc.getInt("G", G_);
    if (G_ < 1) G_ = 1;
    if (vmax_frac_ <= 0.0) vmax_frac_ = 0.2;

    std::string topo = mc.getStr("topology", global_topology_ ? "global" : "local");
    for (auto& ch : topo) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    global_topology_ = (topo == "global" || topo == "gbest" || topo == "g");

    {
        std::string pm = mc.getStr("partner_mode", dynamic_partner_ ? "dynamic" : "fixed");
        for (auto& ch : pm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        dynamic_partner_ = (pm == "dynamic");
    }
    bound_mode_ = psofamily::parseBoundMode(mc.getStr("bound_mode", psofamily::boundModeName(bound_mode_)), bound_mode_);

    std::string lm = mc.getStr("local_method", local_method_);
    for (auto& ch : lm) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    local_method_ = lm;
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = (lr < 0.0) ? 0.0 : (lr > 1.0 ? 1.0 : lr);

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

bool OLPSO::tryEval(const Vec& x, double& f)
{
    if (!prob_ || prob_->calls() >= max_evals_) return false;
    f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return true;
}

void OLPSO::buildOrthogonalArray(int D)
{
    // Two-level OA L_M(2^(M-1)) (paper Appendix), M = 2^ceil(log2(D+1)).
    // 1-based in the paper:
    //   basic columns  b = 2^(k-1), k = 1..u :
    //       L[a][b] = floor((a-1) / 2^(u-k)) mod 2,  a = 1..M
    //   other columns  for k = 2..u, b = 2^(k-1), s = 1..b-1 :
    //       L[a][b+s] = (L[a][s] + L[a][b]) mod 2
    int u = 0;
    while ((1 << u) < D + 1) ++u;
    if (u < 1) u = 1;
    const int M = 1 << u;
    const int N = M - 1;  // columns

    std::vector<std::vector<unsigned char>> full(M, std::vector<unsigned char>(N + 1, 0)); // 1-based cols
    for (int a = 1; a <= M; ++a) {
        for (int k = 1; k <= u; ++k) {
            const int b = 1 << (k - 1);
            full[a - 1][b] = (unsigned char)(((a - 1) >> (u - k)) & 1);
        }
        for (int k = 2; k <= u; ++k) {
            const int b = 1 << (k - 1);
            for (int s = 1; s <= b - 1; ++s)
                full[a - 1][b + s] = (unsigned char)((full[a - 1][s] + full[a - 1][b]) % 2);
        }
    }
    OA_.assign(M, std::vector<unsigned char>(D, 0));
    for (int a = 0; a < M; ++a)
        for (int d = 0; d < D; ++d) OA_[a][d] = full[a][d + 1];
}

int OLPSO::neighbourBest(int i) const
{
    if (global_topology_) return g_;
    const int N = pop_;
    const int a = (i - 1 + N) % N, b = (i + 1) % N;
    int best = i;
    if (FP_[a] < FP_[best]) best = a;
    if (FP_[b] < FP_[best]) best = b;
    return best;
}

void OLPSO::constructPo(int i)
{
    const int D = prob_->dimension();
    int n = neighbourBest(i);
    random_partner_[i] = 0;
    if (n == i) {
        // Pi == Pn: use a random other particle Pr (paper)
        std::uniform_int_distribution<int> Uo(0, pop_ - 2);
        n = Uo(rng_);
        if (n >= i) ++n;
        random_partner_[i] = 1;
    }
    partner_[i] = n;

    const Vec& Pi = P_[i];
    const Vec& Pn = P_[n];
    const int M = (int)OA_.size();

    // Factor analysis accumulators: sum of f and count per (dimension, level)
    std::vector<double> sum0(D, 0.0), sum1(D, 0.0);
    std::vector<int>    cnt0(D, 0),   cnt1(D, 0);

    Vec xt(D);
    int    best_row = 0;
    double best_fv  = FP_[i];  // row 0 == Pi (all level 1), fitness already known
    bool   any      = false;

    for (int a = 0; a < M; ++a) {
        double f;
        bool is_pi = true;
        for (int d = 0; d < D; ++d) if (OA_[a][d]) { is_pi = false; break; }
        if (is_pi) {
            f = FP_[i];
            if (!std::isfinite(f)) continue;
        } else {
            for (int d = 0; d < D; ++d) xt[d] = OA_[a][d] ? Pn[d] : Pi[d];
            if (!tryEval(xt, f)) break;   // budget exhausted: use tests done so far
        }
        any = true;
        for (int d = 0; d < D; ++d) {
            if (OA_[a][d]) { sum1[d] += f; ++cnt1[d]; }
            else           { sum0[d] += f; ++cnt0[d]; }
        }
        if (f < best_fv) { best_fv = f; best_row = a; }
    }

    auto& lev = level_[i];
    lev.assign(D, 0);
    for (int d = 0; d < D; ++d) lev[d] = OA_[best_row][d];   // Xb by default
    if (!any) return;

    // Predicted best combination Xp from the factor analysis (minimisation)
    std::vector<unsigned char> lp(D, 0);
    bool differs = false;
    for (int d = 0; d < D; ++d) {
        const double m0 = cnt0[d] ? sum0[d] / cnt0[d] : std::numeric_limits<double>::infinity();
        const double m1 = cnt1[d] ? sum1[d] / cnt1[d] : std::numeric_limits<double>::infinity();
        lp[d] = (m1 < m0) ? 1 : 0;
        if (lp[d] != lev[d]) differs = true;
    }
    if (!differs) return;  // Xp == Xb, nothing to evaluate

    for (int d = 0; d < D; ++d) xt[d] = lp[d] ? Pn[d] : Pi[d];
    double fp;
    if (tryEval(xt, fp) && fp < best_fv) lev = lp;
}

void OLPSO::maybeLocalSearch(int i)
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
        stay_[i] = 0;
    }
}

void OLPSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (pop_ < 3) pop_ = 3;

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

    buildOrthogonalArray(D);
    level_.assign(pop_, std::vector<unsigned char>(D, 0));
    partner_.assign(pop_, 0);
    random_partner_.assign(pop_, 0);
    stay_.assign(pop_, 0);
    for (int i = 0; i < pop_; ++i) constructPo(i);

    printBest();
}

void OLPSO::one_iteration()
{
    if (!prob_ || prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    const double t = progressFraction();
    const double w = w_start_ + (w_end_ - w_start_) * t;
    const psofamily::BoundMode mode =
        (bound_mode_ == psofamily::BoundMode::Skip && no_eval_gens_ >= psofamily::kMaxGenerationsWithoutEval)
            ? psofamily::BoundMode::Clamp : bound_mode_;
    int evaluated = 0;

    for (int i = 0; i < pop_; ++i) {
        Vec& x = X_[i];
        Vec& v = V_[i];
        const Vec& Pi = P_[i];
        // Pn: the neighbourhood best (re-resolved every generation with
        // partner_mode = dynamic), or the partner fixed when Po was built.
        // A random partner Pr (used when Pi == Pn at build time) stays fixed.
        int n = partner_[i];
        if (dynamic_partner_ && !random_partner_[i]) n = neighbourBest(i);
        const Vec& Pn = P_[n];
        const auto& lev = level_[i];

        for (int d = 0; d < D; ++d) {
            const double po = lev[d] ? Pn[d] : Pi[d];
            v[d] = w * v[d] + c_ * U01(rng_) * (po - x[d]);
        }
        psofamily::clampVelocity(v, Vmax_);
        for (int d = 0; d < D; ++d) x[d] += v[d];
        const bool inside = psofamily::applyBounds(x, v, L, U, mode);

        if (inside) {
            double f;
            if (!tryEval(x, f)) break;
            ++evaluated;
            FX_[i] = f;
            if (f < FP_[i]) {
                P_[i] = x;
                FP_[i] = f;
                stay_[i] = 0;
                if (f < FP_[g_]) g_ = i;
                maybeLocalSearch(i);
            } else {
                ++stay_[i];
            }
        }

        if (stay_[i] >= G_) {
            if (prob_->calls() >= max_evals_) break;
            constructPo(i);
            stay_[i] = 0;
        }
        if (prob_->calls() >= max_evals_) break;
    }
    no_eval_gens_ = evaluated ? 0 : no_eval_gens_ + 1;

    printBest();
    updateStop(FX_);
}

void OLPSO::end()
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
