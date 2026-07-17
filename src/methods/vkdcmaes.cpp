#include "vkdcmaes.h"
#include <numeric>
#include <limits>

namespace optimsolution {

void VkDCMAES::configure(const MethodConfig& mc) {
    lambda_cfg_    = mc.getInt("lambda", lambda_cfg_);
    k_cfg_         = mc.getInt("k", k_cfg_);
    k_init_        = mc.getInt("k_init", k_init_);
    k_max_cfg_     = mc.getInt("k_max", k_max_cfg_);
    sigma0_        = mc.getDbl("sigma0", sigma0_);
    grow_patience_ = mc.getInt("grow_patience", grow_patience_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

double VkDCMAES::safeEval(const Vec& x) {
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

void VkDCMAES::ensureBounds(Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// Sampling transform for ONE stored vector (Akimoto, Auger & Hansen,
// GECCO'14, Section 3.3, "Step 1"): y <- y + c*<y,v>*v with
// c = (sqrt(1+||v||^2)-1)/||v||^2. NOTE: the source paper's printed
// formula omits the "/||v||^2"; this was verified numerically (Monte
// Carlo covariance check against I+vv^T, see delivery notes) to be
// required for the transform to actually produce the claimed
// N(0, I+vv^T) distribution, so it is included here.
void VkDCMAES::applyVjSample(const Vec& vj, double vj_norm2, Vec& y) const {
    if (vj_norm2 < 1e-300) return;
    double dot = 0.0;
    for (size_t k = 0; k < y.size(); ++k) dot += y[k] * vj[k];
    const double c = (std::sqrt(1.0 + vj_norm2) - 1.0) / vj_norm2;
    const double scale = c * dot;
    for (size_t k = 0; k < y.size(); ++k) y[k] += scale * vj[k];
}

// Theorem 3.6 (Akimoto, Auger & Hansen, GECCO'14): O(d) computation of the
// (numerically-stabilised, via the alpha/A/b correction of Proposition 3.5)
// natural gradient of the log-likelihood w.r.t. v and theta_D, for ONE
// vector v_j treated in isolation (see class-level comment for why: the
// k>1 joint derivation was not available to verify).
void VkDCMAES::naturalGradientOneVector(const Vec& ycoord, const Vec& vj, double vj_norm2,
                                         Vec& gs, Vec& gt) const {
    const int D = (int)ycoord.size();
    const double gamma_v = 1.0 + vj_norm2; // "gamma_v" in the paper

    double yv = 0.0; // <y, v>
    for (int k = 0; k < D; ++k) yv += ycoord[k] * vj[k];

    // alpha: this implementation's safe simplification of the paper's
    // Proposition 3.5 (see class-level comment).
    const double gamma = 1.0 / std::sqrt(gamma_v);
    const double alpha = std::min(1.0, gamma);

    const double b = -(1.0 - alpha * alpha) * (vj_norm2 * vj_norm2) / gamma_v
                    + 2.0 * alpha * alpha;

    if ((int)scratch_Ainv_.size() != D) {
        scratch_s_.assign(D, 0.0);
        scratch_t_.assign(D, 0.0);
        scratch_Ainv_.assign(D, 0.0);
        scratch_Ainv_v_.assign(D, 0.0);
        scratch_Ainv_s_.assign(D, 0.0);
    }
    Vec& Ainv = scratch_Ainv_;
    for (int k = 0; k < D; ++k) {
        double Aii = 2.0 - (b + 2.0 * alpha) * vj[k] * vj[k];
        if (Aii < 1e-8) Aii = 1e-8;
        Ainv[k] = 1.0 / Aii;
    }

    Vec& s = scratch_s_;
    Vec& t = scratch_t_;

    // Step 1: s <- y.*y - gamma_v^{-1} <y,v> y.*v - 1
    for (int k = 0; k < D; ++k) {
        s[k] = ycoord[k] * ycoord[k] - (yv / gamma_v) * ycoord[k] * vj[k] - 1.0;
    }
    // Step 2: t <- <y,v> y - 0.5*(<y,v>^2 + gamma_v) v
    {
        const double coeff = 0.5 * (yv * yv + gamma_v);
        for (int k = 0; k < D; ++k) t[k] = yv * ycoord[k] - coeff * vj[k];
    }
    // Step 3: s <- s - alpha*gamma_v^{-1} * [ (2+||v||^2) v.*t - ||v||^2 <v,t> v ]
    {
        double vt = 0.0;
        for (int k = 0; k < D; ++k) vt += vj[k] * t[k];
        const double f1 = alpha / gamma_v;
        for (int k = 0; k < D; ++k) {
            s[k] -= f1 * ((2.0 + vj_norm2) * vj[k] * t[k] - vj_norm2 * vt * vj[k]);
        }
    }
    // Step 4: s <- A^{-1}s - (1+b<v,A^{-1}v>)^{-1} b <s,A^{-1}v> A^{-1}v
    {
        Vec& Ainv_v = scratch_Ainv_v_;
        Vec& Ainv_s = scratch_Ainv_s_;
        for (int k = 0; k < D; ++k) { Ainv_v[k] = Ainv[k] * vj[k]; Ainv_s[k] = Ainv[k] * s[k]; }
        double v_Ainv_v = 0.0, s_Ainv_v = 0.0;
        for (int k = 0; k < D; ++k) { v_Ainv_v += vj[k] * Ainv_v[k]; s_Ainv_v += s[k] * Ainv_v[k]; }
        const double denom = 1.0 + b * v_Ainv_v;
        const double coeff = (std::fabs(denom) > 1e-12) ? (b * s_Ainv_v / denom) : 0.0;
        for (int k = 0; k < D; ++k) s[k] = Ainv_s[k] - coeff * Ainv_v[k];
    }
    // Step 5: t <- t - alpha*[ (2+||v||^2) v.*s - <s,v> v ]
    {
        double sv = 0.0;
        for (int k = 0; k < D; ++k) sv += s[k] * vj[k];
        for (int k = 0; k < D; ++k) {
            t[k] -= alpha * ((2.0 + vj_norm2) * vj[k] * s[k] - sv * vj[k]);
        }
    }

    gs = s; // caller still receives owned copies (gs/gt are accumulated
    const double vnorm = std::sqrt(vj_norm2); // into per-call sums by the caller)
    gt.assign(D, 0.0);
    if (vnorm > 1e-300) {
        for (int k = 0; k < D; ++k) gt[k] = t[k] / vnorm;
    }
}

void VkDCMAES::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;
    const double n = (double)D;

    lambda_ = (lambda_cfg_ > 0) ? lambda_cfg_
                                 : std::max(4, 4 + (int)std::floor(3.0 * std::log(n)));
    mu_ = std::max(1, lambda_ / 2);

    w_.assign(mu_, 0.0);
    double sumw2 = 0.0;
    {
        double denom = 0.0;
        std::vector<double> raw(mu_);
        for (int i = 1; i <= mu_; ++i) {
            raw[i - 1] = std::log((double)mu_ + 0.5) - std::log((double)i);
            denom += raw[i - 1];
        }
        for (int i = 0; i < mu_; ++i) { w_[i] = raw[i] / denom; sumw2 += w_[i] * w_[i]; }
    }
    mu_eff_ = 1.0 / sumw2;

    // Standard CMA-ES constants (Eq. 4), then VD-CMA's rescaling of
    // c_sigma, c1, c_mu (Section 3.3) for the richer per-step learning
    // rate the restricted, lower-dimensional model affords.
    const double c_sigma_std = (mu_eff_ + 2.0) / (n + mu_eff_ + 5.0);
    c_c_ = (4.0 + mu_eff_ / n) / (n + 4.0 + 2.0 * mu_eff_ / n);
    const double c1_std  = 2.0 / ((n + 1.3) * (n + 1.3) + mu_eff_);
    const double cmu_std = std::min(1.0 - c1_std,
                                     2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) / ((n + 2.0) * (n + 2.0) + mu_eff_));

    c_sigma_ = std::sqrt(mu_eff_) / (2.0 * (std::sqrt(n) + std::sqrt(mu_eff_)));
    const double rescale = std::max(1.0, (n - 5.0) / 6.0);
    c1_  = rescale * c1_std;
    cmu_ = std::min(1.0 - c1_, rescale * cmu_std);
    d_sigma_ = 1.0 + c_sigma_ + 2.0 * std::max(0.0, std::sqrt(std::max(0.0, (mu_eff_ - 1.0) / (n + 1.0))) - 1.0);

    // --- k (rank) setup ---
    k_max_ = (k_max_cfg_ > 0) ? k_max_cfg_
                               : std::max(1, std::min(D, 4 + (int)std::floor(3.0 * std::log(n))));
    k_ = (k_cfg_ > 0) ? std::min(k_cfg_, k_max_) : std::max(1, std::min(k_init_, k_max_));
    grow_streak_ = 0;

    V_.assign(k_max_, Vec(D, 0.0));
    c_v_.assign(k_max_, 0.0);
    std::normal_distribution<double> N01small(0.0, 1.0);
    for (int j = 0; j < k_max_; ++j) {
        // Exponentially spaced per-vector learning rate (LM-MA-ES-style),
        // giving each stored direction a different characteristic
        // timescale instead of all k vectors competing for one.
        c_v_[j] = cmu_ / std::pow(1.5, (double)j);
        if (j < k_) {
            // Small random init (per the paper: v ~ N(0, I/d)) so the
            // vector has a well-defined, generic initial direction.
            for (int kk = 0; kk < D; ++kk) V_[j][kk] = N01small(rng_) / std::sqrt(n);
        }
    }

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    mean_.assign(D, 0.0);
    double avg_range = 0.0; int counted = 0;
    for (int j = 0; j < D; ++j) {
        const double lo = L[j], hi = U[j];
        if (std::isfinite(lo) && std::isfinite(hi)) { mean_[j] = 0.5*(lo+hi); avg_range += (hi-lo); ++counted; }
    }
    avg_range = (counted > 0) ? avg_range / counted : 10.0;
    sigma_ = sigma0_ * avg_range;

    Dvec_.assign(D, 1.0);
    p_sigma_.assign(D, 0.0);
    p_c_.assign(D, 0.0);
    t_ = 0;

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    Z_.assign(lambda_, Vec(D, 0.0));
    Y_.assign(lambda_, Vec(D, 0.0));
    F_.assign(lambda_, std::numeric_limits<double>::infinity());
}

void VkDCMAES::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();

    std::normal_distribution<double> N01(0.0, 1.0);

    for (int i = 0; i < lambda_; ++i) {
        if (prob_->calls() >= max_evals_) {
            for (; i < lambda_; ++i) F_[i] = std::numeric_limits<double>::infinity();
            break;
        }
        Vec& z = Z_[i];
        Vec& y = Y_[i];
        for (int k = 0; k < D; ++k) z[k] = N01(rng_);
        y = z;
        for (int j = 0; j < k_; ++j) {
            double vn2 = 0.0; for (double vv : V_[j]) vn2 += vv * vv;
            applyVjSample(V_[j], vn2, y);
        }
        Vec x(D);
        for (int k = 0; k < D; ++k) x[k] = mean_[k] + sigma_ * Dvec_[k] * y[k];
        ensureBounds(x);
        F_[i] = safeEval(x);
    }

    std::vector<int> order(lambda_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return F_[a] < F_[b]; });

    // --- mean update: new_mean = old_mean + sigma * D .* (sum w_i y_{i:lambda}) ---
    Vec mean_old = mean_;
    Vec z_w(D, 0.0), y_w(D, 0.0);
    for (int i = 0; i < mu_; ++i) {
        const int idx = order[i];
        for (int k = 0; k < D; ++k) {
            z_w[k] += w_[i] * Z_[idx][k];
            y_w[k] += w_[i] * Y_[idx][k];
        }
    }
    for (int k = 0; k < D; ++k) mean_[k] = mean_old[k] + sigma_ * Dvec_[k] * y_w[k];

    // --- cumulation ---
    const double ps_scale = std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_);
    for (int k = 0; k < D; ++k) p_sigma_[k] = (1.0 - c_sigma_) * p_sigma_[k] + ps_scale * z_w[k];

    double ps_norm2 = 0.0;
    for (double v : p_sigma_) ps_norm2 += v*v;
    ++t_;
    const double hsig_thresh = (2.0 + 4.0/(D+1.0)) * (1.0 - std::pow(1.0 - c_sigma_, 2.0*t_));
    const bool hsig = (ps_norm2 / D) < hsig_thresh;

    const double pc_scale = hsig ? std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_) / sigma_ : 0.0;
    Vec dmean(D);
    for (int k = 0; k < D; ++k) dmean[k] = mean_[k] - mean_old[k];
    for (int k = 0; k < D; ++k) p_c_[k] = (1.0 - c_c_) * p_c_[k] + pc_scale * dmean[k];

    // --- step size ---
    const double chiD = std::sqrt((double)D) * (1.0 - 1.0/(4.0*D) + 1.0/(21.0*D*D));
    sigma_ *= std::exp((c_sigma_ / d_sigma_) * (std::sqrt(ps_norm2) / chiD - 1.0));

    // --- v_j and D updates (see class-level comment: per-vector Theorem
    // 3.6 update, independent across j; D-update averaged across the
    // active vectors). Each vector j has its own rank-mu rate c_v_[j]
    // (LM-MA-ES-style exponential spacing) and a PROPORTIONALLY scaled
    // rank-1 rate c1_j = c1_ * (c_v_[j]/cmu_), so vector j's rank-1 and
    // rank-mu contributions scale together consistently instead of being
    // mixed in mismatched units. ---
    Vec s_accum(D, 0.0);
    int active_vectors = 0;
    for (int j = 0; j < k_; ++j) {
        double vn2 = 0.0; for (double vv : V_[j]) vn2 += vv*vv;
        if (vn2 < 1e-12) continue; // degenerate vector, skip its contribution this step
        ++active_vectors;

        const double c1_j = c1_ * (c_v_[j] / cmu_);

        Vec grad_s_sum(D, 0.0), grad_t_sum(D, 0.0);
        for (int i = 0; i < mu_; ++i) {
            const int idx = order[i];
            Vec gs, gt;
            naturalGradientOneVector(Y_[idx], V_[j], vn2, gs, gt);
            for (int k = 0; k < D; ++k) {
                grad_s_sum[k] += c_v_[j] * w_[i] * gs[k];
                grad_t_sum[k] += c_v_[j] * w_[i] * gt[k];
            }
        }

        if (!hsig) {
            Vec ycoord(D);
            for (int k = 0; k < D; ++k) ycoord[k] = p_c_[k] / (sigma_ * Dvec_[k]);
            Vec gs, gt;
            naturalGradientOneVector(ycoord, V_[j], vn2, gs, gt);
            for (int k = 0; k < D; ++k) {
                grad_s_sum[k] += c1_j * gs[k];
                grad_t_sum[k] += c1_j * gt[k];
            }
        }

        for (int k = 0; k < D; ++k) V_[j][k] += grad_t_sum[k];
        for (int k = 0; k < D; ++k) s_accum[k] += grad_s_sum[k];
    }
    if (active_vectors > 0) {
        for (int k = 0; k < D; ++k) {
            const double d_update = s_accum[k] / active_vectors;
            // Multiplicative, sign-preserving update (D must stay strictly
            // positive): exp() of the small additive natural-gradient step
            // is a standard, numerically safe way to apply it.
            Dvec_[k] *= std::exp(0.5 * d_update);
            if (Dvec_[k] < 1e-12) Dvec_[k] = 1e-12;
        }
    }

    // --- adaptive k: grow when the newest vector keeps showing a
    // sustained, non-negligible norm (evidence of still-useful anisotropic
    // signal); see class-level comment for the scope of this heuristic ---
    if (k_cfg_ <= 0 && k_ < k_max_) {
        double newest_norm2 = 0.0;
        for (double vv : V_[k_ - 1]) newest_norm2 += vv*vv;
        if (newest_norm2 > 1e-6) ++grow_streak_; else grow_streak_ = 0;
        if (grow_streak_ >= grow_patience_) {
            ++k_;
            grow_streak_ = 0;
            std::normal_distribution<double> N01small(0.0, 1.0);
            for (int kk = 0; kk < D; ++kk) V_[k_ - 1][kk] = N01small(rng_) / std::sqrt((double)D);
        }
    }

    // Optional in-run local search after a successful global-best improvement.
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) { best_f_ = floc; best_x_ = xloc; }
        }
    }

    printBest();
    updateStop(F_);
}

void VkDCMAES::end() {
    if (!end_local_refine_)        return;
    if (!prob_)                    return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    double floc      = refinement.second;
    if (floc < best_f_) { best_f_ = floc; best_x_ = xloc; }
    printBest();
}

} // namespace optimsolution
