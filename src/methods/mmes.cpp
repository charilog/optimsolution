#include "mmes.h"
#include <numeric>
#include <limits>

namespace optimsolution {

namespace { inline double normCdf(double z) { return 0.5 * (1.0 + std::erf(z / 1.4142135623730951)); } }

void MMES::configure(const MethodConfig& mc) {
    lambda_cfg_  = mc.getInt("lambda", lambda_cfg_);
    m_dirs_cfg_  = mc.getInt("m_dirs", m_dirs_cfg_);
    l_mix_       = mc.getInt("l_mix", l_mix_);
    sigma0_      = mc.getDbl("sigma0", sigma0_);
    c_a_cfg_     = mc.getDbl("c_a", c_a_cfg_);
    c_c_cfg_     = mc.getDbl("c_c", c_c_cfg_);
    c_sigma_     = mc.getDbl("c_sigma", c_sigma_);
    d_sigma_     = mc.getDbl("d_sigma", d_sigma_);
    alpha_z_     = mc.getDbl("alpha_z", alpha_z_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

double MMES::safeEval(const Vec& x) {
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

void MMES::ensureBounds(Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

int MMES::sampleTruncatedGeometric() {
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const double u = U01(rng_);
    // Inverse-CDF sample of Geometric(c_a) truncated to {0,...,m_dirs_-1}.
    // gamma_ = 1-(1-c_a_)^m_dirs_ is exactly the truncated distribution's
    // normalising constant, which is why the paper defines gamma this way
    // -- it lets this sampling step reuse the same already-computed value
    // (see class-level comment). Rejection sampling would instead require,
    // on average, 1/gamma_ draws per sample -- for the paper's own
    // parameter choice (c_a=4/n, m=2*ceil(sqrt(n))), 1/c_a = n/4 is much
    // larger than m for any sizeable n, making naive rejection sampling
    // prohibitively slow.
    const double num = std::log(std::max(1e-300, 1.0 - u * gamma_));
    const double den = std::log(1.0 - c_a_);
    int k = (int)std::ceil(num / den) - 1;
    if (k < 0) k = 0;
    if (k > m_dirs_ - 1) k = m_dirs_ - 1;
    return k;
}

void MMES::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;
    const double n = (double)D;

    lambda_ = (lambda_cfg_ > 0) ? lambda_cfg_
                                 : std::max(4, 4 + (int)std::floor(3.0 * std::log(n)));
    mu_ = std::max(1, lambda_ / 2);

    w_.assign(mu_, 0.0);
    {
        double denom = 0.0;
        std::vector<double> raw(mu_);
        for (int i = 1; i <= mu_; ++i) {
            raw[i - 1] = std::log((double)mu_ + 0.5) - std::log((double)i);
            denom += raw[i - 1];
        }
        double sumw2 = 0.0;
        for (int i = 0; i < mu_; ++i) { w_[i] = raw[i] / denom; sumw2 += w_[i] * w_[i]; }
        mu_eff_ = 1.0 / sumw2;
    }

    m_dirs_ = (m_dirs_cfg_ > 0) ? m_dirs_cfg_
                                 : std::max(2, 2 * (int)std::ceil(std::sqrt(n)));

    c_a_ = (c_a_cfg_ > 0.0) ? c_a_cfg_ : 4.0 / n;
    c_c_ = (c_c_cfg_ > 0.0) ? c_c_cfg_ : std::sqrt(0.4 / n);
    T_   = std::max(1, (int)std::ceil(1.0 / c_c_));
    gamma_ = 1.0 - std::pow(1.0 - c_a_, (double)m_dirs_);

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

    p_.assign(D, 0.0);
    W_ = 0.0;

    q_.assign(m_dirs_, Vec(D, 0.0));
    t_stamp_.assign(m_dirs_, 0);
    v_.assign(m_dirs_, 0);
    for (int i = 0; i < m_dirs_; ++i) v_[i] = i;

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    const double f_mean = safeEval(mean_);
    Y_.assign(lambda_, f_mean);
    Y_bak_.assign(lambda_, f_mean);
    X_.assign(lambda_, Vec(D, 0.0));

    n_generations_ = 0;
}

void MMES::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();

    Y_bak_ = Y_; // previous generation's sorted-ascending fitness (for PTA)

    std::normal_distribution<double> N01(0.0, 1.0);
    const double a0 = std::sqrt(1.0 - gamma_);
    const double ak = std::sqrt(gamma_ / (double)l_mix_);

    for (int i = 0; i < lambda_; ++i) {
        if (prob_->calls() >= max_evals_) {
            for (; i < lambda_; ++i) Y_[i] = std::numeric_limits<double>::infinity();
            break;
        }
        Vec z(D);
        for (int j = 0; j < D; ++j) z[j] = a0 * N01(rng_);

        for (int k = 0; k < l_mix_; ++k) {
            const double zk = N01(rng_);
            const int jdraw = sampleTruncatedGeometric();          // 0..m_dirs_-1
            const int logical_pos = (m_dirs_ - 1 - jdraw) % m_dirs_; // favours recent slots
            const int phys = v_[logical_pos];
            const Vec& qj = q_[phys];
            const double coeff = ak * zk;
            for (int j = 0; j < D; ++j) z[j] += coeff * qj[j];
        }

        Vec x(D);
        for (int j = 0; j < D; ++j) x[j] = mean_[j] + sigma_ * z[j];
        ensureBounds(x);
        X_[i] = x;
        Y_[i] = safeEval(x);
    }
    ++n_generations_;

    // --- selection & mean / evolution-path update ---
    std::vector<int> order(lambda_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return Y_[a] < Y_[b]; });

    Vec new_mean(D, 0.0);
    for (int i = 0; i < mu_; ++i) {
        const int idx = order[i];
        for (int j = 0; j < D; ++j) new_mean[j] += w_[i] * X_[idx][j];
    }

    Vec mean_old = mean_;
    const double p_scale = std::sqrt(c_c_ * (2.0 - c_c_) * mu_eff_);
    for (int j = 0; j < D; ++j) {
        p_[j] = (1.0 - c_c_) * p_[j] + p_scale * (new_mean[j] - mean_old[j]) / sigma_;
    }
    mean_ = new_mean;

    // Y_ sorted ascending in place, so the NEXT generation's Y_bak_
    // snapshot has its ranked-by-position fitnesses ready for PTA.
    std::sort(Y_.begin(), Y_.end());

    // --- direction-vector memory eviction (same ring-buffer idea as
    // LM-CMA-ES's updateSet() / Rm-ES's snapshot memory -- see class-level
    // comment; MMES's own paper cites LM-CMA-ES directly for this part) ---
    {
        double min_gap = std::numeric_limits<double>::infinity();
        int kstar = 0; // 0-indexed logical position, 1..m_dirs_-1 range (paper's k* in {2,...,m})
        for (int k = 1; k < m_dirs_; ++k) {
            const double gap = (double)(t_stamp_[v_[k]] - t_stamp_[v_[k - 1]]);
            if (gap < min_gap) { min_gap = gap; kstar = k; }
        }
        if (min_gap >= (double)T_) kstar = 0;

        const int evicted_slot = v_[kstar];
        for (int k = kstar; k + 1 < m_dirs_; ++k) v_[k] = v_[k + 1];
        v_[m_dirs_ - 1] = evicted_slot;
        t_stamp_[evicted_slot] = n_generations_;
        q_[evicted_slot] = p_;
    }

    // --- Paired Test Adaptation (PTA) step-size control ---
    {
        double Lstat = 0.0;
        for (int i = 0; i < mu_; ++i) {
            if (Y_bak_[i] > Y_[i]) Lstat += w_[i];
        }
        const double w_scale = std::sqrt(c_sigma_ * (2.0 - c_sigma_) * mu_eff_);
        W_ = (1.0 - c_sigma_) * W_ + w_scale * (2.0 * Lstat - 1.0);
        sigma_ *= std::exp((1.0 / d_sigma_) * (normCdf(W_) - 1.0 + alpha_z_));
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
    updateStop(Y_);
}

void MMES::end() {
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
