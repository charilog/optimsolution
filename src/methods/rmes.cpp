#include "rmes.h"
#include <numeric>
#include <limits>

namespace optimsolution {

void RmES::configure(const MethodConfig& mc) {
    lambda_cfg_        = mc.getInt("lambda", lambda_cfg_);
    m_paths_           = mc.getInt("m_paths", m_paths_);
    generation_gap_cfg_= mc.getInt("generation_gap", generation_gap_cfg_);
    sigma0_            = mc.getDbl("sigma0", sigma0_);
    c_cov_cfg_         = mc.getDbl("c_cov", c_cov_cfg_);
    c_cfg_             = mc.getDbl("c", c_cfg_);
    c_s_               = mc.getDbl("c_s", c_s_);
    q_star_            = mc.getDbl("q_star", q_star_);
    d_sigma_           = mc.getDbl("d_sigma", d_sigma_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

double RmES::safeEval(const Vec& x) {
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

void RmES::ensureBounds(Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void RmES::init() {
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

    if (m_paths_ < 1) m_paths_ = 1;

    c_cov_ = (c_cov_cfg_ > 0.0) ? c_cov_cfg_ : 1.0 / (3.0 * std::sqrt(n) + 5.0);
    c_     = (c_cfg_ > 0.0)     ? c_cfg_     : 2.0 / (n + 7.0);

    a_   = std::sqrt(1.0 - c_cov_);
    b_   = std::sqrt(c_cov_);
    a_m_ = std::pow(a_, (double)m_paths_);

    p1_ = 1.0 - c_;
    p2_ = std::sqrt(c_ * (2.0 - c_) * mu_eff_);

    generation_gap_ = (generation_gap_cfg_ > 0) ? generation_gap_cfg_ : D;

    rr_full_.assign(2 * mu_, 0);
    for (int i = 0; i < 2 * mu_; ++i) rr_full_[i] = i + 1;

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
    s_ = 0.0;
    mp_.assign(m_paths_, Vec(D, 0.0));
    t_hat_.assign(m_paths_, 0.0);

    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    const double f_mean = safeEval(mean_);
    Y_.assign(lambda_, f_mean); // R1ES/Rm-ES initialize: y <- tile(f(mean), lambda)
    Y_bak_.assign(lambda_, f_mean);
    X_.assign(lambda_, Vec(D, 0.0));

    n_generations_ = 0;
}

void RmES::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();

    Y_bak_ = Y_; // snapshot of the PREVIOUS generation's (already sorted ascending) fitness

    std::normal_distribution<double> N01(0.0, 1.0);

    for (int k = 0; k < lambda_; ++k) {
        if (prob_->calls() >= max_evals_) {
            for (; k < lambda_; ++k) Y_[k] = std::numeric_limits<double>::infinity();
            break;
        }
        Vec z(D);
        for (int j = 0; j < D; ++j) z[j] = N01(rng_);

        Vec sum_p(D, 0.0);
        for (int i = 1; i <= m_paths_; ++i) {
            const double r = N01(rng_);
            const double coeff = std::pow(a_, (double)(m_paths_ - i)) * r;
            const Vec& mpi = mp_[i - 1];
            for (int j = 0; j < D; ++j) sum_p[j] += coeff * mpi[j];
        }

        Vec x(D);
        for (int j = 0; j < D; ++j) x[j] = mean_[j] + sigma_ * (a_m_ * z[j] + b_ * sum_p[j]);
        ensureBounds(x);
        X_[k] = x;
        Y_[k] = safeEval(x);
    }
    ++n_generations_;

    // --- selection & mean/principal-direction update (R1ES core) ---
    std::vector<int> order(lambda_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return Y_[a] < Y_[b]; });

    Vec new_mean(D, 0.0);
    for (int i = 0; i < mu_; ++i) {
        const int idx = order[i];
        for (int j = 0; j < D; ++j) new_mean[j] += w_[i] * X_[idx][j];
    }

    Vec mean_old = mean_;
    for (int j = 0; j < D; ++j) {
        p_[j] = p1_ * p_[j] + p2_ * (new_mean[j] - mean_old[j]) / sigma_;
    }
    mean_ = new_mean;

    // Y_ sorted ascending in place (matches Python's `y.sort()`), so that
    // the NEXT generation's Y_bak_ snapshot has its best-mu at the front.
    std::sort(Y_.begin(), Y_.end());

    // --- rank-based success rule (RSR) step-size control ---
    {
        std::vector<double> combined(2 * mu_);
        for (int i = 0; i < mu_; ++i) combined[i] = Y_bak_[i];
        for (int i = 0; i < mu_; ++i) combined[mu_ + i] = Y_[i];

        std::vector<int> r(2 * mu_);
        std::iota(r.begin(), r.end(), 0);
        std::sort(r.begin(), r.end(), [&](int a, int b){ return combined[a] < combined[b]; });

        std::vector<int> rrA, rrB; // ranks of elements originating from Y_bak_ / Y_ respectively
        rrA.reserve(mu_); rrB.reserve(mu_);
        for (int i = 0; i < 2 * mu_; ++i) {
            if (r[i] < mu_) rrA.push_back(rr_full_[i]);
            else            rrB.push_back(rr_full_[i]);
        }
        const int npairs = std::min((int)rrA.size(), (int)rrB.size());
        double q = 0.0;
        for (int i = 0; i < npairs; ++i) {
            const double rr = (double)(rrA[i] - rrB[i]);
            q += w_[i] * rr;
        }
        q /= (double)mu_;

        s_ = (1.0 - c_s_) * s_ + c_s_ * (q - q_star_);
        sigma_ *= std::exp(s_ / d_sigma_);
    }

    // --- multiple evolution-path snapshot memory (Rm-ES specific) ---
    if (m_paths_ == 1) {
        mp_[0] = p_;
        t_hat_[0] = (double)n_generations_;
    } else {
        double t_min = std::numeric_limits<double>::infinity();
        int i_apostrophe = 0;
        for (int i = 0; i + 1 < m_paths_; ++i) {
            const double diff = t_hat_[i + 1] - t_hat_[i];
            if (diff < t_min) { t_min = diff; i_apostrophe = i + 1; }
        }
        if (t_min > (double)generation_gap_ || n_generations_ < m_paths_) {
            i_apostrophe = 0;
        }
        for (int i = i_apostrophe; i + 1 < m_paths_; ++i) {
            mp_[i] = mp_[i + 1];
            t_hat_[i] = t_hat_[i + 1];
        }
        mp_[m_paths_ - 1] = p_;
        t_hat_[m_paths_ - 1] = (double)n_generations_;
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

void RmES::end() {
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
