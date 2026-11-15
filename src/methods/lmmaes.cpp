#include "lmmaes.h"
#include <numeric>
#include <algorithm>
#include <limits>

namespace optimsolution {

void LMMAES::configure(const MethodConfig& mc) {
    lambda_cfg_ = mc.getInt("lambda", lambda_cfg_);
    // Also honor the generic "population" key used by the other methods in
    // optisolution.cfg. population=0 (or unset) leaves lambda_cfg_ untouched
    // so that init() falls back to the literature-recommended default,
    // lambda = 4 + floor(3*ln(D)); a positive value overrides it.
    const int pop_override = mc.getInt("population", 0);
    if (pop_override > 0) lambda_cfg_ = pop_override;
    m_dirs_cfg_ = mc.getInt("m_dirs", m_dirs_cfg_);
    sigma0_     = mc.getDbl("sigma0", sigma0_);

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

double LMMAES::safeEval(const Vec& x) {
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) {
        best_f_ = f;
        best_x_ = x;
    }
    return f;
}

void LMMAES::ensureBounds(Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void LMMAES::init() {
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    const double n = (double)D;

    // --- population / recombination (standard (mu/mu_w,lambda)-ES sizing) ---
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
        for (int i = 0; i < mu_; ++i) {
            w_[i] = raw[i] / denom;
            sumw2 += w_[i] * w_[i];
        }
        mu_eff_ = 1.0 / sumw2;
    }

    // --- LM-MA-ES specific: number of stored direction vectors and their
    // per-vector, exponentially-spaced learning rates (Loshchilov, Glasmachers
    // & Beyer, arXiv:1705.06693, Algorithm 1, LM-MA-ES branches) ---
    m_dirs_ = (m_dirs_cfg_ > 0) ? m_dirs_cfg_
                                 : std::max(4, 4 + (int)std::floor(3.0 * std::log(n)));
    // BUGFIX (small dimensions): LM-MA-ES's c_sigma = 2*lambda/n (Loshchilov,
    // Glasmachers & Beyer 2019) is a LARGE-scale setting (n >> lambda). For
    // D = 10, lambda = 10 it gives c_sigma = 2: sqrt(c(2-c)) = 0, p_sigma stays
    // 0 and sigma shrinks by e^-1 every generation (premature convergence --
    // sphere-10D ended at f ~ 800); for D < 10 it is > 2 and the square root
    // is NaN. Outside its valid range (2*lambda/n >= 1) the MA-ES value
    // (Beyer & Sendhoff 2017, the full-matrix parent of LM-MA-ES)
    // c_sigma = (mu_eff + 2) / (n + mu_eff + 5) is used; for 2*lambda/n < 1 the
    // paper's setting is unchanged. c_c,j / c_d,j are likewise kept <= 1.
    {
        const double lm = 2.0 * (double)lambda_ / n;
        c_sigma_ = (lm < 1.0) ? lm : (mu_eff_ + 2.0) / (n + mu_eff_ + 5.0);
    }

    c_d_.assign(m_dirs_, 0.0);
    c_c_.assign(m_dirs_, 0.0);
    for (int j = 0; j < m_dirs_; ++j) {
        // 1-indexed i = j+1 in the paper's c_{d,i}=1/(1.5^{i-1} n),
        // c_{c,i}=lambda/(4^{i-1} n) -- (i-1) becomes exactly j here.
        c_d_[j] = std::min(1.0, 1.0 / (std::pow(1.5, (double)j) * n));
        c_c_[j] = std::min(1.0, (double)lambda_ / (std::pow(4.0, (double)j) * n));
    }

    // --- initial state ---
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    y_.assign(D, 0.0);
    double avg_range = 0.0;
    int counted = 0;
    for (int j = 0; j < D; ++j) {
        const double lo = L[j], hi = U[j];
        if (std::isfinite(lo) && std::isfinite(hi)) {
            y_[j] = 0.5 * (lo + hi);
            avg_range += (hi - lo);
            ++counted;
        } else {
            y_[j] = 0.0;
        }
    }
    avg_range = (counted > 0) ? (avg_range / counted) : 10.0;
    sigma_ = sigma0_ * avg_range;

    p_sigma_.assign(D, 0.0);
    m_.assign(m_dirs_, Vec(D, 0.0));

    t_ = 0;
    best_x_.clear();
    best_f_ = std::numeric_limits<double>::infinity();

    Z_.assign(lambda_, Vec(D, 0.0));
    Dd_.assign(lambda_, Vec(D, 0.0));
    F_.assign(lambda_, std::numeric_limits<double>::infinity());
}

void LMMAES::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();

    std::normal_distribution<double> N01(0.0, 1.0);
    const int active = std::min(t_, m_dirs_);

    for (int i = 0; i < lambda_; ++i) {
        if (prob_->calls() >= max_evals_) {
            // Out of budget mid-generation: fill remaining slots with a
            // (very unfavourable) sentinel fitness so sorting/recombination
            // below stays well-defined; they simply will not be selected.
            for (; i < lambda_; ++i) F_[i] = std::numeric_limits<double>::infinity();
            break;
        }

        Vec& z = Z_[i];
        Vec& d = Dd_[i];
        for (int k = 0; k < D; ++k) z[k] = N01(rng_);
        d = z;

        // d <- ((1-c_d,j) I + c_d,j m_j m_j^T) d, applied sequentially for
        // j = 1..min(t, m_dirs_) -- this IS the transformation matrix M
        // applied to z, without M ever being materialised (O(D) per stored
        // vector, O(active*D) total).
        for (int j = 0; j < active; ++j) {
            double proj = 0.0;
            for (int k = 0; k < D; ++k) proj += m_[j][k] * d[k];
            const double cd = c_d_[j];
            for (int k = 0; k < D; ++k) {
                d[k] = (1.0 - cd) * d[k] + cd * proj * m_[j][k];
            }
        }

        Vec x(D);
        for (int k = 0; k < D; ++k) x[k] = y_[k] + sigma_ * d[k];
        ensureBounds(x);
        // BUGFIX (box constraints): only the evaluated point used to be
        // clamped while the update used the unclamped d_i / z_i. The mean then
        // drifted outside the box, where all clamped samples look alike, and
        // the run stalled (shifted sphere-10D: f ~ 3e2..6e2). Lamarckian
        // repair: d_i = (x - y) / sigma for the point actually evaluated, and
        // z_i = T^{-1} d_i, inverting the transform d = T z exactly; each
        // factor A_j = (1-c)I + c m m^T has the Sherman-Morrison inverse
        // A_j^{-1} = [I - c/(1-c+c|m|^2) m m^T] / (1-c), applied in reverse.
        bool clipped = false;
        for (int k = 0; k < D; ++k) if (x[k] != y_[k] + sigma_ * d[k]) { clipped = true; break; }
        if (clipped && sigma_ > 0.0) {
            for (int k = 0; k < D; ++k) d[k] = (x[k] - y_[k]) / sigma_;
            z = d;
            for (int j = active - 1; j >= 0; --j) {
                const double cd = c_d_[j];
                if (cd >= 1.0) continue;   // not invertible (cannot happen: c_d <= 1/n)
                double mm = 0.0, md = 0.0;
                for (int k = 0; k < D; ++k) { mm += m_[j][k] * m_[j][k]; md += m_[j][k] * z[k]; }
                const double beta = cd / (1.0 - cd + cd * mm);
                for (int k = 0; k < D; ++k) z[k] = (z[k] - beta * md * m_[j][k]) / (1.0 - cd);
            }
        }
        F_[i] = safeEval(x);
    }

    // --- selection: best mu_ offspring by fitness ---
    std::vector<int> order(lambda_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return F_[a] < F_[b]; });

    Vec z_w(D, 0.0), d_w(D, 0.0);
    for (int i = 0; i < mu_; ++i) {
        const int idx = order[i];
        const double wi = w_[i];
        const Vec& zi = Z_[idx];
        const Vec& di = Dd_[idx];
        for (int k = 0; k < D; ++k) {
            z_w[k] += wi * zi[k];
            d_w[k] += wi * di[k];
        }
    }

    // --- mean update ---
    for (int k = 0; k < D; ++k) y_[k] += sigma_ * d_w[k];

    // --- isotropic evolution path (step-size control) ---
    const double ps_scale = std::sqrt(mu_eff_ * c_sigma_ * (2.0 - c_sigma_));
    for (int k = 0; k < D; ++k) {
        p_sigma_[k] = (1.0 - c_sigma_) * p_sigma_[k] + ps_scale * z_w[k];
    }

    // --- the m_dirs_ stored direction vectors, each on its own exponentially
    // spaced time scale (this IS the whole LM-MA-ES idea: many cheap,
    // differently-paced fading records instead of one n x n matrix) ---
    for (int j = 0; j < m_dirs_; ++j) {
        const double cc = c_c_[j];
        const double scale = std::sqrt(mu_eff_ * cc * (2.0 - cc));
        Vec& mj = m_[j];
        for (int k = 0; k < D; ++k) {
            mj[k] = (1.0 - cc) * mj[k] + scale * z_w[k];
        }
    }

    // --- step-size update (simplified CSA, no separate damping parameter --
    // see header/class comment: this exact form is Algorithm 1, line 19) ---
    double ps_norm_sq = 0.0;
    for (double v : p_sigma_) ps_norm_sq += v * v;
    sigma_ *= std::exp((c_sigma_ / 2.0) * (ps_norm_sq / (double)D - 1.0));

    ++t_;

    // Optional in-run local search after a successful global-best improvement.
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;
            }
        }
    }

    printBest();
    updateStop(F_);
}

void LMMAES::end() {
    if (!end_local_refine_)        return;
    if (!prob_)                    return;
    if (end_local_method_.empty()) return;

    auto refinement = localSearch(end_local_method_, best_x_);
    const auto& xloc = refinement.first;
    double floc      = refinement.second;

    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }
    printBest();
}

} // namespace optimsolution
