#include "vkdcmaes.h"
#include "init.h"

#include <numeric>
#include <limits>
#include <cctype>

// Port of the official VkD-CMA (Akimoto & Hansen 2016) -- see vkdcmaes.h.
// Comments "[ref: ...]" refer to the corresponding statement of the
// reference implementation (pypop7/optimizers/nes/vkdcma.py, which follows
// Y. Akimoto's official Python code).

namespace optimsolution {

void VkDCMAES::configure(const MethodConfig& mc) {
    lambda_cfg_ = mc.getInt("lambda", lambda_cfg_);
    const int pop_override = mc.getInt("population", 0);
    if (pop_override > 3) lambda_cfg_ = pop_override;
    sigma0_ = mc.getDbl("sigma0", sigma0_);
    if (!(sigma0_ > 0.0)) sigma0_ = 0.3;

    local_method_ = mc.getStr("local_method", local_method_);
    for (char& c : local_method_) c = (char)std::tolower((unsigned char)c);
    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    if (mc.kv.count("end_local_refine")) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    if (mc.kv.count("end_local_method")) end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

double VkDCMAES::safeEval(const Vec& x) {
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = std::numeric_limits<double>::infinity();
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
    return f;
}

void VkDCMAES::clipToBounds(Vec& x) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// [ref: _get_lr(k)]
void VkDCMAES::setLearningRates() {
    const double n = (double)n_, k = (double)k_;
    c1_  = 2.0 / (n * (k + 1.0) + n + 2.0 * (k + 2.0) + mu_eff_);
    cmu_ = std::min(1.0 - c1_, 2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_) /
                               (n * (k + 1.0) + 4.0 * (k + 2.0) + mu_eff_));
    if (cmu_ < 0.0) cmu_ = 0.0;
    cc_  = std::sqrt(c1_);
}

// [ref: _get_det(d, s)] = 2 sum log d + sum_{j<k_a} log(1 + s_j)
double VkDCMAES::logDet() const {
    double v = 0.0;
    for (int i = 0; i < n_; ++i) v += 2.0 * std::log(d_[i]);
    for (int j = 0; j < ka_; ++j) v += std::log(1.0 + s_[j]);
    return v;
}

// [ref: _get_m(d_x, d, v, s)]
double VkDCMAES::mahalanobisSqOfDx() const {
    Vec dxd(n_);
    double a = 0.0;
    for (int i = 0; i < n_; ++i) { dxd[i] = dx_[i] / d_[i]; a += dxd[i] * dxd[i]; }
    double b = 0.0;
    for (int j = 0; j < ka_; ++j) {
        double p = 0.0;
        for (int i = 0; i < n_; ++i) p += v_[j][i] * dxd[i];
        b += p * p * (s_[j] / (s_[j] + 1.0));
    }
    return a - b;
}

void VkDCMAES::symEig(std::vector<double> A, int m, Vec& ev, std::vector<double>& V) {
    V.assign((size_t)m * m, 0.0);
    for (int i = 0; i < m; ++i) V[(size_t)i * m + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (int p = 0; p < m; ++p) {
            diag += A[(size_t)p * m + p] * A[(size_t)p * m + p];
            for (int q = p + 1; q < m; ++q) off += A[(size_t)p * m + q] * A[(size_t)p * m + q];
        }
        if (off <= 1e-30 * std::max(1e-300, diag)) break;
        for (int p = 0; p < m; ++p) {
            for (int q = p + 1; q < m; ++q) {
                const double apq = A[(size_t)p * m + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = A[(size_t)p * m + p], aqq = A[(size_t)q * m + q];
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int r = 0; r < m; ++r) {
                    const double arp = A[(size_t)r * m + p], arq = A[(size_t)r * m + q];
                    A[(size_t)r * m + p] = c * arp - s * arq;
                    A[(size_t)r * m + q] = s * arp + c * arq;
                }
                for (int r = 0; r < m; ++r) {
                    const double apr = A[(size_t)p * m + r], aqr = A[(size_t)q * m + r];
                    A[(size_t)p * m + r] = c * apr - s * aqr;
                    A[(size_t)q * m + r] = s * apr + c * aqr;
                }
                for (int r = 0; r < m; ++r) {
                    const double vrp = V[(size_t)r * m + p], vrq = V[(size_t)r * m + q];
                    V[(size_t)r * m + p] = c * vrp - s * vrq;
                    V[(size_t)r * m + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    ev.assign(m, 0.0);
    for (int i = 0; i < m; ++i) ev[i] = A[(size_t)i * m + i];
}

void VkDCMAES::init() {
    if (!prob_) return;
    n_ = prob_->dimension();
    if (n_ <= 0) return;
    const double n = (double)n_;

    // [ref: ES base] lambda = 4 + floor(3 ln n), mu = floor(lambda/2),
    // w_i = (ln((lambda+1)/2) - ln i) / sum, mu_eff = 1 / sum w^2
    lambda_ = (lambda_cfg_ > 3) ? lambda_cfg_ : 4 + (int)std::floor(3.0 * std::log(n));
    if (lambda_ < 4) lambda_ = 4;
    mu_ = std::max(1, lambda_ / 2);
    w_.assign(mu_, 0.0);
    {
        const double wb = std::log((lambda_ + 1.0) / 2.0);
        double sum = 0.0;
        for (int i = 0; i < mu_; ++i) { w_[i] = wb - std::log(i + 1.0); sum += w_[i]; }
        double s2 = 0.0;
        for (int i = 0; i < mu_; ++i) { w_[i] /= sum; s2 += w_[i] * w_[i]; }
        mu_eff_ = 1.0 / s2;
    }
    setPopulation(lambda_);

    // [ref: initialize()]
    k_ = ka_ = ki_ = 0;
    cs_ = 0.3;
    ds_ = std::sqrt(n);
    injection_ = false;
    setLearningRates();

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double avg_range = 0.0; int counted = 0;
    for (int j = 0; j < n_; ++j)
        if (std::isfinite(L[j]) && std::isfinite(U[j])) { avg_range += U[j] - L[j]; ++counted; }
    avg_range = (counted > 0) ? avg_range / counted : 10.0;
    sigma_ = sigma0_ * avg_range;

    {
        Initializer initSampler;
        initSampler.configure(initopt_);
        auto pts = initSampler.samplePopulation(*prob_, rng_, 1);
        mean_ = pts.empty() ? Vec(n_, 0.0) : pts[0];
        mean_.resize(n_, 0.0);
    }

    d_.assign(n_, 1.0);
    ps_ = 0.0;
    v_.clear();
    s_.assign(n_, 0.0);
    pc_.assign(n_, 0.0);
    dx_.assign(n_, 0.0);
    l_s_ = std::log(sigma_);
    l_d_.assign(n_, 0.0);
    l_c_.assign(n_, 0.0);

    Z_.assign(lambda_, Vec(n_, 0.0));
    F_.assign(lambda_, std::numeric_limits<double>::infinity());

    best_x_ = mean_;
    best_f_ = std::numeric_limits<double>::infinity();
}

void VkDCMAES::one_iteration() {
    if (!prob_ || n_ <= 0) return;
    if (prob_->calls() >= max_evals_) return;
    const double n = (double)n_;
    std::normal_distribution<double> N01(0.0, 1.0);

    // ---- sampling [ref: z = (N(0,I) + N(0,1)*sqrt(s) v) * d] ----
    for (int i = 0; i < lambda_; ++i) {
        Vec& z = Z_[i];
        for (int c = 0; c < n_; ++c) z[c] = N01(rng_);
        for (int j = 0; j < ka_; ++j) {
            const double a = N01(rng_) * std::sqrt(std::max(0.0, s_[j]));
            for (int c = 0; c < n_; ++c) z[c] += a * v_[j][c];
        }
        for (int c = 0; c < n_; ++c) z[c] *= d_[c];
    }
    if (injection_) {
        // [ref: zz = (|N(0,I)| / sqrt(m(dx))) * dx; z0 = zz; z1 = -zz]
        double nrm = 0.0;
        for (int c = 0; c < n_; ++c) { const double g = N01(rng_); nrm += g * g; }
        nrm = std::sqrt(nrm);
        const double m = mahalanobisSqOfDx();
        const double scale = (m > 0.0) ? nrm / std::sqrt(m) : 0.0;
        if (lambda_ >= 2 && scale > 0.0 && std::isfinite(scale)) {
            for (int c = 0; c < n_; ++c) { Z_[0][c] = scale * dx_[c]; Z_[1][c] = -scale * dx_[c]; }
        }
    }

    // ---- evaluation (evaluated point clipped to the box) ----
    for (int i = 0; i < lambda_; ++i) {
        if (prob_->calls() >= max_evals_) {
            // [ref: return before any update when the budget is exhausted]
            printBest();
            updateStop(F_);
            return;
        }
        Vec x(n_);
        for (int c = 0; c < n_; ++c) x[c] = mean_[c] + sigma_ * Z_[i][c];
        clipToBounds(x);
        // Box constraints (not handled by the unconstrained reference code):
        // Lamarckian repair -- the distribution is updated from the point that
        // was actually evaluated, z = (clip(x) - m) / sigma, as this
        // framework's CMA-ES does. Updating from the unclipped z let the mean
        // drift outside the box, where all clipped samples look alike, and
        // the run stalled (sphere-10D: f ~ 4e3).
        if (sigma_ > 0.0)
            for (int c = 0; c < n_; ++c) Z_[i][c] = (x[c] - mean_[c]) / sigma_;
        F_[i] = safeEval(x);
    }

    std::vector<int> order(lambda_);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return F_[a] < F_[b]; });

    // ---- mean [ref: d_x = w . s_z; mean += sigma d_x] ----
    dx_.assign(n_, 0.0);
    for (int i = 0; i < mu_; ++i)
        for (int c = 0; c < n_; ++c) dx_[c] += w_[i] * Z_[order[i]][c];
    for (int c = 0; c < n_; ++c) mean_[c] += sigma_ * dx_[c];

    // ---- step size [ref: injection-based rank success] ----
    double hs = 1.0;
    if (injection_) {
        int r0 = 0, r1 = 0;
        for (int r = 0; r < lambda_; ++r) { if (order[r] == 0) r0 = r; if (order[r] == 1) r1 = r; }
        ps_ += cs_ * ((double)(r1 - r0) / (lambda_ - 1.0) - ps_);
        sigma_ *= std::exp(ps_ / ds_);
        hs = (ps_ < 0.5) ? 1.0 : 0.0;
    } else {
        injection_ = true;
        hs = 1.0;
    }

    // ---- evolution path ----
    {
        const double f = hs * std::sqrt(cc_ * (2.0 - cc_) * mu_eff_);
        for (int c = 0; c < n_; ++c) pc_[c] = (1.0 - cc_) * pc_[c] + f * dx_[c];
    }

    // ---- update matrix U (n x r_u) [ref] ----
    const double alpha = std::sqrt(std::fabs(1.0 - cmu_ - c1_ + c1_ * (1.0 - hs) * cc_ * (2.0 - cc_)));
    std::vector<Vec> cols;   // each column has size n
    cols.reserve(ka_ + mu_ + 1);
    for (int j = 0; j < ka_; ++j) {
        Vec col(n_);
        const double f = std::sqrt(std::max(0.0, s_[j])) * alpha;
        for (int c = 0; c < n_; ++c) col[c] = v_[j][c] * f;
        cols.push_back(std::move(col));
    }
    if (cmu_ > 0.0) {
        for (int i = 0; i < mu_; ++i) {
            Vec col(n_);
            const double f = std::sqrt(cmu_) * std::sqrt(w_[i]);
            for (int c = 0; c < n_; ++c) col[c] = f * Z_[order[i]][c] / d_[c];
            cols.push_back(std::move(col));
        }
    }
    if (c1_ > 0.0) {
        Vec col(n_);
        const double f = std::sqrt(c1_);
        for (int c = 0; c < n_; ++c) col[c] = f * pc_[c] / d_[c];
        cols.push_back(std::move(col));
    }
    const int ru = (int)cols.size();

    // ---- projection onto rank k [ref: eigh of U^T U or U U^T] ----
    if (ru > 0) {
        Vec dd;
        std::vector<double> E;   // eigenvectors (columns)
        const bool small = (n_ > ru);
        const int m = small ? ru : n_;
        std::vector<double> M((size_t)m * m, 0.0);
        if (small) {
            for (int a = 0; a < ru; ++a)
                for (int b = a; b < ru; ++b) {
                    double s = 0.0;
                    for (int c = 0; c < n_; ++c) s += cols[a][c] * cols[b][c];
                    M[(size_t)a * m + b] = M[(size_t)b * m + a] = s;
                }
        } else {
            for (int a = 0; a < n_; ++a)
                for (int b = a; b < n_; ++b) {
                    double s = 0.0;
                    for (int q = 0; q < ru; ++q) s += cols[q][a] * cols[q][b];
                    M[(size_t)a * m + b] = M[(size_t)b * m + a] = s;
                }
        }
        symEig(M, m, dd, E);
        std::vector<int> ie(m);
        std::iota(ie.begin(), ie.end(), 0);
        std::stable_sort(ie.begin(), ie.end(), [&](int a, int b) { return dd[a] > dd[b]; });

        double gamma = 0.0;
        if (ru > k_) {
            double tail = 0.0;
            for (int q = k_; q < m; ++q) tail += dd[ie[q]];
            gamma = tail / (n - (double)k_);
        }
        int npos = 0;
        for (int q = 0; q < m; ++q) if (dd[q] > 1e-300) ++npos;   // ref: dd >= 0 (strict here: avoids 1/sqrt(0))
        ka_ = std::min(npos, k_);
        for (int j = 0; j < ka_; ++j) {
            const int q = ie[j];
            s_[j] = (dd[q] - gamma) / (alpha * alpha + gamma);
            Vec vj(n_, 0.0);
            if (small) {
                const double inv = 1.0 / std::sqrt(dd[q]);
                for (int a = 0; a < ru; ++a) {
                    const double e = E[(size_t)a * m + q];
                    for (int c = 0; c < n_; ++c) vj[c] += cols[a][c] * e;
                }
                for (int c = 0; c < n_; ++c) vj[c] *= inv;
            } else {
                for (int c = 0; c < n_; ++c) vj[c] = E[(size_t)c * m + q];
            }
            v_[j] = std::move(vj);
        }
    }

    // ---- diagonal [ref: d *= sqrt((alpha^2 + sum_cols U^2) / (1 + sum_j s_j v_j^2))] ----
    for (int c = 0; c < n_; ++c) {
        double num = alpha * alpha;
        for (int q = 0; q < ru; ++q) num += cols[q][c] * cols[q][c];
        double den = 1.0;
        for (int j = 0; j < ka_; ++j) den += s_[j] * v_[j][c] * v_[j][c];
        d_[c] *= std::sqrt(num / den);
    }
    {
        const double e = std::exp(logDet() / n / 2.0);
        for (int c = 0; c < n_; ++c) { d_[c] /= e; pc_[c] /= e; }
    }

    // ---- online model selection of k [ref] ----
    ++ki_;
    const double e_s = std::log(sigma_) - l_s_;
    const double l_s_c = e_s / (0.5 * std::min(1.0, (double)lambda_ / n) / 3.0);
    l_s_ = std::log(sigma_);
    Vec ld_new(n_), l_d_c(n_);
    for (int c = 0; c < n_; ++c) {
        double sv = 0.0;
        for (int j = 0; j < k_; ++j) sv += s_[j] * v_[j][c] * v_[j][c];
        ld_new[c] = 2.0 * std::log(d_[c]) + std::log(1.0 + sv);
        l_d_c[c] = (ld_new[c] - l_d_[c]) / (cmu_ + c1_);
    }
    l_d_ = ld_new;
    Vec l_l_c(n_);
    for (int c = 0; c < n_; ++c) {
        const double lc = std::log(1.0 + s_[c]);
        l_l_c[c] = (lc - l_c_[c]) / (cmu_ + c1_);
        l_c_[c] = lc;
    }

    bool k_inc = (ki_ > (2 * n_ - 1)) && (k_ < (n_ - 1)) && (std::fabs(l_s_c) < 0.1);
    for (int j = 0; j < k_ && k_inc; ++j) if (!(1.0 + s_[j] > 30.0)) k_inc = false;
    for (int c = 0; c < n_ && k_inc; ++c) if (!(std::fabs(l_d_c[c]) < 1.0)) k_inc = false;
    std::vector<char> k_dec(k_, 0);
    bool any_dec = false;
    for (int j = 0; j < k_; ++j) {
        k_dec[j] = (1.0 + s_[j] < 30.0) && (l_l_c[j] < 0.0);
        any_dec = any_dec || k_dec[j];
    }

    if (k_inc) {
        ka_ = k_;
        const int new_k = std::min(std::max((int)std::ceil(k_ * 1.414), k_ + 1), n_ - 1);
        for (int j = k_; j < new_k; ++j) v_.push_back(Vec(n_, 0.0));
        k_ = new_k;
        setLearningRates();
        ki_ = 0;
    } else if (ki_ > k_ * (2 * n_ - 1) && any_dec) {
        std::vector<Vec> nv;
        Vec ns(n_, 0.0);
        int new_k = 0;
        for (int j = 0; j < k_; ++j) {
            if (!k_dec[j]) { nv.push_back(v_[j]); ns[new_k] = s_[j]; ++new_k; }
        }
        v_.swap(nv);
        s_ = ns;
        k_ = ka_ = new_k;
        setLearningRates();
    }
    {
        const double e = std::exp(logDet() / n / 2.0);
        for (int c = 0; c < n_; ++c) { d_[c] /= e; pc_[c] /= e; }
    }

    // ---- optional in-run local search (framework feature) ----
    if (local_rate_ > 0.0 && !local_method_.empty() && !best_x_.empty()) {
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
    if (!end_local_refine_ || !prob_ || end_local_method_.empty() || best_x_.empty()) return;
    auto refinement = localSearch(end_local_method_, best_x_);
    if (refinement.second < best_f_) { best_f_ = refinement.second; best_x_ = refinement.first; }
    printBest();
}

} // namespace optimsolution
