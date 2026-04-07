#include "arqdp.h"
#include "init.h"
#include "options.h"

#include <utility>

namespace optimsolution {

static inline std::string to_lower(std::string s) {
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

double ARQDP::clamp01(double x) {
    if (!std::isfinite(x)) return 0.5;
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// Reflect x into [0,1] (avoids boundary-sticking of clamp01).
static inline double reflect01(double x)
{
    if (!std::isfinite(x)) return 0.5;
    for (int k = 0; k < 6 && (x < 0.0 || x > 1.0); ++k) {
        if (x < 0.0) x = -x;
        if (x > 1.0) x = 2.0 - x;
    }
    if (x < 0.0) x = 0.0;
    if (x > 1.0) x = 1.0;
    return x;
}

void ARQDP::ensureDim(Vec& x, int D) const
{
    if (D <= 0) { x.clear(); return; }
    if ((int)x.size() == D) return;

    Vec y(D, 0.0);
    const int m = std::min(D, (int)x.size());
    for (int i = 0; i < m; ++i) y[i] = x[i];

    for (int j = m; j < D; ++j) {
        double L = (bounds_ready_ && j < (int)Lc_.size()) ? Lc_[j] : -1.0;
        double U = (bounds_ready_ && j < (int)Uc_.size()) ? Uc_[j] : +1.0;
        if (!std::isfinite(L)) L = -1.0;
        if (!std::isfinite(U)) U = +1.0;
        if (U <= L) { L = -1.0; U = +1.0; }
        y[j] = 0.5 * (L + U);
    }

    x.swap(y);
}

void ARQDP::prepareBoundsCache()
{
    bounds_ready_ = false;
    Lc_.clear(); Uc_.clear();
    if (!prob_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    const auto& Lraw = prob_->lb();
    const auto& Uraw = prob_->ub();

    Lc_.assign(D, -1.0);
    Uc_.assign(D, +1.0);

    auto get_safe = [&](const std::vector<double>& v, int j, double fallback) -> double {
        if (v.empty()) return fallback;
        if ((int)v.size() == 1) return v[0];
        if (j < (int)v.size()) return v[j];
        return v.back();
    };

    for (int j = 0; j < D; ++j) {
        double Lj = get_safe(Lraw, j, -1.0);
        double Uj = get_safe(Uraw, j, +1.0);

        if (!std::isfinite(Lj)) Lj = -1.0;
        if (!std::isfinite(Uj)) Uj = +1.0;

        if (Uj <= Lj) {
            const double mid = 0.5 * (Lj + Uj);
            Lj = mid - 1.0;
            Uj = mid + 1.0;
        }

        Lc_[j] = Lj;
        Uc_[j] = Uj;
    }

    bounds_ready_ = true;
}

double ARQDP::reflectInto(double x, double L, double U)
{
    if (!std::isfinite(x)) return 0.5 * (L + U);
    if (!std::isfinite(L) || !std::isfinite(U) || U <= L) return x;

    const double range = U - L;
    double t = std::fmod(x - L, 2.0 * range);
    if (t < 0.0) t += 2.0 * range;

    if (t <= range) return L + t;
    return U - (t - range);
}

void ARQDP::ensureBounds(Vec& x)
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (!bounds_ready_) prepareBoundsCache();
    ensureDim(x, D);

    for (int j = 0; j < D; ++j) {
        const double L = (j < (int)Lc_.size() ? Lc_[j] : -1.0);
        const double U = (j < (int)Uc_.size() ? Uc_[j] : +1.0);
        x[j] = reflectInto(x[j], L, U);
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L + U);
    }
}

void ARQDP::sanitizeInPlace(Vec& x)
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) { x.clear(); return; }

    if (!bounds_ready_) prepareBoundsCache();
    ensureDim(x, D);

    for (int j = 0; j < D; ++j) {
        double L = (j < (int)Lc_.size()) ? Lc_[j] : -1.0;
        double U = (j < (int)Uc_.size()) ? Uc_[j] : +1.0;
        if (!std::isfinite(L)) L = -1.0;
        if (!std::isfinite(U)) U = +1.0;
        if (U <= L) { L = -1.0; U = +1.0; }

        if (!std::isfinite(x[j])) x[j] = 0.5 * (L + U);
        x[j] = reflectInto(x[j], L, U);
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L + U);
    }
}


void ARQDP::recomputeBest() {
    best_f_ = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_; ++i) {
        sanitizeInPlace(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
    }
}
ARQDP::Vec ARQDP::sanitizedCopy(const Vec& x)
{
    Vec y = x;
    sanitizeInPlace(y);
    return y;
}

double ARQDP::evalX(const Vec& v)
{
    if (!prob_) return std::numeric_limits<double>::infinity();
    Vec x = sanitizedCopy(v);
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = 1e100;
    return f;
}

int ARQDP::randInt(int lo, int hi)
{
    if (hi < lo) std::swap(lo, hi);
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

int ARQDP::pickIndexExcluding(const std::vector<int>& exclude)
{
    if (N_ <= 0) return -1;
    for (int t = 0; t < 32; ++t) {
        int r = randInt(0, N_ - 1);
        bool ok = true;
        for (int e : exclude) if (r == e) { ok = false; break; }
        if (ok) return r;
    }
    for (int r = 0; r < N_; ++r) {
        bool ok = true;
        for (int e : exclude) if (r == e) { ok = false; break; }
        if (ok) return r;
    }
    return -1;
}


double ARQDP::randU()
{
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double ARQDP::randN01()
{
    std::normal_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double ARQDP::cauchy(double loc, double scale)
{
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

void ARQDP::sampleDistinctExcluding(int N, int k,
                                    const std::vector<int>& exclude,
                                    std::vector<int>& out)
{
    out.clear();
    out.reserve(k);
    if (N <= 0 || k <= 0) return;

    std::vector<int> cand;
    cand.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (std::find(exclude.begin(), exclude.end(), i) == exclude.end())
            cand.push_back(i);
    }
    if ((int)cand.size() <= k) { out = cand; return; }

    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)cand.size() - 1);
        int r = dist(rng_);
        std::swap(cand[i], cand[r]);
        out.push_back(cand[i]);
    }
}

void ARQDP::addToArchive(const Vec& x)
{
    if (!prob_) return;
    Vec y = x;
    sanitizeInPlace(y);
    A_.push_back(std::move(y));
}

void ARQDP::trimArchive()
{
    if (N_ <= 0) return;
    int cap = (int)std::round(archiverate_ * (double)N_);
    if (cap < 0) cap = 0;
    if ((int)A_.size() <= cap) return;
    std::shuffle(A_.begin(), A_.end(), rng_);
    A_.resize(cap);
}

void ARQDP::sortByFitness(std::vector<int>& idx) const
{
    idx.resize(N_);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });
}

double ARQDP::quantile(std::vector<double> v, double q01)
{
    if (v.empty()) return std::numeric_limits<double>::infinity();
    q01 = std::max(0.0, std::min(1.0, q01));

    const double pos  = q01 * (double)(v.size() - 1);
    const size_t k    = (size_t)std::floor(pos);
    const double frac = pos - (double)k;

    std::nth_element(v.begin(), v.begin() + k, v.end());
    double a = v[k];
    if (k + 1 >= v.size()) return a;
    std::nth_element(v.begin(), v.begin() + (k + 1), v.end());
    double b = v[k + 1];
    return a + frac * (b - a);
}

void ARQDP::toBN(const Vec& x, Vec& y) const
{
    if (!prob_) { y.clear(); return; }
    const int D = prob_->dimension();
    if (D <= 0) { y.clear(); return; }

    y.assign(D, 0.0);

    for (int j = 0; j < D; ++j) {
        double xv;
        if (j < (int)x.size() && std::isfinite(x[j])) xv = x[j];
        else {
            double L = (bounds_ready_ && j < (int)Lc_.size()) ? Lc_[j] : -1.0;
            double U = (bounds_ready_ && j < (int)Uc_.size()) ? Uc_[j] : +1.0;
            if (!std::isfinite(L)) L = -1.0;
            if (!std::isfinite(U)) U = +1.0;
            if (U <= L) { L = -1.0; U = +1.0; }
            xv = 0.5 * (L + U);
        }

        double L = (bounds_ready_ && j < (int)Lc_.size()) ? Lc_[j] : -1.0;
        double U = (bounds_ready_ && j < (int)Uc_.size()) ? Uc_[j] : +1.0;
        double denom = U - L;
        if (!std::isfinite(denom) || denom <= 0.0) denom = 1.0;

        y[j] = clamp01((xv - L) / denom);
    }
}

void ARQDP::fromBN(const Vec& y, Vec& x) const
{
    if (!prob_) { x.clear(); return; }
    const int D = prob_->dimension();
    if (D <= 0) { x.clear(); return; }

    x.assign(D, 0.0);
    for (int j = 0; j < D; ++j) {
        double L = (bounds_ready_ && j < (int)Lc_.size()) ? Lc_[j] : -1.0;
        double U = (bounds_ready_ && j < (int)Uc_.size()) ? Uc_[j] : +1.0;
        double denom = U - L;
        if (!std::isfinite(denom) || denom <= 0.0) denom = 1.0;

        double bj = (j < (int)y.size()) ? clamp01(y[j]) : 0.5;
        x[j] = L + bj * denom;
    }
}

double ARQDP::distBN(const Vec& a, const Vec& b) const
{
    Vec ba, bb;
    toBN(a, ba);
    toBN(b, bb);
    double s = 0.0;
    const int D = (int)std::min(ba.size(), bb.size());
    for (int i = 0; i < D; ++i) {
        double d = ba[i] - bb[i];
        s += d * d;
    }
    return std::sqrt(s);
}

double ARQDP::cosineSim(const Vec& a, const Vec& b) const
{
    double na = 0.0, nb = 0.0, dot = 0.0;
    const int D = (int)std::min(a.size(), b.size());
    for (int i = 0; i < D; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if (na <= 0.0 || nb <= 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void ARQDP::clipVec(Vec& v, double clipAbs) const
{
    if (clipAbs <= 0.0) return;
    for (double &x : v) {
        if (!std::isfinite(x)) x = 0.0;
        if (x > clipAbs) x = clipAbs;
        else if (x < -clipAbs) x = -clipAbs;
    }
}

void ARQDP::normalizeVec(Vec& v, double eps) const
{
    double n2 = 0.0;
    for (double x : v) n2 += x * x;
    if (!std::isfinite(n2) || n2 <= eps * eps) return;
    double inv = 1.0 / std::sqrt(n2);
    for (double &x : v) x *= inv;
}


void ARQDP::buildBlendedDirectionBN(Vec& out, double far_weight) const
{
    out.clear();
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    const double w = std::max(0.0, std::min(1.0, far_weight));
    out.assign(D, 0.0);

    const bool has_local = (!dir_path_bn_.empty() && (int)dir_path_bn_.size() == D);
    const bool has_far   = (!dir_far_bn_.empty()  && (int)dir_far_bn_.size()  == D);
    if (!has_local && !has_far) return;

    if (has_local) {
        for (int j = 0; j < D; ++j) out[j] += (1.0 - w) * dir_path_bn_[j];
    }
    if (has_far) {
        for (int j = 0; j < D; ++j) out[j] += w * dir_far_bn_[j];
    }

    clipVec(out, dir_clip_);
    normalizeVec(out, dir_eps_);
}

void ARQDP::updateFarPrediction(double rel_impr, double diversity_bn)
{
    if (!dir_enabled_) return;
    if (!prob_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;
    if (!bounds_ready_) prepareBoundsCache();

    // Track best position trajectory in BN space (scale-free).
    Vec best_bn;
    toBN(best_x_, best_bn);
    if ((int)best_bn.size() != D) best_bn.assign(D, 0.5);

    bool moved = true;
    if (!best_bn_hist_.empty() && (int)best_bn_hist_.back().size() == D) {
        const Vec& last = best_bn_hist_.back();
        double s2 = 0.0;
        for (int j = 0; j < D; ++j) {
            const double d = best_bn[j] - last[j];
            s2 += d * d;
        }
        moved = (s2 > 1e-14);
    }
    if (moved) best_bn_hist_.push_back(best_bn);
    while ((int)best_bn_hist_.size() > dir_hist_len_) best_bn_hist_.pop_front();

    if ((int)best_bn_hist_.size() < 3) {
        if ((int)dir_far_bn_.size() != D) dir_far_bn_.assign(D, 0.0);
        return;
    }

    // (1) Velocity estimate from the best trajectory — "look far".
    Vec vel(D, 0.0);
    double wsum = 0.0;
    const int m = (int)best_bn_hist_.size();
    for (int t = 1; t < m; ++t) {
        const Vec& a = best_bn_hist_[t - 1];
        const Vec& b = best_bn_hist_[t];
        const double w = (double)t; // more weight for recent movement
        wsum += w;
        for (int j = 0; j < D; ++j) vel[j] += w * (b[j] - a[j]);
    }
    if (wsum > 0.0) for (int j = 0; j < D; ++j) vel[j] /= wsum;
    clipVec(vel, 0.50);
    normalizeVec(vel, dir_eps_);

    // (2) Pull away from the current elite centroid — avoid collapsing into a tight local basin.
    std::vector<int> idx;
    sortByFitness(idx);
    int top = (int)std::ceil(pop_elite_frac_ * (double)std::max(4, N_));
    top = std::max(2, std::min(N_, top));

    Vec centroid(D, 0.0);
    for (int t = 0; t < top; ++t) {
        Vec bn;
        toBN(X_[idx[t]], bn);
        if ((int)bn.size() != D) bn.assign(D, 0.5);
        for (int j = 0; j < D; ++j) centroid[j] += bn[j];
    }
    for (int j = 0; j < D; ++j) centroid[j] /= (double)top;

    Vec pull(D, 0.0);
    for (int j = 0; j < D; ++j) pull[j] = best_bn[j] - centroid[j];
    clipVec(pull, 0.50);
    normalizeVec(pull, dir_eps_);

    Vec far(D, 0.0);
    for (int j = 0; j < D; ++j) far[j] = dir_far_mix_ * vel[j] + (1.0 - dir_far_mix_) * pull[j];
    clipVec(far, dir_clip_);
    normalizeVec(far, dir_eps_);
    dir_far_bn_ = far;

    // Dynamic lookahead adaptation: expand when stagnating / low diversity, shrink when real progress happens.
    const bool stagnant = (no_improve_ >= stagnationtrigger_) || (no_improve_ >= escape_trigger_);
    if (stagnant || diversity_bn < pop_div_low_) {
        dir_lookahead_ = std::min(dir_lookahead_max_, dir_lookahead_ * dir_lookahead_grow_);
    } else if (rel_impr >= pop_impr_thr_) {
        dir_lookahead_ = std::max(dir_lookahead_min_, dir_lookahead_ * dir_lookahead_shrink_);
    } else {
        // mild stabilizing drift
        dir_lookahead_ = std::min(dir_lookahead_max_, std::max(dir_lookahead_min_, dir_lookahead_ * 0.995));
    }
}

void ARQDP::resetDirection()
{
    std::fill(dir_path_bn_.begin(), dir_path_bn_.end(), 0.0);
    std::fill(dir_step_sum_bn_.begin(), dir_step_sum_bn_.end(), 0.0);
    dir_step_wsum_ = 0.0;

    dir_trust_ = 0.50;
    dir_bad_iters_ = 0;
    dir_good_iters_ = 0;
    dir_disabled_for_ = 0;
}

void ARQDP::resetRefine()
{
    refine_sigma_ = refine_sigma0_;
}

void ARQDP::accumulateDirectionFromSuccess(int replaced_idx, const Vec& u, double gain, bool used_dir)
{
    if (!dir_enabled_) return;
    if (dir_disabled_for_ > 0) return;
    if (gain <= 0.0) return;
    if (!prob_) return;
    const double passive = dir_passive_factor_;
    const double wmult = used_dir ? 1.0 : std::max(0.0, std::min(1.0, passive));
    if (wmult <= 0.0) return; // disabled passive learning

    if (replaced_idx < 0 || replaced_idx >= (int)X_.size()) return;

    Vec old_bn, new_bn;
    toBN(X_[replaced_idx], old_bn);
    toBN(u, new_bn);

    const int D = prob_->dimension();
    if ((int)dir_step_sum_bn_.size() != D) dir_step_sum_bn_.assign(D, 0.0);

    Vec step(D, 0.0);
    for (int j = 0; j < D; ++j) step[j] = new_bn[j] - old_bn[j];

    clipVec(step, dir_step_clip_);
    normalizeVec(step, dir_eps_);

    double w = std::min(std::max(gain, 0.0), 1e6) * wmult;
for (int j = 0; j < D; ++j) dir_step_sum_bn_[j] += w * step[j];
    dir_step_wsum_ += w;
}

void ARQDP::applyDirectionBias(Vec& v, bool& used_dir)
{
    used_dir = false;
    if (!dir_enabled_) return;
    if (dir_disabled_for_ > 0) return;
    if (!prob_) return;
    if (iter_ < dir_warmup_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    if (dir_path_bn_.empty() && dir_far_bn_.empty()) return;

    ensureDim(v, D);
    if (!bounds_ready_) prepareBoundsCache();

    // Stagnation-boosted usage probability (use direction more aggressively when stuck).
    const double stag = clamp01((double)no_improve_ / (double)std::max(1, escape_trigger_));
    const double p_use = pdir_ * dir_trust_ * (0.40 + 0.60 * stag);
    if (randU() >= p_use) return;

    Vec bn;
    toBN(v, bn);
    if ((int)bn.size() != D) bn.assign(D, 0.5);

    // Progress schedule: slightly larger horizon early, controlled late — but allow stagnation to override.
    const double calls = prob_ ? (double)prob_->calls() : 0.0;
    const double maxc  = (double)max_evals_;
    const double progress = (maxc > 0.0) ? std::min(1.0, std::max(0.0, calls / maxc)) : 0.0;

    double sched = 1.0;
    if (progress < 0.30) sched = 1.25;
    else if (progress > 0.85) sched = 0.85;
    if (stag > 0.25) sched = std::max(sched, 1.10);

    // Compose a "near + far" move in BN space:
    //  - near: learned from successful replacements (local exploitation)
    //  - far : predicted from best-trajectory + elite-centroid geometry (long-horizon escape)
    Vec delta(D, 0.0);

    const bool has_local = (!dir_path_bn_.empty() && (int)dir_path_bn_.size() == D);
    const bool has_far   = (!dir_far_bn_.empty()  && (int)dir_far_bn_.size()  == D);

    const double local_scale = dir_beta_;
    const double far_scale   = dir_beta_ * dir_lookahead_ * (0.35 + 0.65 * stag) * sched;

    for (int j = 0; j < D; ++j) {
        double d = 0.0;
        if (has_local) d += local_scale * dir_path_bn_[j];
        if (has_far)   d += far_scale   * dir_far_bn_[j];
        delta[j] = (double)dir_sign_ * d;
    }

    // Safety: cap the per-dimension move (prevents overshoot in BN, but still allows long-horizon push).
    clipVec(delta, dir_far_clip_);

    for (int j = 0; j < D; ++j) bn[j] = reflect01(bn[j] + delta[j]);

    fromBN(bn, v);
    used_dir = true;
}

void ARQDP::updateDirectionTrust(double success_rate, double rel_impr,
                                 double dir_use_rate,
                                 double dir_succ_rate)
{
    if (!dir_enabled_) return;

    // If direction didn't provide a meaningful aligned step this iteration, penalize trust.
    // Heuristic: if we had no directional learning mass, direction wasn't useful.
    success_rate = std::max(0.0, std::min(1.0, success_rate));
    dir_use_rate = std::max(0.0, std::min(1.0, dir_use_rate));
    dir_succ_rate = std::max(0.0, std::min(1.0, dir_succ_rate));

    const bool used_nontrivial = (dir_use_rate >= 0.05);
    const bool helped_by_success = used_nontrivial && (dir_succ_rate >= 0.85 * success_rate || dir_succ_rate >= 0.20);
    const bool helped_by_progress = (rel_impr >= pop_impr_thr_) && (dir_step_wsum_ > 0.0);
    const bool dir_helped = helped_by_success || helped_by_progress;

    if (dir_helped) {
        dir_trust_ = std::min(dir_trust_max_, dir_trust_ + dir_trust_inc_);
        dir_good_iters_++;
        dir_bad_iters_ = std::max(0, dir_bad_iters_ - 1);
    } else {
        dir_trust_ = std::max(dir_trust_min_, dir_trust_ - dir_trust_dec_);
        dir_bad_iters_++;
        dir_good_iters_ = std::max(0, dir_good_iters_ - 1);
    }

    // Update direction vector only when we have a stable signal
    if (dir_step_wsum_ > 0.0) {
        Vec mean_step = dir_step_sum_bn_;
        for (double &x : mean_step) x /= dir_step_wsum_;

        normalizeVec(mean_step, dir_eps_);
        clipVec(mean_step, dir_clip_);

        if (dir_path_bn_.empty()) dir_path_bn_ = mean_step;
        else {
            for (size_t j = 0; j < dir_path_bn_.size(); ++j) {
                dir_path_bn_[j] = (1.0 - dir_c_) * dir_path_bn_[j] + dir_c_ * mean_step[j];
            }
        }

        clipVec(dir_path_bn_, dir_clip_);
        normalizeVec(dir_path_bn_, dir_eps_);
    } else {
        // decay towards zero when no evidence
        if (!dir_path_bn_.empty()) for (double &x : dir_path_bn_) x *= (1.0 - dir_decay_);
    }

    // Disable direction temporarily if consistently harmful *while being used*
    if (dir_bad_iters_ >= dir_disable_after_ && dir_use_rate >= 0.05) {
        dir_disabled_for_ = dir_disabled_period_;
        dir_bad_iters_ = 0;
        // also shrink beta to avoid huge mis-steps after re-enable
        dir_beta_ = std::max(dir_beta_min_, 0.6 * dir_beta_);
    }

    // If re-enabled, slowly restore beta (aggressive but controlled)
    if (dir_disabled_for_ == 0) {
        dir_beta_ = std::min(dir_beta_max_, std::max(dir_beta_min_, dir_beta_ * (1.0 + 0.01 * dir_trust_)));
    }
}

void ARQDP::sample_F_CR(double& F, double& CR, double muF, double muCR)
{
    F = cauchy(muF, 0.1);
    int tries = 0;
    while (F <= 0.0 && tries < 25) { F = cauchy(muF, 0.1); ++tries; }
    if (F <= 0.0) F = muF;

    if (F > Fmax_) F = Fmax_;
    if (F < Fmin_) F = Fmin_;

    CR = muCR + 0.1 * randN01();
    if (CR > 1.0) CR = 1.0;
    if (CR < 0.0) CR = 0.0;
}

void ARQDP::makeTrialBase(int i, const std::vector<int>& sorted_idx, double F, double CR, double pbestFrac, Vec& u, bool& used_dir)
{
    if (!prob_) { if (i>=0 && i<N_) u = X_[i]; return; }
    const int D = prob_->dimension();
    const int N = N_;
    if (D <= 0 || N < 4 || i < 0 || i >= N_) { if (i>=0 && i<N_) u = X_[i]; return; }

    int pcount = (int)std::ceil(pbestFrac * (double)N);
    pcount = std::max(2, std::min(N, pcount));

    int pbest_idx = sorted_idx[randInt(0, pcount - 1)];

    std::vector<int> pick;
    sampleDistinctExcluding(N, 2, {i, pbest_idx}, pick);
    if ((int)pick.size() < 2) { u = X_[i]; return; }

    int r1 = pick[0];
    int r2 = pick[1];

    Vec xr2 = X_[r2];
    if (!A_.empty() && randU() < 0.5) {
        int aidx = randInt(0, (int)A_.size() - 1);
        xr2 = A_[aidx];
    }

    ensureDim(xr2, D);
    ensureDim(X_[i], D);
    ensureDim(X_[pbest_idx], D);
    ensureDim(X_[r1], D);

    Vec v(D);
    for (int j = 0; j < D; ++j) {
        v[j] = X_[i][j]
             + F * (X_[pbest_idx][j] - X_[i][j])
             + F * (X_[r1][j]       - xr2[j]);
    }

    used_dir = false;
    if (dir_enabled_) applyDirectionBias(v, used_dir);

    sanitizeInPlace(v);

    u = X_[i];
    ensureDim(u, D);

    int jrand = randInt(0, D - 1);
    for (int j = 0; j < D; ++j) if (randU() < CR || j == jrand) u[j] = v[j];
    sanitizeInPlace(u);
}

bool ARQDP::selectionRTR(int parent, const Vec& u, double fu,
                          bool used_dir,
                          double F, double CR,
                          std::vector<double>& SF, std::vector<double>& SCR,
                          std::vector<double>& gains,
                          int& succ_total, int& succ_dir)
{
    if (parent < 0 || parent >= N_) return false;
    if (fu <= FX_[parent]) {
        double g = FX_[parent] - fu;

        if (used_dir) succ_dir++;
        succ_total++;

        accumulateDirectionFromSuccess(parent, u, g, used_dir);

        addToArchive(X_[parent]);
        X_[parent]  = sanitizedCopy(u);
        FX_[parent] = fu;

        SF.push_back(F);
        SCR.push_back(CR);
        gains.push_back(std::max(0.0, g));
        return true;
    }

    int qstar = -1;
    double bestd = std::numeric_limits<double>::infinity();
    for (int t = 0; t < rtr_k_; ++t) {
        int q = randInt(0, N_ - 1);
        double d = distBN(u, X_[q]);
        if (d < bestd) { bestd = d; qstar = q; }
    }
    if (qstar < 0) return false;

    if (fu < FX_[qstar]) {
        double g = FX_[qstar] - fu;

        if (used_dir) succ_dir++;
        succ_total++;

        accumulateDirectionFromSuccess(qstar, u, g, used_dir);

        addToArchive(X_[qstar]);
        X_[qstar]  = sanitizedCopy(u);
        FX_[qstar] = fu;

        SF.push_back(F);
        SCR.push_back(CR);
        gains.push_back(std::max(0.0, g));
        return true;
    }
    return false;
}

void ARQDP::updateMemories(const std::vector<double>& SF,
                           const std::vector<double>& SCR,
                           const std::vector<double>& gains)
{
    if (SF.empty() || SCR.empty() || gains.empty()) return;
    if (SF.size() != SCR.size() || SF.size() != gains.size()) return;

    double sumg = 0.0;
    for (double g : gains) sumg += std::max(0.0, g);
    if (sumg <= 0.0) return;

    double meanF_num = 0.0;
    double meanF_den = 0.0;
    double meanCR    = 0.0;

    for (size_t i = 0; i < SF.size(); ++i) {
        double w = std::max(0.0, gains[i]) / sumg;
        meanF_num += w * SF[i] * SF[i];
        meanF_den += w * SF[i];
        meanCR    += w * SCR[i];
    }

    double meanF = (meanF_den > 0.0) ? (meanF_num / meanF_den) : MF_[k_mem_];

    MF_[k_mem_]  = 0.5 * (MF_[k_mem_]  + meanF);
    MCR_[k_mem_] = 0.5 * (MCR_[k_mem_] + meanCR);

    MF_[k_mem_]  = std::max(Fmin_, std::min(Fmax_, MF_[k_mem_]));
    MCR_[k_mem_] = std::max(0.0,  std::min(1.0,  MCR_[k_mem_]));

    k_mem_ = (k_mem_ + 1) % H_;
}

double ARQDP::estimateDiversityBN(int sampleCount)
{
    if (!prob_ || N_ <= 1) return 0.0;
    const int D = prob_->dimension();
    if (D <= 0) return 0.0;

    int m = sampleCount;
    m = std::max(2, std::min(N_, m));

    std::vector<int> ids(N_);
    std::iota(ids.begin(), ids.end(), 0);
    for (int i = 0; i < m; ++i) {
        std::uniform_int_distribution<int> dist(i, N_ - 1);
        int r = dist(rng_);
        std::swap(ids[i], ids[r]);
    }
    ids.resize(m);

    Vec mean(D, 0.0), bn(D, 0.0);
    for (int id : ids) {
        toBN(X_[id], bn);
        for (int j = 0; j < D; ++j) mean[j] += bn[j];
    }
    for (int j = 0; j < D; ++j) mean[j] /= (double)m;

    double ss = 0.0;
    for (int id : ids) {
        toBN(X_[id], bn);
        double d2 = 0.0;
        for (int j = 0; j < D; ++j) {
            double d = bn[j] - mean[j];
            d2 += d * d;
        }
        ss += d2;
    }
    ss /= (double)m;
    return std::sqrt(std::max(0.0, ss));
}

double ARQDP::relImprovementFromHistory() const
{
    if (best_hist_.size() < 2) return 0.0;
    double first = best_hist_.front();
    double last  = best_hist_.back();
    if (!std::isfinite(first) || !std::isfinite(last)) return 0.0;
    double denom = std::fabs(first) + 1e-12;
    return (first - last) / denom;
}

void ARQDP::injectNewIndividuals(int addCount)
{
    if (!prob_ || addCount <= 0) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    X_.reserve(N_ + addCount);
    FX_.reserve(N_ + addCount);

    Vec best_bn;
    toBN(best_x_, best_bn);

    for (int k = 0; k < addCount; ++k) {
        if (terminated()) break;

        Vec cand(D, 0.0);

        if (randU() < 0.5 && !best_bn.empty()) {
            Vec bn = best_bn;
            for (int j = 0; j < D; ++j) bn[j] = clamp01(bn[j] + rsigma_ * randN01());
            fromBN(bn, cand);
        } else {
            if (!bounds_ready_) prepareBoundsCache();
            for (int j = 0; j < D; ++j) {
                double L = (j < (int)Lc_.size()) ? Lc_[j] : -1.0;
                double U = (j < (int)Uc_.size()) ? Uc_[j] : +1.0;
                double range = U - L;
                if (!std::isfinite(range) || range <= 0.0) range = 1.0;
                cand[j] = L + randU() * range;
            }
        }

        sanitizeInPlace(cand);
        double fc = evalX(cand);

        X_.push_back(std::move(cand));
        FX_.push_back(fc);

        if (fc < best_f_) { best_f_ = fc; best_x_ = X_.back(); }
    }
}

void ARQDP::resizePopulation(int newN, const std::vector<int>& sorted_idx)
{
    newN = std::max(4, newN);
    if (newN == N_) return;

    // Dimension is needed for reinitializing BN-space directional buffers when expanding population.
    const int D = prob_ ? prob_->dimension() : (best_x_.empty() ? 0 : (int)best_x_.size());

    if (newN < N_) {
        int elite = (int)std::round(pop_elite_frac_ * (double)newN);
        elite = std::max(2, std::min(newN, elite));

        int half = std::max(elite, N_ / 2);

        std::vector<int> pool;
        pool.reserve(half);
        for (int i = 0; i < half; ++i) pool.push_back(sorted_idx[i]);

        if ((int)pool.size() < elite) pool = sorted_idx;

        std::vector<int> keep;
        keep.reserve(newN);

        for (int i = 0; i < elite; ++i) keep.push_back(sorted_idx[i]);

        std::shuffle(pool.begin(), pool.end(), rng_);
        for (int i = elite; i < newN; ++i) keep.push_back(pool[(i - elite) % (int)pool.size()]);

        std::vector<Vec> Xnew;
        std::vector<double> FXnew;
        Xnew.reserve(newN);
        FXnew.reserve(newN);

        for (int id : keep) { Xnew.push_back(X_[id]); FXnew.push_back(FX_[id]); }

        X_.swap(Xnew);
        FX_.swap(FXnew);
        N_ = newN;

        trimArchive();
        resetDirection();
        resetRefine();

        best_f_ = std::numeric_limits<double>::infinity();
        for (int i = 0; i < N_; ++i) if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }

        Optimizer::setPopulation(N_);
        return;
    }

    int add = newN - N_;
    if (add <= 0) return;

    injectNewIndividuals(add);
    N_ = (int)X_.size();

    trimArchive();
    resetDirection();
    resetRefine();

    best_bn_hist_.clear();
    dir_far_bn_.assign(D, 0.0);

    best_f_ = std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_; ++i) if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }

    Optimizer::setPopulation(N_);
}

bool ARQDP::maybeAdaptivePopulationLeap(const std::vector<int>& sorted_idx,
                                        double success_rate,
                                        double diversity_bn,
                                        double rel_impr)
{
    if (!adaptive_population_) return false;
    if (N_ < 4) return false;
    if (pop_cooldown_left_ > 0) return false;
    if (pop_check_interval_ > 1 && (iter_ % pop_check_interval_) != 0) return false;

    const int D = prob_ ? prob_->dimension() : 0;

    int hardMin = std::max(4, pop_min_);
    int hardMax = std::max(hardMin, pop_max_);

    int newN = N_;

    // Strong early shrink for large-D when population is likely harmful for best
    if (iter_ <= pop_warmup_iters_ && D >= 60 && N_ > hardMin) {
        bool ineffective = (success_rate < pop_success_thr_) && (rel_impr < pop_impr_thr_);
        if (ineffective) {
            int cand = (int)std::round((double)N_ * pop_shrink_factor_);
            newN = std::max(hardMin, cand);
        }
        if (D >= 100 && N_ >= 80 && (success_rate < 0.15)) newN = hardMin;
    }

    if (newN == N_ && no_improve_ >= stagnationtrigger_ && N_ > hardMin) {
        int cand = (int)std::round((double)N_ * pop_shrink_factor_);
        newN = std::max(hardMin, cand);
    }

    // Expand if collapsed in local basin (low diversity, low success)
    if (newN == N_ && diversity_bn < pop_div_low_ &&
        success_rate < 0.5 * pop_success_thr_ && N_ < hardMax) {
        int cand = (int)std::round((double)N_ * pop_expand_factor_);
        newN = std::min(hardMax, cand);
    }

    // Controlled shrink if too diverse but not improving
    if (newN == N_ && diversity_bn > pop_div_high_ && rel_impr < pop_impr_thr_ &&
        success_rate < pop_success_thr_ && N_ > hardMin) {
        newN = std::max(hardMin, N_ / 2);
    }

    newN = std::max(hardMin, std::min(hardMax, newN));
    if (newN == N_) return false;

    resizePopulation(newN, sorted_idx);
    pop_cooldown_left_ = pop_cooldown_;
    return true;

    return false;
}

bool ARQDP::shouldRefine(double success_rate, double diversity_bn) const
{
    if (!refine_enabled_) return false;
    if (pop_cooldown_left_ > 0) return false;
    if (no_improve_ < refine_trigger_) return false;
    if (refine_every_ > 1 && (iter_ % refine_every_) != 0) return false;

    if (success_rate <= refine_success_thr_) return true;
    if (diversity_bn <= refine_div_thr_) return true;
    if (no_improve_ >= 2 * stagnationtrigger_) return true;

    return false;
}

bool ARQDP::precisionBurst(double diversity_bn)
{
    if (!prob_) return false;
    if (best_x_.empty()) return false;
    if (terminated()) return false;

    const int D = prob_->dimension();
    if (D <= 0) return false;

    int remaining = (int)std::max(0.0, (double)max_evals_ - (double)prob_->calls());
    if (remaining <= 0) return false;

    int budget = std::min(refine_budget_, remaining);
    if (budget <= 0) return false;

    // Endgame cap: keep the tail responsive.
    const double calls = prob_ ? (double)prob_->calls() : 0.0;
    const double maxc  = (double)max_evals_;
    const double progress = (maxc > 0.0) ? std::min(1.0, std::max(0.0, calls / maxc)) : 0.0;
    if (progress >= 0.90) budget = std::min(budget, std::max(40, endgame_refine_cap_));
    else if (progress >= 0.80) budget = std::min(budget, std::max(80, 2 * endgame_refine_cap_));

    sanitizeInPlace(best_x_);
    double sigma = refine_sigma_;
    if (diversity_bn < 0.5 * refine_div_thr_) sigma *= 0.70;
    sigma = std::max(refine_sigma_min_, std::min(refine_sigma_max_, sigma));

    Vec best_bn;
    toBN(best_x_, best_bn);

    Vec comb_dir_bn;
    if (dir_enabled_ && dir_disabled_for_ == 0) {
        buildBlendedDirectionBN(comb_dir_bn, 0.55);
    }

    int k = (int)std::round(refine_subspace_frac_ * (double)D);
    k = std::max(1, std::max(refine_subspace_min_, k));
    k = std::min(D, k);

    std::vector<int> dims(D);
    std::iota(dims.begin(), dims.end(), 0);

    double best_local_f = best_f_;
    Vec    best_local_x = best_x_;

    for (int t = 0; t < budget; ++t) {
        if (terminated()) break;

        std::shuffle(dims.begin(), dims.end(), rng_);
        std::vector<int> sub(dims.begin(), dims.begin() + k);

        Vec delta_bn(D, 0.0);
        const double ru = randU();
        const bool coordOnly = (ru < refine_coord_prob_);
        const bool heavyTail = (!coordOnly && ru < refine_coord_prob_ + refine_cauchy_prob_);

        if (coordOnly) {
            int steps = std::min(4, k);
            for (int s = 0; s < steps; ++s) {
                int j = sub[s];
                delta_bn[j] += sigma * randN01();
            }
        } else if (heavyTail) {
            for (int j : sub) delta_bn[j] += sigma * 0.6 * cauchy(0.0, 1.0);
        } else {
            for (int j : sub) delta_bn[j] += sigma * randN01();
        }

        if (dir_enabled_ && dir_disabled_for_ == 0 && !comb_dir_bn.empty() && refine_dir_mix_ > 0.0) {
            for (int j : sub) delta_bn[j] += refine_dir_mix_ * sigma * comb_dir_bn[j];
        }

        clipVec(delta_bn, refine_clip_);

        Vec cand_bn = best_bn;
        for (int j : sub) cand_bn[j] = clamp01(cand_bn[j] + delta_bn[j]);

        Vec cand;
        fromBN(cand_bn, cand);
        sanitizeInPlace(cand);

        double fc = evalX(cand);
        if (fc < best_local_f) {
            best_local_f = fc;
            best_local_x = std::move(cand);

            sigma *= refine_shrink_;
            sigma = std::max(refine_sigma_min_, std::min(refine_sigma_max_, sigma));
            toBN(best_local_x, best_bn);
        } else {
            if ((t % 25) == 24) {
                sigma *= refine_grow_;
                sigma = std::max(refine_sigma_min_, std::min(refine_sigma_max_, sigma));
            }
        }
    }

    bool improved = (best_local_f < best_f_);
    if (improved) {
        best_f_ = best_local_f;
        best_x_ = best_local_x;

        if (!FX_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) { fw = FX_[i]; worst = i; }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }

        refine_sigma_ = std::max(refine_sigma_min_, std::min(refine_sigma_max_, sigma));
        no_improve_ = 0;
        best_prev_  = best_f_;
    } else {
        refine_sigma_ = std::max(refine_sigma_min_, std::min(refine_sigma_max_, sigma * refine_grow_));
    }

    return improved;
}

bool ARQDP::bestShots(const std::vector<int>& sorted_idx, double diversity_bn)
{
    if (!best_shots_enabled_) return false;
    if (!prob_) return false;
    if (best_x_.empty()) return false;
    if (best_shot_every_ > 1 && (iter_ % best_shot_every_) != 0) return false;

    const int D = prob_->dimension();
    if (D <= 0) return false;

    int remaining = (int)std::max(0.0, (double)max_evals_ - (double)prob_->calls());
    if (remaining <= 0) return false;

    const int Nlocal = (int)sorted_idx.size();
    if (Nlocal <= 0) return false;

    int shots = std::min(best_shots_, remaining);
    if (no_improve_ >= stagnationtrigger_) shots = std::max(shots, (int)std::round(1.25 * (double)best_shots_));

    // Endgame cap: avoid spending a large fraction of the remaining budget on inner loops.
    const double calls = prob_ ? (double)prob_->calls() : 0.0;
    const double maxc  = (double)max_evals_;
    const double progress = (maxc > 0.0) ? std::min(1.0, std::max(0.0, calls / maxc)) : 0.0;
    if (progress >= 0.90) shots = std::min(shots, std::max(2, endgame_shots_cap_));
    else if (progress >= 0.80) shots = std::min(shots, std::max(4, (int)std::round(1.5 * (double)endgame_shots_cap_)));

    shots = std::max(2, std::min(48, shots));
    if (shots <= 0) return false;

    if (!bounds_ready_) prepareBoundsCache();

    Vec elite_dir_bn(D, 0.0);
    if (Nlocal >= 3) {
        Vec b1, b2, b3;
        toBN(X_[sorted_idx[0]], b1);
        toBN(X_[sorted_idx[1]], b2);
        toBN(X_[sorted_idx[2]], b3);
        for (int j = 0; j < D; ++j) {
            elite_dir_bn[j] = (b1[j] - b2[j]) + 0.5 * (b1[j] - b3[j]);
        }
        normalizeVec(elite_dir_bn, dir_eps_);
        clipVec(elite_dir_bn, dir_clip_);
    }

    Vec best_bn;
    toBN(best_x_, best_bn);

    // Blended direction in BN space used for sign probing and directional best-shots.
    // Make it more far-horizon when stagnating / low diversity to help basin-to-basin moves.
    Vec comb_dir_bn;
    if (dir_enabled_ && dir_disabled_for_ == 0) {
        double farw = 0.60;
        if (no_improve_ >= stagnationtrigger_) farw = 0.75;
        if (diversity_bn < pop_div_low_) farw = std::min(0.85, farw + 0.10);
        buildBlendedDirectionBN(comb_dir_bn, farw);
    }

    // Sigma schedule: tighten when diversity is low, expand slightly when stagnating to escape local basin.
    double sigma = best_shot_sigma_;
    if (diversity_bn < pop_div_low_) sigma *= 0.80;
    if (no_improve_ >= stagnationtrigger_) sigma *= 1.25;
    sigma = std::max(1e-6, std::min(0.35, sigma));

    // Direction sign probe: periodically test +dir vs -dir around best and pick the better sign.
    if (dir_enabled_ && dir_disabled_for_ == 0 && iter_ >= dir_warmup_ && !comb_dir_bn.empty() && dir_trust_ > 0.35) {
        const double probeSigma = std::min(0.06, 0.6 * sigma);

        Vec bn_plus = best_bn;
        Vec bn_minus = best_bn;
        for (int j = 0; j < D; ++j) {
            const double step = probeSigma * comb_dir_bn[j];
            bn_plus[j] = reflect01(bn_plus[j] + step);
            bn_minus[j] = reflect01(bn_minus[j] - step);
        }

        Vec x_plus, x_minus;
        fromBN(bn_plus, x_plus);
        fromBN(bn_minus, x_minus);
        sanitizeInPlace(x_plus);
        sanitizeInPlace(x_minus);

        const double f_plus = evalX(x_plus);
        const double f_minus = evalX(x_minus);

        if (std::isfinite(f_plus) && std::isfinite(f_minus)) {
            dir_sign_ = (f_plus <= f_minus) ? +1 : -1;
        }

        if (f_plus < best_f_) {
            best_f_ = f_plus;
            best_x_ = x_plus;
            toBN(best_x_, best_bn);
        }
        if (f_minus < best_f_) {
            best_f_ = f_minus;
            best_x_ = x_minus;
            toBN(best_x_, best_bn);
        }
    }

    bool improved = false;
    double best_local_f = best_f_;
    Vec best_local_x = best_x_;

    for (int s = 0; s < shots; ++s) {
        if (terminated()) break;

        Vec delta_bn(D, 0.0);

        // Core: Gaussian around best + occasional heavy tail.
        for (int j = 0; j < D; ++j) delta_bn[j] = sigma * randN01();
        if (randU() < best_shot_cauchy_prob_) {
            for (int j = 0; j < D; ++j) delta_bn[j] += 0.55 * sigma * cauchy(0.0, 1.0);
        }

        // Mix in elite-based direction for exploitation.
        if (randU() < 0.60) {
            for (int j = 0; j < D; ++j) delta_bn[j] += 0.45 * sigma * elite_dir_bn[j];
        }

        // Mix in learned direction when trusted (signed).
        if (dir_enabled_ && dir_disabled_for_ == 0 && !comb_dir_bn.empty() && best_shot_dir_mix_ > 0.0) {
            const double pmix = std::min(1.0, pdir_ * dir_trust_);
            if (randU() < pmix) {
                for (int j = 0; j < D; ++j) delta_bn[j] += (double)dir_sign_ * best_shot_dir_mix_ * sigma * comb_dir_bn[j];
            }
        }

        clipVec(delta_bn, dir_step_clip_);

        Vec cand_bn = best_bn;
        for (int j = 0; j < D; ++j) cand_bn[j] = reflect01(cand_bn[j] + delta_bn[j]);

        Vec cand;
        fromBN(cand_bn, cand);
        sanitizeInPlace(cand);

        double fc = evalX(cand);
        if (fc < best_local_f) {
            best_local_f = fc;
            best_local_x = std::move(cand);
            improved = true;

            // focus around new best
            sigma = std::max(1e-6, sigma * 0.85);
            toBN(best_local_x, best_bn);
        } else {
            // If we are consistently not improving and direction is involved, occasionally flip sign to avoid misleading guidance.
            if (dir_enabled_ && randU() < 0.02 && no_improve_ >= stagnationtrigger_) dir_sign_ = -dir_sign_;
        }
    }

    if (improved && best_local_f < best_f_) {
        best_f_ = best_local_f;
        best_x_ = best_local_x;
    }

    // Propagate best into population (critical for "best" performance).
    if (best_x_.size() == (size_t)D && !FX_.empty() && (best_f_ + 0.0) < (1e99)) {
        int worst = 0;
        double fw = FX_[0];
        for (int i = 1; i < N_; ++i) {
            if (FX_[i] > fw) { fw = FX_[i]; worst = i; }
        }
        if (best_f_ < FX_[worst]) {
            X_[worst] = best_x_;
            FX_[worst] = best_f_;
        }
    }

    return improved;
}

void ARQDP::escapeBurst(const std::vector<int>& sorted_idx)
{
    if (!escape_enabled_) return;
    if (!prob_) return;
    if (no_improve_ < escape_trigger_) return;
    if (N_ < 4) return;

    const int D = prob_->dimension();
    if (D <= 0) return;

    int remaining = (int)std::max(0.0, (double)max_evals_ - (double)prob_->calls());
    if (remaining <= 0) return;

    int k = (int)std::round(escape_frac_ * (double)N_);
    k = std::max(1, std::min(N_ / 2, k));

    // Choose worst individuals to replace aggressively
    std::vector<int> worst;
    worst.reserve(k);
    for (int i = 0; i < k; ++i) worst.push_back(sorted_idx[N_ - 1 - i]);

    Vec best_bn;
    toBN(best_x_, best_bn);

    for (int t = 0; t < (int)worst.size(); ++t) {
        if (terminated()) break;

        int id = worst[t];

        Vec cand_bn = best_bn;

        const double ru = randU();
        if (ru < escape_opposition_prob_) {
            // Opposition-based around best (in BN): bn' = 1 - bn + noise
            for (int j = 0; j < D; ++j) cand_bn[j] = reflect01(1.0 - best_bn[j] + escape_sigma_ * randN01());
        } else if (ru < escape_opposition_prob_ + escape_cauchy_prob_) {
            for (int j = 0; j < D; ++j) cand_bn[j] = reflect01(best_bn[j] + escape_sigma_ * 0.8 * cauchy(0.0, 1.0));
        } else {
            for (int j = 0; j < D; ++j) cand_bn[j] = reflect01(best_bn[j] + escape_sigma_ * randN01());
        }

        // When direction seems harmful, jump opposite to direction
        if (dir_enabled_ && dir_disabled_for_ > 0 && !dir_path_bn_.empty()) {
            for (int j = 0; j < D; ++j) cand_bn[j] = reflect01(cand_bn[j] - 0.5 * escape_sigma_ * dir_path_bn_[j]);
        }

        Vec cand;
        fromBN(cand_bn, cand);
        sanitizeInPlace(cand);

        double fc = evalX(cand);

        if (fc < FX_[id]) {
            addToArchive(X_[id]);
            X_[id] = cand;
            FX_[id] = fc;
            if (fc < best_f_) { best_f_ = fc; best_x_ = cand; best_prev_ = best_f_; no_improve_ = 0; }
        }
    }

    // After escape, reset direction to avoid re-entering same basin
    resetDirection();
    pop_cooldown_left_ = std::max(pop_cooldown_left_, 2);
    no_improve_ = std::max(0, no_improve_ - 3);
}


bool ARQDP::horizonScan(const std::vector<int>& sorted_idx, double diversity_bn)
{
    if (!horizon_enabled_) return false;
    if (!prob_ || best_x_.empty() || N_ < 4) return false;
    if (horizon_cooldown_left_ > 0) return false;

    // Trigger when stagnating and either diversity is low or stagnation is already pronounced.
    if (no_improve_ < horizon_trigger_) return false;
    if (diversity_bn > horizon_div_thr_ && no_improve_ < 2 * horizon_trigger_) return false;

    const int D = prob_->dimension();
    if (D <= 0) return false;
    if (!bounds_ready_) prepareBoundsCache();

    int remaining = (int)std::max(0.0, (double)max_evals_ - (double)prob_->calls());
    if (remaining <= 0) return false;

    const int shots = std::max(1, std::min(horizon_shots_, remaining));
    if (shots <= 0) return false;

    // Best in BN
    Vec best_bn;
    toBN(best_x_, best_bn);
    ensureDim(best_bn, D);

    // Build a far-looking direction in BN. If unavailable, fall back to random.
    Vec dir_bn;
    if (dir_enabled_) {
        double farw = 0.75;
        if (no_improve_ >= hard_restart_trigger_) farw = 0.85;
        buildBlendedDirectionBN(dir_bn, farw);
    }

    double n2 = 0.0;
    if ((int)dir_bn.size() == D) {
        for (int j = 0; j < D; ++j) n2 += dir_bn[j] * dir_bn[j];
    }
    if ((int)dir_bn.size() != D || n2 <= dir_eps_ * dir_eps_) {
        dir_bn.assign(D, 0.0);
        for (int j = 0; j < D; ++j) dir_bn[j] = randN01();
        normalizeVec(dir_bn, dir_eps_);
        clipVec(dir_bn, dir_clip_);
    }

    // One orthogonal jitter direction (Gram-Schmidt)
    Vec orth_bn(D, 0.0);
    for (int j = 0; j < D; ++j) orth_bn[j] = randN01();
    double dot = 0.0;
    for (int j = 0; j < D; ++j) dot += orth_bn[j] * dir_bn[j];
    for (int j = 0; j < D; ++j) orth_bn[j] -= dot * dir_bn[j];
    normalizeVec(orth_bn, dir_eps_);

    // Quantile gate for injection even if not a strict improvement of current worst.
    std::vector<double> f = FX_;
    const double q_gate = quantile(f, horizon_accept_q_);

    Vec bestCand;
    double bestCandF = std::numeric_limits<double>::infinity();

    // Multi-scale lookahead (geometric), tested in both signs when direction is trusted.
    const double base = horizon_scale0_ * std::max(0.5, dir_lookahead_);
    const double maxs = horizon_scale_max_ * std::max(0.5, dir_lookahead_);
    const bool testBoth = (dir_enabled_ && dir_disabled_for_ == 0 && dir_trust_ > 0.35);

    for (int s = 0; s < shots; ++s) {
        if (terminated()) break;

        const double scale = std::min(maxs, base * std::pow(horizon_scale_growth_, (double)s));
        const double jitter = horizon_orth_sigma_ * (0.40 + 0.60 * randU());

        auto tryOne = [&](double sign)
        {
            Vec cand_bn = best_bn;
            for (int j = 0; j < D; ++j) {
                cand_bn[j] = reflect01(cand_bn[j] + sign * scale * dir_bn[j] + jitter * orth_bn[j]);
            }
            if (randU() < 0.30) {
                for (int j = 0; j < D; ++j) cand_bn[j] = reflect01(cand_bn[j] + 0.15 * jitter * randN01());
            }

            Vec cand;
            fromBN(cand_bn, cand);
            sanitizeInPlace(cand);

            double fc = evalX(cand);
            if (std::isfinite(fc) && fc < bestCandF) {
                bestCandF = fc;
                bestCand  = std::move(cand);
            }
        };

        tryOne(+1.0);
        if (testBoth) tryOne(-1.0);
    }

    if (!std::isfinite(bestCandF)) return false;
    if (sorted_idx.empty()) return false;
    int worst_id = sorted_idx.back();

    bool injected = false;
    if (bestCandF < best_f_) {
        addToArchive(X_[worst_id]);
        X_[worst_id] = bestCand;
        FX_[worst_id] = bestCandF;

        best_f_ = bestCandF;
        best_x_ = bestCand;
        best_prev_ = best_f_;
        no_improve_ = 0;
        injected = true;
    } else {
        const bool longStag = (no_improve_ >= 2 * horizon_trigger_);
        if (longStag && bestCandF < q_gate) {
            // Soft injection: refresh population with a "good" far candidate.
            addToArchive(X_[worst_id]);
            X_[worst_id] = bestCand;
            FX_[worst_id] = bestCandF;
            injected = true;
            no_improve_ = std::max(0, no_improve_ - horizon_trigger_);
        } else if (longStag && horizon_bridge_enabled_) {
            // Barrier-crossing injection (basin bridge): allow a worse but *far* candidate
            // with a temperature gate. This prevents the systematic "plateau" where
            // far moves exist but are always rejected by strict improvement gates.
            const double d = distBN(bestCand, best_x_);
            const double dnorm = d / std::sqrt((double)std::max(1, D));

            // Temperature from population spread (IQR) to make acceptance scale-free.
            std::vector<double> f2 = FX_;
            const double Q1 = quantile(f2, 0.25);
            f2 = FX_;
            const double Q3 = quantile(f2, 0.75);
            const double IQR = std::max(0.0, Q3 - Q1);

            const double delta = std::max(0.0, bestCandF - best_f_);
            const double temp = horizon_bridge_alpha_ * (0.20 * std::abs(best_f_) + 0.50 * IQR + 1e-12);

            if (dnorm >= horizon_bridge_dist_ && temp > 0.0) {
                const double pacc = std::exp(-delta / temp);
                if (randU() < std::min(1.0, std::max(0.0, pacc))) {
                    addToArchive(X_[worst_id]);
                    X_[worst_id] = bestCand;
                    FX_[worst_id] = bestCandF;
                    injected = true;
                    no_improve_ = std::max(0, no_improve_ - std::max(1, horizon_trigger_ / 2));
                }
            }
        }
    }

    if (injected) {
        resetRefine();
        dir_trust_ = std::min(0.99, std::max(0.05, 0.97 * dir_trust_));
        horizon_cooldown_left_ = horizon_cooldown_;
        pop_cooldown_left_ = std::max(pop_cooldown_left_, 2);
    }

    return injected;
}

void ARQDP::hardRestart(const std::vector<int>& sorted_idx)
{
    if (!hard_restart_enabled_) return;
    if (!prob_ || N_ < 4 || best_x_.empty()) return;
    if (hard_restart_cooldown_left_ > 0) return;
    if (hard_restart_max_ > 0 && hard_restart_count_ >= hard_restart_max_) return;
    if (no_improve_ < hard_restart_trigger_) return;

    const int D = prob_->dimension();
    if (D <= 0) return;
    if (!bounds_ready_) prepareBoundsCache();

    int remaining = (int)std::max(0.0, (double)max_evals_ - (double)prob_->calls());
    if (remaining <= 0) return;

    int k = (int)std::round(hard_restart_frac_ * (double)N_);
    k = std::max(1, std::min(N_ - 1, k));
    if ((int)sorted_idx.size() != N_) return;

    Vec best_bn;
    toBN(best_x_, best_bn);
    ensureDim(best_bn, D);

    std::vector<Vec> basins;
    basins.reserve((size_t)basin_memory_);
    for (int i = (int)best_bn_hist_.size() - 1; i >= 0 && (int)basins.size() < basin_memory_; --i) {
        if ((int)best_bn_hist_[i].size() == D) basins.push_back(best_bn_hist_[i]);
    }
    if (basins.empty()) basins.push_back(best_bn);

    auto minDistToBasins = [&](const Vec& bn)->double {
        double md = std::numeric_limits<double>::infinity();
        for (const auto& b : basins) {
            double d2 = 0.0;
            for (int j = 0; j < D; ++j) {
                double t = bn[j] - b[j];
                d2 += t * t;
            }
            md = std::min(md, std::sqrt(d2));
        }
        return md;
    };

    int replaced = 0;
    for (int t = 0; t < N_ && replaced < k; ++t) {
        if (terminated()) break;

        int id = sorted_idx[N_ - 1 - t];
        if (FX_[id] <= best_f_) continue;

        Vec best_bn_cand;
        double best_d = -1.0;

        const int pool = 18;
        for (int r = 0; r < pool; ++r) {
            Vec bn(D, 0.0);
            for (int j = 0; j < D; ++j) bn[j] = randU();
            if (no_improve_ >= 2 * hard_restart_trigger_ && randU() < 0.40) {
                for (int j = 0; j < D; ++j) bn[j] = reflect01(1.0 - best_bn[j] + 0.05 * randN01());
            }
            double d = minDistToBasins(bn);
            if (d > best_d) { best_d = d; best_bn_cand = std::move(bn); }
        }

        Vec cand;
        fromBN(best_bn_cand, cand);
        sanitizeInPlace(cand);
        double fc = evalX(cand);

        addToArchive(X_[id]);
        X_[id] = std::move(cand);
        FX_[id] = fc;
        replaced++;
    }

    recomputeBest();
    if (best_f_ < best_prev_) best_prev_ = best_f_;
    no_improve_ = 0;

    resetDirection();
    resetRefine();

    hard_restart_count_++;
    hard_restart_cooldown_left_ = hard_restart_cooldown_;
    pop_cooldown_left_ = std::max(pop_cooldown_left_, pop_cooldown_);
}


void ARQDP::quarantineAndRestart()
{
    if (!prob_ || N_ < 4) return;
    if (pop_cooldown_left_ > 0) return;

    std::vector<double> f = FX_;
    double Q1 = quantile(f, 0.25);
    f = FX_;
    double Q3 = quantile(f, 0.75);
    double IQR = Q3 - Q1;
    double fence = Q3 + outlier_alpha_ * IQR;

    std::vector<int> idx;
    sortByFitness(idx);

    Vec center_bn(prob_->dimension(), 0.0), bn;
    int half = std::max(1, N_ / 2);
    for (int k = 0; k < half; ++k) {
        toBN(X_[idx[k]], bn);
        for (int j = 0; j < (int)bn.size(); ++j) center_bn[j] += bn[j];
    }
    for (int j = 0; j < (int)center_bn.size(); ++j) center_bn[j] /= (double)half;

    std::vector<int> out;
    for (int i = 0; i < N_; ++i) if (FX_[i] >= fence) out.push_back(i);

    int kfix = (int)std::floor(outlier_rho_ * (double)out.size());
    if (kfix > 0) {
        std::shuffle(out.begin(), out.end(), rng_);
        out.resize(kfix);

        for (int id : out) {
            if (terminated()) break;

            Vec cand_bn = center_bn;
            for (int j = 0; j < (int)cand_bn.size(); ++j) cand_bn[j] = clamp01(cand_bn[j] + qsigma_ * randN01());

            Vec cand;
            fromBN(cand_bn, cand);
            sanitizeInPlace(cand);

            double fc = evalX(cand);
            if (fc < FX_[id]) {
                addToArchive(X_[id]);
                X_[id] = std::move(cand);
                FX_[id] = fc;
            }
        }
    }

    if (no_improve_ < stagnationtrigger_) return;

    int wcount = (int)std::floor(worst_frac_ * (double)N_);
    if (wcount <= 0) { no_improve_ = 0; return; }
    if (wcount > (int)idx.size()) wcount = (int)idx.size();

    Vec best_bn;
    toBN(best_x_, best_bn);

    for (int t = 0; t < wcount; ++t) {
        if (terminated()) break;
        int id = idx[(int)idx.size() - 1 - t];

        Vec cand_bn = best_bn;
        for (int j = 0; j < (int)cand_bn.size(); ++j) cand_bn[j] = clamp01(cand_bn[j] + rsigma_ * randN01());

        Vec cand;
        fromBN(cand_bn, cand);
        sanitizeInPlace(cand);

        double fc = evalX(cand);
        addToArchive(X_[id]);
        X_[id] = std::move(cand);
        FX_[id] = fc;
    }

    resetDirection();
    resetRefine();
    pop_cooldown_left_ = std::max(pop_cooldown_left_, 2);
    no_improve_ = 0;
}

void ARQDP::configure(const MethodConfig& mc)
{
    int p = mc.getInt("population", pop_init_);
    if (p > 3) {
        pop_init_ = p;
        Optimizer::setPopulation(pop_init_);
    }

    adaptive_population_ = mc.getBool("adaptive_population", adaptive_population_);
    pop_min_             = mc.getInt("pop_min", pop_min_);
    pop_max_             = mc.getInt("pop_max", std::max(pop_max_, pop_init_));
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_max_ < pop_min_) pop_max_ = pop_min_;

    pop_warmup_iters_    = mc.getInt("pop_warmup_iters", pop_warmup_iters_);
    if (pop_warmup_iters_ < 0) pop_warmup_iters_ = 0;

    pop_check_interval_  = mc.getInt("pop_check_interval", pop_check_interval_);
    if (pop_check_interval_ < 1) pop_check_interval_ = 1;

    pop_window_          = mc.getInt("pop_window", pop_window_);
    if (pop_window_ < 5) pop_window_ = 5;

    pop_success_thr_     = mc.getDbl("pop_success_thr", pop_success_thr_);
    pop_success_thr_     = std::max(0.0, std::min(1.0, pop_success_thr_));

    pop_impr_thr_        = mc.getDbl("pop_impr_thr", pop_impr_thr_);
    pop_impr_thr_        = std::max(0.0, pop_impr_thr_);

    pop_div_low_         = mc.getDbl("pop_div_low", pop_div_low_);
    pop_div_high_        = mc.getDbl("pop_div_high", pop_div_high_);
    pop_div_low_         = std::max(0.0, pop_div_low_);
    if (pop_div_high_ < pop_div_low_) pop_div_high_ = pop_div_low_;

    pop_shrink_factor_   = mc.getDbl("pop_shrink_factor", pop_shrink_factor_);
    if (pop_shrink_factor_ <= 0.0) pop_shrink_factor_ = 0.25;
    if (pop_shrink_factor_ > 0.95) pop_shrink_factor_ = 0.95;

    pop_expand_factor_   = mc.getDbl("pop_expand_factor", pop_expand_factor_);
    if (pop_expand_factor_ < 1.05) pop_expand_factor_ = 2.0;
    if (pop_expand_factor_ > 10.0) pop_expand_factor_ = 10.0;

    pop_elite_frac_      = mc.getDbl("pop_elite_frac", pop_elite_frac_);
    pop_elite_frac_      = std::max(0.05, std::min(0.80, pop_elite_frac_));

    pop_cooldown_        = mc.getInt("pop_cooldown", pop_cooldown_);
    if (pop_cooldown_ < 0) pop_cooldown_ = 0;

    H_          = mc.getInt("H", H_);
    if (H_ < 2) H_ = 2;

    pbest_      = mc.getDbl("pbest", pbest_);
    pbest_      = std::max(0.01, std::min(0.50, pbest_));

    Fmin_       = mc.getDbl("Fmin", Fmin_);
    Fmax_       = mc.getDbl("Fmax", Fmax_);
    if (Fmin_ <= 0.0) Fmin_ = 0.01;
    if (Fmax_ < Fmin_) std::swap(Fmax_, Fmin_);
    if (Fmax_ > 2.0) Fmax_ = 2.0;

    archiverate_ = mc.getDbl("archiverate", archiverate_);
    if (archiverate_ <= 0.1) archiverate_ = 1.0;

    rtr_k_            = mc.getInt("rtr_k", rtr_k_);
    if (rtr_k_ < 2) rtr_k_ = 2;

    outlier_alpha_    = mc.getDbl("outlier_alpha", outlier_alpha_);
    if (outlier_alpha_ <= 0.0) outlier_alpha_ = 1.5;

    outlier_rho_      = mc.getDbl("outlier_rho", outlier_rho_);
    outlier_rho_      = std::max(0.0, std::min(1.0, outlier_rho_));

    qsigma_           = mc.getDbl("qsigma", qsigma_);
    qsigma_           = std::max(0.0, std::min(2.0, qsigma_));

    worst_frac_       = mc.getDbl("worst_frac", worst_frac_);
    worst_frac_       = std::max(0.0, std::min(0.5, worst_frac_));

    rsigma_           = mc.getDbl("rsigma", rsigma_);
    rsigma_           = std::max(0.0, std::min(2.0, rsigma_));

    stagnationtrigger_ = mc.getInt("stagnationtrigger", stagnationtrigger_);
    if (stagnationtrigger_ < 5) stagnationtrigger_ = 5;

    quarantine_enabled_ = mc.getBool("quarantine_enabled", quarantine_enabled_);

    // Direction settings
    dir_enabled_       = mc.getBool("dir_enabled", dir_enabled_);
    pdir_              = mc.getDbl("pdir", pdir_);
    pdir_              = std::max(0.0, std::min(1.0, pdir_));

    dir_beta_          = mc.getDbl("dirF", mc.getDbl("dir_beta", dir_beta_));
    dir_c_             = mc.getDbl("dir_learn", mc.getDbl("dir_c", dir_c_));
    dir_c_             = std::max(0.01, std::min(1.00, dir_c_));

    dir_clip_          = mc.getDbl("dir_clip", dir_clip_);
    dir_clip_          = std::max(1e-6, std::min(2.0, dir_clip_));

    dir_decay_         = mc.getDbl("dir_decay", dir_decay_);
    dir_decay_         = std::max(0.0, std::min(1.0, dir_decay_));

    dir_warmup_        = mc.getInt("dir_warmup", dir_warmup_);
    if (dir_warmup_ < 0) dir_warmup_ = 0;

    dir_step_clip_     = mc.getDbl("dir_step_clip", dir_step_clip_);
    dir_step_clip_     = std::max(1e-6, std::min(5.0, dir_step_clip_));

    dir_trust_inc_     = mc.getDbl("dir_trust_inc", dir_trust_inc_);
    dir_trust_dec_     = mc.getDbl("dir_trust_dec", dir_trust_dec_);
    dir_disable_after_ = mc.getInt("dir_disable_after", dir_disable_after_);
    dir_disabled_period_= mc.getInt("dir_disabled_period", dir_disabled_period_);
    if (dir_disable_after_ < 2) dir_disable_after_ = 2;
    if (dir_disabled_period_ < 1) dir_disabled_period_ = 1;

    // Passive direction learning factor (learn from all successful steps, even if direction was not injected)
    dir_passive_factor_ = mc.getDbl("dir_passive_factor", dir_passive_factor_);
    dir_passive_factor_ = std::max(0.0, std::min(1.0, dir_passive_factor_));

    // Far-horizon prediction / look-ahead (trajectory-based)
    dir_hist_len_        = mc.getInt("dir_hist_len", dir_hist_len_);
    if (dir_hist_len_ < 3) dir_hist_len_ = 3;
    if (dir_hist_len_ > 60) dir_hist_len_ = 60;

    dir_far_mix_         = mc.getDbl("dir_far_mix", dir_far_mix_);
    dir_far_mix_         = std::max(0.0, std::min(1.0, dir_far_mix_));

    dir_far_clip_        = mc.getDbl("dir_far_clip", dir_far_clip_);
    dir_far_clip_        = std::max(0.05, std::min(1.0, dir_far_clip_));

    dir_lookahead_       = mc.getDbl("dir_lookahead", dir_lookahead_);
    dir_lookahead_min_   = mc.getDbl("dir_lookahead_min", dir_lookahead_min_);
    dir_lookahead_max_   = mc.getDbl("dir_lookahead_max", dir_lookahead_max_);
    dir_lookahead_grow_  = mc.getDbl("dir_lookahead_grow", dir_lookahead_grow_);
    dir_lookahead_shrink_= mc.getDbl("dir_lookahead_shrink", dir_lookahead_shrink_);

    dir_lookahead_min_   = std::max(0.0, dir_lookahead_min_);
    dir_lookahead_max_   = std::max(dir_lookahead_min_, dir_lookahead_max_);
    dir_lookahead_       = std::max(dir_lookahead_min_, std::min(dir_lookahead_max_, dir_lookahead_));

    if (dir_lookahead_grow_ < 1.0) dir_lookahead_grow_ = 1.12;
    if (dir_lookahead_shrink_ <= 0.0 || dir_lookahead_shrink_ >= 1.0) dir_lookahead_shrink_ = 0.92;

    // Endgame caps (performance safeguard)
    endgame_shots_cap_   = mc.getInt("endgame_shots_cap", endgame_shots_cap_);
    endgame_refine_cap_  = mc.getInt("endgame_refine_cap", endgame_refine_cap_);
    if (endgame_shots_cap_ < 2) endgame_shots_cap_ = 2;
    if (endgame_refine_cap_ < 20) endgame_refine_cap_ = 20;

    // Best shots
    best_shots_enabled_   = mc.getBool("best_shots_enabled", best_shots_enabled_);
    best_shots_           = mc.getInt("best_shots", best_shots_);
    best_shot_every_      = mc.getInt("best_shot_every", best_shot_every_);
    best_shot_sigma_      = mc.getDbl("best_shot_sigma", best_shot_sigma_);
    best_shot_cauchy_prob_= mc.getDbl("best_shot_cauchy_prob", best_shot_cauchy_prob_);
    best_shot_dir_mix_    = mc.getDbl("best_shot_dir_mix", best_shot_dir_mix_);
    best_shot_elite_mix_  = mc.getDbl("best_shot_elite_mix", best_shot_elite_mix_);
    best_shot_clip_       = mc.getDbl("best_shot_clip", best_shot_clip_);
    best_shots_ = std::max(0, best_shots_);
    if (best_shot_every_ < 1) best_shot_every_ = 1;

    // Escape
    escape_enabled_       = mc.getBool("escape_enabled", escape_enabled_);
    escape_trigger_       = mc.getInt("escape_trigger", escape_trigger_);
    escape_frac_          = mc.getDbl("escape_frac", escape_frac_);
    escape_sigma_         = mc.getDbl("escape_sigma", escape_sigma_);
    escape_opposition_prob_= mc.getDbl("escape_opposition_prob", escape_opposition_prob_);
    escape_cauchy_prob_   = mc.getDbl("escape_cauchy_prob", escape_cauchy_prob_);
    if (escape_trigger_ < 5) escape_trigger_ = 5;
    escape_frac_ = std::max(0.0, std::min(0.8, escape_frac_));

    
    // Horizon scan (far-looking)
    horizon_enabled_      = mc.getBool("horizon_enabled", horizon_enabled_);
    horizon_trigger_      = mc.getInt("horizon_trigger", horizon_trigger_);
    if (horizon_trigger_ < 3) horizon_trigger_ = 3;
    horizon_shots_        = mc.getInt("horizon_shots", horizon_shots_);
    horizon_shots_        = std::max(3, std::min(40, horizon_shots_));
    horizon_scale0_       = mc.getDbl("horizon_scale0", horizon_scale0_);
    horizon_scale0_       = std::max(0.05, std::min(3.0, horizon_scale0_));
    horizon_scale_growth_ = mc.getDbl("horizon_scale_growth", horizon_scale_growth_);
    horizon_scale_growth_ = std::max(1.05, std::min(2.2, horizon_scale_growth_));
    horizon_scale_max_    = mc.getDbl("horizon_scale_max", horizon_scale_max_);
    horizon_scale_max_    = std::max(0.5, std::min(25.0, horizon_scale_max_));
    horizon_orth_sigma_   = mc.getDbl("horizon_orth_sigma", horizon_orth_sigma_);
    horizon_orth_sigma_   = std::max(0.0, std::min(0.40, horizon_orth_sigma_));
    horizon_accept_q_     = mc.getDbl("horizon_accept_q", horizon_accept_q_);
    horizon_accept_q_     = std::max(0.50, std::min(0.95, horizon_accept_q_));

    horizon_bridge_enabled_ = mc.getBool("horizon_bridge_enabled", horizon_bridge_enabled_);
    horizon_bridge_dist_    = mc.getDbl("horizon_bridge_dist", horizon_bridge_dist_);
    horizon_bridge_dist_    = std::max(0.0, std::min(1.0, horizon_bridge_dist_));
    horizon_bridge_alpha_   = mc.getDbl("horizon_bridge_alpha", horizon_bridge_alpha_);
    horizon_bridge_alpha_   = std::max(0.0, std::min(10.0, horizon_bridge_alpha_));
    horizon_div_thr_      = mc.getDbl("horizon_div_thr", horizon_div_thr_);
    horizon_div_thr_      = std::max(0.0, std::min(1.0, horizon_div_thr_));
    horizon_cooldown_     = mc.getInt("horizon_cooldown", horizon_cooldown_);
    if (horizon_cooldown_ < 0) horizon_cooldown_ = 0;

    // Hard restart
    hard_restart_enabled_       = mc.getBool("hard_restart_enabled", hard_restart_enabled_);
    hard_restart_trigger_       = mc.getInt("hard_restart_trigger", hard_restart_trigger_);
    if (hard_restart_trigger_ < 5) hard_restart_trigger_ = 5;
    hard_restart_frac_          = mc.getDbl("hard_restart_frac", hard_restart_frac_);
    hard_restart_frac_          = std::max(0.05, std::min(0.80, hard_restart_frac_));
    hard_restart_cooldown_      = mc.getInt("hard_restart_cooldown", hard_restart_cooldown_);
    if (hard_restart_cooldown_ < 0) hard_restart_cooldown_ = 0;
    hard_restart_max_           = mc.getInt("hard_restart_max", hard_restart_max_);
    hard_restart_max_           = std::max(0, std::min(20, hard_restart_max_));
    basin_memory_               = mc.getInt("basin_memory", basin_memory_);
    basin_memory_               = std::max(2, std::min(100, basin_memory_));
// Refine
    refine_enabled_      = mc.getBool("refine_enabled", refine_enabled_);
    refine_trigger_      = mc.getInt("refine_trigger", refine_trigger_);
    if (refine_trigger_ < 3) refine_trigger_ = 3;

    refine_every_        = mc.getInt("refine_every", refine_every_);
    if (refine_every_ < 1) refine_every_ = 1;

    refine_div_thr_      = mc.getDbl("refine_div_thr", refine_div_thr_);
    refine_div_thr_      = std::max(0.0, std::min(1.0, refine_div_thr_));

    refine_success_thr_  = mc.getDbl("refine_success_thr", refine_success_thr_);
    refine_success_thr_  = std::max(0.0, std::min(1.0, refine_success_thr_));

    refine_budget_       = mc.getInt("refine_budget", refine_budget_);
    if (refine_budget_ < 50) refine_budget_ = 50;

    refine_subspace_frac_ = mc.getDbl("refine_subspace_frac", refine_subspace_frac_);
    refine_subspace_frac_ = std::max(0.01, std::min(1.0, refine_subspace_frac_));

    refine_sigma0_       = mc.getDbl("refine_sigma0", refine_sigma0_);
    refine_sigma0_       = std::max(1e-8, std::min(1.0, refine_sigma0_));

    refine_shrink_       = mc.getDbl("refine_shrink", refine_shrink_);
    refine_grow_         = mc.getDbl("refine_grow", refine_grow_);
    if (refine_shrink_ <= 0.0 || refine_shrink_ >= 1.0) refine_shrink_ = 0.70;
    if (refine_grow_ < 1.0) refine_grow_ = 1.10;

    refine_clip_         = mc.getDbl("refine_clip", refine_clip_);
    refine_clip_         = std::max(1e-8, std::min(2.0, refine_clip_));

    refine_dir_mix_      = mc.getDbl("refine_dir_mix", refine_dir_mix_);
    refine_dir_mix_      = std::max(0.0, std::min(1.0, refine_dir_mix_));

    local_method_ = to_lower(mc.getStr("local_method", local_method_));
    double lr = mc.getDbl("local_rate", local_rate_);
    local_rate_ = std::max(0.0, std::min(1.0, lr));

    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = to_lower(mc.getStr("end_local_method", end_local_method_));
}

void ARQDP::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    prepareBoundsCache();

    Optimizer::setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);

    for (auto &xi : X_) sanitizeInPlace(xi);

    N_ = (int)X_.size();
    FX_.assign(N_, std::numeric_limits<double>::infinity());

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i = 0; i < N_; ++i) {
        if (terminated()) break;
        FX_[i] = evalX(X_[i]);
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
    }

    MF_.assign(H_, 0.6);
    MCR_.assign(H_, 0.8);
    k_mem_ = 0;

    A_.clear();

    best_prev_ = best_f_;
    no_improve_ = 0;

    iter_ = 0;
    pop_cooldown_left_ = 0;
    best_hist_.clear();
    if (std::isfinite(best_f_)) best_hist_.push_back(best_f_);

    dir_path_bn_.assign(D, 0.0);
    dir_step_sum_bn_.assign(D, 0.0);
    dir_step_wsum_ = 0.0;

    resetDirection();
    resetRefine();

    updateStop(FX_);
    printBest();

    horizon_cooldown_left_ = 0;
    hard_restart_cooldown_left_ = 0;
    hard_restart_count_ = 0;
}

void ARQDP::one_iteration()
{
    if (!prob_ || terminated() || X_.empty() || N_ < 4) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (!bounds_ready_) prepareBoundsCache();

    // cool-down for directional bias (when disabled due to harm)
    if (dir_disabled_for_ > 0) dir_disabled_for_--;
    if (horizon_cooldown_left_ > 0) horizon_cooldown_left_--;
    if (hard_restart_cooldown_left_ > 0) hard_restart_cooldown_left_--;

    trimArchive();

    std::vector<int> idx;
    sortByFitness(idx);

    std::vector<double> SF, SCR, gains;
    SF.reserve((size_t)N_);
    SCR.reserve((size_t)N_);
    gains.reserve((size_t)N_);

    std::fill(dir_step_sum_bn_.begin(), dir_step_sum_bn_.end(), 0.0);
    dir_step_wsum_ = 0.0;

    int succ_total = 0;
    int succ_dir   = 0;
    int dir_used   = 0;

    const double calls = (double)prob_->calls();
    const double maxc  = (double)max_evals_;
    const double progress = (maxc > 0.0) ? std::min(1.0, std::max(0.0, calls / maxc)) : 0.0;
    const int remaining = (int)std::max(0.0, maxc - calls);

    // --- DE/ARQ generation ---
    for (int i = 0; i < N_; ++i) {
        if (terminated()) break;

        int r = randInt(0, H_ - 1);
        double muF  = MF_[r];
        double muCR = MCR_[r];

        double F, CR;
        sample_F_CR(F, CR, muF, muCR);

        double pbest_use = pbest_;

        // IMPORTANT: when stagnating we must *widen* the pbest pool (more exploration),
        // not shrink it. Shrinking pbest makes the method excessively greedy and is a
        // common root-cause of "early good, then plateau" across problems.
        if (no_improve_ >= stagnationtrigger_) {
            pbest_use = std::min(0.35, std::max(0.05, pbest_ * 1.80));
            const double late = (progress > 0.85) ? 0.80 : 1.00;
            F  = std::min(2.0, F * (1.20 * late) + 0.05);
            CR = std::min(1.0, CR + 0.10 * late);

            // Occasional "shock" mutation during stagnation: inject a high-F, varied-CR trial.
            if (randU() < 0.12) {
                F  = std::min(2.0, 0.90 + 0.80 * randU());
                CR = std::min(1.0, 0.10 + 0.90 * randU());
            }
        }

        Vec u;
        bool used_dir = false;
        makeTrialBase(i, idx, F, CR, pbest_use, u, used_dir);

        double fu = evalX(u);

        if (used_dir) dir_used++;

        selectionRTR(i, u, fu, used_dir, F, CR, SF, SCR, gains, succ_total, succ_dir);

        if (fu < best_f_) { best_f_ = fu; best_x_ = sanitizedCopy(u); }
    }

    updateMemories(SF, SCR, gains);
    trimArchive();

    const double success_rate = (N_ > 0) ? ((double)SF.size() / (double)N_) : 0.0;

    auto recomputeBest = [&]() {
        best_f_ = std::numeric_limits<double>::infinity();
        for (int i = 0; i < N_; ++i) {
            sanitizeInPlace(X_[i]);
            if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        }
    };

    // --- Baseline best + stagnation counter ---
    recomputeBest();
    if (best_f_ < best_prev_) { best_prev_ = best_f_; no_improve_ = 0; }
    else { no_improve_++; }

    double diversity_bn = estimateDiversityBN(std::min(N_, 24));

    // --- Best-first attack / escape / polish (budgeted) ---
    (void)bestShots(idx, diversity_bn);

    escapeBurst(idx);

    (void)horizonScan(idx, diversity_bn);
    hardRestart(idx);

    // PBLE (precision burst) can be very costly; cap it in the endgame
    if (remaining > 2 * N_ && shouldRefine(success_rate, diversity_bn)) {
        (void)precisionBurst(diversity_bn);
    }

    if (quarantine_enabled_) quarantineAndRestart();

    // In-run local search: very small rate only if enabled
    if (!local_method_.empty() && local_rate_ > 0.0) {
        for (int i = 0; i < N_; ++i) {
            if (terminated()) break;
            if (randU() < local_rate_) {
                Vec x0 = sanitizedCopy(X_[i]);
                auto res = localSearch(local_method_, x0);
                Vec xr = res.first;
                double fr = res.second;
                sanitizeInPlace(xr);
                if (std::isfinite(fr) && fr < FX_[i]) {
                    X_[i]  = std::move(xr);
                    FX_[i] = fr;
                }
            }
        }
    }

    // Final best recomputation; if auxiliary steps improved best, clear stagnation immediately.
    recomputeBest();
    if (best_f_ < best_prev_) { best_prev_ = best_f_; no_improve_ = 0; }

    // Update history using the final best of this iteration (not the pre-attack best).
    if (std::isfinite(best_f_)) {
        best_hist_.push_back(best_f_);
        while ((int)best_hist_.size() > pop_window_) best_hist_.pop_front();
    }

    const double rel_impr = relImprovementFromHistory();

    // Update long-horizon direction from best trajectory (helps avoid local traps)
    updateFarPrediction(rel_impr, diversity_bn);

    const double dir_use_rate  = (N_ > 0)       ? ((double)dir_used / (double)N_) : 0.0;
    const double dir_succ_rate = (dir_used > 0) ? ((double)succ_dir / (double)dir_used) : 0.0;

    // Direction trust update (gates direction usage next iter)
    updateDirectionTrust(success_rate, rel_impr, dir_use_rate, dir_succ_rate);

    // Adaptive population adjustment (uses up-to-date rel_impr)
    sortByFitness(idx);
    if (maybeAdaptivePopulationLeap(idx, success_rate, diversity_bn, rel_impr)) {
        sortByFitness(idx);
    }

    iter_++;
    if (pop_cooldown_left_ > 0) pop_cooldown_left_--;

    updateStop(FX_);
    printBest();
}


void ARQDP::end()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    sanitizeInPlace(best_x_);

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        Vec x0 = sanitizedCopy(best_x_);
        auto res = localSearch(end_local_method_, x0);
        Vec xr = res.first;
        double fr = res.second;
        sanitizeInPlace(xr);
        if (!xr.empty() && std::isfinite(fr) && fr < best_f_) {
            best_x_ = std::move(xr);
            best_f_ = fr;
        }

        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) { fw = FX_[i]; worst = i; }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }

        printBest();
    }

    updateStop(FX_);
}

} // namespace optimsolution
