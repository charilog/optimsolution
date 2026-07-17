#include "emscso.h"
#include "init.h"
#include <cmath>
#include <limits>

namespace optimsolution {

namespace { constexpr double EMSCSO_PI = 3.141592653589793238462643383279502884; }

void EMSCSO::goodPointSet(int n, int z, std::vector<std::vector<double>>& F) const {
    F.assign(std::max(0, n), std::vector<double>(std::max(0, z), 0.0));
    if (z <= 0 || n <= 0) return;

    auto is_prime = [](int v) {
        if (v < 2) return false;
        for (int d = 2; (long long)d * d <= v; ++d) {
            if (v % d == 0) return false;
        }
        return true;
    };
    int p = 2 * z + 3;
    while (!is_prime(p)) ++p;

    std::vector<double> r(z);
    for (int j = 1; j <= z; ++j) {
        r[j - 1] = 2.0 * std::cos(2.0 * EMSCSO_PI * (double)j / (double)p);
    }

    for (int k = 1; k <= n; ++k) {
        for (int j = 0; j < z; ++j) {
            const double v = (double)k * r[j];
            F[k - 1][j] = v - std::floor(v);
        }
    }
}

void EMSCSO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    X_.assign(pop_, std::vector<double>(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());
    Xbest_.assign(D, 0.0);
    Fbest_ = std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> F;
    goodPointSet(pop_, D, F);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < pop_; ++i) {
        for (int j = 0; j < D; ++j) {
            double lo = L[j], hi = U[j];
            if (!std::isfinite(lo) || !std::isfinite(hi)) { lo = -10.0; hi = 10.0; }
            X_[i][j] = lo + F[i][j] * (hi - lo);
        }
        FX_[i] = eval(X_[i]);
        if (prob_->calls() >= max_evals_) break;
    }

    Xbest_ = X_[0];
    Fbest_ = FX_[0];
    for (int i = 1; i < pop_; ++i) {
        if (FX_[i] < Fbest_) { Fbest_ = FX_[i]; Xbest_ = X_[i]; }
    }

    best_x_ = Xbest_;
    best_f_ = Fbest_;

    no_improve_ = 0;
    polish_mark_f_ = Fbest_;
    polish_mark_calls_ = (long long)prob_->calls();
}

void EMSCSO::ensureBounds(std::vector<double>& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// Mantegna's algorithm for sampling a symmetric Levy-stable step (adapted
// verbatim from SPARQ's own validated sampleLevy(), which cites the same
// source: Mantegna, 1994).
double EMSCSO::sampleLevy() {
    const double beta = levy_beta_;
    const double num = std::tgamma(1.0 + beta) * std::sin(EMSCSO_PI * beta / 2.0);
    const double den = std::tgamma((1.0 + beta) / 2.0) * beta
                      * std::pow(2.0, (beta - 1.0) / 2.0);
    const double sigma_u = std::pow(num / den, 1.0 / beta);

    std::normal_distribution<double> Nu(0.0, sigma_u);
    std::normal_distribution<double> Nv(0.0, 1.0);
    const double u = Nu(rng_);
    const double v = Nv(rng_);
    double step = u / std::pow(std::fabs(v) + 1e-12, 1.0 / beta);

    // Clip extreme tails: Levy draws can produce huge outliers numerically.
    if (step >  50.0) step =  50.0;
    if (step < -50.0) step = -50.0;
    return step;
}

// Normalized population spread: RMS of each dimension's population std-dev
// relative to that dimension's box range. Near 0 means the population has
// collapsed onto (approximately) a single point.
double EMSCSO::normalizedPopSpread() const {
    const int N = (int)X_.size();
    if (N < 2) return 1.0;
    const int D = (int)X_[0].size();
    if (D <= 0) return 1.0;

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double s2sum = 0.0;
    int counted = 0;

    for (int j = 0; j < D; ++j) {
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        const double range = hi - lo;
        if (range <= 0.0) continue;

        double mean = 0.0;
        for (int i = 0; i < N; ++i) mean += X_[i][j];
        mean /= (double)N;

        double var = 0.0;
        for (int i = 0; i < N; ++i) {
            const double d = X_[i][j] - mean;
            var += d * d;
        }
        var /= (double)N;

        const double sd = std::sqrt(var);
        const double z  = sd / range;
        s2sum += z * z;
        ++counted;
    }
    if (counted == 0) return 1.0;
    return std::sqrt(s2sum / (double)counted);
}

// ============================================================================
// oblBasinEscape -- Opposition-Based Learning basin escape (adapted from
// SPARQ's oblBasinEscape(); see header for the full rationale). Self-gated:
// safe to call every iteration, it only acts when BOTH the population has
// genuinely collapsed AND the incumbent has been stuck for obl_trigger_
// iterations.
// ============================================================================
void EMSCSO::oblBasinEscape() {
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N < 4) return;
    if (obl_cooldown_ > 0) { --obl_cooldown_; return; }

    const bool stag      = (no_improve_ >= obl_trigger_);
    const bool collapsed = (normalizedPopSpread() < var_collapse_ratio_);
    if (!(stag && collapsed)) return;

    std::vector<int> ord(N);
    for (int i = 0; i < N; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

    const int count = std::max(1, (int)std::floor(obl_frac_ * (double)N));
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = prob_->dimension();

    int applied = 0;
    for (int t = 0; t < count; ++t) {
        if (prob_->calls() >= max_evals_) break;
        const int idx = ord[N - 1 - t]; // worst-ranked individuals first

        std::vector<double> cand(D);
        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            const double opp = lo + hi - X_[idx][j];
            const double mid = Xbest_[j];
            // 50/50 mix: pure box-opposite vs. quasi-opposition toward best.
            if (randU() < 0.5) cand[j] = opp;
            else               cand[j] = mid + (opp - mid) * randU();
        }
        ensureBounds(cand);
        const double fc = eval(cand);
        ++applied;

        if (fc < FX_[idx]) {
            X_[idx]  = cand;
            FX_[idx] = fc;
            if (fc < Fbest_) {
                Fbest_  = fc;
                Xbest_  = X_[idx];
                best_f_ = Fbest_;
                best_x_ = Xbest_;
            }
        }
    }

    if (applied > 0) {
        obl_cooldown_ = obl_cooldown_init_;
        no_improve_   = 0;
    }
}

// ============================================================================
// elitePolish -- adapted from SPARQ (see header for the full rationale).
// Simplified to two probe modes (single-coordinate, full-D isotropic) since
// MSCSO has no eigen-basis / archive / trajectory-echo machinery to draw on.
// ============================================================================
void EMSCSO::elitePolish() {
    if (!prob_ || best_x_.empty()) return;
    if (polish_cooldown_ > 0) { --polish_cooldown_; return; }

    const double f_before = best_f_;

    double evo_eff = std::numeric_limits<double>::infinity();
    if (std::isfinite(polish_mark_f_)) {
        const double evo_gain  = std::max(0.0, polish_mark_f_ - best_f_);
        const double evo_evals = (double)std::max<long long>(
            1, (long long)prob_->calls() - polish_mark_calls_);
        evo_eff = evo_gain / evo_evals;
    }
    const long long used_before = polish_used_;

    const int N = (int)X_.size();
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    int k = std::max(6, (int)std::round(polish_frac_ * (double)N));
    int fail_streak = 0;
    int wins = 0;

    for (int t = 0; t < k; ++t) {
        if (prob_->calls() >= max_evals_) break;
        if (fail_streak >= 10) break;
        if ((double)polish_used_ >
            polish_budget_ * (double)std::max(1, (int)prob_->calls())) break;

        std::vector<double> cand = best_x_;
        const double mode = randU();

        if (mode < 0.5) {
            // Single-coordinate probe: round-robin sweep, half oppositions
            // (x_j -> lo+hi-x_j, jumps to the mirror basin of a bistable
            // dimension), half adaptive Gaussian steps.
            int j = polish_coord_ptr_;
            polish_coord_ptr_ = (polish_coord_ptr_ + 1) % D;
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            if (randU() < 0.5) cand[j] = lo + hi - cand[j];
            else cand[j] += ps_sigma_c_ * (hi - lo) * gaussN();
        } else {
            // Full-D isotropic Gaussian step.
            for (int j = 0; j < D; ++j) {
                double lo = (j < (int)L.size() ? L[j] : -1.0);
                double hi = (j < (int)U.size() ? U[j] :  1.0);
                if (lo > hi) std::swap(lo, hi);
                cand[j] += ps_sigma_ * (hi - lo) * gaussN();
            }
        }

        ensureBounds(cand);
        const double fc = eval(cand);
        ++polish_used_;

        if (fc < best_f_) {
            best_f_ = fc;
            best_x_ = cand;
            Xbest_  = cand;
            Fbest_  = fc;
            fail_streak = 0;
            ++wins;
            if (mode < 0.5) ps_sigma_c_ *= 1.5;
            else            ps_sigma_   *= 1.5;

            // Injection: propagate the refinement into the population by
            // replacing its current worst individual.
            int worst = 0;
            for (int i = 1; i < N; ++i) if (FX_[i] > FX_[worst]) worst = i;
            X_[worst]  = cand;
            FX_[worst] = fc;
        } else {
            ++fail_streak;
            if (mode < 0.5) ps_sigma_c_ *= 0.90;
            else            ps_sigma_   *= 0.87;
        }

        if (ps_sigma_   < ps_sigma_min_) ps_sigma_   = ps_sigma_min_;
        if (ps_sigma_   > ps_sigma_max_) ps_sigma_   = ps_sigma_max_;
        if (ps_sigma_c_ < 1e-7)          ps_sigma_c_ = 1e-7;
        if (ps_sigma_c_ > 0.5)           ps_sigma_c_ = 0.5;
    }

    // Exponential backoff on fruitless bursts; any success resets it.
    if (wins == 0) {
        polish_backoff_  = std::min(16, std::max(2, polish_backoff_ * 2));
        polish_cooldown_ = polish_backoff_;
    } else {
        polish_backoff_ = 0;
    }

    // Operator-efficiency arbitration: stand down for a while if this burst
    // produced less gain-per-eval than the main loop achieved since the last
    // burst -- the main loop is the better investment on this landscape.
    {
        const double pol_gain  = std::max(0.0, f_before - best_f_);
        const double pol_evals = (double)std::max<long long>(1, polish_used_ - used_before);
        const double pol_eff   = pol_gain / pol_evals;
        if (std::isfinite(evo_eff) && pol_eff < evo_eff) {
            polish_cooldown_ = std::max(polish_cooldown_, 64);
        }
    }
    polish_mark_f_     = best_f_;
    polish_mark_calls_ = (long long)prob_->calls();

    // Landscape self-selection: repeated negligible-gain bursts earn a long
    // cooldown (not a permanent disable -- later in the run refinement may
    // become useful again as the population itself converges further).
    const double denom = std::max(1.0, std::fabs(f_before));
    const double relgain = (f_before - best_f_) / denom;
    if (relgain < polish_min_relgain_) {
        if (++polish_low_streak_ >= 4) {
            polish_cooldown_ = std::max(polish_cooldown_, 64);
            polish_low_streak_ = 0;
        }
    } else {
        polish_low_streak_ = 0;
    }
}

// ============================================================================
// rejuvenate -- adapted from SPARQ (see header for the full rationale).
// Simplified to a single hard-stagnation trigger (no collapsed-spread
// "survival" tier, since MSCSO's population is already reseeded gently by
// its own sparrow-warning mechanism every iteration): once no_improve_
// reaches rejuv_trigger_, the worst (1-rejuv_keep_) fraction of the
// population is re-initialised via a fresh good-point-set sample. The best
// rejuv_keep_ fraction, and best_x_/Xbest_ (never part of the population),
// are always preserved untouched.
// ============================================================================
void EMSCSO::rejuvenate() {
    if (rejuv_cooldown_ > 0) { --rejuv_cooldown_; return; }
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N < 4) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::vector<int> ord(N);
    for (int i = 0; i < N; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

    const int keep = std::max(1, (int)std::round(rejuv_keep_ * (double)N));
    const int nreseed = N - keep;
    if (nreseed <= 0) { no_improve_ = 0; rejuv_cooldown_ = rejuv_cooldown_init_; return; }

    std::vector<std::vector<double>> F;
    goodPointSet(nreseed, D, F);

    for (int r = 0; r < nreseed; ++r) {
        const int idx = ord[keep + r];
        for (int j = 0; j < D; ++j) {
            double lo = L[j], hi = U[j];
            if (!std::isfinite(lo) || !std::isfinite(hi)) { lo = -10.0; hi = 10.0; }
            X_[idx][j] = lo + F[r][j] * (hi - lo);
        }
        FX_[idx] = eval(X_[idx]);
        if (FX_[idx] < Fbest_) {
            Fbest_  = FX_[idx];
            Xbest_  = X_[idx];
            best_f_ = Fbest_;
            best_x_ = Xbest_;
        }
        if (prob_->calls() >= max_evals_) break;
    }

    no_improve_       = 0;
    rejuv_cooldown_    = rejuv_cooldown_init_;
    polish_mark_f_     = best_f_;
    polish_mark_calls_ = (long long)prob_->calls();
}

void EMSCSO::one_iteration() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int n = pop_;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> Uang(1.0, 360.0);
    std::uniform_real_distribution<double> Uk(-1.0, 1.0);
    std::normal_distribution<double>       Nstd(0.0, 1.0);

    const double f_at_iter_start = Fbest_;

    const double t    = (double)iters_;
    const double Tmax = (double)std::max(1, max_iters_);

    // --- Nonlinear adjustment strategy (Eqs. 13-14) ---
    const double f_nl = std::sin((EMSCSO_PI / 4.0) * (t / Tmax));
    const double rg   = SM_ - (SM_ * t / Tmax) * f_nl;

    // --- R, r for this iteration (Eqs. 4-5); shared by the whole population ---
    const double R      = 2.0 * rg * U01(rng_) - rg;
    const double r_sens = rg - U01(rng_);

    // --- Main SCSO position update ---
    for (int i = 0; i < n; ++i) {
        const double theta_rad = Uang(rng_) * (EMSCSO_PI / 180.0);
        const double c_theta   = std::cos(theta_rad);

        std::vector<double> xnew(D);
        if (std::fabs(R) <= 1.0) {
            for (int j = 0; j < D; ++j) {
                const double xrnd = std::fabs(U01(rng_) * (Xbest_[j] - X_[i][j]));
                xnew[j] = Xbest_[j] - r_sens * xrnd * c_theta;
            }
        } else {
            // Search / exploration (Eq. 6, corrected form). The Levy-flight
            // jump is STAGNATION-RAMPED: its effective probability scales
            // linearly with no_improve_ up to levy_ramp_ iterations, so a
            // run that is still making healthy progress sees it rarely (near
            // 0% early on), while a run that has genuinely stalled ramps up
            // to the full levy_prob_ chance of a long-range escape jump.
            // (An earlier unconditional levy_prob_ on every exploration step
            // regressed already-well-behaved landscapes like rastrigin --
            // e.g. success rate dropped from 87% to 37% in testing -- by
            // disrupting runs that did not need any escape mechanism at
            // all.)
            const double ramp = (levy_ramp_ > 0)
                ? std::min(1.0, (double)no_improve_ / (double)levy_ramp_)
                : 1.0;
            if (randU() < levy_prob_ * ramp) {
                for (int j = 0; j < D; ++j) {
                    const double lv = sampleLevy();
                    xnew[j] = X_[i][j] + levy_scale_ * lv * (X_[i][j] - Xbest_[j]);
                }
            } else {
                for (int j = 0; j < D; ++j) {
                    xnew[j] = r_sens * (Xbest_[j] - U01(rng_) * X_[i][j]);
                }
            }
        }

        ensureBounds(xnew);
        const double fnew = eval(xnew);

        X_[i]  = std::move(xnew);
        FX_[i] = fnew;

        if (fnew < Fbest_) { Fbest_ = fnew; Xbest_ = X_[i]; }

        if (prob_->calls() >= max_evals_) break;
    }

    int worst_idx = 0;
    double fworst = FX_[0];
    for (int i = 1; i < n; ++i) {
        if (FX_[i] > fworst) { fworst = FX_[i]; worst_idx = i; }
    }

    // --- Sparrow early-warning mechanism ---
    const int nwarn = std::max(1, (int)std::lround(warning_frac_ * n));
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng_);

    for (int w = 0; w < nwarn; ++w) {
        const int i = idx[w];
        std::vector<double> xcand(D);

        if (FX_[i] > Fbest_) {
            const double beta = Nstd(rng_);
            for (int j = 0; j < D; ++j) {
                xcand[j] = Xbest_[j] + beta * std::fabs(X_[i][j] - Xbest_[j]);
            }
        } else {
            const double K = Uk(rng_);
            const double denom = std::fabs(FX_[i] - fworst) + eps_;
            for (int j = 0; j < D; ++j) {
                xcand[j] = X_[i][j] + K * ((X_[i][j] - X_[worst_idx][j]) / denom);
            }
        }

        ensureBounds(xcand);
        const double fcand = eval(xcand);

        if (fcand < FX_[i]) {
            X_[i]  = std::move(xcand);
            FX_[i] = fcand;
            if (fcand < Fbest_) { Fbest_ = fcand; Xbest_ = X_[i]; }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    best_x_ = Xbest_;
    best_f_ = Fbest_;

    // --- Stagnation tracking, then OBL / elitePolish / rejuvenate ---
    if (Fbest_ < f_at_iter_start) {
        no_improve_ = 0;
    } else {
        ++no_improve_;
    }

    if (no_improve_ >= rejuv_trigger_) {
        rejuvenate();
    } else {
        // Self-gated: only actually acts once the population has collapsed
        // AND no_improve_ has reached obl_trigger_; safe to call every
        // iteration otherwise (immediate no-op).
        oblBasinEscape();
        if (no_improve_ >= polish_trigger_) {
            elitePolish();
        }
    }

    // Optional in-run local search after a successful global-best improvement
    if (local_rate_ > 0.0 && !local_method_.empty()) {
        if (U01(rng_) < local_rate_) {
            auto [xloc, floc] = localSearch(local_method_, best_x_);
            if (floc < best_f_) {
                best_f_ = floc;
                best_x_ = xloc;
                Xbest_  = best_x_;
                Fbest_  = best_f_;

                if (!X_.empty()) {
                    int wi = 0; double wv = FX_[0];
                    for (int i = 1; i < n; ++i) if (FX_[i] > wv) { wv = FX_[i]; wi = i; }
                    X_[wi] = best_x_;
                    FX_[wi] = best_f_;
                }
            }
        }
    }

    printBest();
    updateStop(FX_);
}

void EMSCSO::end() {
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

    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (worst_idx < X_.size()) {
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }
    printBest();
}

} // namespace optimsolution
