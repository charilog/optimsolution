#include "emscso.h"
#include "init.h"
#include <cmath>
#include <limits>
#include <cstdio>

namespace optimsolution {

namespace { constexpr double EMSCSO_PI = 3.141592653589793238462643383279502884; }

double EMSCSO::randU() {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng_);
}

double EMSCSO::gaussN(double mu, double sig) {
    std::normal_distribution<double> d(mu, sig);
    return d(rng_);
}

// ============================================================================
// Good point set (unchanged from MSCSO -- see mscso.cpp for the derivation).
// ============================================================================
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

// ============================================================================
// NLPSR (ported from SPARQ::progress01 / targetPopulationSize / shrinkTo)
// ============================================================================
double EMSCSO::progress01() const {
    if (!prob_ || max_evals_ <= 0) return 1.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

int EMSCSO::targetPopulationSize() const {
    const double p    = progress01();
    const double expo = 1.0 - (1.0 - nlpsr_alpha_) * p;
    const double frac = std::pow(p, expo);
    double N = (double)Ninit_ + ((double)Nmin_ - (double)Ninit_) * frac;
    int Ni = (int)std::round(N);
    if (Ni < Nmin_)  Ni = Nmin_;
    if (Ni > Ninit_) Ni = Ninit_;
    return Ni;
}

void EMSCSO::shrinkTo(int Ntarget) {
    int N = (int)X_.size();
    if (Ntarget >= N) return;
    if (Ntarget < Nmin_) Ntarget = Nmin_;

    std::vector<int> ord(N);
    for (int i = 0; i < N; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec>    nX(Ntarget);
    std::vector<double> nF(Ntarget);
    for (int i = 0; i < Ntarget; ++i) {
        int k = ord[i];
        nX[i] = std::move(X_[k]);
        nF[i] = FX_[k];
    }
    X_.swap(nX);
    FX_.swap(nF);
    this->setPopulation(Ntarget);
}

// ============================================================================
// Quarantine (ported from ARQ::quantile / quarantine_and_restart; the RTR
// selection half of ARQ has been removed from EMSCSO -- see header).
// ============================================================================
double EMSCSO::quantile(std::vector<double> v, double q01) {
    if (v.empty()) return std::numeric_limits<double>::infinity();
    if (q01 < 0.0) q01 = 0.0;
    if (q01 > 1.0) q01 = 1.0;
    const double pos = q01 * (double)(v.size() - 1);
    const size_t k = (size_t)std::floor(pos);
    const double frac = pos - (double)k;

    std::nth_element(v.begin(), v.begin() + k, v.end());
    double a = v[k];
    if (k + 1 >= v.size()) return a;
    std::nth_element(v.begin(), v.begin() + (k + 1), v.end());
    double b = v[k + 1];
    return a + frac * (b - a);
}

int EMSCSO::worstIndex() const {
    int wi = 0;
    double wv = FX_[0];
    for (int i = 1; i < (int)FX_.size(); ++i) {
        if (FX_[i] > wv) { wv = FX_[i]; wi = i; }
    }
    return wi;
}

void EMSCSO::quarantineAndRestart() {
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N < 4) return;
    const int D = prob_->dimension();

    std::vector<int> ord(N);
    for (int i = 0; i < N; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

    // --- quarantine: IQR outlier detection + reseed around robust center ---
    double Q1 = quantile(FX_, 0.25);
    double Q3 = quantile(FX_, 0.75);
    double IQR = Q3 - Q1;
    double theta = Q3 + outlier_alpha_ * IQR;

    const int half = std::max(1, N / 2);
    Vec center(D, 0.0);
    for (int k = 0; k < half; ++k) {
        const Vec& x = X_[ord[k]];
        for (int j = 0; j < D; ++j) center[j] += x[j];
    }
    for (int j = 0; j < D; ++j) center[j] /= (double)half;

    std::vector<int> out;
    for (int i = 0; i < N; ++i) if (FX_[i] >= theta) out.push_back(i);

    if (!out.empty()) {
        int k = (int)std::floor(outlier_rho_ * (double)out.size());
        if (k > 0) {
            std::shuffle(out.begin(), out.end(), rng_);
            out.resize(k);

            const auto& L = prob_->lb();
            const auto& U = prob_->ub();

            for (int idx : out) {
                if (prob_->calls() >= max_evals_) break;

                Vec cand = center;
                for (int j = 0; j < D; ++j) {
                    double lo = (j < (int)L.size() ? L[j] : -1.0);
                    double hi = (j < (int)U.size() ? U[j] :  1.0);
                    if (lo > hi) std::swap(lo, hi);
                    double scale = qsigma_ * (hi - lo);
                    cand[j] += gaussN(0.0, scale);
                }
                ensureBounds(cand);
                double fc = eval(cand);

                if (fc < FX_[idx]) {
                    X_[idx]  = std::move(cand);
                    FX_[idx] = fc;
                    if (fc < Fbest_) { Fbest_ = fc; Xbest_ = X_[idx]; }
                }
            }
        }
    }

    // --- hard-stagnation micro-restart, only once no_improve_ trips it ---
    if (no_improve_ < stagnation_trigger_) return;

    ord.resize(N);
    for (int i = 0; i < N; ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

    int wcount = std::max(1, (int)std::floor(worst_frac_ * (double)N));
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int t = 0; t < wcount; ++t) {
        if (prob_->calls() >= max_evals_) break;
        int idx = ord[N - 1 - t]; // worst

        Vec cand = best_x_;
        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            double scale = rsigma_ * (hi - lo);
            cand[j] += gaussN(0.0, scale);
        }
        ensureBounds(cand);
        double fc = eval(cand);

        if (fc < FX_[idx]) {
            X_[idx]  = std::move(cand);
            FX_[idx] = fc;
            if (fc < Fbest_) { Fbest_ = fc; Xbest_ = X_[idx]; }
        }
    }

    no_improve_ = 0; // reset after restart action
}

// ============================================================================
// Elite Polish (NEW, replaces RTR): stagnation-gated multi-probe local
// intensification of best_x_ ONLY. Strictly elitist -- can only pull Fbest_
// down, never move it -- and self-throttling, so it cannot spend budget the
// main loop needed to keep the run's own average outcome healthy.
// ============================================================================
void EMSCSO::elitePolish() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 1 || D < 1) return;

    // Hard cumulative cap: polish may never have consumed more than
    // polish_budget_ * (evals spent so far) across the WHOLE run.
    const long long callsSoFar = prob_->calls();
    const long long cap = (long long)std::floor(polish_budget_ * (double)std::max<long long>(1, callsSoFar));
    if (polish_used_ >= cap) {
        polish_cooldown_ = std::max(polish_cooldown_, polish_trigger_ * (1 << std::min(polish_backoff_ + 1, 8)));
        return;
    }

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    const int nProbes = std::max(2, (int)std::round(polish_frac_ * (double)N));
    const int half     = nProbes / 2;

    const double f0 = best_f_;
    const long long calls0 = prob_->calls();

    // (a) single-coordinate adaptive Gaussian steps, round-robin over D
    for (int t = 0; t < half; ++t) {
        if (prob_->calls() >= max_evals_) break;
        int j = polish_coord_ptr_ % D;
        polish_coord_ptr_ = (polish_coord_ptr_ + 1) % D;

        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);

        Vec cand = best_x_;
        cand[j] += gaussN(0.0, ps_sigma_c_ * (hi - lo));
        ensureBounds(cand);
        double fc = eval(cand);

        if (fc < best_f_) {
            best_f_ = fc; best_x_ = cand; Xbest_ = cand; Fbest_ = fc;
            ps_sigma_c_ = std::min(ps_sigma_max_, ps_sigma_c_ * 1.5);
        } else {
            ps_sigma_c_ = std::max(ps_sigma_min_, ps_sigma_c_ * 0.87);
        }
    }

    // (b) full-D isotropic adaptive Gaussian steps
    for (int t = half; t < nProbes; ++t) {
        if (prob_->calls() >= max_evals_) break;

        Vec cand = best_x_;
        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            cand[j] += gaussN(0.0, ps_sigma_ * (hi - lo));
        }
        ensureBounds(cand);
        double fc = eval(cand);

        if (fc < best_f_) {
            best_f_ = fc; best_x_ = cand; Xbest_ = cand; Fbest_ = fc;
            ps_sigma_ = std::min(ps_sigma_max_, ps_sigma_ * 1.5);
        } else {
            ps_sigma_ = std::max(ps_sigma_min_, ps_sigma_ * 0.87);
        }
    }

    const long long used = prob_->calls() - calls0;
    polish_used_ += used;

    const bool improved = best_f_ < f0 - 1e-18;
    if (improved) {
        // Propagate the refinement into the swarm without disturbing the
        // rest of the population: only the current worst slot is touched.
        int wi = worstIndex();
        X_[wi] = best_x_;
        FX_[wi] = best_f_;
    }

    // Efficiency self-check: back off exponentially once polish is measured
    // to be a poor investment on this landscape, rather than keep skimming
    // budget from the main loop every time it happens to be eligible.
    const double gain    = std::max(0.0, f0 - best_f_);
    const double relgain = gain / std::max(1.0, std::fabs(f0));
    if (relgain < polish_min_relgain_) {
        ++polish_backoff_;
        polish_cooldown_ = polish_trigger_ * (1 << std::min(polish_backoff_, 6));
    } else {
        polish_backoff_ = 0;
        polish_cooldown_ = 0;
    }
}

// ============================================================================
// Best-Anchored Levy Jump (NEW, replaces RTR): a single heavy-tailed probe
// launched from best_x_ itself. Ramped in with stagnation so it is nearly
// free while a run is converging normally; accepted only if it beats
// Fbest_, so a miss costs one evaluation and touches nothing else.
// ============================================================================
double EMSCSO::sampleLevy() {
    constexpr double kPi = 3.14159265358979323846;
    const double beta = levy_beta_;
    const double num = std::tgamma(1.0 + beta) * std::sin(kPi * beta / 2.0);
    const double den = std::tgamma((1.0 + beta) / 2.0) * beta
                       * std::pow(2.0, (beta - 1.0) / 2.0);
    const double sigma_u = std::pow(num / den, 1.0 / beta);
    std::normal_distribution<double> Nu(0.0, sigma_u);
    std::normal_distribution<double> Nv(0.0, 1.0);
    double u = Nu(rng_);
    double v = Nv(rng_);
    double step = u / std::pow(std::fabs(v) + 1e-12, 1.0 / beta);
    if (step >  50.0) step =  50.0;
    if (step < -50.0) step = -50.0;
    return step;
}

void EMSCSO::bestLevyJump() {
    if (!prob_) return;
    if (levy_prob_ <= 0.0) return;

    const double rampFrac = (levy_ramp_ > 0)
        ? std::min(1.0, (double)no_improve_ / (double)levy_ramp_)
        : 1.0;
    const double p = levy_prob_ * rampFrac;
    if (p <= 0.0 || randU() >= p) return; // usual case: skip, zero cost

    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    Vec cand = best_x_;
    for (int j = 0; j < D; ++j) {
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        cand[j] += levy_scale_ * sampleLevy() * (hi - lo);
    }
    ensureBounds(cand);
    double fc = eval(cand);

    if (fc < Fbest_) {
        Fbest_ = fc; Xbest_ = cand;
        best_f_ = fc; best_x_ = cand;
        int wi = worstIndex();
        X_[wi] = cand;
        FX_[wi] = fc;
    }
    // else: discarded -- exactly one evaluation spent, nothing else touched.
}

// ============================================================================
// init
// ============================================================================
void EMSCSO::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    Ninit_ = (popscale_ > 0) ? std::max(4, popscale_ * D) : pop_;
    if (Ninit_ < Nmin_) Ninit_ = Nmin_;
    pop_ = Ninit_;
    this->setPopulation(Ninit_);

    X_.assign(Ninit_, std::vector<double>(D, 0.0));
    FX_.assign(Ninit_, std::numeric_limits<double>::infinity());
    Xbest_.assign(D, 0.0);
    Fbest_ = std::numeric_limits<double>::infinity();

    std::vector<std::vector<double>> F;
    goodPointSet(Ninit_, D, F);

    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    for (int i = 0; i < Ninit_; ++i) {
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
    for (int i = 1; i < Ninit_; ++i) {
        if (FX_[i] < Fbest_) { Fbest_ = FX_[i]; Xbest_ = X_[i]; }
    }

    best_x_ = Xbest_;
    best_f_ = Fbest_;

    best_prev_  = Fbest_;
    no_improve_ = 0;

    ps_sigma_        = 0.02;
    ps_sigma_c_      = 0.10;
    polish_cooldown_ = 0;
    polish_backoff_  = 0;
    polish_used_     = 0;
    polish_coord_ptr_= 0;
}

void EMSCSO::ensureBounds(Vec& x) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = (int)x.size();
    for (int j = 0; j < D; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

// ============================================================================
// one_iteration
// ============================================================================
void EMSCSO::one_iteration() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int n = (int)X_.size(); // may have shrunk via NLPSR

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> Uang(1.0, 360.0);
    std::uniform_real_distribution<double> Uk(-1.0, 1.0);
    std::normal_distribution<double>       Nstd(0.0, 1.0);

    const double t    = (double)iters_;
    const double Tmax = (double)std::max(1, max_iters_);

    // --- Nonlinear adjustment strategy (Eqs. 13-14) ---
    const double f_nl = std::sin((EMSCSO_PI / 4.0) * (t / Tmax));
    const double rg   = SM_ - (SM_ * t / Tmax) * f_nl;

    // --- R, r for this iteration (Eqs. 4-5) ---
    const double R      = 2.0 * rg * U01(rng_) - rg;
    const double r_sens = rg - U01(rng_);

    // --- Main SCSO position update -- plain MSCSO behaviour (RTR removed):
    //     unconditional replacement, exactly as in the base paper. ---
    for (int i = 0; i < n; ++i) {
        const double theta_rad = Uang(rng_) * (EMSCSO_PI / 180.0);
        const double c_theta   = std::cos(theta_rad);

        Vec xnew(D);
        if (std::fabs(R) <= 1.0) {
            // Attack / exploitation (Eqs. 7-8)
            for (int j = 0; j < D; ++j) {
                const double xrnd = std::fabs(U01(rng_) * (Xbest_[j] - X_[i][j]));
                xnew[j] = Xbest_[j] - r_sens * xrnd * c_theta;
            }
        } else {
            // Search / exploration (Eq. 6, corrected form -- see mscso.h)
            for (int j = 0; j < D; ++j) {
                xnew[j] = r_sens * (Xbest_[j] - U01(rng_) * X_[i][j]);
            }
        }

        ensureBounds(xnew);
        const double fnew = eval(xnew);

        X_[i]  = std::move(xnew);
        FX_[i] = fnew;
        if (fnew < Fbest_) { Fbest_ = fnew; Xbest_ = X_[i]; }

        if (prob_->calls() >= max_evals_) break;
    }

    // Worst individual snapshot for this iteration, used by the warning
    // mechanism below.
    int worst_idx = 0;
    double fworst = FX_[0];
    for (int i = 1; i < n; ++i) {
        if (FX_[i] > fworst) { fworst = FX_[i]; worst_idx = i; }
    }

    // --- Sparrow early-warning mechanism (Eq. 15), unchanged from MSCSO ---
    const int nwarn = std::max(1, (int)std::lround(warning_frac_ * n));
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng_);

    for (int w = 0; w < nwarn; ++w) {
        const int i = idx[w];
        Vec xcand(D);

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
                    int wi = worstIndex();
                    X_[wi] = best_x_;
                    FX_[wi] = best_f_;
                }
            }
        }
    }

    // --- stagnation bookkeeping (drives Elite Polish, Levy Jump ramp, and
    //     the quarantine micro-restart trigger) ---
    if (best_f_ < best_prev_ - 1e-18) {
        best_prev_ = best_f_;
        no_improve_ = 0;
    } else {
        ++no_improve_;
    }

    // --- Elite Polish (NEW, replaces RTR) ---
    if (enable_polish_) {
        if (polish_cooldown_ > 0) {
            --polish_cooldown_;
        } else if (no_improve_ >= polish_trigger_) {
            elitePolish();
        }
    }

    // --- Best-Anchored Levy Jump (NEW, replaces RTR) ---
    if (enable_levy_jump_) {
        bestLevyJump();
    }

    // --- Quarantine maintenance (IQR reseed + stagnation micro-restart) ---
    if (enable_quarantine_) quarantineAndRestart();

    // --- NLPSR shrink (end of iteration, after all indices above are used) ---
    if (enable_nlpsr_) {
        int Ntarget = targetPopulationSize();
        if (Ntarget < (int)X_.size()) {
            shrinkTo(Ntarget);
        }
    }

    printBest();
    updateStop(FX_);
}

void EMSCSO::end() {
    // Executed at the end. Controlled ONLY by [global]. (same pattern as MSCSO)
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
        int wi = worstIndex();
        X_[wi]  = best_x_;
        FX_[wi] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
