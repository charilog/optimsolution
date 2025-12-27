#include "aarq.h"

#include <cmath>
#include <chrono>
#include <cctype>
#include <cstdio>

namespace optimsolution {

// ========================= HELPER FUNCTIONS =========================

static inline uint64_t splitmix64_aarq(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline std::string trim_aarq(std::string s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

static inline std::string to_lower_aarq(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

static inline int parse_int_str_aarq(const std::string& s, int fallback) {
    std::string t = trim_aarq(s);
    if (t.empty()) return fallback;
    try {
        size_t pos = 0;
        long v = std::stol(t, &pos);
        if (pos == t.size()) return (int)v;
    } catch (...) {}
    return fallback;
}

static inline double parse_double_str_aarq(const std::string& s, double fallback) {
    std::string t = trim_aarq(s);
    if (t.empty()) return fallback;
    try {
        size_t pos = 0;
        double v = std::stod(t, &pos);
        if (pos == t.size() && std::isfinite(v)) return v;
    } catch (...) {}
    return fallback;
}

// ========================= CONFIGURE =========================

void AARQ::configure(const MethodConfig& mc) {
    // population override (multiple aliases)
    int p = mc.getInt("population",
            mc.getInt("Population",
            mc.getInt("pop",
            mc.getInt("Pop", -1))));
    if (p < 0) p = parse_int_str_aarq(mc.getStr("population",""), -1);
    if (p < 0) p = parse_int_str_aarq(mc.getStr("Population",""), -1);
    if (p < 0) p = parse_int_str_aarq(mc.getStr("pop",""), -1);
    if (p < 0) p = parse_int_str_aarq(mc.getStr("Pop",""), -1);
    if (p >= 4) {
        pop_override_ = p;
        this->setPopulation(pop_override_);
    }

    agent_fraction_ = std::clamp(mc.getDbl("agent_fraction", agent_fraction_), 0.05, 1.0);

    // Success-history
    muF_   = mc.getDbl("muF_init",  muF_);
    muCR_  = mc.getDbl("muCR_init", muCR_);
    sh_c_  = mc.getDbl("sh_c",      sh_c_);
    F_lo_  = mc.getDbl("F_lo",      F_lo_);
    F_hi_  = mc.getDbl("F_hi",      F_hi_);
    CR_lo_ = mc.getDbl("CR_lo",     CR_lo_);
    CR_hi_ = mc.getDbl("CR_hi",     CR_hi_);

    // p-best & archive
    pbest_frac_   = std::clamp(mc.getDbl("pbest_frac", pbest_frac_), 0.01, 0.9);
    archive_rate_ = mc.getDbl("archive_rate", archive_rate_);

    // RTR
    rtr_pool_             = mc.getInt("rtr_pool", rtr_pool_);
    rtr_min_replace_gain_ = mc.getDbl("rtr_min_replace_gain", rtr_min_replace_gain_);

    // Stagnation & restart
    stagnation_trigger_ = mc.getInt("stagnation_trigger", stagnation_trigger_);
    if (stagnation_trigger_ < 5) stagnation_trigger_ = 5;
    restart_frac_       = mc.getDbl("restart_frac",       restart_frac_);
    restart_sigma_      = mc.getDbl("restart_sigma",      restart_sigma_);

    // In-run local search
    {
        std::string lm = mc.getStr("local_method",
                          mc.getStr("local.method",
                          mc.getStr("inrun_local", local_method_)));
        lm = to_lower_aarq(trim_aarq(lm));

        std::string lr_str = mc.getStr("local_rate",
                               mc.getStr("local.rate",
                               mc.getStr("inrun_rate", std::to_string(local_rate_))));
        double lr = parse_double_str_aarq(lr_str, local_rate_);
        if (lr < 0.0) lr = 0.0;
        if (lr > 1.0) lr = 1.0;

        if (lm == "none" || lm == "off" || lm == "0") {
            local_method_.clear();
            local_rate_ = 0.0;
        } else {
            local_method_ = lm;
            local_rate_   = lr;
        }
    }

    // Seeding
    user_seed_   = static_cast<uint64_t>(mc.getInt("seed", static_cast<int>(user_seed_)));
    run_id_hint_ = mc.getInt("run_id", run_id_hint_);
}

// ========================= BASIC HELPERS =========================

void AARQ::ensureBounds(Vec& v) {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < v.size(); ++j) {
        if (!std::isfinite(v[j])) v[j] = 0.5 * (L[j] + U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

int AARQ::eliteIndexFinite() const {
    int    best_i = 0;
    double best_f = std::numeric_limits<double>::infinity();
    for (int i = 0; i < (int)FX_.size(); ++i) {
        double f = FX_[i];
        if (std::isfinite(f) && f < best_f) {
            best_f = f;
            best_i = i;
        }
    }
    return best_i;
}

int AARQ::pickDistinct(int n, int a, int b, int c) {
    std::uniform_int_distribution<int> I(0, n - 1);
    int r;
    do {
        r = I(rng_);
    } while (r == a || r == b || r == c);
    return r;
}

void AARQ::pushArchive_(const Vec& x) {
    if (archive_rate_ <= 0.0) return;
    archive_.push_back(x);
    const size_t cap = (size_t)std::max(0.0, std::round(archive_rate_ * (double)population()));
    if (cap > 0 && archive_.size() > cap) {
        std::uniform_int_distribution<size_t> A(0, archive_.size() - 1);
        size_t pos = A(rng_);
        archive_.erase(archive_.begin() + (long)pos);
    }
}

int AARQ::pickArchiveIndex_() {
    if (archive_.empty()) return -1;
    std::uniform_int_distribution<int> A(0, (int)archive_.size() - 1);
    return A(rng_);
}

int AARQ::pickPbestIndex_() {
    const int N = population();
    const int P = std::max(1, (int)std::round(pbest_frac_ * (double)N));

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + P, idx.end(),
        [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> J(0, P - 1);
    return idx[J(rng_)];
}

double AARQ::bnDistance_(const Vec& a, const Vec& b) const {
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    double sum = 0.0;
    const double eps = 1e-12;
    for (size_t j = 0; j < a.size(); ++j) {
        double r = U[j] - L[j];
        double d = (a[j] - b[j]) / (r + eps);
        sum += d * d;
    }
    return std::sqrt(sum);
}

int AARQ::pickRTRNeighbor_(const Vec& trial, const std::vector<int>& pool) const {
    int    best   = pool[0];
    double best_d = bnDistance_(trial, X_[best]);
    for (size_t k = 1; k < pool.size(); ++k) {
        int    i = pool[k];
        double d = bnDistance_(trial, X_[i]);
        if (d < best_d) {
            best_d = d;
            best   = i;
        }
    }
    return best;
}

// ========================= DE OPERATOR =========================

void AARQ::trial_pbest1A_bin_(int i, const Vec& xi, Vec& tr, double F, double CR) {
    const int D = prob_->dimension();
    const int N = population();

    const int pbest = pickPbestIndex_();
    const int r1    = pickDistinct(N, i, pbest);

    bool useArch = (!archive_.empty() && U01_(rng_) < 0.5);
    Vec  base_r2;
    if (useArch) {
        int ia = pickArchiveIndex_();
        base_r2 = archive_[ia];
    } else {
        int r2 = pickDistinct(N, i, pbest, r1);
        base_r2 = X_[r2];
    }

    tr = xi;
    std::uniform_int_distribution<int> J(0, D - 1);
    int jrand = J(rng_);
    for (int j = 0; j < D; ++j) {
        if (U01_(rng_) < CR || j == jrand) {
            double v = xi[j]
                     + F * (X_[pbest][j] - xi[j])
                     + F * (X_[r1][j]    - base_r2[j]);
            tr[j] = v;
        }
    }
    ensureBounds(tr);
}

// ========================= MICRO-RESTART (ALWAYS, EXTREMELY AGGRESSIVE) =========================

void AARQ::microRestart_() {
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N <= 1) return;

    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    int elite = idx[0];

    std::uniform_real_distribution<double> UR(0.0, 1.0);
    std::normal_distribution<double>       N0(0.0, 1.0);

    // baseline restart rate per iteration (very aggressive)
    double baseFrac = 0.15;          // always 15% of the worst
    double maxFrac  = restart_frac_; // e.g., 45% when stagnation is high
    if (maxFrac < baseFrac) maxFrac = baseFrac;
    if (maxFrac > 0.80)     maxFrac = 0.80;

    // extra due to stagnation (as stagn_iters_ increases, the rate approaches maxFrac)
    double extra = 0.0;
    if (stagnation_trigger_ > 0) {
        extra = (double)stagn_iters_ / (double)stagnation_trigger_;
        if (extra > 2.0) extra = 2.0; // cap
    }

    double frac = baseFrac + (maxFrac - baseFrac) * std::min(extra, 1.0);
    if (frac > maxFrac) frac = maxFrac;
    if (frac < baseFrac) frac = baseFrac;

    int nreset = std::max(1, (int)std::round(frac * (double)N));
    if (nreset >= N) nreset = N - 1;

    int used = 0;
    for (int k = 0; k < nreset && used < nreset; ++k) {
        int i = idx[N - 1 - k]; // from worst to best
        if (i == elite) continue;
        ++used;

        Vec cand(D);

        // 60% near-best, 40% pure global restart
        bool doNearBest = (std::isfinite(best_f_) && U01_(rng_) < 0.60);

        if (!doNearBest) {
            // global restart
            for (int j = 0; j < D; ++j) {
                double v = L[j] + (U[j] - L[j]) * UR(rng_);
                // adds a small random jitter
                v += 0.05 * (U[j] - L[j]) * N0(rng_);
                cand[j] = clamp_(v, L[j], U[j]);
            }
        } else {
            // restart around best_x_
            for (int j = 0; j < D; ++j) {
                double step = restart_sigma_ * (U[j] - L[j]) * N0(rng_);
                double v    = best_x_[j] + step;
                cand[j] = clamp_(v, L[j], U[j]);
            }
        }

        ensureBounds(cand);
        double f = eval(cand);
        X_[i]  = std::move(cand);
        FX_[i] = f;
        if (f < best_f_) {
            best_f_ = f;
            best_x_ = X_[i];
        }

        if (prob_->calls() >= max_evals_) break;
    }
}

// ========================= INIT / ITERATION / END =========================

void AARQ::init() {
    if (!prob_) return;

    int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    uint64_t now  = (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::random_device rd;
    uint64_t entropy = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    uint64_t mix = now ^ entropy ^ (uint64_t)(uintptr_t)this
                 ^ (runs_started_ * 0x9e3779b97f4a7c15ULL)
                 ^ (uint64_t)prob_->calls();

    if (user_seed_ != 0 || run_id_hint_ >= 0) {
        uint64_t base = (user_seed_ ? user_seed_ : entropy)
                      ^ (uint64_t)(run_id_hint_ < 0 ? 0 : run_id_hint_);
        seed_used_ = splitmix64_aarq(base ^ runs_started_);
    } else {
        seed_used_ = splitmix64_aarq(mix);
    }
    rng_.seed(seed_used_);
    runs_started_++;

    stagn_iters_ = 0;
    start_agent_ = 0;

    const int D = prob_->dimension();
    X_.clear();
    FX_.clear();
    archive_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < N; ++i) {
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    printBest();
    updateStop(FX_);
}

void AARQ::one_iteration() {
    if (!prob_) return;

    const int D = prob_->dimension();
    int N = (int)X_.size();
    if (D <= 0 || N <= 0) {
        updateStop(FX_);
        return;
    }

    // sort by fitness (ascending)
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        double fa = FX_[a], fb = FX_[b];
        bool A = std::isfinite(fa), B = std::isfinite(fb);
        if (A && B) return fa < fb;
        if (A && !B) return true;
        if (!A && B) return false;
        return a < b;
    });

    int elite = idx[0];
    double prevBest = best_f_;
    if (std::isfinite(FX_[elite]) && FX_[elite] < best_f_) {
        best_f_ = FX_[elite];
        best_x_ = X_[elite];
    }

    // number of individuals to update (here: effectively the entire population)
    int batch = std::max(1, (int)std::floor(agent_fraction_ * (double)N));
    if (batch > N) batch = N;

    int s = 0;
    int e = batch;

    std::normal_distribution<double> N0(0.0, 1.0);
    const double PI = 3.14159265358979323846;

    std::vector<double> sF;
    std::vector<double> sCR;
    std::vector<double> wGain;
    sF.reserve(batch);
    sCR.reserve(batch);
    wGain.reserve(batch);

    for (int t = 0; t < batch; ++t) {
        int i = idx[s + t];
        if (i == elite) continue;

        // sample F (Cauchy around muF) and CR (Normal around muCR)
        double F = muF_ + 0.1 * std::tan(PI * (U01_(rng_) - 0.5));
        if (!std::isfinite(F) || F <= 0.0) F = muF_;
        F = std::clamp(F, F_lo_, F_hi_);

        double CR = muCR_ + 0.1 * N0(rng_);
        if (!std::isfinite(CR)) CR = muCR_;
        CR = std::clamp(CR, CR_lo_, CR_hi_);

        Vec xi = X_[i];
        ensureBounds(xi);
        double fi = FX_[i];
        if (!std::isfinite(fi)) {
            fi = eval(xi);
            FX_[i] = fi;
        }

        Vec trial;
        trial_pbest1A_bin_(i, xi, trial, F, CR);
        double ft = eval(trial);

        bool   replaced   = false;
        int    target_idx = i;
        double parent_f   = fi;

        // direct replacement
        if (ft + 1e-12 < fi) {
            pushArchive_(X_[i]);
            X_[i]  = trial;
            FX_[i] = ft;
            replaced   = true;
            target_idx = i;
            parent_f   = fi;
        } else {
            // RTR
            std::vector<int> pool;
            pool.reserve(std::max(1, rtr_pool_));
            for (int kk = 0; kk < rtr_pool_; ++kk) {
                int r;
                do {
                    r = std::uniform_int_distribution<int>(0, N - 1)(rng_);
                } while (r == i);
                pool.push_back(r);
            }
            int jstar = pickRTRNeighbor_(trial, pool);

            double fj = FX_[jstar];
            if (!std::isfinite(fj)) {
                fj = eval(X_[jstar]);
                FX_[jstar] = fj;
            }

            double denom   = std::abs(fj) + 1e-12;
            double relGain = (fj - ft) / denom;

            if (ft < fj && relGain >= rtr_min_replace_gain_) {
                pushArchive_(X_[jstar]);
                X_[jstar]  = trial;
                FX_[jstar] = ft;
                replaced   = true;
                target_idx = jstar;
                parent_f   = fj;
            }
        }

        if (replaced) {
            // optional in-run LS
            if (local_rate_ > 0.0 && !local_method_.empty() && U01_(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, X_[target_idx]);
                if (floc < FX_[target_idx]) {
                    X_[target_idx]  = xloc;
                    FX_[target_idx] = floc;
                }
            }

            if (FX_[target_idx] < best_f_) {
                best_f_ = FX_[target_idx];
                best_x_ = X_[target_idx];
            }

            double gain = std::max(1e-12, parent_f - FX_[target_idx]);
            sF.push_back(F);
            sCR.push_back(CR);
            wGain.push_back(gain);
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // success-history update
    if (!wGain.empty()) {
        double sumw = 0.0;
        for (double w : wGain) sumw += w;
        if (sumw < 1e-12) sumw = 1e-12;

        double wmCR = 0.0, wmF = 0.0, wmF2 = 0.0;
        for (size_t k = 0; k < wGain.size(); ++k) {
            double wk = wGain[k] / sumw;
            wmCR += wk * sCR[k];
            wmF  += wk * sF[k];
            wmF2 += wk * sF[k] * sF[k];
        }
        double LmeanF = (wmF > 1e-12) ? (wmF2 / wmF) : wmF;

        muCR_ = (1.0 - sh_c_) * muCR_ + sh_c_ * std::clamp(wmCR,  CR_lo_, CR_hi_);
        muF_  = (1.0 - sh_c_) * muF_  + sh_c_ * std::clamp(LmeanF, F_lo_,  F_hi_);
    }

    // stagnation update
    if (best_f_ < prevBest - 1e-12) {
        stagn_iters_ = 0;
    } else {
        stagn_iters_++;
    }

    // Always performs aggressive micro-restart (rate is dynamic from stagn_iters_)
    //microRestart_();

    printBest();
    updateStop(FX_);
}

void AARQ::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;
    if (!std::isfinite(best_f_)) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    if (!X_.empty() && !FX_.empty()) {
        size_t worst = 0;
        double fw = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > fw) {
                fw = FX_[k];
                worst = k;
            }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
