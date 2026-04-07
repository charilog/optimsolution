#include "ude.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>
#include <numeric>
#include <cassert>

namespace optimsolution {

// ── Strategy labels (for debug output) ───────────────────────────────────────

static const char* kSN[UDE::NS] = {
    "DE/rand/1/bin",
    "DE/rand/2/bin",
    "DE/best/1/bin",
    "JSO/current-to-pbest-w/1/bin"
};

// ── File-local parsing helpers (mirror DE style) ──────────────────────────────

static inline std::string u_lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
static inline std::string u_trim(std::string s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b - a);
}
static bool u_bool(std::string s, bool fb) {
    s = u_lower(u_trim(s));
    if (s=="1"||s=="true"||s=="on"||s=="yes")  return true;
    if (s=="0"||s=="false"||s=="off"||s=="no") return false;
    return fb;
}
static int u_int(std::string s, int fb) {
    s = u_trim(s);
    if (s.empty()) return fb;
    try { size_t p=0; long v=std::stol(s,&p); if(p==s.size()) return (int)v; } catch(...) {}
    return fb;
}
static double u_dbl(std::string s, double fb) {
    s = u_trim(s);
    if (s.empty()) return fb;
    try { size_t p=0; double v=std::stod(s,&p); if(p==s.size()&&std::isfinite(v)) return v; } catch(...) {}
    return fb;
}

// ═══════════════════════════════════════════════════════════════════════════
//  configure
// ═══════════════════════════════════════════════════════════════════════════

void UDE::configure(const MethodConfig& mc) {

    // ── DE control parameters ─────────────────────────────────────────────
    F_  = u_dbl(mc.getStr("F",  std::to_string(F_)),  F_);
    CR_ = u_dbl(mc.getStr("CR", std::to_string(CR_)), CR_);
    F_  = std::max(0.0, std::min(2.0, F_));
    CR_ = std::max(0.0, std::min(1.0, CR_));

    // jSO p schedule upper bound pmax (with pmin = pmax / 2), capped at 0.5
    pbest_p_ = u_dbl(mc.getStr("pbest_p", std::to_string(pbest_p_)), pbest_p_);
    pbest_p_ = std::max(0.02, std::min(0.5, pbest_p_));

    jso_H_ = u_int(mc.getStr("jso_memory", std::to_string(jso_H_)), jso_H_);
    jso_H_ = std::max(2, jso_H_);

    jso_archive_rate_ = u_dbl(mc.getStr("jso_archive_rate", std::to_string(jso_archive_rate_)),
                              jso_archive_rate_);
    jso_archive_rate_ = std::max(0.25, std::min(4.0, jso_archive_rate_));

    // ── Adaptation hyper-parameters ───────────────────────────────────────
    min_prob_ = u_dbl(mc.getStr("min_prob", std::to_string(min_prob_)), min_prob_);
    min_prob_ = std::max(0.01, std::min(1.0 / NS, min_prob_));   // cap at uniform share

    adapt_window_ = u_int(mc.getStr("adapt_window", std::to_string(adapt_window_)), adapt_window_);
    adapt_window_ = std::max(1, adapt_window_);

    stagnation_limit_ = u_int(mc.getStr("stagnation_limit", std::to_string(stagnation_limit_)), stagnation_limit_);
    stagnation_limit_ = std::max(1, stagnation_limit_);

    // ── In-run local search ───────────────────────────────────────────────
    std::string lm = u_lower(u_trim(
        mc.getStr("local_method",
        mc.getStr("local.method",
        mc.getStr("inrun_local", local_method_)))));

    double lr = u_dbl(
        mc.getStr("local_rate",
        mc.getStr("local.rate",
        mc.getStr("inrun_rate", std::to_string(local_rate_)))), local_rate_);
    lr = std::max(0.0, std::min(1.0, lr));

    if (lm == "none" || lm == "off" || lm == "0") {
        local_method_.clear();
        local_rate_ = 0.0;
    } else {
        local_method_ = lm;
        local_rate_   = lr;
    }

    // ── Population override (aliases: population / Population / pop / Pop) ─
    int p = mc.getInt("population",
            mc.getInt("Population",
            mc.getInt("pop",
            mc.getInt("Pop", -1))));
    if (p < 0) p = u_int(mc.getStr("population", ""), -1);
    if (p < 0) p = u_int(mc.getStr("Population", ""), -1);
    if (p < 0) p = u_int(mc.getStr("pop",        ""), -1);
    if (p < 0) p = u_int(mc.getStr("Pop",        ""), -1);
    if (p >= 4) {
        pop_override_ = p;
        this->setPopulation(pop_override_);  // update base immediately for header printing
    }

    // ── End-of-run local search ───────────────────────────────────────────
    int flg = mc.getInt("end_local_refine",
              mc.getInt("final_local",
              mc.getInt("final.local",
              mc.getInt("end_local_refin", end_local_refine_ ? 1 : 0))));
    flg = mc.getInt("end_local_refin", flg);
    std::string flg_s = mc.getStr("end_local_refine",
                        mc.getStr("final_local",
                        mc.getStr("final.local",
                        mc.getStr("end_local_refin", std::string{}))));
    end_local_refine_ = u_bool(flg_s, flg != 0);

    std::string flm = u_lower(u_trim(
        mc.getStr("end_local_method",
        mc.getStr("final_local_method",
        mc.getStr("final.method", end_local_method_)))));
    end_local_method_ = flm;

    // ── Debug verbosity ───────────────────────────────────────────────────
    debug_ude_ = u_int(mc.getStr("debug_ude", std::to_string(debug_ude_)), debug_ude_);
}

// ═══════════════════════════════════════════════════════════════════════════
//  init
// ═══════════════════════════════════════════════════════════════════════════

void UDE::init() {
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);   // sync with base class (reporter reads this)

    X_.clear();
    FX_.clear();
    archive_.clear();
    jso_SF_.clear();
    jso_SCR_.clear();
    jso_dF_.clear();

    // Sample initial population via the framework Initializer
    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i = 0; i < N; ++i) {
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // Initialise strategy adaptation state to uniform
    strat_prob_.fill(1.0 / NS);
    success_.fill(0);
    trials_.fill(0);
    stag_count_.fill(0);
    iter_ = 0;

    // jSO success-history initialisation:
    // MF starts at 0.3, MCR at 0.8, and the last memory slot is reserved as 0.9.
    jso_MF_.assign(jso_H_, 0.3);
    jso_MCR_.assign(jso_H_, 0.8);
    if (!jso_MF_.empty()) {
        jso_MF_.back()  = 0.9;
        jso_MCR_.back() = 0.9;
    }
    jso_k_ = 0;

    if (debug_ude_) {
        const int top_k = std::max(2, (int)std::round(pbest_p_ * N));
        std::fprintf(stdout,
            "[ude] init -> N=%d (override=%d), F=%.4f, CR=%.4f\n"
            "      pbest_p=%.3f (top %d individuals for JSO)\n"
            "      adapt_window=%d, stag_limit=%d, min_prob=%.3f\n"
            "      in-run local: %s @ %.4f\n"
            "      end-run local: %s (%s)\n"
            "      initial probs: [%.3f %.3f %.3f %.3f]\n",
            N, pop_override_, F_, CR_,
            pbest_p_, top_k,
            adapt_window_, stagnation_limit_, min_prob_,
            local_method_.empty() ? "none" : local_method_.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off",
            end_local_method_.empty() ? "none" : end_local_method_.c_str(),
            strat_prob_[0], strat_prob_[1], strat_prob_[2], strat_prob_[3]);
        std::fflush(stdout);
    }

    printBest();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Low-level helpers
// ═══════════════════════════════════════════════════════════════════════════

int UDE::pickDistinct(int n, int a, int b, int c) {
    std::uniform_int_distribution<int> I(0, n - 1);
    int r;
    do { r = I(rng_); } while (r == a || r == b || r == c);
    return r;
}

void UDE::ensureBounds(Vec& v) {
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j = 0; j < v.size(); ++j) {
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (!std::isfinite(v[j])) v[j] = 0.5 * (lo + hi);
        v[j] = std::max(lo, std::min(hi, v[j]));
    }
}

UDE::Vec UDE::crossover(const Vec& target, const Vec& donor, double CRuse) {
    const int D = (int)target.size();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Jrand(0, std::max(0, D - 1));
    const int jr = Jrand(rng_);
    Vec u = target;
    const double cr = std::max(0.0, std::min(1.0, CRuse));
    for (int j = 0; j < D; ++j)
        if (U01(rng_) < cr || j == jr) u[j] = donor[j];
    return u;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Mutation strategies
// ═══════════════════════════════════════════════════════════════════════════

// [0] DE/rand/1/bin  –  v = x_r1 + F*(x_r2 - x_r3)
// Three distinct random indices, all different from i.
UDE::Vec UDE::mutate_rand1(int i) {
    const int N = (int)X_.size();
    const int D = prob_->dimension();
    int r1 = pickDistinct(N, i);
    int r2 = pickDistinct(N, i, r1);
    int r3 = pickDistinct(N, i, r1, r2);
    Vec v(D);
    for (int j = 0; j < D; ++j)
        v[j] = X_[r1][j] + F_ * (X_[r2][j] - X_[r3][j]);
    return v;
}

// [1] DE/rand/2/bin  –  v = x_r1 + F*(x_r2-x_r3) + F*(x_r4-x_r5)
// Requires 5 indices distinct from i → minimum population 6.
// Falls back to rand/1 when population is too small.
UDE::Vec UDE::mutate_rand2(int i) {
    const int N = (int)X_.size();
    const int D = prob_->dimension();

    if (N < 6) return mutate_rand1(i);   // graceful fallback

    // Shuffle an index pool (excluding i) and take the first five.
    std::vector<int> pool;
    pool.reserve(N - 1);
    for (int k = 0; k < N; ++k) if (k != i) pool.push_back(k);
    std::shuffle(pool.begin(), pool.end(), rng_);

    Vec v(D);
    for (int j = 0; j < D; ++j)
        v[j] = X_[pool[0]][j]
             + F_ * (X_[pool[1]][j] - X_[pool[2]][j])
             + F_ * (X_[pool[3]][j] - X_[pool[4]][j]);
    return v;
}

// [2] DE/best/1/bin  –  v = x_best + F*(x_r1 - x_r2)
// Uses the globally maintained best_x_ (always current).
UDE::Vec UDE::mutate_best1(int i) {
    const int N = (int)X_.size();
    const int D = prob_->dimension();
    int r1 = pickDistinct(N, i);
    int r2 = pickDistinct(N, i, r1);
    Vec v(D);
    for (int j = 0; j < D; ++j)
        v[j] = best_x_[j] + F_ * (X_[r1][j] - X_[r2][j]);
    return v;
}

// [3] JSO / DE/current-to-pbest-w/1/bin
//     v = x_i + Fw*(x_pbest - x_i) + F*(x_r1 - x_r2)
//
// This implementation keeps the jSO-specific machinery local to strategy [3]:
//   • success-history adaptation for F and CR,
//   • external archive of inferior parents,
//   • dynamic p schedule between pmin = pmax/2 and pmax,
//   • weighted current-to-pbest term Fw.
//
// Other UDE strategies still use the original fixed F_ / CR_ behaviour.
std::pair<double,double> UDE::sampleJSOParameters() {
    if (jso_MF_.empty() || jso_MCR_.empty()) {
        jso_MF_.assign(std::max(2, jso_H_), 0.3);
        jso_MCR_.assign(std::max(2, jso_H_), 0.8);
        jso_MF_.back()  = 0.9;
        jso_MCR_.back() = 0.9;
        jso_k_ = 0;
    }

    std::uniform_int_distribution<int> I(0, (int)jso_MF_.size() - 1);
    const int r = I(rng_);

    double muF  = jso_MF_[r];
    double muCR = jso_MCR_[r];
    if (r == (int)jso_MF_.size() - 1) {
        muF  = 0.9;
        muCR = 0.9;
    }

    // Sample F from a Cauchy distribution until it becomes positive, then clip to 1.
    std::cauchy_distribution<double> C(muF, 0.1);
    double Fj = -1.0;
    for (int tries = 0; tries < 64 && Fj <= 0.0; ++tries) Fj = C(rng_);
    if (Fj <= 0.0) Fj = muF;
    Fj = std::min(1.0, Fj);

    // Sample CR from a Gaussian distribution and clip to [0, 1].
    double CRj = 0.0;
    if (muCR < 0.0) {
        CRj = 0.0;
    } else {
        std::normal_distribution<double> N(muCR, 0.1);
        CRj = N(rng_);
        CRj = std::max(0.0, std::min(1.0, CRj));
    }

    // Early-stage safeguards from the jSO/iL-SHADE line.
    const double prog = (max_evals_ > 0)
        ? std::min(1.0, std::max(0.0, (double)prob_->calls() / (double)max_evals_))
        : 1.0;

    if (prog < 0.60 && Fj > 0.7) Fj = 0.7;
    if (prog < 0.25)      CRj = std::max(CRj, 0.7);
    else if (prog < 0.50) CRj = std::max(CRj, 0.6);

    return {Fj, CRj};
}

double UDE::weightedLehmerMean(const std::vector<double>& values,
                               const std::vector<double>& weights) const {
    if (values.empty() || weights.empty() || values.size() != weights.size()) return 0.0;

    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        const double w = std::max(0.0, weights[i]);
        const double v = std::max(0.0, values[i]);
        num += w * v * v;
        den += w * v;
    }
    if (den <= 1e-18) return 0.0;
    return num / den;
}

void UDE::pushArchive(const Vec& x) {
    const int N = (int)X_.size();
    const int maxA = std::max(1, (int)std::round(jso_archive_rate_ * std::max(1, N)));
    if ((int)archive_.size() < maxA) {
        archive_.push_back(x);
        return;
    }
    std::uniform_int_distribution<int> I(0, maxA - 1);
    archive_[I(rng_)] = x;
}

void UDE::updateJSOMemory() {
    if (jso_SF_.empty() || jso_SCR_.empty() || jso_dF_.empty()) return;
    if ((int)jso_MF_.size() < jso_H_) {
        jso_MF_.assign(jso_H_, 0.3);
        jso_MCR_.assign(jso_H_, 0.8);
        jso_MF_.back()  = 0.9;
        jso_MCR_.back() = 0.9;
        jso_k_ = 0;
    }

    const double newMF  = weightedLehmerMean(jso_SF_,  jso_dF_);
    const double newMCR = weightedLehmerMean(jso_SCR_, jso_dF_);

    if (!jso_MF_.empty()) {
        const int pos = std::max(0, std::min(jso_k_, (int)jso_MF_.size() - 1));
        if (newMF  > 0.0) jso_MF_[pos]  = std::max(1e-6, std::min(1.0, newMF));
        if (newMCR >= 0.0) jso_MCR_[pos] = std::max(0.0, std::min(1.0, newMCR));
        jso_k_ = (jso_k_ + 1) % std::max(1, jso_H_);
        jso_MF_.back()  = 0.9;
        jso_MCR_.back() = 0.9;
    }

    jso_SF_.clear();
    jso_SCR_.clear();
    jso_dF_.clear();
}

UDE::Vec UDE::mutate_jso(int i, double Fj, double pj) {
    const int N = (int)X_.size();
    const int D = prob_->dimension();

    const double p = std::max(2.0 / std::max(2, N), std::min(0.5, pj));
    const int top_k = std::max(2, (int)std::ceil(p * N));

    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
                      [this](int a, int b) { return FX_[a] < FX_[b]; });

    std::uniform_int_distribution<int> Ipbest(0, top_k - 1);
    int pbest_idx = idx[Ipbest(rng_)];

    // Select r1 from the population, distinct from i and pbest.
    std::vector<int> pop_pool;
    pop_pool.reserve(std::max(0, N - 2));
    for (int k = 0; k < N; ++k)
        if (k != i && k != pbest_idx) pop_pool.push_back(k);
    if (pop_pool.empty()) return X_[i];

    std::uniform_int_distribution<int> Ir1(0, (int)pop_pool.size() - 1);
    const int r1 = pop_pool[Ir1(rng_)];

    // Select r2 from P ∪ A, distinct from i, pbest and r1.
    std::vector<int> union_ids;
    union_ids.reserve(std::max(0, N - 3) + (int)archive_.size());
    for (int k = 0; k < N; ++k)
        if (k != i && k != pbest_idx && k != r1) union_ids.push_back(k);
    for (int a = 0; a < (int)archive_.size(); ++a)
        union_ids.push_back(N + a);

    if (union_ids.empty()) return X_[i];

    std::uniform_int_distribution<int> Ir2(0, (int)union_ids.size() - 1);
    const int r2_id = union_ids[Ir2(rng_)];

    const Vec& xr2 = (r2_id < N) ? X_[r2_id] : archive_[r2_id - N];

    const double prog = (max_evals_ > 0)
        ? std::min(1.0, std::max(0.0, (double)prob_->calls() / (double)max_evals_))
        : 1.0;
    const double Fw = (prog < 0.20) ? (0.7 * Fj)
                    : (prog < 0.40) ? (0.8 * Fj)
                                    : (1.2 * Fj);

    Vec v(D);
    for (int j = 0; j < D; ++j)
        v[j] = X_[i][j]
             + Fw * (X_[pbest_idx][j] - X_[i][j])
             + Fj * (X_[r1][j]        - xr2[j]);
    return v;
}


// ═══════════════════════════════════════════════════════════════════════════
//  Adaptive strategy selection
// ═══════════════════════════════════════════════════════════════════════════

int UDE::selectStrategy() {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r   = U(rng_);
    double cum = 0.0;
    for (int s = 0; s < NS; ++s) {
        cum += strat_prob_[s];
        if (r <= cum) return s;
    }
    return NS - 1;  // guard against floating-point rounding
}

// Enforce min_prob_ floor on every strategy and renormalise so that the
// probabilities sum to 1.  Excess probability taken from floor-clamped
// strategies is redistributed proportionally among the unclamped ones.
void UDE::normalizeProbabilities() {
    // Step 1 – basic normalisation
    double sum = 0.0;
    for (int s = 0; s < NS; ++s) sum += strat_prob_[s];
    if (sum <= 0.0) { strat_prob_.fill(1.0 / NS); return; }
    for (int s = 0; s < NS; ++s) strat_prob_[s] /= sum;

    // Step 2 – apply floor; accumulate deficit
    double deficit = 0.0;
    for (int s = 0; s < NS; ++s) {
        if (strat_prob_[s] < min_prob_) {
            deficit += min_prob_ - strat_prob_[s];
            strat_prob_[s] = min_prob_;
        }
    }

    // Step 3 – subtract deficit proportionally from unclamped strategies
    if (deficit > 1e-12) {
        double sum_above = 0.0;
        for (int s = 0; s < NS; ++s)
            if (strat_prob_[s] > min_prob_) sum_above += strat_prob_[s];
        if (sum_above > 1e-12)
            for (int s = 0; s < NS; ++s)
                if (strat_prob_[s] > min_prob_)
                    strat_prob_[s] -= deficit * (strat_prob_[s] / sum_above);
    }

    // Step 4 – final renormalise to correct any floating-point drift
    sum = 0.0;
    for (int s = 0; s < NS; ++s) sum += strat_prob_[s];
    if (sum > 0.0)
        for (int s = 0; s < NS; ++s) strat_prob_[s] /= sum;
}

// Called every adapt_window_ iterations.
// Recomputes strat_prob_ from accumulated success rates, then resets the
// window counters.  stag_count_ is NOT touched here.
void UDE::updateProbabilities() {
    double sum_rate = 0.0;
    for (int s = 0; s < NS; ++s) {
        double rate    = (trials_[s] > 0) ? (double)success_[s] / trials_[s] : 0.0;
        strat_prob_[s] = rate;
        sum_rate      += rate;
    }

    if (sum_rate <= 0.0) {
        // No strategy produced an improvement in the whole window:
        // restore uniform distribution so the search does not stall.
        strat_prob_.fill(1.0 / NS);
    }

    normalizeProbabilities();


    // Reset window accumulators; stag_count_ persists independently.
    success_.fill(0);
    trials_.fill(0);
}

// Reset a single stagnated strategy without disturbing the others.
// Its probability is set to the uniform share and the whole vector is
// renormalised; its window and stagnation counters are cleared.
void UDE::resetStrategy(int s) {
    strat_prob_[s] = 1.0 / NS;
    normalizeProbabilities();
    stag_count_[s] = 0;
    success_[s]    = 0;
    trials_[s]     = 0;

    if (s == JSO) {
        archive_.clear();
        jso_SF_.clear();
        jso_SCR_.clear();
        jso_dF_.clear();
        jso_MF_.assign(std::max(2, jso_H_), 0.3);
        jso_MCR_.assign(std::max(2, jso_H_), 0.8);
        if (!jso_MF_.empty()) {
            jso_MF_.back()  = 0.9;
            jso_MCR_.back() = 0.9;
        }
        jso_k_ = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  one_iteration
// ═══════════════════════════════════════════════════════════════════════════

void UDE::one_iteration() {
    if (!prob_) return;
    ++iter_;

    const int N = (int)X_.size();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Per-iteration flags: which strategies were applied and which improved.
    std::array<bool, NS> iter_used    {};
    std::array<bool, NS> iter_success {};

    // ── Main generational loop ─────────────────────────────────────────────
    for (int i = 0; i < N; ++i) {

        // ── Strategy selection (weighted roulette) ─────────────────────────
        const int s = selectStrategy();
        iter_used[s] = true;
        trials_[s]++;

        const Vec parent = X_[i];
        const double f_parent = FX_[i];

        // ── Mutation / parameter sampling ──────────────────────────────────
        Vec donor;
        double CRuse = CR_;
        double Fj_used = F_;
        double CRj_used = CR_;
/*
        switch (s) {
            case RAND1:
                donor = mutate_rand1(i);
                break;

            case RAND2:
                donor = mutate_rand2(i);
                break;

            case BEST1:
                donor = mutate_best1(i);
                break;

            case JSO:
            default: {*/
                const auto [Fj, CRj] = sampleJSOParameters();
                const double prog = (max_evals_ > 0)
                    ? std::min(1.0, std::max(0.0, (double)prob_->calls() / (double)max_evals_))
                    : 1.0;
                const double pmin = std::max(2.0 / std::max(2, N), 0.5 * pbest_p_);
                const double pmax = std::max(pmin, pbest_p_);
                const double pcur = pmin + prog * (pmax - pmin);
                donor = mutate_jso(i, Fj, pcur);
                Fj_used = Fj;
                CRj_used = CRj;
                CRuse = CRj;
                /*break;
            }
        }*/
		

        // ── Crossover + bound repair ───────────────────────────────────────
        Vec u = crossover(X_[i], donor, CRuse);
        ensureBounds(u);

        // ── Evaluation ────────────────────────────────────────────────────
        double fu = eval(u);

        // ── Greedy selection ──────────────────────────────────────────────
        if (fu < FX_[i]) {

            // Optional in-run local search on newly accepted trial
            if (local_rate_ > 0.0 && !local_method_.empty()
                && U01(rng_) < local_rate_)
            {
                auto [xloc, floc] = localSearch(local_method_, u);
                if (std::isfinite(floc) && floc < fu) {
                    u  = std::move(xloc);
                    fu = floc;
                }
            }

            X_[i]  = std::move(u);
            FX_[i] = fu;
            success_[s]++;
            iter_success[s] = true;

            if (s == JSO) {
                const double improvement = std::fabs(f_parent - fu);
                if (improvement > 0.0) {
                    pushArchive(parent);
                    jso_SF_.push_back(Fj_used);
                    jso_SCR_.push_back(CRj_used);
                    jso_dF_.push_back(improvement);
                }
            }

            if (FX_[i] < best_f_) {
                best_f_ = FX_[i];
                best_x_ = X_[i];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    updateJSOMemory();

    // ── Per-strategy stagnation counter update ────────────────────────────
    // Only update the counter for strategies that were actually used this
    // iteration; unused strategies are not penalised.
    for (int s = 0; s < NS; ++s) {
        if (!iter_used[s]) continue;
        if (iter_success[s])
            stag_count_[s] = 0;    // at least one improvement → reset
        else
            stag_count_[s]++;      // no improvement → accumulate
    }

    // ── Stagnation check: reset individual strategies as needed ───────────
    for (int s = 0; s < NS; ++s) {
        if (stag_count_[s] >= stagnation_limit_) {
            if (debug_ude_) {
                std::fprintf(stdout,
                    "[ude] iter=%d  STAGNATION RESET: %s  (stag=%d)\n"
                    "      probs before: [%.3f %.3f %.3f %.3f]\n",
                    iter_, kSN[s], stag_count_[s],
                    strat_prob_[0], strat_prob_[1], strat_prob_[2], strat_prob_[3]);
            }
            resetStrategy(s);
            if (debug_ude_) {
                std::fprintf(stdout,
                    "      probs after:  [%.3f %.3f %.3f %.3f]\n",
                    strat_prob_[0], strat_prob_[1], strat_prob_[2], strat_prob_[3]);
                std::fflush(stdout);
            }
        }
    }

    // ── Probability update (voting) every adapt_window_ iterations ─────────
    if (iter_ % adapt_window_ == 0) {
        if (debug_ude_) {
            // Snapshot before reset inside updateProbabilities
            std::fprintf(stdout,
                "[ude] iter=%d  PROB UPDATE  "
                "suc=[%d %d %d %d]  tri=[%d %d %d %d]\n",
                iter_,
                success_[0], success_[1], success_[2], success_[3],
                trials_[0],  trials_[1],  trials_[2],  trials_[3]);
        }
        updateProbabilities();
        if (debug_ude_) {
            std::fprintf(stdout,
                "      new probs: [%.3f %.3f %.3f %.3f]\n",
                strat_prob_[0], strat_prob_[1], strat_prob_[2], strat_prob_[3]);
            std::fflush(stdout);
        }
    }

    printBest();
    updateStop(FX_);
}

// ═══════════════════════════════════════════════════════════════════════════
//  end
// ═══════════════════════════════════════════════════════════════════════════

void UDE::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())    return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write the refined best into the worst population slot
    // (consistent with DE / GA / BHO practice)
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k)
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        if (worst_idx < X_.size()) {
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    printBest();
}

} // namespace optimsolution
