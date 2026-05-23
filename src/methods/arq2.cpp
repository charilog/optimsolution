#include "arq2.h"
#include <cstdio>
#include <cctype>

namespace optimsolution {

void ARQ2::configure(const MethodConfig& mc) {
    auto trim = [](std::string s){
        size_t a=0, b=s.size();
        while (a<b && std::isspace((unsigned char)s[a])) ++a;
        while (b>a && std::isspace((unsigned char)s[b-1])) --b;
        return s.substr(a, b-a);
    };
    auto parse_int = [&](std::string s, int fb)->int{
        s = trim(s); if (s.empty()) return fb;
        try{ size_t pos=0; long v=std::stol(s,&pos); if(pos==s.size()) return (int)v; }catch(...) {}
        return fb;
    };

    int p = mc.getInt("population", mc.getInt("Population", mc.getInt("pop", mc.getInt("Pop", -1))));
    if (p < 0) p = parse_int(mc.getStr("population", ""), -1);
    if (p >= 4) {
        pop_override_ = p;
        this->setPopulation(pop_override_);
    }

    pbest_           = mc.getDbl("p", pbest_);
    agent_fraction_  = mc.getDbl("agentfraction", agent_fraction_);
    muF_             = mc.getDbl("muF", muF_);
    muCR_            = mc.getDbl("muCR", muCR_);

    outlier_alpha_   = mc.getDbl("alpha", outlier_alpha_);
    outlier_rho_     = mc.getDbl("rho", outlier_rho_);
    qsigma_          = mc.getDbl("qsigma", qsigma_);

    worst_frac_         = mc.getDbl("w", worst_frac_);
    rsigma_             = mc.getDbl("rsigma", rsigma_);
    stagnation_trigger_ = mc.getInt("stagnationtrigger", stagnation_trigger_);

    shc_             = mc.getDbl("shc", shc_);

    Flo_             = mc.getDbl("Flo", Flo_);
    Fhi_             = mc.getDbl("Fhi", Fhi_);

    rtr_pool_        = mc.getInt("rtrpool", rtr_pool_);
    archive_rate_    = mc.getDbl("archiverate", archive_rate_);

    bootstrap_arq_iters_ = mc.getInt("bootstrap_arq_iters", bootstrap_arq_iters_);
    delta_               = mc.getDbl("delta", delta_);
    roulette_normalize_ = mc.getInt("roulette_normalize", roulette_normalize_);
    ide_progress_sync_  = mc.getInt("ide_progress_sync", ide_progress_sync_);
    ide_strict_improve_ = mc.getInt("ide_strict_improve", ide_strict_improve_);

    debug_arq_       = mc.getInt("debug_arq", debug_arq_);

    if (pbest_ < 0.01) pbest_ = 0.01;
    if (pbest_ > 0.5)  pbest_ = 0.5;
    if (agent_fraction_ <= 0.0) agent_fraction_ = 0.5;
    if (agent_fraction_ > 1.0)  agent_fraction_ = 1.0;
    if (Flo_ <= 0.0) Flo_ = 0.01;
    if (Fhi_ < Flo_) std::swap(Fhi_, Flo_);
    if (rtr_pool_ < 2) rtr_pool_ = 2;
    if (archive_rate_ <= 0.1) archive_rate_ = 1.0;
    if (worst_frac_ <= 0.0) worst_frac_ = 0.08;
    if (worst_frac_ > 1.0)  worst_frac_ = 1.0;
    if (rsigma_ <= 0.0) rsigma_ = 0.18;
    if (stagnation_trigger_ < 1) stagnation_trigger_ = 1;
    if (bootstrap_arq_iters_ < 0) bootstrap_arq_iters_ = 0;
    if (delta_ < 0.0 || delta_ >= 1.0) delta_ = 0.0;   // 0 = auto: computed in init() as 1/(5h)
    roulette_normalize_ = roulette_normalize_ ? 1 : 0;
    ide_progress_sync_  = ide_progress_sync_  ? 1 : 0;
    ide_strict_improve_ = ide_strict_improve_ ? 1 : 0;
}

void ARQ2::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear(); FX_.clear(); A_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    best_prev_ = best_f_;
    no_improve_ = 0;

    // roulette
    h_ = 2;
    n0_ = 2;
    delta_ = (delta_ > 0.0) ? delta_ : 1.0 / (5.0 * h_);
    ni_.assign(h_, static_cast<double>(n0_));
    cni_.assign(h_, 0.0);
    success_.assign(h_, 0);
    nrst_ = 0;
    bootstrap_left_ = bootstrap_arq_iters_;

    // IDE schedule from EA4Eig
    g_    = 0;
    gmax_ = std::max(1, (int)std::round((double)max_evals_ / std::max(N, 1)));
    T_    = gmax_ / 10.0;
    gt_   = std::max(1, gmax_ / 2);
    Tcurr_= 0;

    // CBF_/CBCR_ copied from EA4Eig init.
    // FIX #5 (partial): cap the F-retry loop with kMaxCauchyTries.
    CBF_.assign(N, 0.0);
    CBCR_.assign(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double F = -1.0;
        for (int tries = 0; tries < kMaxCauchyTries; ++tries) {
            F = (randU() < 0.5) ? cauchy(0.65, 0.1) : cauchy(1.0, 0.1);
            if (F >= 0.0) break;
        }
        if (F < 0.0) F = 0.65;   // safe fallback after kMaxCauchyTries
        if (F > 1.0) F = 1.0;
        CBF_[i] = F;

        double CR;
        if (randU() < 0.5)
            CR = cauchy(0.1, 0.1);
        else
            CR = cauchy(0.95, 0.1);
        if (CR > 1.0) CR = 1.0;
        if (CR < 0.0) CR = 0.0;
        CBCR_[i] = CR;
    }

    if (debug_arq_) {
        std::fprintf(stdout,
            "[arq2] cfg -> N=%d, p=%.4f, agent=%.3f, muF=%.3f, muCR=%.3f, alpha=%.3f, rho=%.3f, w=%.3f, rsigma=%.3f, stag=%d, rtrpool=%d, bootstrap=%d, normvote=%d, idsync=%d, strictIDE=%d\n",
            N, pbest_, agent_fraction_, muF_, muCR_, outlier_alpha_, outlier_rho_, worst_frac_, rsigma_, stagnation_trigger_, rtr_pool_, bootstrap_arq_iters_, roulette_normalize_, ide_progress_sync_, ide_strict_improve_);
        std::fflush(stdout);
    }

    printBest();
}

// ─────────────────────────────────────────────────────────────────────────────
//  pickDistinct
//  FIX #6: add kMaxPickTries guard — prevents the original unbounded do-while
//  from spinning long when N is small (e.g., N=4 with three exclusions leaves
//  only one valid value, expected ~4 draws per call).
//  After the limit the last drawn value is returned as-is; in degenerate cases
//  (N < 4) callers guard separately so this is a safe fallback.
// ─────────────────────────────────────────────────────────────────────────────
int ARQ2::pickDistinct(int n, int a, int b, int c){
    std::uniform_int_distribution<int> I(0, n-1);
    int r = I(rng_);
    for (int tries = 1; (r == a || r == b || r == c) && tries < kMaxPickTries; ++tries)
        r = I(rng_);
    return r;
}

int ARQ2::randInt(int lo, int hi){
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double ARQ2::randU(){
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double ARQ2::cauchy(double loc, double scale){
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

double ARQ2::progress01() const{
    if (!prob_ || max_evals_ <= 0) return 1.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

int ARQ2::ideGenerationFromProgress() const{
    if (gmax_ <= 1) return 1;
    int geff = (int)std::round(progress01() * (double)gmax_);
    if (geff < 1) geff = 1;
    if (geff > gmax_) geff = gmax_;
    return geff;
}

// ─────────────────────────────────────────────────────────────────────────────
//  resetIDEParamsAt
//  FIX #5 (partial): same bounded Cauchy-retry as init().
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::resetIDEParamsAt(int idx){
    if (idx < 0 || idx >= (int)CBF_.size() || idx >= (int)CBCR_.size()) return;

    double F = -1.0;
    for (int tries = 0; tries < kMaxCauchyTries; ++tries) {
        F = (randU() < 0.5) ? cauchy(0.65, 0.1) : cauchy(1.0, 0.1);
        if (F >= 0.0) break;
    }
    if (F < 0.0) F = 0.65;
    if (F > 1.0) F = 1.0;
    CBF_[idx] = F;

    double CR;
    if (randU() < 0.5)
        CR = cauchy(0.1, 0.1);
    else
        CR = cauchy(0.95, 0.1);
    if (CR > 1.0) CR = 1.0;
    if (CR < 0.0) CR = 0.0;
    CBCR_[idx] = CR;
}

void ARQ2::inheritIDEParams(int dst, int src){
    if (dst < 0 || src < 0) return;
    if (dst >= (int)CBF_.size() || src >= (int)CBF_.size()) return;
    if (dst >= (int)CBCR_.size() || src >= (int)CBCR_.size()) return;
    CBF_[dst]  = CBF_[src];
    CBCR_[dst] = CBCR_[src];
}

void ARQ2::sampleDistinctExcluding(int N, int k,
                                   const std::vector<int>& exclude,
                                   std::vector<int>& out)
{
    out.clear();
    out.reserve(k);
    if (N <= 0 || k <= 0) return;

    std::vector<int> candidates;
    candidates.reserve(N);
    for (int i = 0; i < N; ++i) {
        if (std::find(exclude.begin(), exclude.end(), i) == exclude.end())
            candidates.push_back(i);
    }

    if (candidates.empty()) return;
    if ((int)candidates.size() <= k) {
        out = candidates;
        return;
    }

    for (int i = 0; i < k; ++i) {
        std::uniform_int_distribution<int> dist(i, (int)candidates.size() - 1);
        int r = dist(rng_);
        std::swap(candidates[i], candidates[r]);
        out.push_back(candidates[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  sortByFitness
//  FIX #7: also permute CBF_ and CBCR_ using the same index permutation.
//
//  Original bug: X_ and FX_ were reordered by fitness rank, but CBF_ and
//  CBCR_ (per-individual IDE scaling/crossover parameters) were left in their
//  original positions.  stepIDE() calls this function TWICE (at start and end),
//  so after the first call CBF_[i] and CBCR_[i] no longer belong to X_[i]:
//  they belong to whoever occupied slot i BEFORE the sort.  Every individual
//  therefore uses the wrong F and CR for mutation and crossover throughout the
//  entire IDE generation.  After the second sort the scrambling compounds.
//
//  Fix: apply the same permutation to CBF_ and CBCR_.  We guard against size
//  mismatches (can occur transiently during population resize) with explicit
//  size checks.
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::sortByFitness()
{
    const int N = (int)X_.size();
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec>    newX(N);
    std::vector<double> newF(N);
    std::vector<double> newCBF(N,  0.0);   // FIX #7
    std::vector<double> newCBCR(N, 0.0);   // FIX #7

    const bool hasCBF  = ((int)CBF_.size()  == N);
    const bool hasCBCR = ((int)CBCR_.size() == N);

    for (int i = 0; i < N; ++i) {
        newX[i]  = std::move(X_[idx[i]]);
        newF[i]  = FX_[idx[i]];
        if (hasCBF)  newCBF[i]  = CBF_[idx[i]];   // FIX #7
        if (hasCBCR) newCBCR[i] = CBCR_[idx[i]];  // FIX #7
    }

    X_.swap(newX);
    FX_.swap(newF);
    if (hasCBF)  CBF_.swap(newCBF);    // FIX #7
    if (hasCBCR) CBCR_.swap(newCBCR);  // FIX #7
}

// ─────────────────────────────────────────────────────────────────────────────
//  ensureBounds
//  FIX #1: Replace the unbounded `while` reflection loop with a bounded `for`
//  loop (kMaxBounce iterations) followed by a hard clamp.
//
//  Original bug: for a very large overshoot (e.g., donor coordinate = lo - 1e9
//  when range = [0, 1]), the reflected value 2*lo - v is still far outside the
//  bounds, and each iteration only halves the excess if the range is ≈1.  This
//  could spin for millions of iterations (effectively hanging the algorithm).
//
//  Fix: after kMaxBounce reflections (10 is more than enough for any realistic
//  overshoot from a DE step), clamp the residual to [lo, hi].  The clamp is
//  only reached in truly extreme cases and is mathematically equivalent to a
//  boundary-midpoint repair, which is acceptable.
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::ensureBounds(Vec& v){
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);

        if (!std::isfinite(v[j]))
            v[j] = 0.5 * (lo + hi);

        if (lo == hi) {
            v[j] = lo;
            continue;
        }

        for (int b = 0; b < kMaxBounce && (v[j] < lo || v[j] > hi); ++b) {
            if      (v[j] > hi) v[j] = 2.0 * hi - v[j];
            else if (v[j] < lo) v[j] = 2.0 * lo - v[j];
        }

        // Hard clamp: handles extreme overshoots that survived kMaxBounce
        // reflections (e.g., |v| >> range width), and also covers the rare
        // case where bouncing puts us right on a boundary floating-point edge.
        if (v[j] < lo) v[j] = lo;
        if (v[j] > hi) v[j] = hi;
    }
}

double ARQ2::distBN(const Vec& a, const Vec& b) const {
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    double s = 0.0;
    for (size_t j=0; j<a.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double denom = (hi - lo);
        double z = (denom > 0.0) ? ((a[j] - b[j]) / denom) : (a[j] - b[j]);
        s += z*z;
    }
    return std::sqrt(s);
}

double ARQ2::quantile(std::vector<double> v, double q01){
    if (v.empty()) return std::numeric_limits<double>::infinity();
    if (q01 < 0.0) q01 = 0.0;
    if (q01 > 1.0) q01 = 1.0;
    const double pos = q01 * (double)(v.size()-1);
    const size_t k = (size_t)std::floor(pos);
    const double frac = pos - (double)k;

    std::nth_element(v.begin(), v.begin()+k, v.end());
    double a = v[k];
    if (k+1 >= v.size()) return a;
    std::nth_element(v.begin(), v.begin()+(k+1), v.end());
    double b = v[k+1];
    return a + frac*(b-a);
}

void ARQ2::archivePush(const Vec& x){
    A_.push_back(x);
}

void ARQ2::archiveTrim(int N){
    const int cap = std::max(1, (int)std::floor(archive_rate_ * (double)N));
    if ((int)A_.size() <= cap) return;
    std::shuffle(A_.begin(), A_.end(), rng_);
    A_.resize(cap);
}

void ARQ2::sample_F_CR(double& F, double& CR,
                       std::cauchy_distribution<double>& cauchyF,
                       std::normal_distribution<double>& normCR) {
    for (int tries=0; tries<50; ++tries){
        F = cauchyF(rng_);
        if (F > 0.0) break;
    }
    if (F <= 0.0) F = muF_;
    if (F < Flo_) F = Flo_;
    if (F > Fhi_) F = Fhi_;

    CR = normCR(rng_);
    if (CR < 0.0) CR = 0.0;
    if (CR > 1.0) CR = 1.0;
}

void ARQ2::update_mu_from_success(const std::vector<double>& SF,
                                  const std::vector<double>& SCR,
                                  const std::vector<double>& SG) {
    if (SG.empty()) return;
    double sumG = std::accumulate(SG.begin(), SG.end(), 0.0);
    if (!(sumG > 0.0)) return;

    std::vector<double> w(SG.size());
    for (size_t i=0; i<w.size(); ++i) w[i] = SG[i] / sumG;

    double num=0.0, den=0.0;
    for (size_t i=0; i<SF.size(); ++i){
        num += w[i] * SF[i] * SF[i];
        den += w[i] * SF[i];
    }
    double newMuF  = (den > 0.0) ? (num/den) : muF_;

    double newMuCR = 0.0;
    for (size_t i=0; i<SCR.size(); ++i) newMuCR += w[i] * SCR[i];

    muF_  = (1.0 - shc_) * muF_  + shc_ * newMuF;
    muCR_ = (1.0 - shc_) * muCR_ + shc_ * newMuCR;

    if (muF_ < Flo_) muF_ = Flo_;
    if (muF_ > Fhi_) muF_ = Fhi_;
    if (muCR_ < 0.0) muCR_ = 0.0;
    if (muCR_ > 1.0) muCR_ = 1.0;
}

std::pair<int,double> ARQ2::rouletteSelect() const
{
    const int h = h_;
    if (h <= 0) return {0, 0.0};

    double sumni = 0.0;
    for (int i = 0; i < h; ++i) sumni += ni_[i];

    auto* self = const_cast<ARQ2*>(this);

    if (sumni <= 0.0) {
        std::uniform_int_distribution<int> dist(0, h - 1);
        int idx = dist(self->rng_);
        double p = 1.0 / h;
        return {idx, p};
    }

    std::vector<double> p(h);
    double pmin = 1.0;
    for (int i = 0; i < h; ++i) {
        p[i] = ni_[i] / sumni;
        if (p[i] < pmin) pmin = p[i];
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(self->rng_);
    double acc = 0.0;
    for (int i = 0; i < h; ++i) {
        acc += p[i];
        if (r <= acc) return {i, pmin};
    }
    return {h - 1, pmin};
}

double ARQ2::computeK(double F) const {
    const double progress = (max_evals_ > 0)
        ? std::max(0.0, std::min(1.0, (double)prob_->calls() / (double)max_evals_))
        : 1.0;

    if (progress < 0.2) return 0.7 * F;
    if (progress < 0.4) return 0.8 * F;
    return 1.2 * F;
}

// ─────────────────────────────────────────────────────────────────────────────
//  makeTrialARQ
//  FIX #8: Ensure ipbest ≠ i  (blanket, confirmed by benchmark).
//
//  When ipbest == i the mutation degrades silently to DE/rand/1 because
//  K*(X_[i] - X_[i]) = 0, removing all exploitation pressure toward the
//  global best.  Enforcing ipbest ≠ i unconditionally is correct: even for
//  elite individuals the forced attraction to a DIFFERENT top-p% member
//  provides useful positional diversity without losing the pbest signal.
//
//  A conditional variant (allow self-selection when i is elite) was tested
//  and produced worse results on all 5 benchmark problems — it amplified
//  the bimodal behaviour on multimodal landscapes (weierstrass mean ×1.9,
//  SD ×1.9 vs fix_10).  The blanket version is kept.
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::makeTrialARQ(int i, const std::vector<int>& ord, double F, double CR, Vec& u){
    const int D = prob_->dimension();
    const int N = (int)X_.size();

    int pcount = std::max(2, (int)std::ceil(pbest_ * (double)N));
    if (pcount > N) pcount = N;
    std::uniform_int_distribution<int> Ip(0, pcount-1);

    int ipbest;
    {
        int pb_tries = 0;
        do {
            ipbest = ord[Ip(rng_)];
            ++pb_tries;
        } while (ipbest == i && pcount > 1 && pb_tries < kMaxPickTries);
    }
    const Vec& xpbest = X_[ipbest];

    int r1 = pickDistinct(N, i);
    bool useA = (!A_.empty()) && (randU() < 0.5);

    Vec r2v(D, 0.0);
    if (useA) {
        std::uniform_int_distribution<int> Ia(0, (int)A_.size()-1);
        r2v = A_[Ia(rng_)];
    } else {
        int r2 = pickDistinct(N, i, r1);
        r2v = X_[r2];
    }

    const double K = computeK(F);

    Vec v(D, 0.0);
    for (int j=0; j<D; ++j)
        v[j] = X_[i][j] + K*(xpbest[j] - X_[i][j]) + F*(X_[r1][j] - r2v[j]);

    u = X_[i];
    int jr = randInt(0, std::max(0, D-1));
    for (int j=0; j<D; ++j) {
        if (randU() < CR || j == jr)
            u[j] = v[j];
    }
    ensureBounds(u);
}

// ─────────────────────────────────────────────────────────────────────────────
//  selectionRTR
//  FIX #3: Exclude parentIndex from the RTR random pool.
//
//  Original bug: the pool was sampled uniformly from [0, N-1], so it could
//  pick parentIndex.  We already know fu >= FX_[parentIndex] (the direct
//  selection at the top of the function failed), so the RTR check
//  `fu < FX_[qstar]` trivially fails for qstar == parentIndex, wasting a pool
//  slot and reducing the effective neighbourhood size.
//
//  Fix: skip any sample that lands on parentIndex so all rtr_pool_ candidates
//  are genuinely distinct from the parent.  The while-guard terminates quickly
//  because P(q == parentIndex) = 1/N ≤ 0.25 for N ≥ 4.
// ─────────────────────────────────────────────────────────────────────────────
bool ARQ2::selectionRTR(int parentIndex, const Vec& u, double fu,
                        double F, double CR,
                        std::vector<double>& SF,
                        std::vector<double>& SCR,
                        std::vector<double>& SG) {
    if (fu < FX_[parentIndex]) {
        double gain = FX_[parentIndex] - fu;
        archivePush(X_[parentIndex]);
        X_[parentIndex] = u;
        FX_[parentIndex] = fu;

        SF.push_back(F);
        SCR.push_back(CR);
        SG.push_back(gain);
        return true;
    }

    const int N = (int)X_.size();
    int qstar = -1;
    double bestD = std::numeric_limits<double>::infinity();

    for (int k=0; k<rtr_pool_; ++k){
        // FIX #3: reject the parent index so it cannot appear in the RTR pool.
        // The parent already failed the direct comparison above; including it
        // in the pool wastes a slot and guarantees a failed RTR check.
        int q;
        int attempts = 0;
        do {
            q = randInt(0, N-1);
            ++attempts;
        } while (q == parentIndex && attempts < kMaxPickTries);
        if (q == parentIndex) continue;   // degenerate fallback (N==1, impossible in practice)

        double d = distBN(u, X_[q]);
        if (d < bestD) { bestD = d; qstar = q; }
    }
    if (qstar < 0) return false;

    if (fu < FX_[qstar]) {
        double gain = FX_[qstar] - fu;
        archivePush(X_[qstar]);
        X_[qstar] = u;
        FX_[qstar] = fu;
        inheritIDEParams(qstar, parentIndex);

        SF.push_back(F);
        SCR.push_back(CR);
        SG.push_back(gain);
        return true;
    }

    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  stepARQ
//  FIX #4: Remove the pointless per-individual distribution reconstruction.
//
//  Original bug: lines
//      cauchyF = std::cauchy_distribution<double>(muF_,  0.1);
//      normCR  = std::normal_distribution<double>(muCR_, 0.1);
//  were executed inside the agent loop on every iteration.  Since muF_ and
//  muCR_ are only updated AFTER the loop (via update_mu_from_success), these
//  assignments recreate identical distribution objects in a tight inner loop —
//  wasting CPU and resetting the distribution's internal state, which can
//  introduce subtle autocorrelation in the sample sequence depending on the
//  STL implementation.
//
//  Fix: create the distributions once before the loop and leave them in place.
//  muF_ / muCR_ are read by sample_F_CR only as a fallback when F ≤ 0.
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::stepARQ(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 4) return;

    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    int m = std::max(1, (int)std::ceil(agent_fraction_ * (double)N));
    if (m > N) m = N;

    std::vector<int> idx = ord;
    std::shuffle(idx.begin(), idx.end(), rng_);
    idx.resize(m);

    std::vector<double> SF, SCR, SG;

    // FIX #4: construct distributions once; muF_/muCR_ are stable for the
    // entire agent loop — they are updated only after update_mu_from_success().
    std::cauchy_distribution<double> cauchyF(muF_, 0.1);
    std::normal_distribution<double> normCR(muCR_, 0.1);

    for (int t=0; t<m; ++t){
        if (prob_->calls() >= max_evals_) break;

        int i = idx[t];

        double F, CR;
        // Do NOT recreate cauchyF / normCR here — see FIX #4 above.
        sample_F_CR(F, CR, cauchyF, normCR);

        Vec u(D, 0.0);
        makeTrialARQ(i, ord, F, CR, u);

        double fu = eval(u);
        if (selectionRTR(i, u, fu, F, CR, SF, SCR, SG)) {
            if (fu < best_f_) {
                best_f_ = fu;
                best_x_ = u;
            }
        }
    }

    update_mu_from_success(SF, SCR, SG);
    success_[0] += (int)SF.size();
    if (roulette_normalize_) {
        const double arq_attempts = (double)std::max(1, m);
        ni_[0] += (double)SF.size() / arq_attempts;
    } else {
        ni_[0] += (double)SF.size();
    }
}

void ARQ2::stepIDE(){
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 4) return;

    if (ide_progress_sync_)
        g_ = std::max(g_ + 1, ideGenerationFromProgress());
    else
        ++g_;
    if (g_ > gmax_) g_ = gmax_;

    sortByFitness();
    std::vector<Vec> Q(N, Vec(D));
    std::vector<double> QF(N, std::numeric_limits<double>::infinity());

    double IDEps = 0.1 + 0.9 * std::pow(10.0, 5.0 * ((double)g_ / (double)gmax_ - 1.0));
    double SRT = (g_ < gt_) ? 0.0 : 0.1;

    for (int i = 0; i < N; ++i) {
        std::vector<int> idx;
        sampleDistinctExcluding(N, 4, {i}, idx);
        if ((int)idx.size() < 4) continue;
        int o  = idx[0];
        int r1 = idx[1];
        int r2 = idx[2];
        int r3 = idx[3];

        const Vec& xo = X_[o];
        const Vec* xr1ptr = nullptr;

        // ── Elite-guided xr1 selection ────────────────────────────────────────
        // FIX #9 (restored): elite pick ≠ i.
        // Reverted version was tested and performed worse across all problems —
        // fix #9 is confirmed beneficial and is kept active.
        auto pickElite = [&](int high_ind_S) -> int {
            if (high_ind_S <= 1) return 0;
            int pick, tries = 0;
            do {
                pick = randInt(0, high_ind_S - 1);
                ++tries;
            } while (pick == i && tries < kMaxPickTries);
            return pick;
        };

        if (g_ <= gt_) {
            double probSup = 0.9 * IDEps;
            if (randU() < probSup) {
                int high_ind_S = std::max(2, (int)std::round(IDEps * N));
                if (high_ind_S > N) high_ind_S = N;
                xr1ptr = &X_[pickElite(high_ind_S)];
            } else {
                xr1ptr = &X_[r1];
            }
        } else {
            int high_ind_S = std::max(2, (int)std::round(IDEps * N));
            if (high_ind_S > N) high_ind_S = N;
            if (randU() < 0.5)
                xr1ptr = &X_[pickElite(high_ind_S)];
            else
                xr1ptr = &X_[r1];
        }

        const Vec& xr1 = *xr1ptr;
        const Vec& xr2 = X_[r2];
        const Vec& xr3 = X_[r3];

        double Fo = CBF_[i];
        Vec v(D);
        if (g_ > gt_ && randU() < 0.5) {
            for (int j = 0; j < D; ++j)
                v[j] = X_[i][j] + Fo * (xr1[j] - xo[j]) + Fo * (xr2[j] - xr3[j]);
        } else {
            for (int j = 0; j < D; ++j)
                v[j] = xo[j] + Fo * (xr1[j] - xo[j]) + Fo * (xr2[j] - xr3[j]);
        }
        ensureBounds(v);
        Q[i] = v;
    }

    for (int i = 0; i < N; ++i) {
        Vec y = X_[i];
        const Vec& v = Q[i];
        double CR = CBCR_[i];
        int jrand = randInt(0, D - 1);
        for (int j = 0; j < D; ++j) {
            if (randU() < CR || j == jrand)
                y[j] = v[j];
        }
        ensureBounds(y);
        Q[i] = y;
    }

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;
        double fy = eval(Q[i]);
        QF[i] = fy;
    }

    std::vector<int> indsucc;
    for (int i = 0; i < N; ++i) {
        const bool improved = ide_strict_improve_ ? (QF[i] < FX_[i]) : (QF[i] <= FX_[i]);
        if (improved)
            indsucc.push_back(i);
    }

    success_[1] += (int)indsucc.size();
    if (roulette_normalize_) {
        const double ide_attempts = (double)std::max(1, N);
        ni_[1] += (double)indsucc.size() / ide_attempts;
    } else {
        ni_[1] += (double)indsucc.size();
    }

    double SR = (N > 0) ? ((double)indsucc.size() / (double)N) : 0.0;
    if (g_ < gt_) {
        if (SR <= SRT) ++Tcurr_;
        else Tcurr_ = 0;
        if ((double)Tcurr_ >= T_) {
            gt_ = g_;
        }
    }

    for (int idx : indsucc) {
        X_[idx]  = Q[idx];
        FX_[idx] = QF[idx];
        if (FX_[idx] < best_f_) {
            best_f_ = FX_[idx];
            best_x_ = X_[idx];
        }
    }

    // ── FIX #10: Remove the trailing sortByFitness() call. ─────────────────
    //
    //  Original code + FIX #7 interaction (critical):
    //  stepIDE() called sortByFitness() TWICE — at the start and at the end.
    //  After FIX #7, sortByFitness() correctly carries CBF_/CBCR_ alongside
    //  X_/FX_.  The end-of-step sort re-locks the updated CBF_/CBCR_ to the
    //  re-ranked individuals.  Over successive IDE calls this creates a
    //  parameter convergence feedback loop:
    //    generation g  : sort → elite get their own good params → use them
    //    generation g+1: sort → same elite still have same params → reinforce
    //    ...
    //  On unimodal / structured problems this accelerates convergence (good).
    //  On multimodal problems it locks F/CR to values that work for the
    //  current LOCAL optimum, preventing escape → observed bimodal behaviour
    //  (SD×6 for weierstrass, mean +900% for weierstrass, +59% for polyphase).
    //
    //  The trailing sort is REDUNDANT:
    //  - stepARQ() computes its own `ord` from scratch.
    //  - quarantine() computes its own `ord` from scratch.
    //  - The NEXT stepIDE() call sorts at its start — providing the correct
    //    parameter alignment for THAT generation's operation.
    //
    //  Removing it breaks the convergence loop while preserving FIX #7's
    //  benefit (correct CBF_/CBCR_ alignment at the start of each IDE step).
    //  Parameter diversity is maintained because, between two IDE calls, ARQ
    //  and quarantine can reorder/replace individuals without carrying the
    //  sorted parameter order forward.
}

void ARQ2::quarantine(){
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N < 4) return;

    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    std::vector<double> fits = FX_;
    double Q1 = quantile(fits, 0.25);
    fits = FX_;
    double Q3 = quantile(fits, 0.75);
    double IQR = Q3 - Q1;
    double theta = Q3 + outlier_alpha_ * IQR;

    const int half = std::max(1, N/2);
    const int D = prob_->dimension();
    Vec center(D, 0.0);
    for (int k=0; k<half; ++k){
        const Vec& x = X_[ord[k]];
        for (int j=0; j<D; ++j) center[j] += x[j];
    }
    for (int j=0; j<D; ++j) center[j] /= (double)half;

    std::vector<int> out;
    for (int i=0; i<N; ++i)
        if (FX_[i] >= theta) out.push_back(i);

    if (!out.empty()){
        int k = (int)std::floor(outlier_rho_ * (double)out.size());
        if (k > 0){
            std::shuffle(out.begin(), out.end(), rng_);
            out.resize(k);

            const Vec& L = prob_->lb();
            const Vec& U = prob_->ub();

            for (int idx : out){
                if (prob_->calls() >= max_evals_) break;

                Vec cand = center;
                for (int j=0; j<D; ++j){
                    double lo = (j < (int)L.size() ? L[j] : -1.0);
                    double hi = (j < (int)U.size() ? U[j] :  1.0);
                    if (lo > hi) std::swap(lo, hi);
                    double scale = qsigma_ * (hi - lo);
                    std::normal_distribution<double> N0(0.0, scale);
                    cand[j] += N0(rng_);
                }
                ensureBounds(cand);
                double fc = eval(cand);

                if (fc < FX_[idx]) {
                    archivePush(X_[idx]);
                    X_[idx] = std::move(cand);
                    FX_[idx] = fc;
                    resetIDEParamsAt(idx);
                    if (fc < best_f_) {
                        best_f_ = fc;
                        best_x_ = X_[idx];
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  microRestartARQ
//  FIX #2: Make the stagnation counter reset conditional on actual improvement.
//
//  Original bug: `no_improve_ = 0` appeared unconditionally at the end of the
//  function, so even when every perturbed candidate failed to improve on its
//  target, the counter was zeroed.  The next stagnation event could then not
//  fire for another `stagnation_trigger_` ARQ steps — defeating the purpose of
//  the mechanism entirely.
//
//  Fix: track whether any replacement occurred (`any_replaced`).  Only reset
//  `no_improve_` when at least one candidate was accepted.  If the restart
//  produced no improvement, leave the counter at its current value (≥
//  stagnation_trigger_) so the mechanism can try again on the very next ARQ
//  step that still finds no improvement.  This prevents the algorithm from
//  silently stagnating in a region where the micro-restart perturbation σ is
//  too small to escape.
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::microRestartARQ(){
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N < 4) return;
    if (no_improve_ < stagnation_trigger_) return;

    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    int wcount = std::max(1, (int)std::floor(worst_frac_ * (double)N));
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    const int D = prob_->dimension();

    bool any_replaced = false;  // FIX #2: track whether any slot was improved

    for (int t=0; t<wcount; ++t){
        if (prob_->calls() >= max_evals_) break;
        int idx = ord[N-1-t];

        Vec cand = best_x_;
        for (int j=0; j<D; ++j){
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            double scale = rsigma_ * (hi - lo);
            std::normal_distribution<double> N0(0.0, scale);
            cand[j] += N0(rng_);
        }
        ensureBounds(cand);
        double fc = eval(cand);

        if (fc < FX_[idx]) {
            archivePush(X_[idx]);
            X_[idx] = std::move(cand);
            FX_[idx] = fc;
            resetIDEParamsAt(idx);
            any_replaced = true;   // FIX #2

            if (fc < best_f_) {
                best_f_ = fc;
                best_x_ = X_[idx];
            }
        }
    }

    // FIX #2: only reset the counter when at least one replacement happened.
    // If nothing improved, leaving no_improve_ ≥ stagnation_trigger_ lets the
    // mechanism retry immediately on the next eligible ARQ step.
    if (any_replaced) {
        no_improve_ = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  one_iteration
//  FIX #2 (complement): Track stagnation during IDE steps as well.
//
//  Original bug: `no_improve_` was only incremented / reset inside the ARQ
//  branch.  When the roulette selected IDE for several consecutive iterations,
//  `no_improve_` stalled, making microRestartARQ blind to stagnation that
//  occurred during IDE phases.  The fix mirrors the existing ARQ tracking
//  logic in the IDE branch.
// ─────────────────────────────────────────────────────────────────────────────
void ARQ2::one_iteration(){
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;

    const int N = (int)X_.size();
    if (N < 4) return;

    archiveTrim(N);

    int hh = 0;
    double pmin = 1.0 / std::max(1, h_);

    if (bootstrap_left_ > 0) {
        hh = 0;
        --bootstrap_left_;
    } else {
        auto sel = rouletteSelect();
        hh   = sel.first;
        pmin = sel.second;

        if (pmin < delta_) {
            for (int i = 0; i < h_; ++i)
                cni_[i] += ni_[i] - (double)n0_;
            ni_.assign(h_, (double)n0_);
            ++nrst_;
        }
    }

    switch (hh) {
        case 1:
            stepIDE();

            // FIX #2: mirror ARQ stagnation tracking so microRestartARQ can
            // fire even when IDE dominates the roulette for many iterations.
            if (best_f_ < best_prev_) {
                best_prev_ = best_f_;
                no_improve_ = 0;
            } else {
                no_improve_++;
            }
            // Note: microRestartARQ is intentionally NOT called for IDE steps
            // (it targets ARQ-specific worst individuals), but the counter must
            // advance so that when ARQ is next selected the trigger fires
            // correctly.
            break;

        case 0:
        default:
            stepARQ();
            quarantine();

            if (best_f_ < best_prev_) {
                best_prev_ = best_f_;
                no_improve_ = 0;
            } else {
                no_improve_++;
            }

            microRestartARQ();
            break;
    }

    archiveTrim((int)X_.size());

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
