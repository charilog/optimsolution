#include "arq3.h"
#include <cstdio>
#include <cctype>


namespace optimsolution {

// ============================================================================
// configure
// ============================================================================
void ARQ3::configure(const MethodConfig& mc) {
    auto trim = [](std::string s) {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace((unsigned char)s[a])) ++a;
        while (b > a && std::isspace((unsigned char)s[b-1])) --b;
        return s.substr(a, b - a);
    };
    auto parse_int = [&](std::string s, int fb) -> int {
        s = trim(s);
        if (s.empty()) return fb;
        try { size_t pos = 0; long v = std::stol(s, &pos); if (pos == s.size()) return (int)v; }
        catch (...) {}
        return fb;
    };

    // Population: explicit "population" param overrides Ninit = pop_scale * D.
    int p = mc.getInt("population", mc.getInt("Population",
             mc.getInt("pop", mc.getInt("Pop", -1))));
    if (p < 0) p = parse_int(mc.getStr("population", ""), -1);
    if (p >= 4) { Ninit_ = p; this->setPopulation(Ninit_); }

    pop_scale_       = mc.getInt("popscale", pop_scale_);
    Nmin_            = mc.getInt("Nmin", Nmin_);
    nlpsr_alpha_     = mc.getDbl("nlpsralpha", nlpsr_alpha_);

    // SHADE memory
    H_               = mc.getInt("H", H_);
    MF_terminal_     = mc.getDbl("MFterminal", MF_terminal_);
    MCR_terminal_    = mc.getDbl("MCRterminal", MCR_terminal_);

    Flo_             = mc.getDbl("Flo", Flo_);
    Fhi_             = mc.getDbl("Fhi", Fhi_);

    // jSO schedules
    pbest_max_       = mc.getDbl("pmax", pbest_max_);
    pbest_min_       = mc.getDbl("pmin", pbest_min_);

    // RSP
    kr_init_         = mc.getDbl("krinit",  kr_init_);
    kr_final_        = mc.getDbl("krfinal", kr_final_);

    // Eigen crossover
    p_eig_           = mc.getDbl("peig",        p_eig_);
    eig_period_      = mc.getInt("eigperiod",   eig_period_);
    eig_frac_        = mc.getDbl("eigfrac",     eig_frac_);

    // RTR / archive
    rtr_pool_        = mc.getInt("rtrpool",     rtr_pool_);
    rtr_pool_frac_   = mc.getDbl("rtrpoolfrac", rtr_pool_frac_);
    archive_rate_    = mc.getDbl("archiverate", archive_rate_);

    // Thompson bandit
    bandit_decay_    = mc.getDbl("banditdecay",         bandit_decay_);
    bootstrap_arq_iters_ = mc.getInt("bootstrap_arq_iters", bootstrap_arq_iters_);

    // Quarantine (Levy)
    outlier_alpha_   = mc.getDbl("alpha",    outlier_alpha_);
    outlier_rho_     = mc.getDbl("rho",      outlier_rho_);
    levy_beta_       = mc.getDbl("levybeta", levy_beta_);
    qscale_          = mc.getDbl("qscale",   qscale_);

    // OBL basin escape
    stag_trigger_         = mc.getInt("stagnationtrigger",  stag_trigger_);
    var_collapse_ratio_   = mc.getDbl("varcollapseratio",   var_collapse_ratio_);
    obl_cooldown_init_    = mc.getInt("oblcooldown",        obl_cooldown_init_);
    obl_frac_             = mc.getDbl("oblfrac",            obl_frac_);

    // Agent fraction
    agent_fraction_       = mc.getDbl("agentfraction",      agent_fraction_);

    // IDE flags
    ide_progress_sync_    = mc.getInt("ide_progress_sync",  ide_progress_sync_);
    ide_strict_improve_   = mc.getInt("ide_strict_improve", ide_strict_improve_);

    debug_                = mc.getInt("debug_arq", debug_);

    // Sanity clamps
    if (H_ < 2) H_ = 2;
    if (Nmin_ < 4) Nmin_ = 4;
    if (pop_scale_ < 4) pop_scale_ = 4;
    if (nlpsr_alpha_ <= 0.0) nlpsr_alpha_ = 0.5;
    if (Flo_ <= 0.0) Flo_ = 0.01;
    if (Fhi_ < Flo_) std::swap(Fhi_, Flo_);
    if (pbest_max_ < 0.01) pbest_max_ = 0.01;
    if (pbest_max_ > 0.5)  pbest_max_ = 0.5;
    if (pbest_min_ < 0.01) pbest_min_ = 0.01;
    if (pbest_min_ > pbest_max_) pbest_min_ = pbest_max_;
    if (p_eig_ < 0.0) p_eig_ = 0.0;
    if (p_eig_ > 1.0) p_eig_ = 1.0;
    if (eig_period_ < 1) eig_period_ = 1;
    if (eig_frac_ <= 0.0 || eig_frac_ > 1.0) eig_frac_ = 0.5;
    if (rtr_pool_ < 2) rtr_pool_ = 2;
    if (rtr_pool_frac_ <= 0.0) rtr_pool_frac_ = 0.1;
    if (archive_rate_ <= 0.1) archive_rate_ = 1.0;
    if (bandit_decay_ <= 0.0 || bandit_decay_ > 1.0) bandit_decay_ = 0.97;
    if (bootstrap_arq_iters_ < 0) bootstrap_arq_iters_ = 0;
    if (outlier_rho_ < 0.0) outlier_rho_ = 0.0;
    if (outlier_rho_ > 1.0) outlier_rho_ = 1.0;
    if (levy_beta_ < 1.01) levy_beta_ = 1.01;
    if (levy_beta_ > 2.0)  levy_beta_ = 2.0;
    if (qscale_ <= 0.0) qscale_ = 0.1;
    if (stag_trigger_ < 1) stag_trigger_ = 1;
    if (var_collapse_ratio_ < 0.0) var_collapse_ratio_ = 0.0;
    if (obl_cooldown_init_ < 0) obl_cooldown_init_ = 0;
    if (obl_frac_ <= 0.0 || obl_frac_ > 1.0) obl_frac_ = 0.3;
    if (agent_fraction_ <= 0.0 || agent_fraction_ > 1.0) agent_fraction_ = 1.0;
    ide_progress_sync_  = ide_progress_sync_  ? 1 : 0;
    ide_strict_improve_ = ide_strict_improve_ ? 1 : 0;
}

// ============================================================================
// init
// ============================================================================
void ARQ3::init() {
    if (!prob_) return;

    const int D = prob_->dimension();

    // Decide Ninit.  If user explicitly provided population, Ninit_ already set.
    if (Ninit_ < 4) {
        Ninit_ = std::max(4, pop_scale_ * D);
    }
    int N = std::max(Ninit_, Nmin_);
    this->setPopulation(N);

    X_.clear(); FX_.clear(); A_.clear();
    CBF_.clear(); CBCR_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    // Evaluate initial population
    for (int i = 0; i < N; ++i) {
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

    // SHADE circular memory
    initMemory();

    // Thompson bandit priors -- mildly optimistic for ARQ (a_=2,b_=1) to get
    // some warm-up bias; IDE starts flat.
    h_ = 2;
    bandit_a_.assign(h_, 1.0);
    bandit_b_.assign(h_, 1.0);
    bandit_a_[0] = 2.0;
    bootstrap_left_ = bootstrap_arq_iters_;

    // IDE schedule setup
    g_    = 0;
    gmax_ = std::max(1, (int)std::round((double)max_evals_ / std::max(N, 1)));
    T_    = gmax_ / 10.0;
    gt_   = std::max(1, gmax_ / 2);
    Tcurr_ = 0;

    // Per-individual IDE params (bimodal Cauchy, as in EA4Eig)
    CBF_.assign(N, 0.0);
    CBCR_.assign(N, 0.0);
    for (int i = 0; i < N; ++i) sampleIDEParamsAt(i);

    // Eigen basis: lazy, computed later
    B_rot_.clear();
    eig_valid_ = false;
    iters_since_eig_ = 0;

    if (debug_) {
        std::fprintf(stdout,
            "[arq3] cfg -> Ninit=%d Nmin=%d D=%d H=%d alphaNL=%.2f "
            "pmax=%.2f pmin=%.2f krI=%.1f krF=%.1f peig=%.2f eper=%d "
            "rtr=%d arch=%.2f levyB=%.2f\n",
            Ninit_, Nmin_, D, H_, nlpsr_alpha_, pbest_max_, pbest_min_,
            kr_init_, kr_final_, p_eig_, eig_period_, rtr_pool_,
            archive_rate_, levy_beta_);
        std::fflush(stdout);
    }

    printBest();
}

// ============================================================================
// Low-level utilities
// ============================================================================
int ARQ3::pickDistinct(int n, int a, int b, int c) {
    std::uniform_int_distribution<int> I(0, n - 1);
    int r;
    do { r = I(rng_); } while (r == a || r == b || r == c);
    return r;
}

int ARQ3::randInt(int lo, int hi) {
    if (hi < lo) std::swap(lo, hi);
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double ARQ3::randU() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double ARQ3::cauchy(double loc, double scale) {
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

double ARQ3::gaussN(double mu, double sig) {
    std::normal_distribution<double> dist(mu, sig);
    return dist(rng_);
}

// Mantegna 1994 Levy step (symmetric alpha-stable approximation).
// Returns a centered Levy-distributed scalar.  Caller must scale by range.
double ARQ3::sampleLevy() {
    constexpr double kPi = 3.14159265358979323846;
    const double beta = levy_beta_;
    const double num = std::tgamma(1.0 + beta)
                       * std::sin(kPi * beta / 2.0);
    const double den = std::tgamma((1.0 + beta) / 2.0)
                       * beta
                       * std::pow(2.0, (beta - 1.0) / 2.0);
    double sigma_u = std::pow(num / den, 1.0 / beta);
    double u = gaussN(0.0, sigma_u);
    double v = gaussN(0.0, 1.0);
    double step = u / std::pow(std::fabs(v) + 1e-12, 1.0 / beta);
    // Clip extreme tails: levy can produce huge outliers numerically.
    if (step >  50.0) step =  50.0;
    if (step < -50.0) step = -50.0;
    return step;
}

double ARQ3::progress01() const {
    if (!prob_ || max_evals_ <= 0) return 1.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

void ARQ3::ensureBounds(Vec& v) {
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j = 0; j < v.size(); ++j) {
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (!std::isfinite(v[j])) v[j] = 0.5 * (lo + hi);
        if (lo == hi) { v[j] = lo; continue; }
        // Reflection within a bounded number of iterations.
        int guard = 0;
        while ((v[j] < lo || v[j] > hi) && guard++ < 50) {
            if (v[j] > hi)      v[j] = 2.0 * hi - v[j];
            else if (v[j] < lo) v[j] = 2.0 * lo - v[j];
        }
        // Hard clamp fallback (in case reflection oscillates numerically)
        if (v[j] < lo) v[j] = lo;
        if (v[j] > hi) v[j] = hi;
    }
}

double ARQ3::distBN(const Vec& a, const Vec& b) const {
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    double s = 0.0;
    for (size_t j = 0; j < a.size(); ++j) {
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double denom = (hi - lo);
        double z = (denom > 0.0) ? ((a[j] - b[j]) / denom) : (a[j] - b[j]);
        s += z * z;
    }
    return std::sqrt(s);
}

double ARQ3::quantile(std::vector<double> v, double q01) {
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

double ARQ3::normalizedPopSpread() const {
    const int N = (int)X_.size();
    if (N < 2) return 0.0;
    const int D = (N > 0 ? (int)X_[0].size() : 0);
    if (D <= 0) return 0.0;
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    double s2sum = 0.0;
    int    counted = 0;
    for (int j = 0; j < D; ++j) {
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double range = hi - lo;
        if (range <= 0.0) continue;
        double mean = 0.0;
        for (int i = 0; i < N; ++i) mean += X_[i][j];
        mean /= (double)N;
        double var = 0.0;
        for (int i = 0; i < N; ++i) {
            double d = X_[i][j] - mean;
            var += d * d;
        }
        var /= (double)N;
        s2sum += std::sqrt(var) / range;
        ++counted;
    }
    return (counted > 0) ? (s2sum / (double)counted) : 0.0;
}

void ARQ3::sortByFitness() {
    const int N = (int)X_.size();
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });
    std::vector<Vec>    nX(N);
    std::vector<double> nF(N);
    std::vector<double> nF2((int)CBF_.size());
    std::vector<double> nCR((int)CBCR_.size());
    for (int i = 0; i < N; ++i) {
        nX[i] = std::move(X_[idx[i]]);
        nF[i] = FX_[idx[i]];
        if (!CBF_.empty())  nF2[i] = CBF_[idx[i]];
        if (!CBCR_.empty()) nCR[i] = CBCR_[idx[i]];
    }
    X_.swap(nX);
    FX_.swap(nF);
    if (!CBF_.empty())  CBF_.swap(nF2);
    if (!CBCR_.empty()) CBCR_.swap(nCR);
}



// ============================================================================
// NLPSR: compute target population size from current budget progress.
//   N(t) = round( Ninit + (Nmin - Ninit) * progress(t)^alpha )
// alpha == 1  -> linear LSHADE reduction
// alpha  < 1  -> slower shrink early, faster shrink near the end (NL-SHADE-RSP)
// ============================================================================
int ARQ3::targetPopulationSize() const {
    double p = progress01();
    double frac = std::pow(p, nlpsr_alpha_);
    double N = (double)Ninit_ + ((double)Nmin_ - (double)Ninit_) * frac;
    int Ni = (int)std::round(N);
    if (Ni < Nmin_) Ni = Nmin_;
    if (Ni > Ninit_) Ni = Ninit_;
    return Ni;
}

// Shrink population to N by removing worst individuals.  Keeps per-individual
// IDE parameters aligned.  Does NOT call eval.
void ARQ3::shrinkTo(int Ntarget) {
    int N = (int)X_.size();
    if (Ntarget >= N) return;
    if (Ntarget < Nmin_) Ntarget = Nmin_;

    // sort ascending by fitness; we then keep the first Ntarget
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec>    nX(Ntarget);
    std::vector<double> nF(Ntarget);
    std::vector<double> nCBF(Ntarget);
    std::vector<double> nCBCR(Ntarget);
    for (int i = 0; i < Ntarget; ++i) {
        int k = ord[i];
        nX[i]    = std::move(X_[k]);
        nF[i]    = FX_[k];
        nCBF[i]  = (k < (int)CBF_.size())  ? CBF_[k]  : 0.5;
        nCBCR[i] = (k < (int)CBCR_.size()) ? CBCR_[k] : 0.5;
    }
    X_.swap(nX);
    FX_.swap(nF);
    CBF_.swap(nCBF);
    CBCR_.swap(nCBCR);
    this->setPopulation(Ntarget);

    // any cached eigen basis may no longer reflect distribution; force refresh
    eig_valid_ = false;
    iters_since_eig_ = 0;
}

// ============================================================================
// jSO-style schedules
// ============================================================================
double ARQ3::currentKR() const {
    double pr = progress01();
    return kr_init_ + (kr_final_ - kr_init_) * pr;
}

double ARQ3::currentPbest() const {
    double pr = progress01();
    // linear from pbest_max_ down to pbest_min_
    return pbest_max_ + (pbest_min_ - pbest_max_) * pr;
}

// ============================================================================
// LSHADE-RSP rank-biased picking: weight index r (rank 0 = best) by (N-r)^kr
// ============================================================================
int ARQ3::rankBasedPick(const std::vector<int>& ord, int forbid) const {
    const int N = (int)ord.size();
    if (N <= 0) return forbid;
    if (N == 1) return ord[0];

    auto* self = const_cast<ARQ3*>(this);
    const double kr = currentKR();

    std::vector<double> cw(N);
    double acc = 0.0;
    for (int r = 0; r < N; ++r) {
        acc += std::pow((double)(N - r), kr);
        cw[r] = acc;
    }

    std::uniform_real_distribution<double> U(0.0, acc);
    for (int tries = 0; tries < 20; ++tries) {
        double u = U(self->rng_);
        int lo = 0, hi = N - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (cw[mid] < u) lo = mid + 1;
            else hi = mid;
        }
        int idx = ord[lo];
        if (idx != forbid) return idx;
    }
    for (int r = 0; r < N; ++r)
        if (ord[r] != forbid) return ord[r];
    return ord[0];
}

// ============================================================================
// SHADE circular memory (jSO-style with fixed terminal slot)
// ============================================================================
void ARQ3::initMemory() {
    MF_.assign(H_,  0.5);
    MCR_.assign(H_, 0.5);
    if (H_ >= 1) {
        MF_[H_ - 1]  = MF_terminal_;
        MCR_[H_ - 1] = MCR_terminal_;
    }
    memK_ = 0;
}

void ARQ3::sampleFCR(double& F, double& CR) {
    // random memory slot (including terminal)
    int r = randInt(0, H_ - 1);
    double muF  = MF_[r];
    double muCR = MCR_[r];

    // CR ~ Normal(muCR, 0.1) clamped to [0,1]; if muCR is "terminal" (>= 1.0
    // numerically would be very large) we still clamp to 1.
    CR = gaussN(muCR, 0.1);
    if (CR < 0.0) CR = 0.0;
    if (CR > 1.0) CR = 1.0;

    // F ~ Cauchy(muF, 0.1) with >0 rejection, then clamp to [Flo_, Fhi_]
    double Fs = 0.0;
    for (int t = 0; t < 50; ++t) {
        Fs = cauchy(muF, 0.1);
        if (Fs > 0.0) break;
    }
    if (Fs <= 0.0) Fs = muF;
    if (Fs < Flo_) Fs = Flo_;
    if (Fs > Fhi_) Fs = Fhi_;
    F = Fs;

    // jSO-style CR floor (early: stronger crossover)
    double pr = progress01();
    if (pr < 0.25) CR = std::max(CR, 0.7);
    else if (pr < 0.50) CR = std::max(CR, 0.6);

    // F cap in early/middle phase: avoid too-aggressive F
    if (pr < 0.60) F = std::min(F, 0.7 + 0.3 * pr);   // 0.7 -> ~0.88
}

void ARQ3::updateMemoryFromSuccess(const std::vector<double>& SF,
                                   const std::vector<double>& SCR,
                                   const std::vector<double>& SG) {
    if (SF.empty()) return;
    double sumG = 0.0;
    for (double g : SG) sumG += g;
    if (!(sumG > 0.0)) return;

    // weighted Lehmer mean for F, weighted arithmetic for CR
    double num = 0.0, den = 0.0, newMuCR = 0.0;
    for (size_t i = 0; i < SF.size(); ++i) {
        double w = SG[i] / sumG;
        num += w * SF[i] * SF[i];
        den += w * SF[i];
        newMuCR += w * SCR[i];
    }
    double newMuF = (den > 0.0) ? (num / den) : MF_[memK_];

    if (newMuF  < Flo_) newMuF  = Flo_;
    if (newMuF  > Fhi_) newMuF  = Fhi_;
    if (newMuCR < 0.0)  newMuCR = 0.0;
    if (newMuCR > 1.0)  newMuCR = 1.0;

    // Only cycle over non-terminal slots 0..H-2 (terminal stays at 0.9/0.9)
    int cyclic_H = std::max(1, H_ - 1);
    MF_[memK_]  = 0.5 * (MF_[memK_]  + newMuF);
    MCR_[memK_] = 0.5 * (MCR_[memK_] + newMuCR);
    memK_ = (memK_ + 1) % cyclic_H;
}

// ============================================================================
// Archive: FIFO-ish with overflow pruning
// ============================================================================
void ARQ3::archivePush(const Vec& x) {
    A_.push_back(x);
}

void ARQ3::archiveTrim(int N) {
    const int cap = std::max(1, (int)std::floor(archive_rate_ * (double)N));
    if ((int)A_.size() <= cap) return;
    // evict oldest entries first (FIFO) -- retains recent failures
    int excess = (int)A_.size() - cap;
    A_.erase(A_.begin(), A_.begin() + excess);
}

// ============================================================================
// Jacobi eigen-decomposition for small symmetric matrices.
// Dimensionality here is the problem dimension D (typically <= 100).
// ============================================================================
void ARQ3::jacobiEigen(const Mat& Ain, Mat& V, std::vector<double>& w) const {
    const int n = (int)Ain.size();
    Mat A = Ain;
    V.assign(n, Vec(n, 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;
    w.assign(n, 0.0);

    const int max_sweeps = 50;
    const double tol = 1e-12;

    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q)
                off += A[p][q] * A[p][q];
        if (off < tol) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double Apq = A[p][q];
                if (std::fabs(Apq) < 1e-15) continue;
                double App = A[p][p];
                double Aqq = A[q][q];

                double theta = (Aqq - App) / (2.0 * Apq);
                double t;
                if (std::fabs(theta) > 1e15) {
                    t = 1.0 / (2.0 * theta);
                } else {
                    double sign = (theta >= 0.0) ? 1.0 : -1.0;
                    t = sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                }
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                A[p][p] = App - t * Apq;
                A[q][q] = Aqq + t * Apq;
                A[p][q] = 0.0;
                A[q][p] = 0.0;

                for (int r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    double Arp = A[r][p];
                    double Arq = A[r][q];
                    A[r][p] = c * Arp - s * Arq;
                    A[r][q] = s * Arp + c * Arq;
                    A[p][r] = A[r][p];
                    A[q][r] = A[r][q];
                }
                for (int r = 0; r < n; ++r) {
                    double Vrp = V[r][p];
                    double Vrq = V[r][q];
                    V[r][p] = c * Vrp - s * Vrq;
                    V[r][q] = s * Vrp + c * Vrq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) w[i] = A[i][i];
}

void ARQ3::recomputeEigenBasis() {
    if (!prob_) { eig_valid_ = false; return; }
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    int top = std::max(D + 2, (int)std::round(eig_frac_ * (double)N));
    if (top > N) top = N;
    if (D < eig_min_D_ || top < std::max(D + 2, 4)) { eig_valid_ = false; return; }

    // Assumes X_ is sorted by fitness (caller's responsibility).
    Vec mean(D, 0.0);
    for (int i = 0; i < top; ++i)
        for (int j = 0; j < D; ++j) mean[j] += X_[i][j];
    for (int j = 0; j < D; ++j) mean[j] /= (double)top;

    Mat C(D, Vec(D, 0.0));
    for (int i = 0; i < top; ++i) {
        const Vec& x = X_[i];
        for (int a = 0; a < D; ++a) {
            double da = x[a] - mean[a];
            for (int b = a; b < D; ++b) {
                C[a][b] += da * (x[b] - mean[b]);
            }
        }
    }
    double invN = 1.0 / std::max(1, top - 1);
    for (int a = 0; a < D; ++a)
        for (int b = a; b < D; ++b) {
            C[a][b] *= invN;
            C[b][a] = C[a][b];
        }
    // Regularize diagonal to avoid singular cases when population collapses
    double trace_over_D = 0.0;
    for (int a = 0; a < D; ++a) trace_over_D += C[a][a];
    trace_over_D = std::fabs(trace_over_D) / std::max(1, D);
    double reg = 1e-12 + 1e-8 * trace_over_D;
    for (int a = 0; a < D; ++a) C[a][a] += reg;

    std::vector<double> w;
    jacobiEigen(C, B_rot_, w);
    eig_valid_ = true;
    iters_since_eig_ = 0;
}

// out = B^T x
void ARQ3::applyBt(const Mat& B, const Vec& x, Vec& out) const {
    const int D = (int)x.size();
    out.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double xi = x[i];
        for (int j = 0; j < D; ++j) {
            out[j] += B[i][j] * xi;
        }
    }
}

// out = B x
void ARQ3::applyB(const Mat& B, const Vec& x, Vec& out) const {
    const int D = (int)x.size();
    out.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double acc = 0.0;
        for (int j = 0; j < D; ++j) acc += B[i][j] * x[j];
        out[i] = acc;
    }
}

void ARQ3::eigenBinomialCrossover(int D, const Vec& base, const Vec& v,
                                  double CR, Vec& u) {
    Vec base_e, v_e;
    applyBt(B_rot_, base, base_e);
    applyBt(B_rot_, v,    v_e);
    Vec u_e = base_e;
    int jr = randInt(0, D - 1);
    for (int j = 0; j < D; ++j) {
        if (randU() < CR || j == jr) u_e[j] = v_e[j];
    }
    applyB(B_rot_, u_e, u);
}

// ============================================================================
// Thompson sampling bandit
// ============================================================================
int ARQ3::thompsonPick() {
    std::vector<double> samples(h_, 0.0);
    for (int k = 0; k < h_; ++k) {
        double a = std::max(1e-3, bandit_a_[k]);
        double b = std::max(1e-3, bandit_b_[k]);
        // Sample Beta(a,b) via two Gamma draws: X/(X+Y)
        std::gamma_distribution<double> gA(a, 1.0);
        std::gamma_distribution<double> gB(b, 1.0);
        double x = gA(rng_);
        double y = gB(rng_);
        double s = x + y;
        samples[k] = (s > 0.0) ? (x / s) : 0.5;
    }
    int best = 0;
    for (int k = 1; k < h_; ++k)
        if (samples[k] > samples[best]) best = k;
    return best;
}

void ARQ3::banditDecay() {
    // Non-stationary forgetting: decay toward uniform prior (1,1).
    for (int k = 0; k < h_; ++k) {
        bandit_a_[k] = 1.0 + (bandit_a_[k] - 1.0) * bandit_decay_;
        bandit_b_[k] = 1.0 + (bandit_b_[k] - 1.0) * bandit_decay_;
    }
}

void ARQ3::banditRecord(int k, int successes, int attempts) {
    if (k < 0 || k >= h_) return;
    if (attempts <= 0) return;
    int failures = std::max(0, attempts - successes);
    // Normalize by attempts so one heavy IDE sweep doesn't dominate one ARQ sweep
    double norm = 1.0 / (double)attempts;
    bandit_a_[k] += (double)successes * norm;
    bandit_b_[k] += (double)failures  * norm;
}

// ============================================================================
// IDE parameter sampling (unchanged from EA4Eig IDE)
// ============================================================================
void ARQ3::sampleIDEParamsAt(int idx) {
    if (idx < 0 || idx >= (int)CBF_.size() || idx >= (int)CBCR_.size()) return;
    double F;
    if (randU() < 0.5) F = cauchy(0.65, 0.1);
    else                F = cauchy(1.00, 0.1);
    while (F < 0.0) {
        if (randU() < 0.5) F = cauchy(0.65, 0.1);
        else                F = cauchy(1.00, 0.1);
    }
    if (F > 1.0) F = 1.0;
    CBF_[idx] = F;

    double CR;
    if (randU() < 0.5) CR = cauchy(0.10, 0.1);
    else                CR = cauchy(0.95, 0.1);
    if (CR > 1.0) CR = 1.0;
    if (CR < 0.0) CR = 0.0;
    CBCR_[idx] = CR;
}

void ARQ3::inheritIDEParams(int dst, int src) {
    if (dst < 0 || src < 0) return;
    if (dst >= (int)CBF_.size()  || src >= (int)CBF_.size())  return;
    if (dst >= (int)CBCR_.size() || src >= (int)CBCR_.size()) return;
    CBF_[dst]  = CBF_[src];
    CBCR_[dst] = CBCR_[src];
}

// ============================================================================
// Restricted Tournament Replacement.  Adaptive pool size.
// ============================================================================
bool ARQ3::selectionRTR(int parentIndex, const Vec& u, double fu,
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
    // Dynamic pool: baseline rtr_pool_, but capped by N-1 and by rtr_pool_frac_*N
    int pool = rtr_pool_;
    int fcap = (int)std::round(rtr_pool_frac_ * (double)N);
    if (pool < fcap) pool = fcap;
    if (pool > N - 1) pool = N - 1;
    if (pool < 2)    pool = std::min(2, std::max(1, N - 1));

    int qstar = -1;
    double bestD = std::numeric_limits<double>::infinity();
    for (int k = 0; k < pool; ++k) {
        int q = randInt(0, N - 1);
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

// ============================================================================
// jSO-style K(F) (time-varying attraction weight on pbest)
// ============================================================================
double ARQ3::computeK(double F) const {
    const double pr = progress01();
    if (pr < 0.2) return 0.7 * F;
    if (pr < 0.4) return 0.8 * F;
    return 1.2 * F;
}



// ============================================================================
// makeTrialARQ: current-to-pbest/1 with
//   - rank-biased r1,
//   - archive-aware r2,
//   - jSO K(F) weight,
//   - optional eigen-coordinate binomial crossover.
// ============================================================================
void ARQ3::makeTrialARQ(int i, const std::vector<int>& ord,
                        double F, double CR, Vec& u) {
    const int D = prob_->dimension();
    const int N = (int)X_.size();

    // pbest (uniform from top pcount)
    int pcount = std::max(2, (int)std::ceil(currentPbest() * (double)N));
    if (pcount > N) pcount = N;
    std::uniform_int_distribution<int> Ip(0, pcount - 1);
    int ipbest = ord[Ip(rng_)];
    const Vec& xpbest = X_[ipbest];

    // r1: rank-biased from current population (forbid i)
    int r1 = rankBasedPick(ord, i);

    // r2: either from archive (JADE-style) or from population (rank-biased, forbid i and r1)
    Vec r2v;
    bool useA = (!A_.empty()) && (randU() < 0.5);
    if (useA) {
        std::uniform_int_distribution<int> Ia(0, (int)A_.size() - 1);
        r2v = A_[Ia(rng_)];
    } else {
        int r2 = rankBasedPick(ord, i);
        int guard = 0;
        while (r2 == r1 && guard++ < 20) r2 = rankBasedPick(ord, i);
        r2v = X_[r2];
    }

    // Mutant
    const double K = computeK(F);
    Vec v(D);
    for (int j = 0; j < D; ++j)
        v[j] = X_[i][j] + K * (xpbest[j] - X_[i][j]) + F * (X_[r1][j] - r2v[j]);
    ensureBounds(v);

    // Crossover: eigen-space with probability p_eig_ (if basis is valid),
    // otherwise classical binomial.
    if (eig_valid_ && D >= eig_min_D_ && randU() < p_eig_) {
        eigenBinomialCrossover(D, X_[i], v, CR, u);
    } else {
        u = X_[i];
        int jr = randInt(0, D - 1);
        for (int j = 0; j < D; ++j) {
            if (randU() < CR || j == jr) u[j] = v[j];
        }
    }
    ensureBounds(u);
}

// ============================================================================
// ARQ step (one sweep)
// ============================================================================
void ARQ3::stepARQ() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 4) return;

    // Rebuild sorted order and (periodically) the eigen basis
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    if (!eig_valid_ || iters_since_eig_ >= eig_period_) {
        // recompute on sorted view: to keep code simple we physically sort once
        sortByFitness();
        ord.assign(N, 0);
        std::iota(ord.begin(), ord.end(), 0);
        recomputeEigenBasis();
    }
    ++iters_since_eig_;

    // Choose how many parents to try
    int m = std::max(1, (int)std::ceil(agent_fraction_ * (double)N));
    if (m > N) m = N;

    // Randomized parent order (but use ord for rank lookups)
    std::vector<int> parents = ord;
    std::shuffle(parents.begin(), parents.end(), rng_);
    if ((int)parents.size() > m) parents.resize(m);

    std::vector<double> SF, SCR, SG;
    int attempts = 0, successes = 0;

    for (int t = 0; t < m; ++t) {
        if (prob_->calls() >= max_evals_) break;
        int i = parents[t];

        double F, CR;
        sampleFCR(F, CR);

        Vec u(D, 0.0);
        makeTrialARQ(i, ord, F, CR, u);
        double fu = eval(u);
        ++attempts;

        if (selectionRTR(i, u, fu, F, CR, SF, SCR, SG)) {
            ++successes;
            if (fu < best_f_) {
                best_f_ = fu;
                best_x_ = u;
            }
        }
    }

    updateMemoryFromSuccess(SF, SCR, SG);
    banditRecord(0, successes, attempts);
}

// ============================================================================
// IDE step (EA4Eig IDE strategy, unchanged semantics)
// ============================================================================
void ARQ3::stepIDE() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 4) return;

    if (ide_progress_sync_) {
        int geff = std::max(1, (int)std::round(progress01() * (double)gmax_));
        g_ = std::max(g_ + 1, geff);
    } else {
        ++g_;
    }
    if (g_ > gmax_) g_ = gmax_;

    sortByFitness();
    std::vector<Vec> Q(N, Vec(D));
    std::vector<double> QF(N, std::numeric_limits<double>::infinity());

    double IDEps = 0.1 + 0.9 * std::pow(10.0, 5.0 * ((double)g_ / (double)gmax_ - 1.0));
    double SRT = (g_ < gt_) ? 0.0 : 0.1;

    for (int i = 0; i < N; ++i) {
        // four distinct indices != i
        std::vector<int> cand;
        cand.reserve(N - 1);
        for (int k = 0; k < N; ++k) if (k != i) cand.push_back(k);
        if ((int)cand.size() < 4) continue;
        // shuffle-pick 4
        for (int pass = 0; pass < 4; ++pass) {
            int r = randInt(pass, (int)cand.size() - 1);
            std::swap(cand[pass], cand[r]);
        }
        int o = cand[0], r1 = cand[1], r2 = cand[2], r3 = cand[3];

        const Vec& xo  = X_[o];
        const Vec* xr1ptr = nullptr;

        if (g_ <= gt_) {
            double probSup = 0.9 * IDEps;
            if (randU() < probSup) {
                int high_ind_S = std::max(2, (int)std::round(IDEps * N));
                if (high_ind_S > N) high_ind_S = N;
                int pick = randInt(0, high_ind_S - 1);
                xr1ptr = &X_[pick];
            } else {
                xr1ptr = &X_[r1];
            }
        } else {
            int high_ind_S = std::max(2, (int)std::round(IDEps * N));
            if (high_ind_S > N) high_ind_S = N;
            if (randU() < 0.5) {
                int pick = randInt(0, high_ind_S - 1);
                xr1ptr = &X_[pick];
            } else {
                xr1ptr = &X_[r1];
            }
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

    // Binomial crossover (axis-aligned, per IDE spec)
    for (int i = 0; i < N; ++i) {
        Vec y = X_[i];
        const Vec& v = Q[i];
        double CR = CBCR_[i];
        int jrand = randInt(0, D - 1);
        for (int j = 0; j < D; ++j) {
            if (randU() < CR || j == jrand) y[j] = v[j];
        }
        ensureBounds(y);
        Q[i] = y;
    }

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;
        QF[i] = eval(Q[i]);
    }

    std::vector<int> indsucc;
    for (int i = 0; i < N; ++i) {
        bool improved = ide_strict_improve_ ? (QF[i] < FX_[i]) : (QF[i] <= FX_[i]);
        if (improved) indsucc.push_back(i);
    }

    double SR = (N > 0) ? ((double)indsucc.size() / (double)N) : 0.0;
    if (g_ < gt_) {
        if (SR <= SRT) ++Tcurr_;
        else Tcurr_ = 0;
        if ((double)Tcurr_ >= T_) gt_ = g_;
    }

    for (int idx : indsucc) {
        X_[idx]  = Q[idx];
        FX_[idx] = QF[idx];
        if (FX_[idx] < best_f_) {
            best_f_ = FX_[idx];
            best_x_ = X_[idx];
        }
    }

    banditRecord(1, (int)indsucc.size(), std::max(1, N));

    // IDE reshuffles ordering; force eigen recomputation
    eig_valid_ = false;
    iters_since_eig_ = 0;

    sortByFitness();
}

// ============================================================================
// Quarantine with Levy-flight perturbation (replaces Gaussian).
// Targets IQR outliers and perturbs them around the top-half centroid.
// ============================================================================
void ARQ3::quarantineLevy() {
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N < 4) return;

    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<double> fits = FX_;
    double Q1 = quantile(fits, 0.25);
    fits = FX_;
    double Q3 = quantile(fits, 0.75);
    double IQR = Q3 - Q1;
    double theta = Q3 + outlier_alpha_ * IQR;

    // Centroid of top half
    const int half = std::max(1, N / 2);
    const int D = prob_->dimension();
    Vec center(D, 0.0);
    for (int k = 0; k < half; ++k) {
        const Vec& x = X_[ord[k]];
        for (int j = 0; j < D; ++j) center[j] += x[j];
    }
    for (int j = 0; j < D; ++j) center[j] /= (double)half;

    // Identify outliers
    std::vector<int> out;
    for (int i = 0; i < N; ++i) if (FX_[i] >= theta) out.push_back(i);

    if (out.empty()) return;
    int k = (int)std::floor(outlier_rho_ * (double)out.size());
    if (k <= 0) return;

    std::shuffle(out.begin(), out.end(), rng_);
    out.resize(k);

    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int idx : out) {
        if (prob_->calls() >= max_evals_) break;

        Vec cand = center;
        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            double range = (hi - lo);
            double step = sampleLevy();
            cand[j] += qscale_ * range * step;
        }
        ensureBounds(cand);
        double fc = eval(cand);

        if (fc < FX_[idx]) {
            archivePush(X_[idx]);
            X_[idx] = std::move(cand);
            FX_[idx] = fc;
            sampleIDEParamsAt(idx);
            if (fc < best_f_) {
                best_f_ = fc;
                best_x_ = X_[idx];
            }
        }
    }
}

// ============================================================================
// On-demand OBL basin escape.
// Triggered when (a) no_improve_ >= stag_trigger_, (b) population spread has
// collapsed below var_collapse_ratio_, and (c) cooldown elapsed.
// Replaces worst obl_frac_ fraction with opposite points if they're better.
// ============================================================================
void ARQ3::oblBasinEscape() {
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N < 4) return;
    if (obl_cooldown_ > 0) { --obl_cooldown_; return; }

    bool stag = (no_improve_ >= stag_trigger_);
    bool collapsed = (normalizedPopSpread() < var_collapse_ratio_);
    if (!(stag && collapsed)) return;

    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    int count = std::max(1, (int)std::floor(obl_frac_ * (double)N));
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    const int D = prob_->dimension();

    int applied = 0;
    for (int t = 0; t < count; ++t) {
        if (prob_->calls() >= max_evals_) break;
        int idx = ord[N - 1 - t];

        Vec cand(D);
        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            // Quasi-opposition: midpoint between best and opposite-of-x.
            double opp = lo + hi - X_[idx][j];
            double mid = best_x_[j];
            // 50/50 mix: pure opposition vs. opposition toward best (quasi).
            if (randU() < 0.5) cand[j] = opp;
            else cand[j] = mid + (opp - mid) * randU();
        }
        ensureBounds(cand);
        double fc = eval(cand);
        ++applied;

        if (fc < FX_[idx]) {
            archivePush(X_[idx]);
            X_[idx] = std::move(cand);
            FX_[idx] = fc;
            sampleIDEParamsAt(idx);
            if (fc < best_f_) {
                best_f_ = fc;
                best_x_ = X_[idx];
            }
        }
    }

    if (applied > 0) {
        obl_cooldown_ = obl_cooldown_init_;
        no_improve_ = 0;
        eig_valid_ = false;
        iters_since_eig_ = 0;
    }
}

// ============================================================================
// one_iteration (main loop body)
// ============================================================================
void ARQ3::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if ((int)X_.size() < 4) return;

    archiveTrim((int)X_.size());

    // Strategy selection: warmup ARQ, then Thompson sampling
    int hh = 0;
    if (bootstrap_left_ > 0) {
        hh = 0;
        --bootstrap_left_;
    } else {
        hh = thompsonPick();
    }

    switch (hh) {
        case 1:
            stepIDE();
            break;
        case 0:
        default:
            stepARQ();
            quarantineLevy();
            break;
    }

    // Stagnation tracking (applies to both strategies)
    if (best_f_ < best_prev_ - 1e-18) {
        best_prev_ = best_f_;
        no_improve_ = 0;
    } else {
        ++no_improve_;
    }

    // On-demand OBL (only after enough stagnation + variance collapse)
    oblBasinEscape();

    // NLPSR shrink (end of iteration so all indices above are still valid)
    int Ntarget = targetPopulationSize();
    if (Ntarget < (int)X_.size()) {
        shrinkTo(Ntarget);
    }

    // Bandit book-keeping: decay posteriors (non-stationary world)
    banditDecay();

    archiveTrim((int)X_.size());
    updateStop(FX_);
    printBest();

    if (debug_ && ((int)prob_->calls() % 1000) == 0) {
        std::fprintf(stdout,
            "[arq3] calls=%d N=%d best=%.6g noimpr=%d banditA=(%.2f,%.2f) B=(%.2f,%.2f) spread=%.3g\n",
            (int)prob_->calls(), (int)X_.size(), best_f_, no_improve_,
            bandit_a_[0], bandit_b_[0],
            bandit_a_[1], bandit_b_[1],
            normalizedPopSpread());
        std::fflush(stdout);
    }
}

} // namespace optimsolution
