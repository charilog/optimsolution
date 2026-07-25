#include "sparq.h"
#include <cstdio>
#include <cctype>


namespace optimsolution {

// ============================================================================
// configure
// ============================================================================
void SPARQ::configure(const MethodConfig& mc) {
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
    agent_fraction_Dthreshold_ = mc.getDbl("agentfractiondthreshold", agent_fraction_Dthreshold_);
    polish_progress_trig_  = mc.getDbl("polishprogresstrig",  polish_progress_trig_);
    polish_progress_burst_ = mc.getDbl("polishprogressburst", polish_progress_burst_);
    rejuv_progress_cutoff_ = mc.getDbl("rejuvprogresscutoff", rejuv_progress_cutoff_);

    // IDE flags
    ide_progress_sync_    = mc.getInt("ide_progress_sync",  ide_progress_sync_);
    ide_strict_improve_   = mc.getInt("ide_strict_improve", ide_strict_improve_);

    cr_sort_              = mc.getInt("crsort",             cr_sort_);
    rejuv_weak_streak_limit_  = mc.getInt("rejuvweaklimit",   rejuv_weak_streak_limit_);
    rejuv_strong_fail_limit_  = mc.getInt("rejuvstronglimit", rejuv_strong_fail_limit_);
    polish_trigger_       = mc.getInt("polishtrigger",      polish_trigger_);
    polish_frac_          = mc.getDbl("polishfrac",         polish_frac_);
    polish_budget_        = mc.getDbl("polishbudget",       polish_budget_);
    rejuv_factor_         = mc.getInt("rejuvfactor",        rejuv_factor_);
    rejuv_keep_           = mc.getDbl("rejuvkeep",          rejuv_keep_);
    rejuv_cooldown_init_  = mc.getInt("rejuvcooldown",      rejuv_cooldown_init_);

    debug_                = mc.getInt("debug_arq", debug_);

    // Ablation-study switches (see sparq.h for per-flag semantics).
    enable_ide_     = mc.getInt("enable_ide",     enable_ide_)     ? 1 : 0;
    enable_levy_    = mc.getInt("enable_levy",    enable_levy_)    ? 1 : 0;
    enable_polish_  = mc.getInt("enable_polish",  enable_polish_)  ? 1 : 0;
    enable_obl_     = mc.getInt("enable_obl",     enable_obl_)     ? 1 : 0;
    enable_rejuv_   = mc.getInt("enable_rejuv",   enable_rejuv_)   ? 1 : 0;
    enable_eigen_   = mc.getInt("enable_eigen",   enable_eigen_)   ? 1 : 0;
    enable_rtr_     = mc.getInt("enable_rtr",     enable_rtr_)     ? 1 : 0;
    enable_archive_ = mc.getInt("enable_archive", enable_archive_) ? 1 : 0;
    enable_nlpsr_   = mc.getInt("enable_nlpsr",   enable_nlpsr_)   ? 1 : 0;
    enable_echo_    = mc.getInt("enable_echo",    enable_echo_)    ? 1 : 0;
    enable_shade_   = mc.getInt("enable_shade",   enable_shade_)   ? 1 : 0;

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
    if (agent_fraction_Dthreshold_ < 0.0) agent_fraction_Dthreshold_ = 0.0;
    if (polish_progress_trig_  < 0.0 || polish_progress_trig_  > 1.0) polish_progress_trig_  = 0.75;
    if (polish_progress_burst_ < 0.0 || polish_progress_burst_ > 1.0) polish_progress_burst_ = 0.80;
    if (rejuv_progress_cutoff_ < 0.0 || rejuv_progress_cutoff_ > 1.0) rejuv_progress_cutoff_ = 0.90;
    ide_progress_sync_  = ide_progress_sync_  ? 1 : 0;
    ide_strict_improve_ = ide_strict_improve_ ? 1 : 0;
    cr_sort_ = cr_sort_ ? 1 : 0;
    if (polish_trigger_ < 1) polish_trigger_ = 1;
    if (polish_frac_ <= 0.0 || polish_frac_ > 1.0) polish_frac_ = 0.10;
    if (polish_budget_ < 0.0) polish_budget_ = 0.0;
    if (polish_budget_ > 0.5) polish_budget_ = 0.5;
    if (rejuv_factor_ < 2) rejuv_factor_ = 2;
    if (rejuv_keep_ <= 0.0 || rejuv_keep_ >= 1.0) rejuv_keep_ = 0.25;
    if (rejuv_weak_streak_limit_ < 1) rejuv_weak_streak_limit_ = 1;
    if (rejuv_strong_fail_limit_ < 1) rejuv_strong_fail_limit_ = 1;
    if (rejuv_cooldown_init_ < 0) rejuv_cooldown_init_ = 0;
}

// ============================================================================
// init
// ============================================================================
void SPARQ::init() {
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
    polish_stag_mark_f_ = best_f_;
    polish_stag_count_  = 0;

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

    // UPGRADE state
    ps_sigma_ = 0.02;
    ps_sigma_c_ = 0.10;
    rejuv_cooldown_ = 0;
    rejuv_watch_f_ = std::numeric_limits<double>::infinity();
    rejuv_weak_streak_ = 0;
    rejuv_strong_fail_streak_ = 0;
    rejuv_strong_watch_f_ = std::numeric_limits<double>::infinity();
    polish_cooldown_ = 0;
    polish_backoff_ = 0;
    polish_used_ = 0;
    polish_low_streak_ = 0;
    polish_disabled_ = false;
    polish_mark_f_ = std::numeric_limits<double>::infinity();
    polish_mark_calls_ = 0;
    polish_coord_ptr_ = 0;
    sw_bias_.assign(D, 0.0);
    sw_rho_ = 0.02;
    echo_steps_.assign(kEchoCapacity, Vec());
    echo_ptr_ = 0;
    echo_count_ = 0;
    echo_scale_ = 1.0;
    echo_fail_streak_ = 0;
    echo_disabled_ = false;

    if (debug_) {
        std::fprintf(stdout,
            "[sparq] cfg -> Ninit=%d Nmin=%d D=%d H=%d alphaNL=%.2f "
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
int SPARQ::pickDistinct(int n, int a, int b, int c) {
    std::uniform_int_distribution<int> I(0, n - 1);
    int r;
    do { r = I(rng_); } while (r == a || r == b || r == c);
    return r;
}

int SPARQ::randInt(int lo, int hi) {
    if (hi < lo) std::swap(lo, hi);
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng_);
}

double SPARQ::randU() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

double SPARQ::cauchy(double loc, double scale) {
    std::cauchy_distribution<double> dist(loc, scale);
    return dist(rng_);
}

double SPARQ::gaussN(double mu, double sig) {
    std::normal_distribution<double> dist(mu, sig);
    return dist(rng_);
}

// Mantegna 1994 Levy step (symmetric alpha-stable approximation).
// Returns a centered Levy-distributed scalar.  Caller must scale by range.
double SPARQ::sampleLevy() {
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

double SPARQ::progress01() const {
    if (!prob_ || max_evals_ <= 0) return 1.0;
    double p = (double)prob_->calls() / (double)max_evals_;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    return p;
}

void SPARQ::ensureBounds(Vec& v) {
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

double SPARQ::distBN(const Vec& a, const Vec& b) const {
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

double SPARQ::quantile(std::vector<double> v, double q01) {
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

double SPARQ::normalizedPopSpread() const {
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

void SPARQ::sortByFitness() {
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
int SPARQ::targetPopulationSize() const {
    double p = progress01();
    // FIX (logic): the old code used pow(p, alpha) with alpha = 0.5, which
    // contradicts both the documented semantics ("1.0 = linear, <1 = slower
    // shrink early, faster near the end") and the reference it names
    // (NL-SHADE-RSP, whose reduction follows p^(1-p)): being concave, it had
    // completed 22% of the reduction at 5% of the budget and 50% at 25% —
    // i.e. it collapsed the population early instead of late. The exponent is
    // now interpolated from 1 (at p=0) down to alpha (at p=1):
    //   alpha == 1 -> exponent 1 -> exactly linear (as documented);
    //   alpha  < 1 -> near-linear early, accelerating shrink near the end,
    // which for the default alpha = 0.5 tracks NL-SHADE-RSP's p^(1-p) curve
    // closely in the critical early phase (0.054 vs 0.058 at p = 0.05).
    const double expo = 1.0 - (1.0 - nlpsr_alpha_) * p;
    double frac = std::pow(p, expo);
    double N = (double)Ninit_ + ((double)Nmin_ - (double)Ninit_) * frac;
    int Ni = (int)std::round(N);
    if (Ni < Nmin_) Ni = Nmin_;
    if (Ni > Ninit_) Ni = Ninit_;
    return Ni;
}

// Shrink population to N by removing worst individuals.  Keeps per-individual
// IDE parameters aligned.  Does NOT call eval.
void SPARQ::shrinkTo(int Ntarget) {
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
double SPARQ::currentKR() const {
    double pr = progress01();
    return kr_init_ + (kr_final_ - kr_init_) * pr;
}

double SPARQ::currentPbest() const {
    double pr = progress01();
    // linear from pbest_max_ down to pbest_min_
    return pbest_max_ + (pbest_min_ - pbest_max_) * pr;
}

// ============================================================================
// LSHADE-RSP rank-biased picking: weight index r (rank 0 = best) by (N-r)^kr
// ============================================================================
int SPARQ::rankBasedPick(const std::vector<int>& ord, int forbid) const {
    const int N = (int)ord.size();
    if (N <= 0) return forbid;
    if (N == 1) return ord[0];

    auto* self = const_cast<SPARQ*>(this);
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
void SPARQ::initMemory() {
    MF_.assign(H_,  0.5);
    MCR_.assign(H_, 0.5);
    if (H_ >= 1) {
        MF_[H_ - 1]  = MF_terminal_;
        MCR_[H_ - 1] = MCR_terminal_;
    }
    memK_ = 0;
}

void SPARQ::sampleFCR(double& F, double& CR) {
    // random memory slot (including terminal)
    int r = randInt(0, H_ - 1);
    double muF  = MF_[r];
    double muCR = MCR_[r];

    // Ablation: enable_shade=0 removes the success-history ADAPTATION only
    // -- F/CR are sampled around fixed classical-DE means 0.5/0.5 with the
    // exact same sampling machinery (Cauchy/Gaussian draws, clamps, jSO
    // schedules) so the memory's learning is the sole thing switched off.
    if (!enable_shade_) { muF = 0.5; muCR = 0.5; }

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

void SPARQ::updateMemoryFromSuccess(const std::vector<double>& SF,
                                   const std::vector<double>& SCR,
                                   const std::vector<double>& SG) {
    if (!enable_shade_) return; // ablation: memory never learns
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
void SPARQ::archivePush(const Vec& x) {
    // Ablation: with enable_archive=0 nothing is ever pushed, so A_ stays
    // empty for the whole run -- makeTrialARQ's own `useA = !A_.empty() &&
    // ...` guard then automatically draws r2 from the population instead,
    // with no further special-casing needed anywhere else.
    if (!enable_archive_) return;
    A_.push_back(x);
}

void SPARQ::archiveTrim(int N) {
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
void SPARQ::jacobiEigen(const Mat& Ain, Mat& V, std::vector<double>& w) const {
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

void SPARQ::recomputeEigenBasis() {
    if (!prob_) { eig_valid_ = false; return; }
    const int D = prob_->dimension();
    // FIX (omission): the Jacobi decomposition is O(D^3) per sweep with up to
    // 50 sweeps; without a dimensionality cap a single basis refresh at
    // D = 200 costs ~10^10 operations and dominates the entire run. EA4Eig
    // caps the eigen machinery at D <= 100 for exactly this reason; above the
    // cap the crossover silently falls back to classical binomial.
    if (D > 100) { eig_valid_ = false; return; }
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
    // UPGRADE (B'): keep per-axis sqrt(eigenvalue) scales so the elite polish
    // can sample ANISOTROPIC Gaussian steps aligned with the population shape
    // (long steps along the valley, short across it).
    eig_scale_.assign(D, 1.0);
    double wmax = 0.0;
    for (double v : w) wmax = std::max(wmax, std::fabs(v));
    if (wmax > 0.0) {
        for (int j = 0; j < D; ++j) {
            double s = std::sqrt(std::max(0.0, w[j]) / wmax);
            eig_scale_[j] = std::min(1.0, std::max(0.05, s));
        }
    }
    eig_valid_ = true;
    iters_since_eig_ = 0;
}

// out = B^T x
void SPARQ::applyBt(const Mat& B, const Vec& x, Vec& out) const {
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
void SPARQ::applyB(const Mat& B, const Vec& x, Vec& out) const {
    const int D = (int)x.size();
    out.assign(D, 0.0);
    for (int i = 0; i < D; ++i) {
        double acc = 0.0;
        for (int j = 0; j < D; ++j) acc += B[i][j] * x[j];
        out[i] = acc;
    }
}

void SPARQ::eigenBinomialCrossover(int D, const Vec& base, const Vec& v,
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
int SPARQ::thompsonPick() {
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

void SPARQ::banditDecay() {
    // Non-stationary forgetting: decay toward uniform prior (1,1).
    for (int k = 0; k < h_; ++k) {
        bandit_a_[k] = 1.0 + (bandit_a_[k] - 1.0) * bandit_decay_;
        bandit_b_[k] = 1.0 + (bandit_b_[k] - 1.0) * bandit_decay_;
    }
}

void SPARQ::banditRecord(int k, int successes, int attempts) {
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
void SPARQ::sampleIDEParamsAt(int idx) {
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

void SPARQ::inheritIDEParams(int dst, int src) {
    if (dst < 0 || src < 0) return;
    if (dst >= (int)CBF_.size()  || src >= (int)CBF_.size())  return;
    if (dst >= (int)CBCR_.size() || src >= (int)CBCR_.size()) return;
    CBF_[dst]  = CBF_[src];
    CBCR_[dst] = CBCR_[src];
}

// ============================================================================
// Restricted Tournament Replacement.  Adaptive pool size.
// ============================================================================
// Push the accepted displacement (to - from) into the fixed-size ring
// buffer. O(D) cost, no allocation after the first fill (assign() below
// only reuses/resizes the existing Vec slot).
void SPARQ::recordEchoStep(const Vec& from, const Vec& to) {
    // Ablation: enable_echo=0 means no step is ever recorded, so
    // echo_count_ stays 0 for the whole run and the polish's echo probe
    // mode (gated on echo_count_ >= 1) can never fire -- one central
    // guard covers every recording site and the consumption site alike.
    if (!enable_echo_) return;
    if (echo_steps_.empty()) return;
    const int D = (int)from.size();
    Vec& slot = echo_steps_[echo_ptr_];
    slot.resize(D);
    for (int j = 0; j < D; ++j)
        slot[j] = to[j] - from[j];
    echo_ptr_ = (echo_ptr_ + 1) % kEchoCapacity;
    if (echo_count_ < kEchoCapacity) ++echo_count_;
}

bool SPARQ::selectionRTR(int parentIndex, const Vec& u, double fu,
                        double F, double CR,
                        std::vector<double>& SF,
                        std::vector<double>& SCR,
                        std::vector<double>& SG) {
    if (fu < FX_[parentIndex]) {
        double gain = FX_[parentIndex] - fu;
        recordEchoStep(X_[parentIndex], u);
        archivePush(X_[parentIndex]);
        X_[parentIndex] = u;
        FX_[parentIndex] = fu;
        SF.push_back(F);
        SCR.push_back(CR);
        SG.push_back(gain);
        return true;
    }
    // Ablation: the RTR mechanism proper is the nearest-neighbour
    // replacement fallback below; the branch above is plain greedy parent
    // replacement. enable_rtr=0 therefore stops here -- selection becomes
    // exactly classical DE greedy selection, with identical
    // success-stat/echo/archive bookkeeping on acceptance.
    if (!enable_rtr_) return false;
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
        // FIX (inconsistency): the parent must be excluded from the RTR pool.
        // The greedy parent comparison has already failed in the first branch
        // (fu >= FX_[parent]); since the offspring is typically closest to its
        // own parent, drawing the parent into the pool made it the nearest
        // element most of the time and the fallback silently re-tested a known
        // failure instead of giving the trial a chance against the truly
        // nearest OTHER individual — defeating the purpose of RTR.
        int q = randInt(0, N - 1);
        if (q == parentIndex) continue;
        double d = distBN(u, X_[q]);
        if (d < bestD) { bestD = d; qstar = q; }
    }
    if (qstar < 0) return false;
    if (fu < FX_[qstar]) {
        double gain = FX_[qstar] - fu;
        recordEchoStep(X_[qstar], u);
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
double SPARQ::computeK(double F) const {
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
void SPARQ::makeTrialARQ(int i, const std::vector<int>& ord,
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
    // otherwise classical binomial. Ablation: enable_eigen=0 forces the
    // classical binomial path unconditionally.
    if (enable_eigen_ && eig_valid_ && D >= eig_min_D_ && randU() < p_eig_) {
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
void SPARQ::stepARQ() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 4) return;

    // Rebuild sorted order and (periodically) the eigen basis
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    if (enable_eigen_ && (!eig_valid_ || iters_since_eig_ >= eig_period_)) {
        // recompute on sorted view: to keep code simple we physically sort once
        sortByFitness();
        ord.assign(N, 0);
        std::iota(ord.begin(), ord.end(), 0);
        recomputeEigenBasis();
    }
    ++iters_since_eig_;

    // Choose how many parents to try (D-dependent agentfraction, see header
    // comment on effectiveAgentFraction()).
    int m = std::max(1, (int)std::ceil(effectiveAgentFraction(D) * (double)N));
    if (m > N) m = N;

    // Randomized parent order (but use ord for rank lookups)
    std::vector<int> parents = ord;
    std::shuffle(parents.begin(), parents.end(), rng_);
    if ((int)parents.size() > m) parents.resize(m);

    std::vector<double> SF, SCR, SG;
    int attempts = 0, successes = 0;

    // UPGRADE (A) — NL-SHADE-RSP CR sorting: pre-sample one (F,CR) pair per
    // selected parent, then reassign the CR values so that better-ranked
    // parents receive SMALLER crossover rates (fine, conservative moves near
    // the top of the population; aggressive recombination at the bottom).
    // This is the ingredient that gave NL-SHADE-LBC its consistency edge on
    // rugged landscapes; F stays paired with its memory draw.
    std::vector<double> Fs(m), CRs(m);
    for (int t = 0; t < m; ++t) sampleFCR(Fs[t], CRs[t]);
    if (cr_sort_) {
        std::vector<int> rankpos(N, 0);
        for (int rpos = 0; rpos < N; ++rpos) rankpos[ord[rpos]] = rpos;
        // rank order of the selected parents (best first)
        std::vector<int> psel(m);
        std::iota(psel.begin(), psel.end(), 0);
        std::sort(psel.begin(), psel.end(),
                  [&](int a, int b){ return rankpos[parents[a]] < rankpos[parents[b]]; });
        std::vector<double> crs_sorted = CRs;
        std::sort(crs_sorted.begin(), crs_sorted.end());
        std::vector<double> CRas(m);
        for (int q = 0; q < m; ++q) CRas[psel[q]] = crs_sorted[q];
        CRs.swap(CRas);
    }

    for (int t = 0; t < m; ++t) {
        if (prob_->calls() >= max_evals_) break;
        int i = parents[t];

        double F = Fs[t], CR = CRs[t];

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
void SPARQ::stepIDE() {
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

    int evaluated = 0;
    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;
        QF[i] = eval(Q[i]);
        ++evaluated;
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
        recordEchoStep(X_[idx], Q[idx]);
        X_[idx]  = Q[idx];
        FX_[idx] = QF[idx];
        if (FX_[idx] < best_f_) {
            best_f_ = FX_[idx];
            best_x_ = X_[idx];
        }
    }

    // FIX (inconsistency): the bandit's "attempts" for IDE must count the
    // trials that were actually EVALUATED, exactly as stepARQ counts them.
    // Charging a fixed N even when the budget cut the evaluation loop short
    // deflated IDE's apparent success rate relative to ARQ's.
    banditRecord(1, (int)indsucc.size(), std::max(1, evaluated));

    // IDE reshuffles ordering; force eigen recomputation
    eig_valid_ = false;
    iters_since_eig_ = 0;

    sortByFitness();
}

// ============================================================================
// Quarantine with Levy-flight perturbation (replaces Gaussian).
// Targets IQR outliers and perturbs them around the top-half centroid.
// ============================================================================
void SPARQ::quarantineLevy() {
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
void SPARQ::oblBasinEscape() {
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
// UPGRADE (B): stagnation-gated elite (1+1)-ES polish with 1/5 success rule.
// Deep exploitation around the incumbent best: a short burst of Gaussian
// trials with a self-adaptive step size (success -> sigma up, failure ->
// sigma down; ~1/5 rule). Fired only while the main loop is not improving, so
// it costs nothing when evolution is making progress. Improvements are also
// written into the population's worst slot so they propagate.
// ============================================================================
void SPARQ::elitePolish() {
    if (!prob_ || best_x_.empty() || polish_disabled_) return;
    const double f_before = best_f_;
    // Efficiency of the evolutionary loop since the last polish activation
    // (gain per evaluation); the polish must beat this to keep its budget.
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
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    // Burst budget: proportional to N, but never trivial; terminate early
    // after a streak of failures so budget is not wasted when nothing works.
    int k = std::max(6, (int)std::round(polish_frac_ * (double)N));
    // In the exploitation tail the burst scales with the DIMENSION: the
    // round-robin coordinate sweep must be able to cover the space within a
    // reasonable number of bursts (a 12-eval burst covers ~4 coordinates —
    // hopeless at D = 200). The global 8%-of-budget cap still applies.
    if (progress01() > polish_progress_burst_) k = std::max(k, std::min(std::max(12, D), 64));
    int fail_streak = 0;
    int wins = 0;

    for (int t = 0; t < k; ++t) {
        if (prob_->calls() >= max_evals_) return;
        if (fail_streak >= 10) break;
        // Hard budget cap: the polish may never consume more than
        // polish_budget_ of the evaluations spent so far. On landscapes where
        // refinement is cheapily productive this cap is never binding; on
        // far-basin landscapes it guarantees the evolutionary loop keeps
        // >= (1 - polish_budget_) of the budget no matter what.
        if ((double)polish_used_ >
            polish_budget_ * (double)std::max(1, (int)prob_->calls())) break;

        Vec cand = best_x_;
        const double mode = randU();
        if (mode < 0.30) {
            // (i) single-coordinate probe: repairs "one dimension off" local
            // minima (Rastrigin/Styblinski/Tersoff-type lattices) one axis at
            // a time. Half the probes are COORDINATE OPPOSITIONS
            // (x_j -> lo+hi-x_j), which jump directly to the mirror basin of
            // bistable dimensions; the other half are adaptive Gaussian steps.
            // Round-robin sweep guarantees every coordinate is visited, which
            // matters in high dimension (random picks leave dimensions
            // uncovered for a long time at D = 200).
            int j = polish_coord_ptr_;
            polish_coord_ptr_ = (polish_coord_ptr_ + 1) % D;
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            if (randU() < 0.5) cand[j] = lo + hi - cand[j];
            else cand[j] += ps_sigma_c_ * (hi - lo) * gaussN(0.0, 1.0);
        } else if (mode < 0.65 && (int)eig_scale_.size() == D
                   && (int)B_rot_.size() == D) {
            // NOTE: the polish deliberately uses the LAST KNOWN basis even
            // when eig_valid_ is false (late populations of size < D+2 cannot
            // refresh it). A slightly stale rotation is far better than
            // axis-aligned probing on rotated landscapes.
            // (ii) anisotropic full-D step aligned with the population shape
            Vec z(D);
            for (int j = 0; j < D; ++j)
                z[j] = ps_sigma_ * eig_scale_[j] * gaussN(0.0, 1.0);
            Vec step;
            applyB(B_rot_, z, step);
            for (int j = 0; j < D; ++j) {
                double lo = (j < (int)L.size() ? L[j] : -1.0);
                double hi = (j < (int)U.size() ? U[j] :  1.0);
                if (lo > hi) std::swap(lo, hi);
                cand[j] += (hi - lo) * step[j];
            }
        } else if (mode < 0.80 && echo_count_ >= 1 && !echo_disabled_ && D <= 150) {
            // (iii) INNOVATION: "Trajectory Echo" — see the header comment
            // on echo_steps_ for the full rationale. Draw a random linear
            // combination of the DE core's own recent successful step
            // directions and apply it directly (these are already real
            // displacement vectors in the problem's own units, not
            // fractions of the box range like the other probe modes).
            // Normalizing each coefficient by 1/sqrt(echo_count_) keeps the
            // combined step's expected magnitude comparable to a single
            // one of its ingredients, regardless of how many are buffered.
            Vec step(D, 0.0);
            const double norm = 1.0 / std::sqrt((double)echo_count_);
            for (int k = 0; k < echo_count_; ++k) {
                if ((int)echo_steps_[k].size() != D) continue; // safety guard
                const double c = norm * gaussN(0.0, 1.0);
                for (int j = 0; j < D; ++j)
                    cand[j] += echo_scale_ * c * echo_steps_[k][j];
            }
        } else {
            // (iv) INNOVATION: Solis-Wets adaptive random search with
            // directional bias and reflection, replacing the old memoryless
            // isotropic probe. Draw one random step; try it in the "biased"
            // direction first, and if that fails, try the exact REFLECTION
            // (same random draw, opposite sign) before giving up — a
            // successful reflection means the accumulated bias was pointing
            // the wrong way, and the update below corrects it immediately.
            // This never touches or blends any OTHER individual's
            // information; it only ever proposes points along a single
            // trajectory anchored at the current incumbent, so it cannot
            // create the structurally-incoherent "averaged" candidates that
            // made population-recombination-based operators unsafe on
            // permutation-symmetric cluster problems.
            Vec delta(D);
            for (int j = 0; j < D; ++j) {
                double lo = (j < (int)L.size() ? L[j] : -1.0);
                double hi = (j < (int)U.size() ? U[j] :  1.0);
                if (lo > hi) std::swap(lo, hi);
                delta[j] = sw_rho_ * (hi - lo) * gaussN(0.0, 1.0);
            }
            Vec plus(D), minus(D);
            for (int j = 0; j < D; ++j) {
                plus[j]  = best_x_[j] + sw_bias_[j] + delta[j];
                minus[j] = best_x_[j] - sw_bias_[j] - delta[j];
            }
            ensureBounds(plus);
            double fplus = eval(plus);
            ++polish_used_;
            if (fplus < best_f_) {
                best_f_ = fplus;
                best_x_ = plus;
                for (int j = 0; j < D; ++j)
                    sw_bias_[j] = 0.2 * delta[j] + 0.4 * sw_bias_[j];
                sw_rho_ = std::min(ps_sigma_max_, sw_rho_ * 1.5);
                fail_streak = 0;
                ++wins;
                int worst = 0;
                for (int i = 1; i < N; ++i) if (FX_[i] > FX_[worst]) worst = i;
                archivePush(X_[worst]);
                X_[worst]  = plus;
                FX_[worst] = fplus;
            } else if (prob_->calls() < max_evals_ &&
                       (double)polish_used_ <=
                           polish_budget_ * (double)std::max(1, (int)prob_->calls())) {
                ensureBounds(minus);
                double fminus = eval(minus);
                ++polish_used_;
                if (fminus < best_f_) {
                    best_f_ = fminus;
                    best_x_ = minus;
                    for (int j = 0; j < D; ++j)
                        sw_bias_[j] = 0.6 * sw_bias_[j] - 0.2 * delta[j];
                    fail_streak = 0;
                    ++wins;
                    int worst = 0;
                    for (int i = 1; i < N; ++i) if (FX_[i] > FX_[worst]) worst = i;
                    archivePush(X_[worst]);
                    X_[worst]  = minus;
                    FX_[worst] = fminus;
                } else {
                    // Both directions failed: the bias was uninformative
                    // here — decay it toward zero (classic Solis-Wets rule)
                    // and contract the step, concentrating the next attempts
                    // instead of continuing to wander.
                    for (int j = 0; j < D; ++j) sw_bias_[j] *= 0.5;
                    sw_rho_ = std::max(ps_sigma_min_, sw_rho_ * 0.87);
                    ++fail_streak;
                }
            } else {
                for (int j = 0; j < D; ++j) sw_bias_[j] *= 0.5;
                sw_rho_ = std::max(ps_sigma_min_, sw_rho_ * 0.87);
                ++fail_streak;
            }
            continue;   // this branch already evaluated/injected/adapted directly
        }
        ensureBounds(cand);
        double fc = eval(cand);
        ++polish_used_;
        if (fc < best_f_) {
            best_f_ = fc;
            best_x_ = cand;
            fail_streak = 0;
            ++wins;
            if (mode < 0.30)      ps_sigma_c_  *= 1.5;
            else if (mode < 0.65) ps_sigma_    *= 1.5;
            else                  { echo_scale_ *= 1.5; echo_fail_streak_ = 0; }
            // Injection: the improvement replaces the population's current
            // worst. This (a) propagates the corrected structure into the
            // mutation pool and (b) keeps similarity-based stopping rules
            // (doublebox) alive for as long as the polish keeps producing
            // real progress — without it, a converged population terminates
            // the run while refinement is still paying off. Diversity damage
            // is bounded by the polish budget cap and the efficiency
            // arbitration below.
            int worst = 0;
            for (int i = 1; i < N; ++i) if (FX_[i] > FX_[worst]) worst = i;
            archivePush(X_[worst]);
            X_[worst]  = cand;
            FX_[worst] = fc;
        } else {
            ++fail_streak;
            if (mode < 0.30)      ps_sigma_c_ *= 0.90;
            else if (mode < 0.65) ps_sigma_   *= 0.87;
            else {
                echo_scale_ *= 0.87;
                // Circuit breaker: 5 fruitless echo attempts in a row (out
                // of what is already a minority 15% probe share) is strong
                // enough evidence on a landscape this size that replaying
                // combined historical directions isn't paying off here —
                // stand down for the rest of the run rather than keep
                // spending budget (and risking harm) on it.
                if (++echo_fail_streak_ >= 5) echo_disabled_ = true;
            }
        }
        if (ps_sigma_  < ps_sigma_min_) ps_sigma_  = ps_sigma_min_;
        if (ps_sigma_  > ps_sigma_max_) ps_sigma_  = ps_sigma_max_;
        if (echo_scale_ < 0.05) echo_scale_ = 0.05;
        if (echo_scale_ > 4.0)  echo_scale_ = 4.0;
        if (ps_sigma_c_ < 1e-7) ps_sigma_c_ = 1e-7;
        if (ps_sigma_c_ > 0.5)  ps_sigma_c_ = 0.5;
    }

    // Exponential backoff: on landscapes where local refinement of the
    // incumbent is systematically useless (the next better basin is FAR away,
    // e.g. Schwefel-type), a fruitless activation doubles the cooldown, so
    // the eval budget flows back to the evolutionary loop. Any success resets
    // the backoff, so on refinement-friendly landscapes the polish stays hot.
    if (wins == 0) {
        polish_backoff_ = std::min(16, std::max(2, polish_backoff_ * 2));
        polish_cooldown_ = polish_backoff_;
    } else {
        polish_backoff_ = 0;
    }

    // Operator-efficiency arbitration: if this burst produced less gain per
    // evaluation than the evolutionary loop achieved since the last burst,
    // the evolution is the better investment — stand down for a while. This
    // is what shuts the polish out on far-basin landscapes (Schwefel-type),
    // where the tail evolution still makes large hops that dwarf any local
    // refinement, without any hand-tuned landscape detection.
    {
        const double pol_gain  = std::max(0.0, f_before - best_f_);
        const double pol_evals = (double)std::max<long long>(1, polish_used_ - used_before);
        const double pol_eff   = pol_gain / pol_evals;
        if (std::isfinite(evo_eff) && pol_eff < evo_eff) {
            // FIX: on problems where the DE core is already the more
            // productive investment (e.g. potential, polyphase — both
            // showed small but real Value/Mean gaps vs. plain arq3), a short
            // 32-iteration cooldown let polish come right back and keep
            // skimming a percent or two of budget the core could have used
            // instead. Doubling the penalty gives the core a longer,
            // uninterrupted stretch once polish has been measured to be the
            // worse investment, while genuinely-polish-friendly landscapes
            // (hydrothermal, fmsynth) are unaffected: their bursts keep
            // beating the core's own rate and never trigger this branch.
            polish_cooldown_ = std::max(polish_cooldown_, 64);
        }
    }
    polish_mark_f_     = best_f_;
    polish_mark_calls_ = (long long)prob_->calls();

    // Landscape self-selection: if the VALUE of the polish is negligible
    // (relative gain < polish_min_relgain_ for two consecutive activations),
    // switch it off permanently for this run. On far-basin landscapes
    // (Schwefel-type) the polish produces frequent-but-worthless micro-wins
    // that keep it "hot" while starving the evolutionary loop; measuring the
    // payoff instead of the win count shuts it down after ~2 bursts. On
    // refinement-friendly landscapes (Rastrigin lattices, narrow valleys,
    // molecular PES) the gains are orders of magnitude above the threshold.
    const double denom = std::max(1.0, std::fabs(f_before));
    const double relgain = (f_before - best_f_) / denom;
    if (relgain < polish_min_relgain_) {
        // A LONG cooldown instead of a permanent switch-off: with the
        // stagnation-driven activation the polish may fire early in the run,
        // where micro-gains are normal; a permanent disable there would rob
        // the exploitation tail of its refinement engine. The cooldown keeps
        // useless polishing shut out for a long stretch while always allowing
        // a fresh attempt later in the run.
        // Mild penalty: the round-robin coordinate sweep legitimately needs
        // many bursts to cover a high-dimensional space (at D = 200 only a
        // few probes per burst land on any given coordinate), so most bursts
        // are "fruitless" by construction while the sweep is mid-cycle. A
        // heavy lockout here was measured to starve the polish to ~0.2% of
        // the budget and cost the final coordinate fixes entirely.
        if (++polish_low_streak_ >= 4) {
            polish_cooldown_ = std::max(polish_cooldown_, 64);
            polish_low_streak_ = 0;
        }
    } else {
        polish_low_streak_ = 0;
    }
}

// ============================================================================
// UPGRADE (C): hard-stagnation rejuvenation (partial restart). When the run
// has been stuck for rejuv_factor_ * stag_trigger_ iterations (i.e. well past
// the point where quarantine/OBL could help), the worst (1 - rejuv_keep_)
// fraction is re-initialised uniformly, their IDE parameters are re-seeded,
// the SHADE memory is reset (terminal slot preserved) and the polish step is
// re-opened. This rescues the stuck runs that drag the MEAN down while never
// touching the elite (the best-so-far is preserved by construction).
// ============================================================================
void SPARQ::rejuvenate() {
    if (rejuv_cooldown_ > 0) { --rejuv_cooldown_; return; }
    if (!prob_) return;
    const int N = (int)X_.size();
    if (N < 4) return;

    // Two firing modes:
    //  (1) hard stagnation (the original trigger), and
    //  (2) SURVIVAL (IPOP-style): the population's relative fitness spread
    //      has collapsed to ~zero. Similarity-based stopping rules
    //      (doublebox) terminate such a run within a few iterations, killing
    //      it at the FIRST converged basin — long before the budget is spent
    //      and while refinement/restarts are still profitable. Re-seeding the
    //      worst part of the population revives the spread, so the run stays
    //      alive under ANY stopping rule for exactly as long as it keeps
    //      producing progress (the classic IPOP-CMA-ES rationale).
    double flo = FX_[0], fhi = FX_[0];
    for (double v : FX_) { flo = std::min(flo, v); fhi = std::max(fhi, v); }
    const double frel = (fhi - flo) / std::max(1.0, std::fabs(flo));
    // The detection threshold must be LOOSER than any plausible similarity
    // stopping rule (typically ~1e-8), otherwise the run is terminated before
    // the survival restart can fire.
    const bool collapsed = (frel < 1e-6) ||
                           (normalizedPopSpread() < std::max(1e-4, var_collapse_ratio_));
    bool hard_stag = (no_improve_ >= rejuv_factor_ * stag_trigger_) &&
                     (progress01() <= rejuv_progress_cutoff_) &&
                     (normalizedPopSpread() <= 10.0 * var_collapse_ratio_);

    // FIX (escalation): on deceptive-basin problems the weak survival reseed
    // fires repeatedly (every ~20 iterations, hundreds of times over a run)
    // because each shallow 5% refresh briefly revives the population spread
    // — which resets the COLLAPSE signal and so keeps no_improve_ from ever
    // accumulating the 120+ consecutive stagnant iterations hard_stag needs,
    // even though the run is genuinely, durably trapped. Track consecutive
    // weak-path activations that produce no MEANINGFUL improvement and force
    // one strong escape once that streak gets long, independent of whether
    // hard_stag's own raw iteration-count condition has technically fired.
    if (collapsed && !hard_stag) {
        const double denom = std::max(1.0, std::fabs(rejuv_watch_f_));
        const double relgain = std::isfinite(rejuv_watch_f_)
            ? (rejuv_watch_f_ - best_f_) / denom : 1.0;
        if (relgain >= 1e-4) {
            rejuv_watch_f_ = best_f_;
            rejuv_weak_streak_ = 0;
        } else if (++rejuv_weak_streak_ >= rejuv_weak_streak_limit_) {
            hard_stag = true;              // force the strong escape this time
            rejuv_weak_streak_ = 0;
            rejuv_watch_f_ = best_f_;
        }
    } else {
        rejuv_watch_f_ = best_f_;
        rejuv_weak_streak_ = 0;
    }

    // FIX (priority bug): "survival" (loose collapse check) and "hard_stag"
    // (deep, confirmed stagnation) are NOT mutually exclusive in practice —
    // once a run has genuinely stalled, its population naturally converges
    // too, so BOTH conditions end up true at the same time. The old code
    // checked `survival` first in the branch below, so it ALWAYS took the
    // weak 5%-reseed keep-alive path in that case, even when hard_stag's own
    // (stricter) condition was ALSO satisfied and the run actually needed the
    // strong 75%-reseed escape. This left every run that got trapped in a
    // deceptive local basin (the classic FM-synth / cluster-type trap) stuck
    // on keep-alive treatment for the rest of the budget — never escaping —
    // which is exactly the "spectacular best, dismal mean" signature seen on
    // fmsynth, tersoffc, eld4 and eld1. hard_stag now takes priority whenever
    // its own stricter condition fires; survival's weak reseed is reserved
    // for the case where the population has merely converged (as intended,
    // for keeping similarity-based stopping rules alive) WITHOUT also having
    // crossed the deep-stagnation bar.
    const bool survival  = collapsed && !hard_stag;
    if (!survival && !hard_stag) return;
    const int D = prob_->dimension();

    // FIX (escalation, tier 2): a 25%-elite strong reseed can still get
    // "reconquered" by the surviving elite via crossover on very sticky
    // deceptive basins (the classic FM-synth trap), firing repeatedly with
    // zero net progress. Track consecutive strong-tier firings without
    // meaningful gain and, once that streak is long enough, escalate to a
    // near-full reset (keep only a tiny safety sliver) so fresh exploration
    // isn't immediately pulled back by an entrenched elite. best_x_/best_f_
    // are never part of the population and are therefore never at risk.
    bool very_strong = false;
    if (hard_stag) {
        const double denom = std::max(1.0, std::fabs(rejuv_strong_watch_f_));
        const double relgain = std::isfinite(rejuv_strong_watch_f_)
            ? (rejuv_strong_watch_f_ - best_f_) / denom : 1.0;
        if (relgain >= 1e-2) {
            rejuv_strong_watch_f_ = best_f_;
            rejuv_strong_fail_streak_ = 0;
        } else if (++rejuv_strong_fail_streak_ >= rejuv_strong_fail_limit_) {
            very_strong = true;
            rejuv_strong_fail_streak_ = 0;
            rejuv_strong_watch_f_ = best_f_;
        }
    } else if (!collapsed) {
        // FIX (bug in this very escalation logic): rejuvenate() runs every
        // iteration, but hard_stag is only momentarily true on the rare
        // iterations that actually fire. Resetting the streak whenever
        // hard_stag is merely false (i.e. on almost every call) wiped it out
        // between firings and made the tier-3 escalation unreachable. The
        // streak must persist across the (many) non-firing iterations in
        // between and reset only when the population has genuinely
        // recovered (stopped being collapsed) — not on every quiet iteration
        // while it is still stuck.
        rejuv_strong_watch_f_ = best_f_;
        rejuv_strong_fail_streak_ = 0;
    }

    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(),
              [&](int a, int b){ return FX_[a] < FX_[b]; });

    // Survival mode re-seeds only a SLIVER of the population: its job is to
    // keep the fitness spread non-degenerate for similarity-based stopping
    // rules, and a handful of fresh points suffices. Re-seeding 75% here (as
    // the hard-stagnation mode does) was measured to drain hundreds of
    // thousands of evaluations into random points over a long run.
    const int keep = very_strong
        ? std::max(2, (int)std::round(0.05 * (double)N))
        : hard_stag
            ? std::max(2, (int)std::round(rejuv_keep_ * (double)N))
            : (N - std::max(2, (int)std::ceil(0.05 * (double)N)));
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int t = keep; t < N; ++t) {
        if (prob_->calls() >= max_evals_) break;
        int idx = ord[t];
        Vec cand(D);
        for (int j = 0; j < D; ++j) {
            double lo = (j < (int)L.size() ? L[j] : -1.0);
            double hi = (j < (int)U.size() ? U[j] :  1.0);
            if (lo > hi) std::swap(lo, hi);
            cand[j] = lo + (hi - lo) * randU();
        }
        archivePush(X_[idx]);
        X_[idx]  = std::move(cand);
        FX_[idx] = eval(X_[idx]);
        sampleIDEParamsAt(idx);
        if (FX_[idx] < best_f_) { best_f_ = FX_[idx]; best_x_ = X_[idx]; }
    }

    // NOTE: the SHADE memory is deliberately PRESERVED — it encodes F/CR
    // statistics that remain useful for the refreshed population; resetting it
    // proved harmful on boundary-optimum landscapes (Schwefel-type).
    eig_valid_ = false;
    iters_since_eig_ = 0;
    if (hard_stag) {
        // Strong escape fired: fully reset stagnation bookkeeping and reopen
        // the polish step, with a long cooldown before another rejuvenation
        // is allowed (give the fresh population a real chance to develop).
        ps_sigma_ = 0.02;
        no_improve_ = 0;
        rejuv_cooldown_ = rejuv_cooldown_init_;
    } else {
        // Survival restarts fire repeatedly (each converged basin triggers
        // one) and must NOT touch the stagnation bookkeeping: no_improve_
        // drives the elite polish, and zeroing it here starves the very
        // refinement engine the restart is buying time for.
        rejuv_cooldown_ = 10;
    }
}

// ============================================================================
// one_iteration (main loop body)
// ============================================================================
void SPARQ::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if ((int)X_.size() < 4) return;

    archiveTrim((int)X_.size());

    // Strategy selection: warmup ARQ, then Thompson sampling.
    // Ablation: with enable_ide=0 the bandit is never consulted and every
    // iteration runs the ARQ step (the IDE strategy is fully switched off).
    int hh = 0;
    if (!enable_ide_) {
        hh = 0;
    } else if (bootstrap_left_ > 0) {
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
            if (enable_levy_) quarantineLevy();
            break;
    }

    // Stagnation tracking (applies to both strategies)
    if (best_f_ < best_prev_ - 1e-18) {
        best_prev_ = best_f_;
        no_improve_ = 0;
    } else {
        ++no_improve_;
    }

    // Separate, MEANINGFUL-gain stagnation counter dedicated to polish (see
    // header comment on polish_stag_count_): counts iterations where best_f_
    // has not improved by at least a small RELATIVE amount, independent of
    // no_improve_ (which OBL/rejuvenate still use unchanged).
    {
        const double denom = std::max(1.0, std::fabs(polish_stag_mark_f_));
        const double relgain = (polish_stag_mark_f_ - best_f_) / denom;
        if (relgain >= kPolishStagRelGain) {
            polish_stag_mark_f_ = best_f_;
            polish_stag_count_  = 0;
        } else {
            ++polish_stag_count_;
        }
    }

    // UPGRADE (B): elite polish while the main loop is stagnating; in the
    // exploitation tail the trigger drops to a single stagnant iteration
    // (never unconditional, so a still-improving evolution keeps its budget).
    // Polish activation is STAGNATION-DRIVEN, not budget-driven. Frameworks
    // that terminate on population similarity (doublebox) may stop a run long
    // before 75% of max_evals is consumed, so a progress-gated polish would
    // simply never fire there; the stagnation counter works under any
    // stopping rule. In the budget tail the trigger additionally drops to a
    // single stagnant iteration for deep final refinement.
    //
    // FIX (large-N starvation): the trigger now reads polish_stag_count_
    // (meaningful-gain based) instead of no_improve_ (any-improvement-based).
    // At high D, population size scales with D (pop_scale_*D), so only a few
    // dozen total generations fit the evaluation budget and SOME individual
    // improves the incumbent by a vanishing amount almost every generation —
    // no_improve_ then never reaches polish_trigger_ and polish starves
    // exactly on the large-N family (hydrothermal/ded*/eld4-5) where its
    // coordinate refinement measurably helps. polish_stag_count_ only resets
    // on genuinely meaningful progress, so it reaches the trigger normally
    // regardless of population size.
    // FIX (protect very-high-D flagships): on fully separable, very-high-D
    // problems (D=200-class, e.g. test2n) direct A/B testing showed the more
    // frequent polish activations from polish_stag_count_ are a WASH at best
    // — sometimes marginally better, sometimes measurably worse on a given
    // seed, because injecting a polished point into the population disturbs
    // the DE core's own SHADE/eigen-crossover dynamics slightly differently
    // depending on the exact iteration it lands on (chaotic sensitivity).
    // Since arq3's ORIGINAL no_improve_-based trigger is already proven
    // excellent there, above this boundary the trigger reverts to the
    // untouched original behavior — byte-for-byte matching arq3 — while
    // D<=150 problems (hydrothermal, ded1, eld4/5, ...) keep the fix that
    // measurably helps them.
    const int D_guard = prob_ ? prob_->dimension() : 0;
    const int polish_stag_signal = (D_guard <= 150) ? polish_stag_count_ : no_improve_;

    {
        const int trig = (progress01() > polish_progress_trig_) ? 1 : polish_trigger_;
        if (!enable_polish_) {
            // Ablation: polish fully disabled; cooldown bookkeeping is
            // irrelevant in this mode.
        } else if (polish_cooldown_ > 0) {
            --polish_cooldown_;
        } else if (polish_stag_signal >= trig) {
            elitePolish();
        }
    }
    if (best_f_ < best_prev_ - 1e-18) { best_prev_ = best_f_; no_improve_ = 0; }
    if (best_f_ < polish_stag_mark_f_ - 1e-18) {
        polish_stag_mark_f_ = best_f_;
        polish_stag_count_  = 0;
    }

    // On-demand OBL (only after enough stagnation + variance collapse)
    if (enable_obl_) oblBasinEscape();

    // UPGRADE (C): hard-stagnation partial restart
    if (enable_rejuv_) rejuvenate();

    // NLPSR shrink (end of iteration so all indices above are still valid)
    if (enable_nlpsr_) {
        int Ntarget = targetPopulationSize();
        if (Ntarget < (int)X_.size()) {
            shrinkTo(Ntarget);
        }
    }

    // Bandit book-keeping: decay posteriors (non-stationary world)
    banditDecay();

    archiveTrim((int)X_.size());
    updateStop(FX_);
    printBest();

    if (debug_ && ((int)prob_->calls() % 1000) == 0) {
        std::fprintf(stdout,
            "[sparq] calls=%d N=%d best=%.6g noimpr=%d banditA=(%.2f,%.2f) B=(%.2f,%.2f) spread=%.3g\n",
            (int)prob_->calls(), (int)X_.size(), best_f_, no_improve_,
            bandit_a_[0], bandit_b_[0],
            bandit_a_[1], bandit_b_[1],
            normalizedPopSpread());
        std::fflush(stdout);
    }
}

} // namespace optimsolution
