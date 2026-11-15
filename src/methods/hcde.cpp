#include "hcde.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

// ===========================================================================
//  helpers
// ===========================================================================

void HCDE::sanitize(){
    if (f_min_ < 1e-6)  f_min_ = 1e-6;
    if (f_max_ < f_min_) std::swap(f_min_, f_max_);
    if (f_hi_  < f_lo_)  std::swap(f_lo_,  f_hi_);
    f_lo_ = std::max(f_lo_, f_min_);
    f_hi_ = std::min(f_hi_, f_max_);

    if (cr_hi_ < cr_lo_) std::swap(cr_lo_, cr_hi_);
    cr_lo_ = std::min(std::max(cr_lo_, 0.0), 1.0);
    cr_hi_ = std::min(std::max(cr_hi_, 0.0), 1.0);

    if (pcross_max_ < pcross_min_) std::swap(pcross_min_, pcross_max_);
    pcross_min_ = std::min(std::max(pcross_min_, 0.0), 1.0);
    pcross_max_ = std::min(std::max(pcross_max_, 0.0), 1.0);

    if (w_max_ < w_min_) std::swap(w_min_, w_max_);
    if (w_min_ <= 0.0)   w_min_ = 1e-3;

    c_up_  = std::min(std::max(c_up_,  0.0), 1.0);
    c_dn_  = std::min(std::max(c_dn_,  0.0), 1.0);
    c_cr_  = std::min(std::max(c_cr_,  0.0), 1.0);
    c_sr_  = std::min(std::max(c_sr_,  0.0), 1.0);
    lam_   = std::min(std::max(lam_,   0.0), 1.0);
    lam_f_ = std::min(std::max(lam_f_, 0.0), 1.0);
    alpha_ = std::min(std::max(alpha_, 0.0), 1.0);

    if (beta_    < 0.0) beta_    = 0.0;
    if (sigmaF0_ < 0.0) sigmaF0_ = 0.0;
    if (sigmaCR_ < 0.0) sigmaCR_ = 0.0;
    if (div_exp_ < 0.1) div_exp_ = 0.1;
    if (arc_rate_< 0.0) arc_rate_= 0.0;
    if (n_min_   < 4)   n_min_   = 4;

    if (pbest_hi_ < pbest_lo_) std::swap(pbest_lo_, pbest_hi_);
    pbest_lo_ = std::min(std::max(pbest_lo_, 0.01), 1.0);
    pbest_hi_ = std::min(std::max(pbest_hi_, 0.01), 1.0);
    p_rand_max_ = std::min(std::max(p_rand_max_, 0.0), 1.0);
    ps_gain_    = std::min(std::max(ps_gain_,   0.0), 1.0);
}

void HCDE::clampBounds(Vec& v){
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        if (!std::isfinite(v[j])) v[j] = 0.5*(lo + hi);
        if (v[j] < lo) v[j] = lo;
        if (v[j] > hi) v[j] = hi;
    }
}

// Midpoint reflection: a violating coordinate is pulled halfway back towards
// the parent instead of being snapped onto the bound. This avoids the
// artificial clustering on the box faces that plain clamping produces and
// keeps the diversity measure used by DAHM/PS meaningful.
void HCDE::repairMidpoint(Vec& v, const Vec& parent){
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (size_t j=0; j<v.size(); ++j){
        double lo = (j < L.size() ? L[j] : -1.0);
        double hi = (j < U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double pj = (j < parent.size() ? parent[j] : 0.5*(lo+hi));
        if (!std::isfinite(v[j]))  v[j] = 0.5*(lo + hi);
        else if (v[j] < lo)        v[j] = 0.5*(lo + pj);
        else if (v[j] > hi)        v[j] = 0.5*(hi + pj);
    }
    clampBounds(v);
}

int HCDE::randDistinct(int n, int a, int b, int c){
    if (n <= 1) return 0;
    std::uniform_int_distribution<int> I(0, n-1);
    int r, guard = 0;
    do { r = I(rng_); } while ((r==a || r==b || r==c) && ++guard < 200);
    return r;
}

double HCDE::sampleCauchyPositive(double mu, double sigma){
    std::cauchy_distribution<double> C(mu, std::max(1e-9, sigma));
    double v = 0.0;
    for (int t=0; t<64; ++t){
        v = C(rng_);
        if (std::isfinite(v) && v > 0.0) break;
        v = mu;
    }
    if (!std::isfinite(v) || v <= 0.0) v = std::max(f_min_, mu);
    if (v > f_max_) v = f_max_;
    if (v < f_min_) v = f_min_;
    return v;
}

void HCDE::buildOrder(){
    const int N = (int)X_.size();
    order_.resize(N);
    std::iota(order_.begin(), order_.end(), 0);
    std::sort(order_.begin(), order_.end(),
              [&](int a, int b){ return FX_[a] < FX_[b]; });
}

void HCDE::trimArchive(){
    const int N = (int)X_.size();
    const size_t cap = (size_t)std::max(0, (int)std::round(arc_rate_ * (double)N));
    std::uniform_int_distribution<size_t> I(0, ARC_.empty() ? 0 : ARC_.size()-1);
    while (ARC_.size() > cap){
        std::uniform_int_distribution<size_t> J(0, ARC_.size()-1);
        size_t k = J(rng_);
        ARC_[k] = ARC_.back();
        ARC_.pop_back();
    }
    (void)I;
}

// ===========================================================================
//  Population State  (shared substrate of all three mechanisms)
// ===========================================================================

double HCDE::measureDiversity() const {
    const int N = (int)X_.size();
    if (N < 2) return 0.0;
    const int D = (int)X_[0].size();

    Vec c(D, 0.0);
    for (int i=0; i<N; ++i)
        for (int j=0; j<D; ++j) c[j] += X_[i][j];
    for (int j=0; j<D; ++j) c[j] /= (double)N;

    double acc = 0.0;
    for (int i=0; i<N; ++i){
        double s = 0.0;
        for (int j=0; j<D; ++j){ double d = X_[i][j] - c[j]; s += d*d; }
        acc += std::sqrt(s);
    }
    return (acc / (double)N) / std::max(1e-30, diagScale_);
}

// state_ in [-1,+1]:  negative = stagnating / collapsed,  positive = healthy
void HCDE::updatePopulationState(double progress){
    div_       = measureDiversity();
    divTarget_ = div0_ * std::pow(std::max(0.0, 1.0 - progress), div_exp_);
    const double dRel = (div_ - divTarget_) / std::max(divTarget_, 1e-12);
    state_ = std::tanh(ps_a_ * std::min(std::max(dRel, -5.0), 5.0)
                     + ps_b_ * (srEma_ - sr_ref_));
}

// ===========================================================================
//  MECHANISM 1 - DAHM : Diversity-Anchored Hysteretic Momentum  ->  F
// ===========================================================================

// Population-level anchor. The sigmoid argument is the *relative diversity
// deficit*: positive when the swarm has converged faster than the schedule
// prescribes, which immediately raises F* and re-opens exploration.
double HCDE::anchorF() const {
    const double deficit = (divTarget_ - div_) / std::max(divTarget_, 1e-12);
    const double s = 1.0 / (1.0 + std::exp(-kappa_ * std::min(std::max(deficit, -20.0), 20.0)));
    return f_lo_ + (f_hi_ - f_lo_) * s;
}

// Individual-level deviation from the anchor, driven by hysteretic momentum.
// mom_i > 0 -> individual is on a winning streak -> shrink F -> exploit.
// mom_i < 0 -> individual keeps failing        -> grow  F -> explore.
double HCDE::sampleF(int i){
    const double mu = std::min(std::max(Fstar_ * std::exp(-eta_ * mom_[i]), f_min_), f_max_);
    // the Cauchy scale itself is state-driven: when the whole population is
    // failing, F is sampled from a much heavier tail.
    const double sig = sigmaF0_ * (1.0 + (1.0 - srEma_));
    return sampleCauchyPositive(mu, sig);
}

// Asymmetric update = hysteresis loop. c_up_ (failure) > c_dn_ (success):
// the individual escapes fast but settles slowly.
void HCDE::updateMomentum(int i, bool success){
    if (success) mom_[i] = (1.0 - c_dn_) * mom_[i] + c_dn_ * ( 1.0);
    else         mom_[i] = (1.0 - c_up_) * mom_[i] + c_up_ * (-1.0);
    mom_[i] = std::min(std::max(mom_[i], -1.0), 1.0);
}

// ===========================================================================
//  MECHANISM 2 - ECCA : Entropy-regulated Coordinate Credit Allocation -> CR
// ===========================================================================

// Normalised Shannon entropy of the credit vector. 1 = no coordinate has been
// identified as useful; 0 = all credit sits on one coordinate.
double HCDE::creditEntropy() const {
    const int D = (int)W_.size();
    if (D <= 1) return 1.0;
    double s = 0.0;
    for (int j=0; j<D; ++j) s += std::max(0.0, W_[j]);
    if (s <= 1e-30) return 1.0;
    double H = 0.0;
    for (int j=0; j<D; ++j){
        double p = std::max(0.0, W_[j]) / s;
        if (p > 1e-30) H -= p * std::log(p);
    }
    return std::min(1.0, std::max(0.0, H / std::log((double)D)));
}

// Softmax tilt over standardised credits. q_ sums to 1, so the expected number
// of inherited coordinates stays CR_i * D: CR keeps its classical meaning
// ("how many"), while q_ decides "which ones".
void HCDE::buildCrossoverTilt(){
    const int D = (int)W_.size();
    q_.assign(D, 1.0 / std::max(1, D));
    if (D <= 1 || beta_ <= 0.0) return;

    double m = 0.0;
    for (int j=0; j<D; ++j) m += W_[j];
    m /= (double)D;
    double v = 0.0;
    for (int j=0; j<D; ++j){ double d = W_[j]-m; v += d*d; }
    const double sd = std::sqrt(v / (double)D);
    if (sd < 1e-12) return;                       // credits still uniform

    double zmax = -1e300;
    std::vector<double> z(D);
    for (int j=0; j<D; ++j){ z[j] = beta_ * (W_[j]-m) / sd; zmax = std::max(zmax, z[j]); }
    double s = 0.0;
    for (int j=0; j<D; ++j){ q_[j] = std::exp(z[j]-zmax); s += q_[j]; }
    if (s <= 1e-30) { q_.assign(D, 1.0/(double)D); return; }
    for (int j=0; j<D; ++j) q_[j] /= s;
}

// Credit rate per coordinate = (total improvement obtained while j was crossed)
// divided by (number of times j was crossed). The division is what makes this
// a proper credit *rate* instead of a usage histogram.
void HCDE::updateCredit(const std::vector<double>& num,
                        const std::vector<double>& den){
    const int D = (int)W_.size();
    std::vector<double> rate(D, 0.0);
    double mean = 0.0;
    for (int j=0; j<D; ++j){
        rate[j] = num[j] / (den[j] + 1.0);
        mean += rate[j];
    }
    mean /= std::max(1, D);

    if (mean <= 1e-30){
        // Nothing worked this generation: forget towards uniform. This raises
        // the entropy, which in turn raises mu_CR -> automatic escape.
        for (int j=0; j<D; ++j)
            W_[j] = (1.0 - lam_f_) * W_[j] + lam_f_ * 1.0;
    } else {
        for (int j=0; j<D; ++j){
            const double rhat = rate[j] / mean;              // mean-1 normalised
            W_[j] = (1.0 - lam_) * W_[j] + lam_ * rhat;
            W_[j] = std::min(std::max(W_[j], w_min_), w_max_);
        }
    }
}

// mu_CR is pulled by two forces: the success-weighted Lehmer mean of the CR
// values that actually worked (SHADE-like), and the entropy target.
void HCDE::updateMuCR(const std::vector<double>& S_CR,
                      const std::vector<double>& S_dF){
    Hnorm_ = creditEntropy();
    const double muTarget = cr_lo_ + (cr_hi_ - cr_lo_) * Hnorm_;

    double target = muTarget;
    if (!S_CR.empty()){
        double wsum = 0.0;
        for (double d : S_dF) wsum += d;
        double n2 = 0.0, n1 = 0.0;
        if (wsum > 1e-30){
            for (size_t k=0; k<S_CR.size(); ++k){
                const double w = S_dF[k] / wsum;
                n2 += w * S_CR[k] * S_CR[k];
                n1 += w * S_CR[k];
            }
        } else {
            for (size_t k=0; k<S_CR.size(); ++k){ n2 += S_CR[k]*S_CR[k]; n1 += S_CR[k]; }
        }
        const double lehmer = (n1 > 1e-30) ? (n2 / n1) : muTarget;
        target = alpha_ * lehmer + (1.0 - alpha_) * muTarget;
    }
    muCR_ = (1.0 - c_cr_) * muCR_ + c_cr_ * target;
    muCR_ = std::min(std::max(muCR_, cr_lo_), cr_hi_);
}

// ===========================================================================
//  MECHANISM 3 - PS-NLPSR
// ===========================================================================

void HCDE::applyNLPSR(double progress){
    if (!use_nlpsr_) return;
    const int N = (int)X_.size();
    if (N <= n_min_) return;

    // Population-state warp of the progress variable. The bell factor
    // p*(1-p) vanishes at both ends, so N(0)=N0 and N(1)=Nmin are preserved
    // exactly; only the *shape* of the schedule is modulated.
    double pe = progress + ps_gain_ * state_ * progress * (1.0 - progress);
    pe = std::min(1.0, std::max(0.0, pe));

    // NLPSR:  N(p) = round( (Nmin - Nmax) * p^(1-p) + Nmax )
    const double shape = std::pow(pe, 1.0 - pe);
    int Nt = (int)std::round((double)(n_min_ - N0_) * shape + (double)N0_);
    Nt = std::min(N0_, std::max(n_min_, Nt));
    if (Nt >= N) return;

    buildOrder();                                  // keep the Nt best
    std::vector<Vec>    nX;  nX.reserve(Nt);
    std::vector<double> nF;  nF.reserve(Nt);
    std::vector<double> nFi, nCR, nMom;
    nFi.reserve(Nt); nCR.reserve(Nt); nMom.reserve(Nt);
    for (int k=0; k<Nt; ++k){
        const int idx = order_[k];
        nX.push_back(X_[idx]);
        nF.push_back(FX_[idx]);
        nFi.push_back(Fi_[idx]);
        nCR.push_back(CRi_[idx]);
        nMom.push_back(mom_[idx]);                 // adaptive state travels with
    }                                              // the individual, not the slot
    X_.swap(nX); FX_.swap(nF); Fi_.swap(nFi); CRi_.swap(nCR); mom_.swap(nMom);

    trimArchive();
    this->setPopulation((int)X_.size());           // keep the reporter consistent
}

// ===========================================================================
//  Optimizer interface
// ===========================================================================

void HCDE::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    // population=0 (or unset): internal method, no external published
    // literature default; N=100 is this codebase's general convention.
    // Previously population=0 silently fell back to [global] population -- fixed here.
    const int N = std::max(std::max(6, n_min_),
                           (pop_override_ >= 4 ? pop_override_ : 100));

    N0_ = N;
    this->setPopulation(N);

    X_.clear(); FX_.clear(); ARC_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_){ best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // adaptive state
    Fi_.assign(N, 0.5);
    CRi_.assign(N, 0.5);
    mom_.assign(N, 0.0);
    W_.assign(D, 1.0);
    q_.assign(D, 1.0 / std::max(1, D));
    muCR_  = 0.5 * (cr_lo_ + cr_hi_);
    Hnorm_ = 1.0;
    srEma_ = sr_ref_;
    state_ = 0.0;
    gen_   = 0;

    // diagonal of the box, used to normalise the diversity measure
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    double s = 0.0;
    for (int j=0; j<D; ++j){
        const double lo = (j < (int)L.size() ? L[j] : -1.0);
        const double hi = (j < (int)U.size() ? U[j] :  1.0);
        const double d  = hi - lo;
        s += d*d;
    }
    diagScale_ = std::sqrt(std::max(1e-30, s));

    div_ = div0_ = std::max(1e-12, measureDiversity());
    divTarget_ = div0_;
    Fstar_ = anchorF();

    if (debug_){
        std::string lm  = (local_method_.empty()     ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[hcde] N0=%d D=%d | DAHM(F* in [%.2f,%.2f], kappa=%.2f, eta=%.2f, up/dn=%.2f/%.2f) "
            "| ECCA(CR in [%.2f,%.2f], beta=%.2f, lambda=%.2f) "
            "| NLPSR(%s, Nmin=%d, gain=%.2f) | in-run: %s @ %.3f | final: %s (%s)\n",
            N0_, D, f_lo_, f_hi_, kappa_, eta_, c_up_, c_dn_,
            cr_lo_, cr_hi_, beta_, lam_,
            use_nlpsr_ ? "on" : "off", n_min_, ps_gain_,
            lm.c_str(), local_rate_, end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void HCDE::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    int N = (int)X_.size();
    if (N < 4 || D < 1) return;

    const double progress =
        std::min(1.0, (double)prob_->calls() / std::max(1.0, (double)max_evals_));

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double>       NRM(0.0, 1.0);
    std::uniform_int_distribution<int>     Jrand(0, std::max(0, D-1));

    // ---- population state, then the two adaptation anchors ------------
    updatePopulationState(progress);
    Fstar_ = anchorF();                 // DAHM anchor for this generation
    buildCrossoverTilt();               // ECCA coordinate tilt for this generation
    buildOrder();

    // pbest fraction: shrinks with progress, widens again when stagnating
    double pRate = pbest_hi_ + (pbest_lo_ - pbest_hi_) * progress;
    pRate *= (1.0 + 0.5 * std::max(0.0, -state_));
    pRate = std::min(1.0, std::max(0.01, pRate));
    const int pn = std::max(2, (int)std::round(pRate * (double)N));

    // stagnation-triggered DE/rand/1 branch (pure exploration)
    const double pRand = p_rand_max_ * 0.5 * (1.0 - state_);

    // generation accumulators
    std::vector<Vec>    U(N);
    std::vector<double> FU(N, std::numeric_limits<double>::infinity());
    std::vector<char>   improved(N, 0);
    std::vector<double> credNum(D, 0.0), credDen(D, 0.0);
    std::vector<double> S_CR, S_dF;
    S_CR.reserve(N); S_dF.reserve(N);
    int evaluated = 0;

    for (int i=0; i<N; ++i){
        // ---- (1) DAHM: this individual's F ---------------------------
        Fi_[i] = sampleF(i);

        // ---- (2) ECCA: this individual's CR --------------------------
        double cr = muCR_ + sigmaCR_ * NRM(rng_);
        cr = std::min(std::max(cr, cr_lo_), cr_hi_);
        CRi_[i] = cr;

        // ---- mutation -------------------------------------------------
        Vec v(D, 0.0);
        if (U01(rng_) < pRand){
            // DE/rand/1 : no attraction to the elite at all
            const int a = randDistinct(N, i);
            const int b = randDistinct(N, i, a);
            const int c = randDistinct(N, i, a, b);
            for (int j=0; j<D; ++j)
                v[j] = X_[a][j] + Fi_[i] * (X_[b][j] - X_[c][j]);
        } else {
            // DE/current-to-pbest/1 with external archive
            std::uniform_int_distribution<int> P(0, pn-1);
            const int pb = order_[P(rng_)];
            const int r1 = randDistinct(N, i, pb);
            const int nA = (int)ARC_.size();
            int r2 = 0; const Vec* xr2 = nullptr;
            for (int t=0; t<200; ++t){
                std::uniform_int_distribution<int> R(0, N + nA - 1);
                r2 = R(rng_);
                if (r2 < N){ if (r2 == i || r2 == r1) continue; xr2 = &X_[r2]; }
                else       { xr2 = &ARC_[r2 - N]; }
                break;
            }
            if (!xr2) xr2 = &X_[randDistinct(N, i, r1)];
            for (int j=0; j<D; ++j)
                v[j] = X_[i][j]
                     + Fi_[i] * (X_[pb][j]  - X_[i][j])
                     + Fi_[i] * (X_[r1][j]  - (*xr2)[j]);
        }

        // ---- crossover: ECCA per-coordinate probabilities -------------
        Vec u = X_[i];
        std::vector<char> crossed(D, 0);
        const int jr = Jrand(rng_);
        const double K = CRi_[i] * (double)D;          // expected #crossed coords
        for (int j=0; j<D; ++j){
            double pj = K * q_[j];
            pj = std::min(std::max(pj, pcross_min_), pcross_max_);
            if (U01(rng_) < pj || j == jr){
                u[j] = v[j];
                crossed[j] = 1;
                credDen[j] += 1.0;
            }
        }
        repairMidpoint(u, X_[i]);

        // ---- evaluation ----------------------------------------------
        double fu = eval(u);
        ++evaluated;

        if (fu < FX_[i]){
            // optional in-run local refinement, identical contract to DE
            if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
                auto [xloc, floc] = localSearch(local_method_, u);
                if (std::isfinite(floc) && floc < fu){ u = std::move(xloc); fu = floc; }
            }
            improved[i] = 1;

            const double dF = (FX_[i] - fu) / (std::fabs(FX_[i]) + 1e-30);
            const double dFc = std::min(std::max(dF, 0.0), 1e6);
            S_CR.push_back(CRi_[i]);
            S_dF.push_back(dFc);
            for (int j=0; j<D; ++j) if (crossed[j]) credNum[j] += dFc;
        }

        U[i]  = std::move(u);
        FU[i] = fu;

        if (prob_->calls() >= max_evals_){
            for (int k=i+1; k<N; ++k){ U[k] = X_[k]; FU[k] = FX_[k]; }
            break;
        }
    }

    // ---- synchronous selection + archive -----------------------------
    int succ = 0;
    for (int i=0; i<N; ++i){
        const bool ok = (improved[i] != 0) && (FU[i] < FX_[i]);
        updateMomentum(i, ok);                       // (1) DAHM hysteresis
        if (ok){
            ++succ;
            if (arc_rate_ > 0.0) ARC_.push_back(X_[i]);   // parent goes to archive
            X_[i]  = std::move(U[i]);
            FX_[i] = FU[i];
            if (FX_[i] < best_f_){ best_f_ = FX_[i]; best_x_ = X_[i]; }
        }
    }
    trimArchive();

    // ---- end-of-generation adaptation --------------------------------
    updateCredit(credNum, credDen);                  // (2) ECCA credit vector
    updateMuCR(S_CR, S_dF);                          // (2) ECCA mu_CR
    const double sr = (evaluated > 0) ? (double)succ / (double)evaluated : 0.0;
    srEma_ = (1.0 - c_sr_) * srEma_ + c_sr_ * sr;    // (3) PS success rate

    updatePopulationState(progress);                 // refresh state after selection
    applyNLPSR(progress);                            // (3) NLPSR resize

    ++gen_;
    if (debug_ && (gen_ % 25 == 0)){
        std::fprintf(stdout,
            "[hcde] g=%lld p=%.3f N=%d | div=%.4f/%.4f state=%+.3f SR=%.3f "
            "| F*=%.3f <F>=%.3f | H=%.3f muCR=%.3f\n",
            gen_, progress, (int)X_.size(), div_, divTarget_, state_, srEma_,
            Fstar_,
            (Fi_.empty() ? 0.0 : std::accumulate(Fi_.begin(), Fi_.end(), 0.0)/(double)Fi_.size()),
            Hnorm_, muCR_);
        std::fflush(stdout);
    }

    printBest();
    updateStop(FX_);
}

void HCDE::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_){
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // the refined best replaces the worst slot (same convention as DE/GA/BHO)
    if (!X_.empty() && !FX_.empty()){
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k)
            if (FX_[k] > worst_val){ worst_val = FX_[k]; worst_idx = k; }
        if (worst_idx < X_.size()){
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    printBest();
}

} // namespace optimsolution
