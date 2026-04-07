#include "arq.h"
#include <cstdio>

namespace optimsolution {

static inline std::string to_lower(std::string s){
    for (auto &c: s) c = (char)std::tolower((unsigned char)c);
    return s;
}

void ARQ::configure(const MethodConfig& mc) {
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
    auto parse_double = [&](std::string s, double fb)->double{
        s = trim(s); if (s.empty()) return fb;
        try{ size_t pos=0; double v=std::stod(s,&pos); if(pos==s.size() && std::isfinite(v)) return v; }catch(...) {}
        return fb;
    };

    // population override
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

    worst_frac_      = mc.getDbl("w", worst_frac_);
    rsigma_          = mc.getDbl("rsigma", rsigma_);
    stagnation_trigger_ = mc.getInt("stagnationtrigger", stagnation_trigger_);

    shc_             = mc.getDbl("shc", shc_);

    Flo_             = mc.getDbl("Flo", Flo_);
    Fhi_             = mc.getDbl("Fhi", Fhi_);

    rtr_pool_        = mc.getInt("rtrpool", rtr_pool_);
    archive_rate_    = mc.getDbl("archiverate", archive_rate_);

    debug_arq_       = mc.getInt("debug_arq", debug_arq_);

    // sanitize
    if (pbest_ < 0.01) pbest_ = 0.01;
    if (pbest_ > 0.5)  pbest_ = 0.5;
    if (agent_fraction_ <= 0.0) agent_fraction_ = 0.5;
    if (agent_fraction_ >  1.0) agent_fraction_ = 1.0;
    if (Flo_ <= 0.0) Flo_ = 0.01;
    if (Fhi_ < Flo_) std::swap(Fhi_, Flo_);
    if (rtr_pool_ < 2) rtr_pool_ = 2;
    if (archive_rate_ <= 0.1) archive_rate_ = 1.0;
}

void ARQ::init(){
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

    if (debug_arq_) {
        std::fprintf(stdout,
            "[arq] cfg -> N=%d, p=%.4f, agent=%.3f, muF=%.3f, muCR=%.3f, alpha=%.3f, rho=%.3f, w=%.3f, stag=%d, rtrpool=%d\n",
            N, pbest_, agent_fraction_, muF_, muCR_, outlier_alpha_, outlier_rho_, worst_frac_, stagnation_trigger_, rtr_pool_);
        std::fflush(stdout);
    }

    printBest();
}

int ARQ::pickDistinct(int n, int a, int b, int c){
    std::uniform_int_distribution<int> I(0, n-1);
    int r;
    do { r = I(rng_); } while (r==a || r==b || r==c);
    return r;
}

void ARQ::ensureBounds(Vec& v){
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

double ARQ::distBN(const Vec& a, const Vec& b) const {
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

double ARQ::quantile(std::vector<double> v, double q01){
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

void ARQ::archivePush(const Vec& x){
    A_.push_back(x);
}

void ARQ::archiveTrim(int N){
    const int cap = std::max(1, (int)std::floor(archive_rate_ * (double)N));
    if ((int)A_.size() <= cap) return;
    std::shuffle(A_.begin(), A_.end(), rng_);
    A_.resize(cap);
}

void ARQ::sample_F_CR(double& F, double& CR,
                      std::cauchy_distribution<double>& cauchyF,
                      std::normal_distribution<double>& normCR) {
    // sample F until positive
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

void ARQ::update_mu_from_success(const std::vector<double>& SF,
                                 const std::vector<double>& SCR,
                                 const std::vector<double>& SG) {
    if (SG.empty()) return;
    double sumG = std::accumulate(SG.begin(), SG.end(), 0.0);
    if (!(sumG > 0.0)) return;

    // weights
    std::vector<double> w(SG.size());
    for (size_t i=0; i<w.size(); ++i) w[i] = SG[i] / sumG;

    // Lehmer mean for F: sum w*F^2 / sum w*F
    double num=0.0, den=0.0;
    for (size_t i=0; i<SF.size(); ++i){
        num += w[i] * SF[i] * SF[i];
        den += w[i] * SF[i];
    }
    double newMuF  = (den > 0.0) ? (num/den) : muF_;

    // weighted mean for CR
    double newMuCR = 0.0;
    for (size_t i=0; i<SCR.size(); ++i) newMuCR += w[i] * SCR[i];

    // smooth with shc
    muF_  = (1.0 - shc_) * muF_  + shc_ * newMuF;
    muCR_ = (1.0 - shc_) * muCR_ + shc_ * newMuCR;

    if (muF_ < Flo_) muF_ = Flo_;
    if (muF_ > Fhi_) muF_ = Fhi_;
    if (muCR_ < 0.0) muCR_ = 0.0;
    if (muCR_ > 1.0) muCR_ = 1.0;
}

void ARQ::makeTrial(int i, const std::vector<int>& ord, double F, double CR, Vec& u){
    const int D = prob_->dimension();
    const int N = (int)X_.size();

    // pbest index from top p fraction
    int pcount = std::max(2, (int)std::ceil(pbest_ * (double)N));
    if (pcount > N) pcount = N;
    std::uniform_int_distribution<int> Ip(0, pcount-1);
    int ipbest = ord[Ip(rng_)];
    const Vec& xpbest = X_[ipbest];

    int r1 = pickDistinct(N, i);
    // r2 from population or archive (if available)
    bool useA = (!A_.empty()) && (std::uniform_real_distribution<double>(0.0,1.0)(rng_) < 0.5);

    Vec r2v(D, 0.0);
    if (useA) {
        std::uniform_int_distribution<int> Ia(0, (int)A_.size()-1);
        r2v = A_[Ia(rng_)];
    } else {
        int r2 = pickDistinct(N, i, r1);
        r2v = X_[r2];
    }

    // mutation: current-to-pbest/1 with archive support
    Vec v(D, 0.0);
    for (int j=0; j<D; ++j){
        v[j] = X_[i][j] + F*(xpbest[j] - X_[i][j]) + F*(X_[r1][j] - r2v[j]);
    }

    // binomial crossover with jrand
    u = X_[i];
    std::uniform_real_distribution<double> U01(0.0,1.0);
    std::uniform_int_distribution<int> Jrand(0, std::max(0, D-1));
    int jr = Jrand(rng_);
    for (int j=0; j<D; ++j){
        if (U01(rng_) < CR || j == jr) u[j] = v[j];
    }
    ensureBounds(u);
}

bool ARQ::selectionRTR(int parentIndex, const Vec& u, double fu,
                       double F, double CR,
                       std::vector<double>& SF,
                       std::vector<double>& SCR,
                       std::vector<double>& SG) {
    // parent first
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

    // RTR: nearest in a random pool using bounds-normalized distance
    const int N = (int)X_.size();
    int qstar = -1;
    double bestD = std::numeric_limits<double>::infinity();
    std::uniform_int_distribution<int> I(0, N-1);

    for (int k=0; k<rtr_pool_; ++k){
        int q = I(rng_);
        double d = distBN(u, X_[q]);
        if (d < bestD) { bestD = d; qstar = q; }
    }
    if (qstar < 0) return false;

    if (fu < FX_[qstar]) {
        double gain = FX_[qstar] - fu;
        archivePush(X_[qstar]);
        X_[qstar] = u;
        FX_[qstar] = fu;

        // critical: log success even for neighbor replacement
        SF.push_back(F);
        SCR.push_back(CR);
        SG.push_back(gain);
        return true;
    }

    return false;
}

void ARQ::quarantine_and_restart(){
    if (!prob_) return;

    const int N = (int)X_.size();
    if (N < 4) return;

    // build ordering by fitness (ascending)
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    // quartiles on fitness
    std::vector<double> fits = FX_;
    double Q1 = quantile(fits, 0.25);
    fits = FX_;
    double Q3 = quantile(fits, 0.75);
    double IQR = Q3 - Q1;
    double theta = Q3 + outlier_alpha_ * IQR;

    // robust center = mean of best 50%
    const int half = std::max(1, N/2);
    const int D = prob_->dimension();
    Vec center(D, 0.0);
    for (int k=0; k<half; ++k){
        const Vec& x = X_[ord[k]];
        for (int j=0; j<D; ++j) center[j] += x[j];
    }
    for (int j=0; j<D; ++j) center[j] /= (double)half;

    // outliers
    std::vector<int> out;
    for (int i=0; i<N; ++i) if (FX_[i] >= theta) out.push_back(i);

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
                }
            }
        }
    }

    // micro-restart only on stagnation trigger
    if (no_improve_ < stagnation_trigger_) return;

    // refresh ordering after quarantine
    ord.resize(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    int wcount = std::max(1, (int)std::floor(worst_frac_ * (double)N));
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int t=0; t<wcount; ++t){
        if (prob_->calls() >= max_evals_) break;
        int idx = ord[N-1-t]; // worst

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
        }
    }

    no_improve_ = 0; // reset after restart action
}

void ARQ::one_iteration(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N < 4) return;

    // keep archive bounded
    archiveTrim(N);

    // ordering by fitness (needed for pbest + stable geometry)
    std::vector<int> ord(N);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FX_[a] < FX_[b]; });

    // mini-batch size
    int m = std::max(1, (int)std::ceil(agent_fraction_ * (double)N));
    if (m > N) m = N;

    // choose m distinct indices
    std::vector<int> idx = ord;
    std::shuffle(idx.begin(), idx.end(), rng_);
    idx.resize(m);

    // success logs
    std::vector<double> SF, SCR, SG;

    std::cauchy_distribution<double> cauchyF(muF_, 0.1);
    std::normal_distribution<double> normCR(muCR_, 0.1);

    // main mini-batch loop
    for (int t=0; t<m; ++t){
        if (prob_->calls() >= max_evals_) break;

        int i = idx[t];

        double F, CR;
        // re-center distributions each draw
        cauchyF = std::cauchy_distribution<double>(muF_, 0.1);
        normCR  = std::normal_distribution<double>(muCR_, 0.1);
        sample_F_CR(F, CR, cauchyF, normCR);

        Vec u(D, 0.0);
        makeTrial(i, ord, F, CR, u);

        double fu = eval(u);

        bool improved = selectionRTR(i, u, fu, F, CR, SF, SCR, SG);
        if (improved) {
            // update best quickly (safe scan is OK for N=100)
            for (int k=0; k<N; ++k){
                if (FX_[k] < best_f_) { best_f_ = FX_[k]; best_x_ = X_[k]; }
            }
        }
    }

    // update success-history means
    update_mu_from_success(SF, SCR, SG);

    // maintenance: quarantine + (optional) restart
    quarantine_and_restart();

    // update stagnation
    if (best_f_ < best_prev_) {
        best_prev_ = best_f_;
        no_improve_ = 0;
    } else {
        no_improve_++;
    }

    // keep archive bounded
    archiveTrim(N);

    printBest();
    updateStop(FX_);
}

} // namespace optimsolution
