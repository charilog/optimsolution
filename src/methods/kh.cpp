#include "kh.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static inline double clamp01(double x){ return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

double KH::meanRange() const {
    if (!prob_) return 1.0;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int D = prob_->dimension();
    double s = 0.0;
    for (int j=0; j<D; ++j){
        double lo = (j < (int)L.size() ? L[j] : -1.0);
        double hi = (j < (int)U.size() ? U[j] :  1.0);
        if (lo > hi) std::swap(lo, hi);
        double r = hi - lo;
        if (!std::isfinite(r) || r <= 0.0) r = 1.0;
        s += r;
    }
    s /= std::max(1, D);
    if (!std::isfinite(s) || s <= 0.0) s = 1.0;
    return s;
}

KH::Vec KH::unitDir(const Vec& a, const Vec& b) const {
    // returns unit vector of (a-b)
    const int D = (int)a.size();
    Vec d(D, 0.0);
    double n2 = 0.0;
    for (int j=0; j<D; ++j){
        d[j] = a[j] - b[j];
        n2 += d[j] * d[j];
    }
    double n = std::sqrt(n2);
    if (!std::isfinite(n) || n <= 1e-12) return Vec(D, 0.0);
    for (int j=0; j<D; ++j) d[j] /= n;
    return d;
}

void KH::ensureBounds(Vec& v){
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

void KH::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    // population=0 (or unset): literature convention. Gandomi & Alavi's
    // original Krill Herd paper (2012) and the great majority of
    // follow-up studies use a krill population of N=25. Previously
    // population=0 silently fell back to [global] population (default
    // 100) instead of this documented default -- fixed here.
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : 25));
    this->setPopulation(N);

    X_.clear(); FX_.clear(); Nprev_.clear(); Fprev_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, N);

    FX_.assign(N, std::numeric_limits<double>::infinity());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<N; ++i){
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    Nprev_.assign(N, Vec(D, 0.0));
    Fprev_.assign(N, Vec(D, 0.0));
    P_ = X_; FP_ = FX_;
    iter_ = 0;

    if (debug_kh_) {
        std::fprintf(stdout,
            "[kh] cfg -> N=%d (population() now=%d, override=%d), nmax=%.6g, vf=%.6g, dmax0=%.6g, dt=%.4g, wn=%.3f, wf=%.3f, c_best=%.3f, c_food=%.3f, k=%d, greedy=%s, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, nmax_, vf_, dmax0_, dt_, wn_, wf_, c_best_, c_food_, neighbor_k_,
            greedy_ ? "on" : "off",
            (local_method_.empty() ? "none" : local_method_.c_str()), local_rate_,
            end_local_refine_ ? "on" : "off", (end_local_method_.empty() ? "none" : end_local_method_.c_str()));
        std::fflush(stdout);
    }

    printBest();
}

void KH::one_iteration(){
    if (!prob_) return;

    if (paper_model_) { paperIteration(); return; }

    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    // iteration horizon estimate based on max_evals and population
    const long long T = std::max(1LL, (long long)std::max(1, (int)(max_evals_ / std::max(1, N))));
    const double progress = clamp01((double)iter_ / (double)T);

    const double mr = meanRange();
    const double nmax_abs  = (nmax_  <= 1.0 ? nmax_  * mr : nmax_);
    const double vf_abs    = (vf_    <= 1.0 ? vf_    * mr : vf_);
    const double dmax_abs0 = (dmax0_ <= 1.0 ? dmax0_ * mr : dmax0_);
    const double dmax_abs  = dmax_abs0 * (1.0 - progress);

    // masses (higher mass => better fitness)
    double fbest = FX_[0], fworst = FX_[0];
    size_t ibest = 0, iworst = 0;
    for (int i=1; i<N; ++i){
        if (FX_[i] < fbest) { fbest = FX_[i]; ibest = (size_t)i; }
        if (FX_[i] > fworst){ fworst= FX_[i]; iworst= (size_t)i; }
    }
    if (fbest < best_f_) { best_f_ = fbest; best_x_ = X_[ibest]; }

    const double denom = (fworst - fbest) + 1e-12;
    std::vector<double> mass(N, 0.0);
    for (int i=0; i<N; ++i){
        double m = (fworst - FX_[i]) / denom;
        if (!std::isfinite(m)) m = 0.0;
        if (m < 0.0) m = 0.0;
        if (m > 1.0) m = 1.0;
        mass[i] = m;
    }

    // food position (mass-weighted)
    Vec food(D, 0.0);
    double wsum = 0.0;
    for (int i=0; i<N; ++i){
        wsum += mass[i];
        for (int j=0; j<D; ++j) food[j] += mass[i] * X_[i][j];
    }
    if (wsum > 1e-12) {
        for (int j=0; j<D; ++j) food[j] /= wsum;
    } else {
        food = best_x_;
    }

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> U11(-1.0, 1.0);
    std::uniform_int_distribution<int> Iidx(0, N-1);

    // precompute best unit directions and food unit directions on-demand
    for (int i=0; i<N; ++i){
        // induced motion direction
        Vec alpha(D, 0.0);
        const int k = std::min(std::max(1, neighbor_k_), N-1);
        for (int t=0; t<k; ++t){
            int j = Iidx(rng_);
            if (j == i) { j = (j + 1) % N; }
            Vec udir = unitDir(X_[j], X_[i]);
            const double w = (mass[j] - mass[i]);
            for (int d=0; d<D; ++d) alpha[d] += w * udir[d];
        }
        // best guidance
        {
            Vec bdir = unitDir(best_x_, X_[i]);
            for (int d=0; d<D; ++d) alpha[d] += c_best_ * bdir[d];
        }
        const double invk = 1.0 / (double)(k + 1);
        for (int d=0; d<D; ++d) alpha[d] *= invk;

        Vec Ni(D, 0.0);
        for (int d=0; d<D; ++d) {
            Ni[d] = nmax_abs * alpha[d] + wn_ * Nprev_[i][d];
        }

        // foraging motion direction
        Vec beta(D, 0.0);
        {
            Vec fdir = unitDir(food, X_[i]);
            Vec bdir = unitDir(best_x_, X_[i]);
            for (int d=0; d<D; ++d) beta[d] = c_food_ * fdir[d] + c_best_ * bdir[d];
        }
        for (int d=0; d<D; ++d) beta[d] *= 0.5;

        Vec Fi(D, 0.0);
        for (int d=0; d<D; ++d) {
            Fi[d] = vf_abs * beta[d] + wf_ * Fprev_[i][d];
        }

        // diffusion
        Vec Di(D, 0.0);
        if (dmax_abs > 0.0) {
            for (int d=0; d<D; ++d) Di[d] = dmax_abs * U11(rng_);
        }

        // velocity & position update
        Vec xnew = X_[i];
        for (int d=0; d<D; ++d){
            const double v = Ni[d] + Fi[d] + Di[d];
            xnew[d] = xnew[d] + dt_ * v;
        }
        ensureBounds(xnew);

        // Evaluate candidate
        double fnew = eval(xnew);

        // Always update motion memories (so dynamics evolve even if greedy rejects)
        Nprev_[i] = Ni;
        Fprev_[i] = Fi;

        bool accept = true;
        if (greedy_) accept = (fnew < FX_[i]);

        if (accept) {
            if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_) {
                auto [xloc, floc] = localSearch(local_method_, xnew);
                if (std::isfinite(floc) && floc < fnew) { xnew = std::move(xloc); fnew = floc; }
            }
            X_[i]  = std::move(xnew);
            FX_[i] = fnew;
        }

        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (std::isfinite(fnew) && fnew < best_f_) { best_f_ = fnew; best_x_ = accept ? X_[i] : xnew; }

        if (prob_->calls() >= max_evals_) break;
    }

    // Elitism: write best to the current worst slot for consistent reporting
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        if (worst_idx < X_.size()) {
            X_[worst_idx]  = best_x_;
            FX_[worst_idx] = best_f_;
        }
    }

    ++iter_;
    printBest();
    updateStop(FX_);
}

// ---------------------------------------------------------------------------
// model = paper (default): the Krill Herd motion of Gandomi & Alavi (2012),
// "Krill herd: A new bio-inspired optimization algorithm", Commun. Nonlinear
// Sci. Numer. Simul. 17, 4831-4845, eqs. (2)-(19):
//   induced motion  N_i = Nmax*(a_local + a_target) + wn*N_i_old
//     a_local  = sum_{j: ||Xj-Xi|| < ds_i} K_ij X_ij,  K_ij = (K_i-K_j)/(K_w-K_b),
//                X_ij = (X_j-X_i)/(||X_j-X_i||+eps),  ds_i = sum_j ||X_i-X_j|| / (5N)
//     a_target = C_best K_ib X_ib,  C_best = 2 (rand + I/Imax)
//   foraging        F_i = Vf*(b_food + b_best) + wf*F_i_old
//     b_food = C_food K_if X_if,  C_food = 2 (1 - I/Imax),  b_best = K_i,ibest X_i,ibest
//   diffusion       D_i = Dmax (1 - I/Imax) delta,  delta ~ U[-1,1]^n
//   X_i(t+dt) = X_i + dt (N_i + F_i + D_i)
// FIX: the previous update (now model = legacy) used CONSTANT unit-vector
// pulls (no fitness coefficients K, no C_best / C_food schedules, random
// neighbours instead of the sensing distance), so every krill kept moving
// with a step of fixed length and the method could not refine (sphere-10D
// stalled at f ~ 25). Framework adaptations: Nmax, Vf, Dmax <= 1 are
// fractions of the mean box width; the food position is the fitness-weighted
// centre (the paper's sum X_i/K_i assumes K_i > 0); I/Imax = evaluation
// progress; c_best / c_food scale the schedules (2 = paper).
// ---------------------------------------------------------------------------
void KH::paperIteration(){
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0 || prob_->calls() >= max_evals_) { printBest(); updateStop(FX_); return; }
    if ((int)P_.size() != N) { P_ = X_; FP_ = FX_; }

    const double prog = clamp01(progressFraction());
    const double mr = meanRange();
    const double nmax_abs  = (nmax_  <= 1.0 ? nmax_  * mr : nmax_);
    const double vf_abs    = (vf_    <= 1.0 ? vf_    * mr : vf_);
    const double dmax_abs  = (dmax0_ <= 1.0 ? dmax0_ * mr : dmax0_) * (1.0 - prog);
    const double eps = 1e-300;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_real_distribution<double> U11(-1.0, 1.0);

    // K_best = best so far, K_worst = worst current krill
    double Kw = -std::numeric_limits<double>::infinity();
    for (int i=0; i<N; ++i) if (std::isfinite(FX_[i]) && FX_[i] > Kw) Kw = FX_[i];
    const double Kb = best_f_;
    const double den = Kw - Kb;
    const double invden = (std::isfinite(den) && den > 0.0) ? 1.0 / den : 0.0;
    auto Kc = [&](double a, double b)->double{
        if (!std::isfinite(a) || !std::isfinite(b)) return 0.0;
        double v = (a - b) * invden;
        return std::isfinite(v) ? v : 0.0;
    };

    // food centre (fitness-weighted) and its fitness
    Vec food(D, 0.0);
    {
        double ws = 0.0;
        for (int i=0; i<N; ++i){
            double m = (invden > 0.0 && std::isfinite(FX_[i])) ? (Kw - FX_[i]) * invden : 0.0;
            if (!(m > 0.0)) continue;
            ws += m;
            for (int j=0; j<D; ++j) food[j] += m * X_[i][j];
        }
        if (ws > 0.0) { for (int j=0; j<D; ++j) food[j] /= ws; }
        else food = best_x_;
        ensureBounds(food);
    }
    double Kfood = std::numeric_limits<double>::quiet_NaN();
    if (prob_->calls() < max_evals_) {
        Kfood = eval(food);
        if (Kfood < best_f_) { best_f_ = Kfood; best_x_ = food; }
    }

    // pairwise distances for the sensing distance
    std::vector<double> dist((size_t)N * (size_t)N, 0.0);
    for (int i=0; i<N; ++i)
        for (int k=i+1; k<N; ++k){
            double s2 = 0.0;
            for (int j=0; j<D; ++j){ const double d = X_[k][j] - X_[i][j]; s2 += d*d; }
            dist[(size_t)i*N + k] = dist[(size_t)k*N + i] = std::sqrt(s2);
        }

    const std::vector<Vec> Xc = X_;            // positions at the start of the iteration
    const std::vector<double> Fc = FX_;

    for (int i=0; i<N; ++i){
        if (prob_->calls() >= max_evals_) break;
        const Vec& xi = Xc[(size_t)i];
        const double Ki = Fc[(size_t)i];

        // --- induced motion ---
        double ds = 0.0;
        for (int k=0; k<N; ++k) ds += dist[(size_t)i*N + k];
        ds /= (5.0 * N);
        Vec alpha(D, 0.0);
        for (int k=0; k<N; ++k){
            if (k == i) continue;
            const double dik = dist[(size_t)i*N + k];
            if (!(dik < ds)) continue;
            const double kij = Kc(Ki, Fc[(size_t)k]);
            if (kij == 0.0) continue;
            for (int j=0; j<D; ++j) alpha[j] += kij * (Xc[(size_t)k][j] - xi[j]) / (dik + eps);
        }
        {
            const double Cb = c_best_ * (U01(rng_) + prog);
            const double kib = Kc(Ki, Kb);
            Vec u = unitDir(best_x_, xi);
            for (int j=0; j<D; ++j) alpha[j] += Cb * kib * u[j];
        }
        Vec Ni(D);
        for (int j=0; j<D; ++j) Ni[j] = nmax_abs * alpha[j] + wn_ * Nprev_[i][j];

        // --- foraging motion ---
        Vec beta(D, 0.0);
        if (std::isfinite(Kfood)) {
            const double Cf = c_food_ * (1.0 - prog);
            const double kif = Kc(Ki, Kfood);
            Vec u = unitDir(food, xi);
            for (int j=0; j<D; ++j) beta[j] += Cf * kif * u[j];
        }
        {
            const double kibest = Kc(Ki, FP_[(size_t)i]);
            if (kibest != 0.0) {
                Vec u = unitDir(P_[(size_t)i], xi);
                for (int j=0; j<D; ++j) beta[j] += kibest * u[j];
            }
        }
        Vec Fi(D);
        for (int j=0; j<D; ++j) Fi[j] = vf_abs * beta[j] + wf_ * Fprev_[i][j];

        // --- physical diffusion + position update ---
        Vec xnew = xi;
        for (int j=0; j<D; ++j){
            const double Dij = (dmax_abs > 0.0 ? dmax_abs * U11(rng_) : 0.0);
            xnew[j] += dt_ * (Ni[j] + Fi[j] + Dij);
        }
        ensureBounds(xnew);
        double fnew = eval(xnew);
        Nprev_[i] = std::move(Ni);
        Fprev_[i] = std::move(Fi);

        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_
            && prob_->calls() < max_evals_) {
            auto [xloc, floc] = localSearch(local_method_, xnew);
            if (std::isfinite(floc) && floc < fnew) { xnew = std::move(xloc); fnew = floc; }
        }

        const bool accept = greedy_ ? (fnew < FX_[(size_t)i]) : true;
        if (accept) { X_[(size_t)i] = xnew; FX_[(size_t)i] = fnew; }
        if (fnew < FP_[(size_t)i]) { P_[(size_t)i] = xnew; FP_[(size_t)i] = fnew; }
        if (fnew < best_f_) { best_f_ = fnew; best_x_ = xnew; }
    }

    // elitism: the best-so-far replaces the worst krill
    if (!X_.empty()) {
        size_t wi = 0;
        for (size_t k=1; k<FX_.size(); ++k) if (FX_[k] > FX_[wi]) wi = k;
        if (best_f_ < FX_[wi]) {
            X_[wi] = best_x_; FX_[wi] = best_f_;
            Nprev_[wi].assign((size_t)D, 0.0); Fprev_[wi].assign((size_t)D, 0.0);
            if (best_f_ < FP_[wi]) { P_[wi] = best_x_; FP_[wi] = best_f_; }
        }
    }

    ++iter_;
    printBest();
    updateStop(FX_);
}

void KH::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    // Write refined best to worst position
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
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
