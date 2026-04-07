#include "hho.h"
#include "init.h"
#include <cstdio>
#include <cmath>
#include <limits>

namespace optimsolution {

static constexpr double kPI = 3.141592653589793238462643383279502884;

void HHO::ensureBounds(Vec& v){
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

HHO::Vec HHO::meanPop() const {
    const int N = (int)X_.size();
    if (N <= 0) return {};
    const int D = (int)X_[0].size();
    Vec m(D, 0.0);
    for (int i=0; i<N; ++i){
        for (int j=0; j<D; ++j) m[j] += X_[i][j];
    }
    for (int j=0; j<D; ++j) m[j] /= (double)N;
    return m;
}

HHO::Vec HHO::levyFlight(int D){
    // Mantegna's algorithm
    const double beta = beta_;
    const double num = std::tgamma(1.0 + beta) * std::sin(kPI * beta / 2.0);
    const double den = std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0);
    double sigma_u = std::pow(num / den, 1.0 / beta);

    std::normal_distribution<double> Nu(0.0, sigma_u);
    std::normal_distribution<double> Nv(0.0, 1.0);

    Vec step(D, 0.0);
    for (int j=0; j<D; ++j){
        double u = Nu(rng_);
        double v = Nv(rng_);
        double denomv = std::pow(std::fabs(v) + 1e-12, 1.0 / beta);
        step[j] = u / denomv;
        if (!std::isfinite(step[j])) step[j] = 0.0;
    }
    return step;
}

void HHO::init(){
    if (!prob_) return;

    const int D = prob_->dimension();
    const int N = std::max(4, (pop_override_ >= 4 ? pop_override_ : population()));
    this->setPopulation(N);

    X_.clear(); FX_.clear();

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

    iter_ = 0;

    if (debug_hho_) {
        std::string lm  = (local_method_.empty() ? std::string("none") : local_method_);
        std::string fem = (end_local_method_.empty() ? std::string("none") : end_local_method_);
        std::fprintf(stdout,
            "[hho] cfg -> N=%d (population() now=%d, override=%d), beta=%.4f, in-run: %s @ %.4f, final@end: %s (%s)\n",
            N, population(), pop_override_, beta_, lm.c_str(), local_rate_,
            end_local_refine_ ? "on" : "off", fem.c_str());
        std::fflush(stdout);
    }

    printBest();
}

void HHO::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    const int N = (int)X_.size();
    if (N <= 0) return;

    const int T_est = std::max(1, (int)(max_evals_ / std::max(1, N)));
    const double ratio = (T_est <= 1 ? 1.0 : std::min(1.0, (double)iter_ / (double)(T_est - 1)));

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Iidx(0, std::max(0, N-1));
    std::normal_distribution<double>       N01(0.0, 1.0);

    Vec meanX = meanPop();
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();

    for (int i=0; i<N; ++i){
        if (prob_->calls() >= max_evals_) break;

        const double r = U01(rng_);
        const double E0 = 2.0 * U01(rng_) - 1.0;
        const double E  = 2.0 * E0 * (1.0 - ratio);
        const double J  = 2.0 * (1.0 - U01(rng_));

        Vec Xnew = X_[i];
        double fnew = std::numeric_limits<double>::infinity();
        bool f_computed = false;

        if (std::fabs(E) >= 1.0){
            // Exploration
            const int q = Iidx(rng_);
            const Vec& Xrand = X_[q];

            if (r < 0.5){
                for (int j=0; j<D; ++j){
                    double r1 = U01(rng_);
                    double r2 = U01(rng_);
                    Xnew[j] = Xrand[j] - r1 * std::fabs(Xrand[j] - 2.0 * r2 * X_[i][j]);
                }
            } else {
                for (int j=0; j<D; ++j){
                    double lo = (j < (int)L.size() ? L[j] : -1.0);
                    double hi = (j < (int)U.size() ? U[j] :  1.0);
                    if (lo > hi) std::swap(lo, hi);
                    double r3 = U01(rng_);
                    double r4 = U01(rng_);
                    Xnew[j] = (best_x_[j] - meanX[j]) - r3 * (lo + r4 * (hi - lo));
                }
            }
        } else {
            // Exploitation
            if (r >= 0.5 && std::fabs(E) >= 0.5){
                // Soft besiege
                for (int j=0; j<D; ++j){
                    Xnew[j] = best_x_[j] - E * std::fabs(J * best_x_[j] - X_[i][j]);
                }
            } else if (r >= 0.5 && std::fabs(E) < 0.5){
                // Hard besiege
                for (int j=0; j<D; ++j){
                    Xnew[j] = best_x_[j] - E * std::fabs(best_x_[j] - X_[i][j]);
                }
            } else if (r < 0.5 && std::fabs(E) >= 0.5){
                // Soft besiege + progressive rapid dives
                Vec Y(D, 0.0);
                for (int j=0; j<D; ++j){
                    Y[j] = best_x_[j] - E * std::fabs(J * best_x_[j] - X_[i][j]);
                }
                ensureBounds(Y);
                double fY = eval(Y);
                if (prob_->calls() >= max_evals_) {
                    // Accept Y if better; stop early
                    if (fY < FX_[i]) { X_[i] = Y; FX_[i] = fY; if (fY < best_f_) { best_f_=fY; best_x_=Y; } }
                    break;
                }

                Vec LF = levyFlight(D);
                Vec Z = Y;
                for (int j=0; j<D; ++j){
                    Z[j] = Y[j] + U01(rng_) * LF[j] * std::fabs(Y[j] - X_[i][j]);
                }
                ensureBounds(Z);
                double fZ = eval(Z);

                if (fY <= fZ) { Xnew = std::move(Y); fnew = fY; }
                else          { Xnew = std::move(Z); fnew = fZ; }
                f_computed = true;
            } else {
                // Hard besiege + progressive rapid dives
                Vec Y(D, 0.0);
                for (int j=0; j<D; ++j){
                    Y[j] = best_x_[j] - E * std::fabs(J * best_x_[j] - meanX[j]);
                }
                ensureBounds(Y);
                double fY = eval(Y);
                if (prob_->calls() >= max_evals_) {
                    if (fY < FX_[i]) { X_[i] = Y; FX_[i] = fY; if (fY < best_f_) { best_f_=fY; best_x_=Y; } }
                    break;
                }

                Vec LF = levyFlight(D);
                Vec Z = Y;
                for (int j=0; j<D; ++j){
                    Z[j] = Y[j] + U01(rng_) * LF[j] * std::fabs(Y[j] - meanX[j]);
                }
                ensureBounds(Z);
                double fZ = eval(Z);

                if (fY <= fZ) { Xnew = std::move(Y); fnew = fY; }
                else          { Xnew = std::move(Z); fnew = fZ; }
                f_computed = true;
            }
        }

        ensureBounds(Xnew);

        if (!f_computed){
            fnew = eval(Xnew);
        }

        // Greedy selection + optional in-run local
        if (fnew < FX_[i]){
            if (local_rate_ > 0.0 && !local_method_.empty()){
                if (U01(rng_) < local_rate_){
                    auto [xloc, floc] = localSearch(local_method_, Xnew);
                    if (std::isfinite(floc) && floc < fnew){
                        Xnew = std::move(xloc);
                        fnew = floc;
                    }
                }
            }
            X_[i] = std::move(Xnew);
            FX_[i] = fnew;

            if (fnew < best_f_){
                best_f_ = fnew;
                best_x_ = X_[i];
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // elitism: write best to worst
    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    iter_++;

    printBest();
    updateStop(FX_);
}

void HHO::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty())     return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (std::isfinite(floc) && floc < best_f_) {
        best_f_ = floc;
        best_x_ = std::move(xloc);
    }

    if (!X_.empty() && !FX_.empty()) {
        size_t worst_idx = 0;
        double worst_val = FX_[0];
        for (size_t k=1; k<FX_.size(); ++k){
            if (FX_[k] > worst_val) { worst_val = FX_[k]; worst_idx = k; }
        }
        X_[worst_idx]  = best_x_;
        FX_[worst_idx] = best_f_;
    }

    printBest();
}

} // namespace optimsolution
