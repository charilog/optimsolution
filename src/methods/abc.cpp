#include "abc.h"
#include "init.h"
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

namespace optimsolution {

void ABC::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

int ABC::pickOtherIndex(int i){
    if (pop_ <= 1) return 0;
    std::uniform_int_distribution<int> Ui(0, pop_-1);
    int r = Ui(rng_);
    if (r == i) r = (r+1) % pop_;
    return r;
}

void ABC::neighborFrom(int i, int j, std::vector<double>& v){
    // v starts from X_[i] and tweaks neighbor_dims_ coordinates using j
    const int D = (int)v.size();
    v = X_[i]; // base
    std::uniform_real_distribution<double> Uphi(-1.0, 1.0);
    std::uniform_int_distribution<int> Udim(0, D-1);

    // pick dims and perturb
    for (int c=0; c<neighbor_dims_; ++c){
        int k = Udim(rng_);
        double phi = Uphi(rng_);
        v[k] = X_[i][k] + phi * (X_[i][k] - X_[j][k]);
    }
    ensureBounds(v);
}

void ABC::elitismInject(){
    if (X_.empty()) return;
    // place best into worst slot
    size_t worst = 0; double fw = FX_[0];
    for (size_t i=1; i<FX_.size(); ++i){
        if (FX_[i] > fw){ fw = FX_[i]; worst = i; }
    }
    X_[worst]  = best_x_;
    FX_[worst] = best_f_;
    trials_[worst] = 0;
}

void ABC::init(){
    if (!prob_) return;
    const int D = prob_->dimension();

    // Initialize sources (pop_ = #employed sources)
    Initializer initSampler;
    initSampler.configure(initopt_);

    X_ = initSampler.samplePopulation(*prob_, rng_, std::max(pop_, 2));
    FX_.assign((size_t)pop_, std::numeric_limits<double>::infinity());
    trials_.assign((size_t)pop_, 0);

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (int i=0; i<pop_; ++i){
        ensureBounds(X_[i]);
        double f = eval(X_[i]);
        FX_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // Elitism so the stop controller can also see the best inside FX_
    elitismInject();

    // Stop update → printing (as in DE)
    updateStop(FX_);
    printBest();
}

void ABC::employedPhase(){
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    for (int i=0; i<pop_; ++i){
        if (prob_->calls() >= max_evals_) break;

        int j = pickOtherIndex(i);
        std::vector<double> v(D);
        neighborFrom(i, j, v);

        double fv = eval(v);

        // In-run local search (optional)
        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [vl, fl] = localSearch(local_method_, v);
            if (fl < fv){ v = std::move(vl); fv = fl; }
        }

        if (fv <= FX_[i]) {
            X_[i] = std::move(v);
            FX_[i] = fv;
            trials_[i] = 0;
            if (fv < best_f_) { best_f_ = fv; best_x_ = X_[i]; }
        } else {
            trials_[i] += 1;
        }
    }
}

void ABC::onlookerPhase(){
    if (onlooker_frac_ <= 0.0) return;
    const int D = prob_->dimension();

    // Fitness mapping for minimization -> positive probabilities
    double fmin = *std::min_element(FX_.begin(), FX_.end());
    double eps = 1e-12;
    std::vector<double> fit(pop_);
    double sumfit = 0.0;
    for (int i=0; i<pop_; ++i){
        double val = 1.0 / (1.0 + std::max(0.0, FX_[i] - fmin) + eps);
        fit[i] = val;
        sumfit += val;
    }
    std::vector<double> cdf(pop_);
    double acc=0.0;
    for (int i=0; i<pop_; ++i){
        acc += fit[i] / (sumfit > 0.0 ? sumfit : 1.0);
        cdf[i] = (i==pop_-1) ? 1.0 : acc;
    }

    int onlookers = (int)std::round(pop_ * onlooker_frac_);
    if (onlookers < 1) onlookers = 1;

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    auto sampleByCDF = [&](void)->int{
        double r = U01(rng_);
        for (int i=0; i<pop_; ++i) if (r <= cdf[i]) return i;
        return pop_-1;
    };

    for (int t=0; t<onlookers; ++t){
        if (prob_->calls() >= max_evals_) break;

        int i = sampleByCDF();
        int j = pickOtherIndex(i);

        std::vector<double> v(D);
        neighborFrom(i, j, v);
        double fv = eval(v);

        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [vl, fl] = localSearch(local_method_, v);
            if (fl < fv){ v = std::move(vl); fv = fl; }
        }

        if (fv <= FX_[i]) {
            X_[i] = std::move(v);
            FX_[i] = fv;
            trials_[i] = 0;
            if (fv < best_f_) { best_f_ = fv; best_x_ = X_[i]; }
        } else {
            trials_[i] += 1;
        }
    }
}

void ABC::scoutPhase(){
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();

    auto makeRandomPoint = [&](void){
        std::vector<double> x(D);
        for (int j=0; j<D; ++j){
            std::uniform_real_distribution<double> Uj(L[j], U[j]);
            x[j] = Uj(rng_);
        }
        return x;
    };

    // 1) rule-based scouts (trial>limit)
    int scouts = 0;
    for (int i=0; i<pop_; ++i){
        if (trials_[i] > limit_) {
            std::vector<double> x = makeRandomPoint();
            ensureBounds(x);
            double f = eval(x);
            X_[i] = std::move(x);
            FX_[i] = f;
            trials_[i] = 0;
            ++scouts;
            if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
            if (prob_->calls() >= max_evals_) break;
        }
    }

    // 2) optional probabilistic scout (sparse)
    if (scout_rate_ > 0.0) {
        std::uniform_real_distribution<double> U01(0.0, 1.0);
        if (U01(rng_) < scout_rate_) {
            // Replace the worst
            size_t worst = 0; double fw = FX_[0];
            for (size_t i=1; i<FX_.size(); ++i){
                if (FX_[i] > fw){ fw = FX_[i]; worst = i; }
            }
            std::vector<double> x = makeRandomPoint();
            ensureBounds(x);
            double f = eval(x);
            X_[worst]  = std::move(x);
            FX_[worst] = f;
            trials_[worst] = 0;
            if (f < best_f_) { best_f_ = f; best_x_ = X_[worst]; }
        }
    }
}

void ABC::one_iteration(){
    if (!prob_) return;

    employedPhase();
    onlookerPhase();
    scoutPhase();

    // ELITISM (so the stop controller can see the best within FX_)
    elitismInject();

    // Order: stop update → printing
    updateStop(FX_);
    printBest();
}

void ABC::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) { best_f_ = floc; best_x_ = xloc; }

    // Elitism also at the end
    elitismInject();

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
