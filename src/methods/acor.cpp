#include "acor.h"
#include "init.h"
#include <random>
#include <limits>
#include <numeric>

namespace optimsolution {

void ACOR::ensureBounds(std::vector<double>& x){
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j=0; j<x.size(); ++j){
        if (!std::isfinite(x[j])) x[j] = 0.5*(L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void ACOR::buildWeights(){
    // Rank-based Gaussian weights as in the ACOR literature:
    // w_i ∝ exp( - (i-1)^2 / (2*(q*k)^2) ), i=1..k (k=archive_)
    // Then normalization and a CDF for roulette-wheel selection.
    int k = (int)FA_.size();
    if (k <= 0) { W_.clear(); WCDF_.clear(); return; }

    W_.assign(k, 0.0);
    double denom = 2.0 * std::pow(q_ * k, 2.0);
    if (denom <= 0.0) denom = 1e-9;

    for (int i=0; i<k; ++i){
        double a = (double)i; // i = 0..k-1 (instead of 1..k) — equivalent up to a shift
        W_[i] = std::exp(-(a*a)/denom);
    }
    double sumw = std::accumulate(W_.begin(), W_.end(), 0.0);
    if (sumw <= 0.0) {
        // fallback: uniform
        W_.assign(k, 1.0/(double)k);
    } else {
        for (double& w : W_) w /= sumw;
    }
    // CDF
    WCDF_.resize(k);
    double acc = 0.0;
    for (int i=0; i<k; ++i){
        acc += W_[i];
        WCDF_[i] = (i==k-1) ? 1.0 : acc;
    }
}

int ACOR::sampleIndexByWeights(){
    int k = (int)WCDF_.size();
    if (k <= 0) return 0;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    double r = U01(rng_);
    for (int i=0; i<k; ++i){
        if (r <= WCDF_[i]) return i;
    }
    return k-1;
}

double ACOR::sigma_ij(int i, int j) const {
    // sigma_{i,j} = xi * (1/(k-1)) * sum_{r != i} |A[i][j] - A[r][j]|
    // (classic in the ACOR core). A small minimum is added for numerical stability.
    int k = (int)A_.size();
    if (k <= 1) return 1e-12;
    double mu = A_[i][j];
    double s = 0.0;
    for (int r=0; r<k; ++r){
        if (r == i) continue;
        s += std::fabs(mu - A_[r][j]);
    }
    s /= (double)(k-1);
    double sig = xi_ * s;
    if (!(sig > 0.0) || !std::isfinite(sig)) sig = 1e-12;
    return sig;
}

void ACOR::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    // Seed the archive using the initializer
    Initializer initSampler;
    initSampler.configure(initopt_);

    // Take enough samples to populate a good archive.
    int seedN = std::max({archive_, pop_, 2*archive_});
    auto X0 = initSampler.samplePopulation(*prob_, rng_, seedN);

    std::vector<double> F0; F0.reserve(X0.size());
    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (auto& x : X0){
        ensureBounds(x);
        double f = eval(x);
        F0.push_back(f);
        if (f < best_f_) { best_f_ = f; best_x_ = x; }
        if (prob_->calls() >= max_evals_) break;
    }

    // Sort and keep the k best in the archive
    std::vector<int> ord(X0.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return F0[a] < F0[b]; });

    int k = std::min(archive_, (int)ord.size());
    A_.assign(k, std::vector<double>(D, 0.0));
    FA_.assign(k, std::numeric_limits<double>::infinity());

    for (int i=0; i<k; ++i){
        A_[i]  = std::move(X0[ord[i]]);
        FA_[i] = F0[ord[i]];
    }

    // Prepare working buffers for ants
    X_.assign(pop_, std::vector<double>(D, 0.0));
    FX_.assign(pop_, std::numeric_limits<double>::infinity());

    // Pre-build weights
    buildWeights();

    updateStop(FA_);
	printBest();
}

void ACOR::one_iteration(){
    if (!prob_) return;
    const int D = prob_->dimension();
    std::normal_distribution<double> N01(0.0, 1.0);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // Sampling pop_ ants
    for (int i=0; i<pop_; ++i){
        for (int j=0; j<D; ++j){
            int s = sampleIndexByWeights();   // select a "center" from the archive
            double mu = A_[s][j];
            double sg = sigma_ij(s, j);
            double val = mu + sg * N01(rng_);
            X_[i][j] = val;
        }
        ensureBounds(X_[i]);

        double f = eval(X_[i]);

        // optional in-run local search
        if (local_rate_ > 0.0 && !local_method_.empty() && U01(rng_) < local_rate_){
            auto [xl, fl] = localSearch(local_method_, X_[i]);
            if (fl < f){ X_[i] = std::move(xl); f = fl; }
        }
        FX_[i] = f;

        if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // Merge: Archive U Ants, keep the k best
    std::vector<std::vector<double>> C = A_;
    std::vector<double> FC = FA_;

    // add ants
    for (int i=0; i<pop_; ++i){
        if (std::isfinite(FX_[i])) {
            C.push_back(X_[i]);
            FC.push_back(FX_[i]);
        }
    }

    std::vector<int> ord(C.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return FC[a] < FC[b]; });

    int k = std::min(archive_, (int)ord.size());
    A_.assign(k, std::vector<double>(D, 0.0));
    FA_.assign(k, std::numeric_limits<double>::infinity());
    for (int i=0; i<k; ++i){
        A_[i]  = std::move(C[ord[i]]);
        FA_[i] = FC[ord[i]];
    }

    // refresh weights (rank-based) based on the new ranking
    buildWeights();

	updateStop(FA_);
	printBest();
}

void ACOR::end(){
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    // replace the worst archive member with the best (alignment with other methods)
    if (!A_.empty() && !FA_.empty()){
        size_t worst = 0; double fw = FA_[0];
        for (size_t k=1; k<FA_.size(); ++k){
            if (FA_[k] > fw){ fw = FA_[k]; worst = k; }
        }
        A_[worst]  = best_x_;
        FA_[worst] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
