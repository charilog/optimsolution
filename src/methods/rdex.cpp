#include "rdex.h"
#include "init.h"
#include <numeric>

namespace optimsolution {

int RDEx::pickRandomExcept(int exclude1, int exclude2) {
    const int N = static_cast<int>(X_.size());
    if (N <= 1) return 0;
    std::uniform_int_distribution<int> ui(0, N - 1);
    int r = ui(rng_);
    int guard = 0;
    while ((r == exclude1 || r == exclude2) && guard < 100) {
        r = ui(rng_);
        ++guard;
    }
    return r;
}

double RDEx::sampleTruncGaussian(double mean, double sigma, double lo, double hi) {
    std::normal_distribution<double> nd(mean, sigma > 0.0 ? sigma : 1e-6);
    for (int t = 0; t < 25; ++t) {
        const double v = nd(rng_);
        if (v >= lo && v <= hi) return v;
    }
    return std::min(std::max(mean, lo), hi);
}

double RDEx::sampleCauchyParam(double loc, double scale, double lo, double hi) {
    std::cauchy_distribution<double> cd(loc, scale > 0.0 ? scale : 1e-6);
    for (int t = 0; t < 25; ++t) {
        const double v = cd(rng_);
        if (v >= lo && v <= hi) return v;
    }
    return std::min(std::max(loc, lo), hi);
}

void RDEx::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    pop_ = N0_; // front size N^(0) <- N0 (Alg. 1, line 1)

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, std::max(pop_, 2));
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (size_t i = 0; i < X_.size(); ++i) {
        const double f = eval(X_[i]);
        FX_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    // Memories initialised to 0.5 (standard SHADE/L-SHADE convention; Alg. 1, line 2).
    mem_F_.assign(H_, 0.5);
    mem_CR_.assign(H_, 0.5);
    mem_pos_ = 0;
    rho_eb_  = rho_eb0_;
    sr_prev_ = 0.0;

    updateStop(FX_);
    printBest();
}

void RDEx::one_iteration() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int N = static_cast<int>(X_.size());
    if (N < 4) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> UH(0, H_ - 1);
    std::uniform_int_distribution<int> Ujrand(0, D - 1);

    // Selection window size p^(g) (Eq. 3).
    int p_window = static_cast<int>(std::floor(double(N) * xi_ * std::exp(-k_ * sr_prev_)));
    if (p_window < 2) p_window = 2;
    if (p_window > N) p_window = N;
    std::uniform_int_distribution<int> Uwin(0, p_window - 1);

    // Fitness-ascending order of the current front, needed for pbest/EB donors.
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

    // Mean of the standard-branch F distribution, driven by the last success rate (Eq. 7).
    const double mu_F = 0.4 + 0.25 * std::tanh(5.0 * sr_prev_);
    const double progress = (max_evals_ > 0)
        ? double(prob_->calls()) / double(max_evals_)
        : 1.0;

    std::vector<Vec>    Xnew(N);
    std::vector<double> Fnew(N, std::numeric_limits<double>::infinity());

    // Bookkeeping for the memory update (Eq. 8-9) and the hybrid-rate update (Eq. 6).
    std::vector<double> succ_F, succ_CR, succ_dF;
    double sumDf_eb = 0.0, sumDf_std = 0.0;
    int n_success = 0;

    for (int i = 0; i < N; ++i) {
        const bool useEB = (U01(rng_) < rho_eb_);

        // ---- sample (F_i, CR_i) from the success-history memories ----
        double F_i, CR_i;
        if (!useEB) {
            F_i = sampleTruncGaussian(mu_F, sigma_F_, 0.0, 1.0);
            const int r = UH(rng_);
            CR_i = sampleTruncGaussian(mem_CR_[r], sigma_cr_, 0.0, 1.0);
        } else {
            const int r = UH(rng_);
            double locF = mem_F_[r];
            if (!(locF > 0.0)) locF = 0.5; // fallback reference value (report, Sec. II-E)
            F_i = sampleCauchyParam(locF, 0.1, 0.0, 1.0);
            // Early-stage lower bound on CR to encourage larger crossovers (report, Sec. II-E).
            const double crFloor = (progress < 0.25) ? 0.5 : 0.0;
            const double crRaw = sampleTruncGaussian(mem_CR_[r], sigma_cr_, 0.0, 1.0);
            CR_i = std::max(crRaw, crFloor);
        }

        // ---- donor vector v_i ----
        Vec v(D);
        if (!useEB) {
            // Standard branch: current-to-pbest/1 + extra difference (Eq. 2).
            const int pbestIdx = order[Uwin(rng_)];
            const int r1 = pickRandomExcept(i);
            const int r2 = pickRandomExcept(i, r1);
            for (int j = 0; j < D; ++j) {
                v[j] = X_[i][j]
                     + F_i * (X_[pbestIdx][j] - X_[i][j])
                     + F_i * (X_[r1][j] - X_[r2][j]);
            }
        } else {
            // EB branch: ordered best/mid/worst donor set (Eq. 5).
            const int aIdx = order[Uwin(rng_)];
            int b = pickRandomExcept(i, aIdx);
            int c = pickRandomExcept(i, aIdx);
            int guard = 0;
            while (c == b && guard++ < 20) c = pickRandomExcept(i, aIdx);
            int trio[3] = { aIdx, b, c };
            std::sort(trio, trio + 3, [&](int x, int y) { return FX_[x] < FX_[y]; });
            const int bestI = trio[0], midI = trio[1], worstI = trio[2];
            for (int j = 0; j < D; ++j) {
                v[j] = X_[i][j]
                     + F_i * (X_[bestI][j] - X_[i][j])
                     + F_i * (X_[midI][j] - X_[worstI][j]);
            }
        }

        // ---- binomial crossover + bound repair by resampling (Eq. 4) ----
        Vec u = X_[i];
        std::vector<bool> fromDonor(D, false);
        const int jrand = Ujrand(rng_);
        for (int j = 0; j < D; ++j) {
            if (U01(rng_) < CR_i || j == jrand) {
                double val = v[j];
                if (!std::isfinite(val) || val < L[j] || val > U[j]) {
                    std::uniform_real_distribution<double> ud(L[j], U[j]);
                    val = ud(rng_);
                }
                u[j] = val;
                fromDonor[j] = true;
            }
        }

        // ---- Cauchy local perturbation on ONE non-crossover dimension (Eq. 10) ----
        // BUGFIX: applying the p_r gate independently to every non-crossover
        // dimension makes the expected number of perturbed dimensions grow
        // linearly with D (e.g. ~0.3 dims at D=6 but ~10+ dims at D=200), so
        // the heavy-tailed Cauchy noise becomes an increasingly disruptive
        // force exactly as D grows -- this reproduced the observed pattern of
        // RDEx performing fine on low-D problems but catastrophically on
        // high-D ones. The standard iLSHADE-RSP-style local perturbation
        // instead targets a single randomly chosen non-crossover dimension
        // per individual, so its disruptiveness is D-independent by design.
        if (U01(rng_) < p_r_) {
            std::vector<int> nonCrossoverDims;
            nonCrossoverDims.reserve(D);
            for (int j = 0; j < D; ++j) {
                if (!fromDonor[j]) nonCrossoverDims.push_back(j);
            }
            if (!nonCrossoverDims.empty()) {
                std::uniform_int_distribution<int> Ud(0, static_cast<int>(nonCrossoverDims.size()) - 1);
                const int j = nonCrossoverDims[Ud(rng_)];
                std::cauchy_distribution<double> cd(u[j], sigma_loc_ * (U[j] - L[j]));
                double val = cd(rng_);
                if (!std::isfinite(val) || val < L[j] || val > U[j]) {
                    std::uniform_real_distribution<double> ud(L[j], U[j]);
                    val = ud(rng_);
                }
                u[j] = val;
            }
        }

        double fu = eval(u);

        // ---- greedy selection ----
        if (fu <= FX_[i]) {
            const double df = FX_[i] - fu;
            Xnew[i] = std::move(u);
            Fnew[i] = fu;
            if (df > 0.0) { // strict improvements feed the adaptation signals
                succ_F.push_back(F_i);
                succ_CR.push_back(CR_i);
                succ_dF.push_back(df);
                if (useEB) sumDf_eb += df; else sumDf_std += df;
                ++n_success;
            }
        } else {
            Xnew[i] = X_[i];
            Fnew[i] = FX_[i];
        }

        if (local_rate_ > 0.0 && local_method_ != "none" && !local_method_.empty()
            && U01(rng_) < local_rate_) {
            auto [xl, fl] = localSearch(local_method_, Xnew[i]);
            if (fl < Fnew[i]) { Xnew[i] = std::move(xl); Fnew[i] = fl; }
        }

        if (Fnew[i] < best_f_) { best_f_ = Fnew[i]; best_x_ = Xnew[i]; }

        if (prob_->calls() >= max_evals_) {
            for (int k = i + 1; k < N; ++k) { Xnew[k] = X_[k]; Fnew[k] = FX_[k]; }
            break;
        }
    }

    X_  = std::move(Xnew);
    FX_ = std::move(Fnew);

    // Success rate of this generation, used by mu_F and p^(g+1) next generation.
    sr_prev_ = (N > 0) ? double(n_success) / double(N) : 0.0;

    // ---- adaptive hybrid rate rho_EB (Eq. 6) ----
    if (sumDf_eb + sumDf_std > 0.0) {
        rho_eb_ = sumDf_eb / (sumDf_eb + sumDf_std);
        // Keep both branches alive (avoid the hybrid rate collapsing to 0 or 1).
        rho_eb_ = std::min(std::max(rho_eb_, 0.05), 0.95);
    }

    // ---- success-history memory update, weighted Lehmer mean (Eq. 8-9) ----
    if (n_success > 0) {
        double sumW = 0.0;
        for (double df : succ_dF) sumW += df;
        if (sumW > 0.0) {
            double numF = 0.0, denF = 0.0, numCR = 0.0, denCR = 0.0;
            for (int t = 0; t < n_success; ++t) {
                const double w = succ_dF[t] / sumW;
                numF  += w * succ_F[t]  * succ_F[t];
                denF  += w * succ_F[t];
                numCR += w * succ_CR[t] * succ_CR[t];
                denCR += w * succ_CR[t];
            }
            if (denF > 1e-12)  mem_F_[mem_pos_]  = numF / denF;
            if (denCR > 1e-12) mem_CR_[mem_pos_] = 0.5 * (mem_CR_[mem_pos_] + numCR / denCR);
            mem_pos_ = (mem_pos_ + 1) % H_;
        }
    }

    // ---- linear population size reduction (Eq. 11) ----
    const long long nfe = prob_->calls();
    const long long budget = std::max<long long>(1, max_evals_);
    int Nnext = static_cast<int>(std::floor(
        double(N0_) + (double(Nmin_) - double(N0_)) * (double(nfe) / double(budget))));
    if (Nnext < Nmin_) Nnext = Nmin_;
    if (Nnext > static_cast<int>(X_.size())) Nnext = static_cast<int>(X_.size()); // never grow back

    if (Nnext < static_cast<int>(X_.size())) {
        std::vector<int> idx(X_.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

        std::vector<Vec>    Xk(Nnext);
        std::vector<double> Fk(Nnext);
        for (int i = 0; i < Nnext; ++i) { Xk[i] = X_[idx[i]]; Fk[i] = FX_[idx[i]]; }
        X_  = std::move(Xk);
        FX_ = std::move(Fk);
        pop_ = Nnext; // keep the base class's population() accessor consistent
    }

    updateStop(FX_);
    printBest();
}

void RDEx::end() {
    if (!end_local_refine_ || !prob_) return;
    if (end_local_method_.empty()) return;

    auto [xloc, floc] = localSearch(end_local_method_, best_x_);
    if (floc < best_f_) {
        best_f_ = floc;
        best_x_ = xloc;
    }

    if (!X_.empty() && !FX_.empty()) {
        size_t worst = 0; double fw = FX_[0];
        for (size_t k = 1; k < FX_.size(); ++k) {
            if (FX_[k] > fw) { fw = FX_[k]; worst = k; }
        }
        X_[worst]  = best_x_;
        FX_[worst] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
