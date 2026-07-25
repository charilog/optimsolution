#include "rde.h"
#include "init.h"
#include <numeric>
#include <utility>

namespace optimsolution {

double RDE::sampleTruncGaussian(double mean, double sigma, double lo, double hi) {
    std::normal_distribution<double> nd(mean, sigma > 0.0 ? sigma : 1e-6);
    for (int t = 0; t < 25; ++t) {
        const double v = nd(rng_);
        if (v >= lo && v <= hi) return v;
    }
    return std::min(std::max(mean, lo), hi);
}

double RDE::sampleCauchyParam(double loc, double scale, double lo, double hi) {
    std::cauchy_distribution<double> cd(loc, scale > 0.0 ? scale : 1e-6);
    for (int t = 0; t < 25; ++t) {
        const double v = cd(rng_);
        if (v >= lo && v <= hi) return v;
    }
    return std::min(std::max(loc, lo), hi);
}

std::discrete_distribution<int> RDE::buildRspDistribution(const std::vector<int>& order) const {
    // Rank_i = kr*(M-i)+1, i = 1-based rank (order[0] is rank 1 = best).
    const int M = static_cast<int>(order.size());
    std::vector<double> w(std::max(1, M));
    for (int pos = 0; pos < M; ++pos) {
        const double rank_i = kr_ * double(M - (pos + 1)) + 1.0;
        w[pos] = std::max(1e-9, rank_i);
    }
    if (M == 0) w[0] = 1.0;
    return std::discrete_distribution<int>(w.begin(), w.end());
}

void RDE::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    int Nmax = (fixed_population_ > 3)
        ? fixed_population_
        : std::max(Nmin_ + 1, static_cast<int>(std::round(Nmax_mult_ * double(D))));

    // Budget-aware cap: if max_evals_ is fixed regardless of D (as in this
    // codebase's [global] config, unlike the report's own max_nfes=10000*D
    // protocol), an unmodified Nmax=18*D can consume the entire budget in a
    // handful of generations for high-D problems. Ensure at least
    // min_generations_ generations remain feasible, using the average of
    // Nmax and Nmin as a rough per-generation cost estimate (population
    // shrinks linearly over the run, cf. Eq. 21).
    if (fixed_population_ <= 3 && min_generations_ > 0 && max_evals_ > 0) {
        // avgPop ~= (Nmax+Nmin)/2 evals per generation; solve for the Nmax
        // that keeps avgPop*min_generations_ <= max_evals_.
        const double budgetPerGen = double(max_evals_) / double(min_generations_);
        const int cappedNmax = static_cast<int>(std::floor(2.0 * budgetPerGen - double(Nmin_)));
        if (cappedNmax >= Nmin_ + 1 && cappedNmax < Nmax) {
            Nmax = cappedNmax;
        }
    }

    Nmax_ = std::max(Nmin_ + 1, Nmax);
    pop_ = Nmax_;

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

    mem_F_.assign(H_, muF0_);
    mem_CR_.assign(H_, muCR0_);
    if (H_ >= 1) {
        mem_F_[H_ - 1]  = 0.9; // fixed terminal slot (Eq. 17)
        mem_CR_[H_ - 1] = 0.9;
    }
    mem_pos_ = 0;
    gamma1_  = gamma1_0_;

    archiveX_.clear();
    archiveF_.clear();

    updateStop(FX_);
    printBest();
}

void RDE::one_iteration() {
    if (!prob_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int N = static_cast<int>(X_.size());
    if (N < 4) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> UH(0, H_ - 1);
    std::uniform_int_distribution<int> Ujrand(0, D - 1);

    const double progress = (max_evals_ > 0) ? double(prob_->calls()) / double(max_evals_) : 1.0;

    // Linear p reduction (Eq. 20): p in [pmax/2, pmax], deterministic in nfes.
    const double p_frac = pmax_ * (1.0 - 0.5 * progress);
    int p_window = std::max(2, static_cast<int>(std::round(p_frac * double(N))));
    if (p_window > N) p_window = N;

    // Fitness-ascending order of the population (rank 1 = best = order[0]).
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });
    std::discrete_distribution<int> popRsp = buildRspDistribution(order);

    // p-best window: RSP re-applied WITHIN the top p_window individuals only
    // (Eq. 9-10 extended to the "p" selection, per Sec. II-B-4).
    std::vector<int> pbestOrder(order.begin(), order.begin() + p_window);
    std::discrete_distribution<int> pbestRsp = buildRspDistribution(pbestOrder);

    // Joint population+archive pool, RSP-weighted, for r2 in the
    // current-to-pbest branch only (Sec. II-B-1).
    const int NA = static_cast<int>(archiveX_.size());
    std::vector<int> unionOrder(N + NA);
    {
        for (int i = 0; i < N; ++i) unionOrder[i] = i;
        for (int i = 0; i < NA; ++i) unionOrder[N + i] = N + i;
        auto fitnessOf = [&](int idx) { return idx < N ? FX_[idx] : archiveF_[idx - N]; };
        std::sort(unionOrder.begin(), unionOrder.end(),
                  [&](int a, int b) { return fitnessOf(a) < fitnessOf(b); });
    }
    std::discrete_distribution<int> unionRsp = buildRspDistribution(unionOrder);
    auto unionX = [&](int idx) -> const Vec& { return idx < N ? X_[idx] : archiveX_[idx - N]; };

    std::vector<Vec>    Xnew(N);
    std::vector<double> Fnew(N, std::numeric_limits<double>::infinity());

    // Per-strategy average-improvement bookkeeping for the gamma1 update (Eq. 6-8).
    double sumImprove1 = 0.0, sumImprove2 = 0.0;
    int n1 = 0, n2 = 0;

    // Success-history bookkeeping for the F/Cr memory update (Eq. 14-16).
    std::vector<double> succ_F, succ_CR, succ_dF;

    // Parents replaced by a strictly-better trial, queued for the archive.
    std::vector<std::pair<Vec, double>> replacedParents;

    for (int i = 0; i < N; ++i) {
        const bool useOrderPbest = (U01(rng_) < gamma1_);

        // ---- sample F_i, Cr_i (jSO-style memory + early-stage clamps, Eq. 11-13, 17-19) ----
        const int r = UH(rng_);
        double F_i  = sampleCauchyParam(mem_F_[r], 0.1, 0.0, 1.0);
        double CR_i = sampleTruncGaussian(mem_CR_[r], 0.1, 0.0, 1.0);

        if (progress < 0.6 && F_i > 0.7) F_i = 0.7;                          // Eq. 18
        if (progress < 0.25 && F_i < 0.7) CR_i = std::max(CR_i, 0.7);        // Eq. 19
        else if (progress < 0.5 && F_i < 0.6) CR_i = std::max(CR_i, 0.6);    // Eq. 19

        // ---- donor vector v_i ----
        Vec v(D);
        const int donorA = pbestOrder[pbestRsp(rng_)]; // "p" (Eq. 2) / 'a' (Eq. 5)

        if (!useOrderPbest) {
            // Standard branch: current-to-pbest/1, r2 from population+archive (Eq. 2).
            int r1 = order[popRsp(rng_)];
            for (int g = 0; (r1 == i || r1 == donorA) && g < 50; ++g) r1 = order[popRsp(rng_)];

            int r2u = unionOrder[unionRsp(rng_)];
            for (int g = 0; (r2u == i || r2u == donorA || r2u == r1) && g < 50; ++g) {
                r2u = unionOrder[unionRsp(rng_)];
            }
            const Vec& xr2 = unionX(r2u);

            for (int j = 0; j < D; ++j) {
                v[j] = X_[i][j]
                     + F_i * (X_[donorA][j] - X_[i][j])
                     + F_i * (X_[r1][j] - xr2[j]);
            }
        } else {
            // Order-pbest branch: ordered best/median/worst donor set (Eq. 5).
            int b = order[popRsp(rng_)];
            for (int g = 0; (b == i || b == donorA) && g < 50; ++g) b = order[popRsp(rng_)];
            int c = order[popRsp(rng_)];
            for (int g = 0; (c == i || c == donorA || c == b) && g < 50; ++g) c = order[popRsp(rng_)];

            int trio[3] = { donorA, b, c };
            std::sort(trio, trio + 3, [&](int x, int y) { return FX_[x] < FX_[y]; });
            const int bestI = trio[0], midI = trio[1], worstI = trio[2];
            for (int j = 0; j < D; ++j) {
                v[j] = X_[i][j]
                     + F_i * (X_[bestI][j] - X_[i][j])
                     + F_i * (X_[midI][j] - X_[worstI][j]);
            }
        }

        // ---- crossover + bound repair by resampling ----
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

        // ---- Cauchy local perturbation on ONE non-crossover dimension (Eq. 22, adapted) ----
        // See the class-level comment in rde.h for why this targets a single
        // dimension rather than gating every non-crossover dimension
        // independently (the latter's disruptiveness would grow with D).
        if (U01(rng_) < pr_) {
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

        const double fPrev = FX_[i];
        const double fu = eval(u);

        if (fu <= fPrev) {
            const double improve = fPrev - fu;
            if (fu < fPrev) {
                replacedParents.push_back({ X_[i], fPrev }); // archive the replaced parent
                succ_F.push_back(F_i);
                succ_CR.push_back(CR_i);
                succ_dF.push_back(improve);
            }
            if (useOrderPbest) { sumImprove1 += improve; ++n1; } else { sumImprove2 += improve; ++n2; }
            Xnew[i] = std::move(u);
            Fnew[i] = fu;
        } else {
            if (useOrderPbest) ++n1; else ++n2;
            Xnew[i] = X_[i];
            Fnew[i] = fPrev;
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

    // ---- external archive update (Sec. II-B-1) ----
    for (auto& pr : replacedParents) {
        archiveX_.push_back(std::move(pr.first));
        archiveF_.push_back(pr.second);
    }
    const int archiveCap = std::max(0, static_cast<int>(std::round(Ar_ * double(X_.size()))));
    while (static_cast<int>(archiveX_.size()) > archiveCap) {
        std::uniform_int_distribution<int> Uevict(0, static_cast<int>(archiveX_.size()) - 1);
        const int idx = Uevict(rng_);
        std::swap(archiveX_[idx], archiveX_.back()); archiveX_.pop_back();
        std::swap(archiveF_[idx], archiveF_.back()); archiveF_.pop_back();
    }

    // ---- gamma1/gamma2 resource-allocation update (Eq. 6-8) ----
    const double omega1 = (n1 > 0) ? (sumImprove1 / double(n1)) : 0.0;
    const double omega2 = (n2 > 0) ? (sumImprove2 / double(n2)) : 0.0;
    if (omega1 <= 0.0 && omega2 <= 0.0) {
        gamma1_ = 0.5;
    } else {
        gamma1_ = omega1 / (omega1 + omega2);
        gamma1_ = std::min(std::max(gamma1_, 0.05), 0.95); // keep both branches alive
    }

    // ---- success-history memory update, weighted Lehmer mean (Eq. 14-16) ----
    // The terminal slot (h = H-1) is fixed at 0.9 and never overwritten (Eq. 17),
    // so the rotation pointer only cycles over the H-1 updatable slots.
    if (!succ_dF.empty() && H_ > 1) {
        double sumW = 0.0;
        for (double df : succ_dF) sumW += df;
        if (sumW > 0.0) {
            double numF = 0.0, denF = 0.0, numCR = 0.0, denCR = 0.0;
            for (size_t t = 0; t < succ_dF.size(); ++t) {
                const double w = succ_dF[t] / sumW;
                numF  += w * succ_F[t]  * succ_F[t];
                denF  += w * succ_F[t];
                numCR += w * succ_CR[t] * succ_CR[t];
                denCR += w * succ_CR[t];
            }
            if (denF > 1e-12)  mem_F_[mem_pos_]  = numF / denF;
            if (denCR > 1e-12) mem_CR_[mem_pos_] = numCR / denCR;
        }
        mem_pos_ = (mem_pos_ + 1) % (H_ - 1);
    }

    // ---- linear population size reduction, Nmax -> Nmin (Eq. 21) ----
    const long long nfe = prob_->calls();
    const long long budget = std::max<long long>(1, max_evals_);
    int Nnext = static_cast<int>(std::llround(
        (double(Nmin_) - double(Nmax_)) * (double(nfe) / double(budget)) + double(Nmax_)));
    if (Nnext < Nmin_) Nnext = Nmin_;
    if (Nnext > static_cast<int>(X_.size())) Nnext = static_cast<int>(X_.size());

    if (Nnext < static_cast<int>(X_.size())) {
        std::vector<int> idx(X_.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return FX_[a] < FX_[b]; });

        std::vector<Vec>    Xk(Nnext);
        std::vector<double> Fk(Nnext);
        for (int i = 0; i < Nnext; ++i) { Xk[i] = X_[idx[i]]; Fk[i] = FX_[idx[i]]; }
        X_  = std::move(Xk);
        FX_ = std::move(Fk);
        pop_ = Nnext;
    }

    updateStop(FX_);
    printBest();
}

void RDE::end() {
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
