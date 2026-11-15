#include "rde.h"
#include "init.h"
#include <numeric>
#include <utility>

namespace optimsolution {

// Line references "RDE.cpp:NNN" point to the authors' code/rde-cec2024/RDE.cpp (v1.1.0).

namespace {

// Ordering of three donors as in the authors' EB_order() (RDE.cpp:422-501),
// including its tie-breaking: returns best, medium, worst.
struct Trio { const Vec* best; const Vec* mid; const Vec* worst; };
Trio orderTrio(const Vec* p1, double f1, const Vec* p3, double f3, const Vec* p4, double f4) {
    if (f1 <= f3 && f1 <= f4) return (f3 <= f4) ? Trio{p1, p3, p4} : Trio{p1, p4, p3};
    if (f3 <= f1 && f3 <= f4) return (f1 <= f4) ? Trio{p3, p1, p4} : Trio{p3, p4, p1};
    if (f4 <= f1 && f4 <= f3) return (f1 <= f3) ? Trio{p4, p1, p3} : Trio{p4, p3, p1};
    return Trio{p1, p3, p4}; // only reachable with NaN fitness values
}

std::vector<int> ascendingOrder(const std::vector<double>& f, int n) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return f[a] < f[b]; });
    return idx;
}

} // namespace

double RDE::progress() const {
    return (max_evals_ > 0) ? double(prob_->calls()) / double(max_evals_) : 1.0;
}

double RDE::meanWL(const std::vector<double>& v, const std::vector<double>& w) const {
    // RDE.cpp:360-379. With all weights 0 the authors' code divides 0/0 and,
    // through the NaN, falls back to 0.5; that outcome is made explicit here.
    double sumW = 0.0;
    for (double x : w) sumW += x;
    if (!(sumW > 0.0)) return 0.5;
    double sumSq = 0.0, sum = 0.0;
    for (size_t t = 0; t < v.size(); ++t) {
        const double wt = w[t] / sumW;
        sumSq += wt * v[t] * v[t];
        sum   += wt * v[t];
    }
    return (std::fabs(sum) > 1e-6) ? sumSq / sum : 0.5;
}

void RDE::findLimits(Vec& x, const Vec& parent) const {
    // RDE.cpp:166-173: midpoint between the violated bound and the parent.
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = parent[j];
        if (x[j] < L[j]) x[j] = 0.5 * (L[j] + parent[j]);
        if (x[j] > U[j]) x[j] = 0.5 * (U[j] + parent[j]);
    }
}

void RDE::addToArchive(const Vec& x, double f) {
    // RDE.cpp:380-392: append while there is room, otherwise overwrite a random slot.
    if (archiveCap_ <= 0) return;
    if (static_cast<int>(archiveX_.size()) < archiveCap_) {
        archiveX_.push_back(x);
        archiveF_.push_back(f);
    } else {
        std::uniform_int_distribution<int> ui(0, archiveCap_ - 1);
        const int k = ui(rng_);
        archiveX_[k] = x;
        archiveF_[k] = f;
    }
}

void RDE::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    int Nmax = (fixed_population_ > 3)
        ? fixed_population_
        : std::max(Nmin_ + 1, static_cast<int>(Nmax_mult_ * double(D))); // RDE.cpp:897 int(18*D)

    // optimsolution safeguard (not in the authors' code): with a budget that does
    // not scale with D, keep at least min_generations_ generations feasible.
    // With max_evals = 10000*D it never triggers (cap = 200*D - Nmin > 18*D).
    if (fixed_population_ <= 3 && min_generations_ > 0 && max_evals_ > 0) {
        const double budgetPerGen = double(max_evals_) / double(min_generations_);
        const int cappedNmax = static_cast<int>(std::floor(2.0 * budgetPerGen - double(Nmin_)));
        if (cappedNmax >= Nmin_ + 1 && cappedNmax < Nmax) Nmax = cappedNmax;
    }

    Nmax_ = std::max(Nmin_ + 1, Nmax);
    pop_ = Nmax_;

    Initializer initSampler;
    initSampler.configure(initopt_);
    X_ = initSampler.samplePopulation(*prob_, rng_, pop_);
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (size_t i = 0; i < X_.size(); ++i) {
        if (prob_->calls() >= max_evals_) break;
        const double f = eval(X_[i]);
        FX_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = X_[i]; }
    }

    mem_F_.assign(H_, muF0_);   // RDE.cpp:331-334
    mem_CR_.assign(H_, muCR0_);
    mem_pos_ = 0;
    gamma1_  = gamma1_0_;       // RDE.cpp:30 EB_hybrid_rate = 0.5

    archiveX_.clear();
    archiveF_.clear();
    archiveCap_ = static_cast<int>(double(Nmax_) * Ar_); // RDE.cpp:285

    updateStop(FX_);
    printBest();
}

void RDE::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int N = static_cast<int>(X_.size());
    if (N < 4) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> UHp1(0, H_);     // {0..H}: H = the fixed 0.9 entry
    std::uniform_int_distribution<int> Ujrand(0, D - 1);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::cauchy_distribution<double> C01(0.0, 1.0);

    // ---- rankings and RSP selectors (RDE.cpp:551-606) ----
    const std::vector<int> order = ascendingOrder(FX_, N);
    std::vector<double> wPop(N);
    for (int i = 0; i < N; ++i) wPop[i] = kr_ * double(N - i);
    std::discrete_distribution<int> popRsp(wPop.begin(), wPop.end());

    const double psize = pmax_ * (1.0 - 0.5 * progress());
    int psizeval = static_cast<int>(double(N) * psize);
    if (psizeval <= 1) psizeval = 2;
    if (psizeval > N) psizeval = N;
    std::discrete_distribution<int> pbestRsp(wPop.begin(), wPop.begin() + psizeval);

    const int NA = static_cast<int>(archiveX_.size());
    std::vector<int> orderA;
    std::discrete_distribution<int> archRsp;
    if (NA > 0) {
        orderA = ascendingOrder(archiveF_, NA);
        std::vector<double> wA(NA);
        for (int i = 0; i < NA; ++i) wA[i] = kr_ * double(NA - i);
        archRsp = std::discrete_distribution<int>(wA.begin(), wA.end());
    }

    std::vector<Vec>    Xt(N);
    std::vector<double> Ft(N, std::numeric_limits<double>::infinity());
    std::vector<char>   isEB(N, 0);
    std::vector<double> succ_F, succ_CR, succ_dF;

    for (int i = 0; i < N; ++i) {
        const int r = UHp1(rng_);

        // ---- donors (RDE.cpp:610-639) ----
        int prand = order[pbestRsp(rng_)];
        for (int g = 0; prand == i && progress() < 0.5 && g < 1000; ++g) prand = order[pbestRsp(rng_)];
        int r1 = order[popRsp(rng_)];
        for (int g = 0; r1 == prand && g < 1000; ++g) r1 = order[popRsp(rng_)];
        int r2 = order[popRsp(rng_)];
        for (int g = 0; (r2 == prand || r2 == r1) && g < 1000; ++g) r2 = order[popRsp(rng_)];

        // ---- F: Cauchy, redrawn while < 0, clipped at 1, <= 0.7 before 60% (RDE.cpp:642-665) ----
        const double locF = (r < H_) ? mem_F_[r] : 0.9;
        double F_i = locF + 0.1 * C01(rng_);
        for (int g = 0; F_i < 0.0 && g < 1000; ++g) F_i = locF + 0.1 * C01(rng_);
        if (F_i < 0.0) F_i = 0.0;
        if (F_i > 1.0) F_i = 1.0;
        if (progress() < 0.6 && F_i > 0.7) F_i = 0.7;

        // ---- mutation (RDE.cpp:669-720) ----
        const bool useEB = (U01(rng_) < gamma1_);
        isEB[i] = useEB ? 1 : 0;
        const bool fromArchive = (NA > 0) && (U01(rng_) < double(NA) / double(NA + N));
        const Vec*   x3 = fromArchive ? nullptr : &X_[r2];
        double       f3 = fromArchive ? 0.0 : FX_[r2];
        if (fromArchive) {
            const int a = orderA[archRsp(rng_)];
            x3 = &archiveX_[a];
            f3 = archiveF_[a];
        }

        Vec v(D);
        if (useEB) {
            const Trio t = orderTrio(&X_[prand], FX_[prand], &X_[r1], FX_[r1], x3, f3);
            for (int j = 0; j < D; ++j)
                v[j] = X_[i][j] + F_i * ((*t.best)[j] - X_[i][j]) + F_i * ((*t.mid)[j] - (*t.worst)[j]);
        } else {
            for (int j = 0; j < D; ++j)
                v[j] = X_[i][j] + F_i * (X_[prand][j] - X_[i][j]) + F_i * (X_[r1][j] - (*x3)[j]);
        }
        findLimits(v, X_[i]); // RDE.cpp:738

        // ---- Cr (RDE.cpp:743-773) ----
        double CR_i;
        if (r < H_) CR_i = (mem_CR_[r] < 0.0) ? 0.0 : mem_CR_[r] + 0.1 * N01(rng_);
        else        CR_i = 0.9 + 0.1 * N01(rng_);
        if (CR_i >= 1.0) CR_i = 1.0;
        if (CR_i <= 0.0) CR_i = 0.0;
        if (progress() < 0.25) CR_i = std::max(CR_i, 0.7);
        if (progress() < 0.5)  CR_i = std::max(CR_i, 0.6);

        // ---- crossover + Cauchy perturbation of the non-crossover dimensions (RDE.cpp:776-785) ----
        const int jrand = Ujrand(rng_);
        const bool perturbation = (U01(rng_) < pr_);
        Vec u(D);
        for (int j = 0; j < D; ++j) {
            if (U01(rng_) < CR_i || j == jrand) {
                u[j] = v[j];
            } else if (perturbation) {
                u[j] = X_[i][j] + sigma_loc_ * (U[j] - L[j]) * C01(rng_);
            } else {
                u[j] = X_[i][j];
            }
        }
        findLimits(u, X_[i]); // RDE.cpp:794 (v1.1.0)

        const double fu = eval(u);
        if (fu < best_f_) { best_f_ = fu; best_x_ = u; }
        if (fu <= FX_[i]) {
            succ_F.push_back(F_i);
            succ_CR.push_back(CR_i);
            succ_dF.push_back(std::fabs(FX_[i] - fu));
        }
        Xt[i] = std::move(u);
        Ft[i] = fu;

        if (prob_->calls() >= max_evals_) break; // remaining trials stay at +inf (not accepted)
    }

    // ---- EB share update: sums of improvements (RDE.cpp:503-538) ----
    {
        double sumEB = 0.0, sumStd = 0.0;
        for (int i = 0; i < N; ++i) {
            if (Ft[i] <= FX_[i]) {
                if (isEB[i]) sumEB += FX_[i] - Ft[i];
                else         sumStd += FX_[i] - Ft[i];
            }
        }
        if (sumEB != 0.0 && sumStd != 0.0) {
            gamma1_ = sumEB / (sumEB + sumStd);
            gamma1_ = std::min(std::max(gamma1_, 0.0), 1.0);
        } else {
            gamma1_ = gamma1_0_;
        }
    }

    // ---- synchronous replacement + archive (RDE.cpp:806-813) ----
    for (int i = 0; i < N; ++i) {
        if (Ft[i] <= FX_[i]) {
            addToArchive(X_[i], FX_[i]);
            X_[i]  = std::move(Xt[i]);
            FX_[i] = Ft[i];
        }
        if (local_rate_ > 0.0 && local_method_ != "none" && !local_method_.empty()
            && U01(rng_) < local_rate_ && prob_->calls() < max_evals_) {
            auto [xl, fl] = localSearch(local_method_, X_[i]);
            if (fl < FX_[i]) { X_[i] = std::move(xl); FX_[i] = fl; }
            if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        }
    }

    // ---- linear population and archive size reduction (RDE.cpp:815-827) ----
    const double nfe    = double(prob_->calls());
    const double maxfe  = double(std::max<long long>(1, (long long)max_evals_));
    int newN = static_cast<int>(double(Nmin_ - Nmax_) / maxfe * nfe + double(Nmax_));
    if (newN < Nmin_) newN = Nmin_;
    if (newN > N)     newN = N;

    int newArch = static_cast<int>((maxfe - nfe) / maxfe * (Ar_ * double(Nmax_ - Nmin_)));
    if (newArch < Nmin_) newArch = Nmin_;
    archiveCap_ = newArch;
    if (static_cast<int>(archiveX_.size()) > archiveCap_) {
        archiveX_.resize(archiveCap_);
        archiveF_.resize(archiveCap_);
    }

    // RemoveWorst (RDE.cpp:403-420): drop the worst one at a time, keeping the order of the rest.
    while (static_cast<int>(X_.size()) > newN) {
        size_t worst = 0;
        for (size_t k = 1; k < FX_.size(); ++k) if (FX_[k] > FX_[worst]) worst = k;
        X_.erase(X_.begin() + worst);
        FX_.erase(FX_.begin() + worst);
    }
    pop_ = static_cast<int>(X_.size());

    // ---- success-history memory update (RDE.cpp:342-359) ----
    if (!succ_dF.empty()) {
        const double oldCR = mem_CR_[mem_pos_];
        const double maxCR = *std::max_element(succ_CR.begin(), succ_CR.end());
        if (oldCR == -1.0 || maxCR == 0.0) mem_CR_[mem_pos_] = -1.0;   // terminal value
        else                               mem_CR_[mem_pos_] = 0.5 * (meanWL(succ_CR, succ_dF) + oldCR);
        mem_F_[mem_pos_] = meanWL(succ_F, succ_dF);
        mem_pos_ = (mem_pos_ + 1) % H_;
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
