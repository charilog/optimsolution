#include "ude.h"
#include "init.h"
#include <numeric>
#include <utility>

namespace optimsolution {

namespace {

std::vector<int> ascendingOrder(const std::vector<double>& f, int n) {
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return f[a] < f[b]; });
    return idx;
}

} // namespace

double UDE::progress() const {
    return (max_evals_ > 0) ? double(prob_->calls()) / double(max_evals_) : 1.0;
}

double UDE::meanWL(const std::vector<double>& v, const std::vector<double>& w) const {
    // Weighted Lehmer mean (as in RDE); all-zero weights fall back to 0.5.
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

double UDE::weightedQuantile(const std::vector<double>& v, const std::vector<double>& w) const {
    // Smallest value whose cumulative weight (values in ascending order)
    // reaches q_ of the total weight. All-zero weights: the largest value.
    std::vector<int> id(v.size());
    std::iota(id.begin(), id.end(), 0);
    std::sort(id.begin(), id.end(), [&](int a, int b) { return v[a] < v[b]; });
    double tot = 0.0;
    for (double x : w) tot += x;
    if (!(tot > 0.0)) return v[id.back()];
    double acc = 0.0;
    for (int k : id) {
        acc += w[k];
        if (acc >= q_ * tot) return v[k];
    }
    return v[id.back()];
}

void UDE::findLimits(Vec& x, const Vec& parent) const {
    // Midpoint between the violated bound and the parent value (as in RDE).
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = parent[j];
        if (x[j] < L[j]) x[j] = 0.5 * (L[j] + parent[j]);
        if (x[j] > U[j]) x[j] = 0.5 * (U[j] + parent[j]);
    }
}

void UDE::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    int Nmax = (fixed_population_ > 3)
        ? fixed_population_
        : std::max(Nmin_ + 1, static_cast<int>(Nmax_mult_ * double(D)));

    // optimsolution safeguard (as in rde.cpp): with a budget that does not
    // scale with D, keep at least min_generations_ generations feasible.
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

    mem_CR_.assign(H_, muCR0_);
    mem_F_.assign(H_, muF0_);
    mem_pos_ = 0;
    qCr_  = muCR0_;
    qF_   = muF0_;
    sr_   = sr0_;
    piT_  = 0.0;
    age_.assign(X_.size(), 0);
    trail_.clear();
    lineage_ = false;
    histT_.clear();
    histS_.clear();
    histP_.clear();

    updateStop(FX_);
    printBest();
}

void UDE::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const int N = static_cast<int>(X_.size());
    if (N < 4) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> UN(0, N - 1);
    std::uniform_int_distribution<int> USlot(0, use_qentry_ ? H_ : H_ - 1); // H = quantile entry (Cr)
    std::uniform_int_distribution<int> Ujrand(0, D - 1);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::cauchy_distribution<double> C01(0.0, 1.0);

    const std::vector<int> order = ascendingOrder(FX_, N);
    int psize = static_cast<int>(double(N) * pbest_);
    if (psize < 2) psize = 2;
    if (psize > N) psize = N;
    std::uniform_int_distribution<int> Upbest(0, psize - 1);

    // ---- age guidance: freshness ranks (age 0 first), ties broken by fitness rank ----
    if ((int)age_.size() != N) age_.assign(N, 0);   // safety; sizes are kept in step below
    std::vector<int> freshAll, freshPool;            // whole population / best psize, freshest first
    std::discrete_distribution<int> pickAll, pickPool;
    if (use_age_) {
        freshAll = order;
        std::stable_sort(freshAll.begin(), freshAll.end(), [&](int a, int b) { return age_[a] < age_[b]; });
        freshPool.assign(order.begin(), order.begin() + psize);
        std::stable_sort(freshPool.begin(), freshPool.end(), [&](int a, int b) { return age_[a] < age_[b]; });
        std::vector<double> wa(N), wp(psize);           // linear ranking weights
        for (int k = 0; k < N; ++k)     wa[k] = double(N - k);
        for (int k = 0; k < psize; ++k) wp[k] = double(psize - k);
        pickAll  = std::discrete_distribution<int>(wa.begin(), wa.end());
        pickPool = std::discrete_distribution<int>(wp.begin(), wp.end());
    }

    // ---- F centre from the success rate of the previous generation (LSRTDE's rule) ----
    const double meanF = f_base_ + f_range_ * std::tanh(f_slope_ * sr_);

    std::vector<Vec>    Xt(N);
    std::vector<double> Ft(N, std::numeric_limits<double>::infinity());
    std::vector<double> Fi(N, 0.0), CRi(N, 0.0);
    std::vector<char>   evaluated(N, 0);

    for (int i = 0; i < N; ++i) {
        const int r = USlot(rng_);
        const bool qEntry = (use_qentry_ && r == H_);

        int prand, r1, r2;
        if (use_age_) {
            // ---- age-guided donors: pbest = fresh elite, r1 = fresh member, r2 uniform,
            //      pair oriented so that x_r1 - x_r2 points from the staler to the fresher ----
            prand = freshPool[pickPool(rng_)];
            for (int g = 0; prand == i && progress() < 0.5 && g < 1000; ++g) prand = freshPool[pickPool(rng_)];
            r2 = UN(rng_);
            for (int g = 0; r2 == prand && g < 1000; ++g) r2 = UN(rng_);
            r1 = freshAll[pickAll(rng_)];
            for (int g = 0; (r1 == prand || r1 == r2) && g < 1000; ++g) r1 = freshAll[pickAll(rng_)];
            if (age_[r2] < age_[r1]) std::swap(r1, r2);
        } else {
            // ---- donors: pbest uniform among the best psize, r1/r2 uniform (RDE's exclusions) ----
            prand = order[Upbest(rng_)];
            for (int g = 0; prand == i && progress() < 0.5 && g < 1000; ++g) prand = order[Upbest(rng_)];
            r1 = UN(rng_);
            for (int g = 0; r1 == prand && g < 1000; ++g) r1 = UN(rng_);
            r2 = UN(rng_);
            for (int g = 0; (r2 == prand || r2 == r1) && g < 1000; ++g) r2 = UN(rng_);
        }

        const Vec* x2 = &X_[r2];
        double F;
        if (lineage_) {
            // ---- lineage mode: trail difference with probability piT_ (as in v2-v3) ----
            const bool trailOn = (U01(rng_) < piT_);
            if (trailOn && !trail_.empty()) {
                std::uniform_int_distribution<int> UPT(0, N + (int)trail_.size() - 1);
                int k = UPT(rng_);
                for (int g = 0; k < N && (k == prand || k == r1) && g < 1000; ++g) k = UPT(rng_);
                x2 = (k < N) ? &X_[k] : &trail_[k - N];
            }
            // ---- F: Cauchy around the chosen memory entry, redrawn while < 0, clipped at 1 ----
            const double locF = qEntry ? qF_ : mem_F_[r];
            F = locF + 0.1 * C01(rng_);
            for (int g = 0; F < 0.0 && g < 1000; ++g) F = locF + 0.1 * C01(rng_);
            if (F < 0.0) F = 0.0;
            if (F > 1.0) F = 1.0;
        } else {
            // ---- turnover mode: F ~ N(meanF, f_sigma), redrawn until inside [0,1] ----
            F = meanF + f_sigma_ * N01(rng_);
            for (int g = 0; (F < 0.0 || F > 1.0) && g < 1000; ++g) F = meanF + f_sigma_ * N01(rng_);
            F = std::min(std::max(F, 0.0), 1.0);
        }
        Fi[i] = F;

        // ---- Cr: Normal around the chosen entry, clipped to [0,1] ----
        double CR;
        if (qEntry)               CR = qCr_ + 0.1 * N01(rng_);
        else if (mem_CR_[r] < 0.) CR = 0.0;
        else                      CR = mem_CR_[r] + 0.1 * N01(rng_);
        CR = std::min(std::max(CR, 0.0), 1.0);
        CRi[i] = CR;

        // ---- current-to-pbest/1 + binomial crossover ----
        Vec v(D);
        for (int j = 0; j < D; ++j)
            v[j] = X_[i][j] + F * (X_[prand][j] - X_[i][j]) + F * (X_[r1][j] - (*x2)[j]);
        findLimits(v, X_[i]);
        const int jrand = Ujrand(rng_);
        Vec u(D);
        for (int j = 0; j < D; ++j) u[j] = (U01(rng_) < CR || j == jrand) ? v[j] : X_[i][j];
        findLimits(u, X_[i]);

        const double fu = eval(u);
        if (fu < best_f_) { best_f_ = fu; best_x_ = u; }
        Xt[i] = std::move(u);
        Ft[i] = fu;
        evaluated[i] = 1;

        if (prob_->calls() >= max_evals_) break;
    }

    // ---- successful trials (target-wise: trial <= its target) ----
    std::vector<int>    succ;
    std::vector<double> succ_F, succ_CR, succ_dF;
    int nEval = 0;
    for (int i = 0; i < N; ++i) {
        if (!evaluated[i]) continue;
        ++nEval;
        if (Ft[i] <= FX_[i]) {
            succ.push_back(i);
            succ_F.push_back(Fi[i]);
            succ_CR.push_back(CRi[i]);
            succ_dF.push_back(std::fabs(FX_[i] - Ft[i]));
        }
    }
    const double SR = (nEval > 0) ? double(succ.size()) / nEval : 1.0;
    sr_ = SR;

    // ---- lineage mode: the trail keeps the parents that are replaced in this generation ----
    if (lineage_) {
        std::uniform_int_distribution<int> UT(0, std::max(0, N - 1));
        for (int k : succ) {
            if ((int)trail_.size() < N) trail_.push_back(X_[k]);
            else trail_[UT(rng_) % trail_.size()] = X_[k];
        }
    }

    // ---- 1/5-rule controller of the turnover share (turnover mode) / trail share (lineage mode) ----
    if (use_turnover_)
        piT_ = std::min(1.0, std::max(0.0, piT_ + turnover_gain_ * (0.2 - SR) / 0.2));

    // ---- ages: 0 for members whose trial succeeded, +1 for the others ----
    for (int i = 0; i < N; ++i) age_[i] = (evaluated[i] && Ft[i] <= FX_[i]) ? 0 : age_[i] + 1;

    // ---- replacement ----
    if (use_turnover_ && !lineage_) {
        // Each success becomes a new member with probability piT_ (the parent is
        // kept, the stalest unsuccessful member gives up its slot); otherwise it
        // replaces its parent as usual.
        std::vector<char> isSucc(N, 0);
        for (int k : succ) isSucc[k] = 1;
        std::vector<int> born;
        std::vector<char> inPlace(N, 0);
        for (int k : succ) {
            if (U01(rng_) < piT_) born.push_back(k);
            else inPlace[k] = 1;
        }
        for (int k : succ) if (inPlace[k]) { X_[k] = Xt[k]; FX_[k] = Ft[k]; }
        std::vector<int> stale;                       // unsuccessful members, stalest first
        for (int k = 0; k < N; ++k) if (!isSucc[k]) stale.push_back(k);
        std::stable_sort(stale.begin(), stale.end(), [&](int a, int b) {
            return age_[a] != age_[b] ? age_[a] > age_[b] : FX_[a] > FX_[b];
        });
        size_t q = 0;
        for (int k : born) {
            if (q >= stale.size()) { X_[k] = Xt[k]; FX_[k] = Ft[k]; continue; }
            const int sl = stale[q++];
            X_[sl] = Xt[k]; FX_[sl] = Ft[k]; age_[sl] = 0;
        }
    } else {
        for (int k : succ) { X_[k] = Xt[k]; FX_[k] = Ft[k]; }
    }

    // ---- optional in-run local search ----
    if (local_rate_ > 0.0 && local_method_ != "none" && !local_method_.empty()) {
        for (int i = 0; i < N; ++i) {
            if (prob_->calls() >= max_evals_) break;
            if (U01(rng_) >= local_rate_) continue;
            auto [xl, fl] = localSearch(local_method_, X_[i]);
            if (fl < FX_[i]) { X_[i] = std::move(xl); FX_[i] = fl; }
            if (FX_[i] < best_f_) { best_f_ = FX_[i]; best_x_ = X_[i]; }
        }
    }

    // ---- linear population size reduction (remove the worst) ----
    const double nfe   = double(prob_->calls());
    const double maxfe = double(std::max<long long>(1, (long long)max_evals_));
    int newN = static_cast<int>(double(Nmin_ - Nmax_) / maxfe * nfe + double(Nmax_));
    if (newN < Nmin_) newN = Nmin_;
    if (newN > N)     newN = N;
    while (static_cast<int>(X_.size()) > newN) {
        size_t worst = 0;
        for (size_t k = 1; k < FX_.size(); ++k) if (FX_[k] > FX_[worst]) worst = k;
        X_.erase(X_.begin() + worst);
        FX_.erase(FX_.begin() + worst);
        age_.erase(age_.begin() + worst);
    }
    pop_ = static_cast<int>(X_.size());
    if (static_cast<int>(trail_.size()) > newN) trail_.resize(newN);

    if (!succ_dF.empty()) {
        // ---- success-history memory update of Cr (as in RDE) ----
        const double oldCR = mem_CR_[mem_pos_];
        const double maxCR = *std::max_element(succ_CR.begin(), succ_CR.end());
        if (oldCR == -1.0 || maxCR == 0.0) mem_CR_[mem_pos_] = -1.0;
        else                               mem_CR_[mem_pos_] = 0.5 * (meanWL(succ_CR, succ_dF) + oldCR);
        mem_F_[mem_pos_] = meanWL(succ_F, succ_dF);   // used in lineage mode
        mem_pos_ = (mem_pos_ + 1) % H_;

        // ---- success-quantile entries ----
        if (use_qentry_) {
            qF_  = 0.5 * (qF_  + weightedQuantile(succ_F,  succ_dF));
            qCr_ = 0.5 * (qCr_ + weightedQuantile(succ_CR, succ_dF));
        }
    }

    // ---- switch criterion: premature collapse under turnover -> lineage mode for the rest ----
    if (use_switch_ && use_turnover_ && !lineage_) {
        const int n2 = static_cast<int>(X_.size());
        double v = 0.0;                                    // mean variance per dimension / range^2
        for (int j = 0; j < D; ++j) {
            double m = 0.0, q = 0.0;
            for (int k = 0; k < n2; ++k) m += X_[k][j];
            m /= n2;
            for (int k = 0; k < n2; ++k) q += (X_[k][j] - m) * (X_[k][j] - m);
            const double rg = prob_->ub()[j] - prob_->lb()[j];
            v += q / n2 / (rg * rg);
        }
        v = std::max(v / D, 1e-300);
        const double t = progress();
        histT_.push_back(t);
        histS_.push_back(0.5 * std::log(v));               // log of the spread
        histP_.push_back(piT_);                            // turnover share
        size_t j0 = 0, j5 = 0; bool ok = false, ok5 = false;
        for (size_t qd = 0; qd < histT_.size(); ++qd) {
            if (histT_[qd] <= t - switch_window_)       { j0 = qd; ok = true; }
            if (histT_[qd] <= t - 0.5 * switch_window_) { j5 = qd; ok5 = true; }
        }
        if (ok && ok5 && t >= switch_at_ && t <= switch_at_ + 0.02 && t - histT_[j0] > 0.0 && t - histT_[j5] > 0.0) {
            // contraction rate d ln(spread)/d(budget) over the whole window and over its second half
            const double rate  = (histS_.back() - histS_[j0]) / (t - histT_[j0]);
            const double rate2 = (histS_.back() - histS_[j5]) / (t - histT_[j5]);
            // mean turnover share over the window
            double pb = 0.0; int nb = 0;
            for (size_t qd = j0 + 1; qd < histT_.size(); ++qd) { pb += histP_[qd]; ++nb; }
            const bool turnoverDriven = (nb > 0) && (pb / nb >= switch_share_) && (piT_ >= switch_share_);
            if (rate < -switch_rate_ && rate2 < -switch_rate_ && turnoverDriven) lineage_ = true;
        }
    }

    updateStop(FX_);
    printBest();
}

void UDE::end() {
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
