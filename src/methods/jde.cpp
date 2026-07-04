#include "jde.h"
#include "init.h"

#include <numeric>   // iota

namespace optimsolution  {

void jDE::configure(const MethodConfig& mc)
{
    // Per-method population. If provided, the base value is updated immediately for correct reporting.
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    // jDE parameters
    F_min_   = mc.getDbl("F_min",   F_min_);
    F_max_   = mc.getDbl("F_max",   F_max_);
    CR_init_ = mc.getDbl("CR_init", CR_init_);
    tau1_    = mc.getDbl("tau1",    tau1_);
    tau2_    = mc.getDbl("tau2",    tau2_);

    if (F_min_ < 0.0) F_min_ = 0.0;
    if (F_max_ <= F_min_) F_max_ = F_min_ + 0.9;
    if (F_max_ > 1.5) F_max_ = 1.5; // typical values around 0.9-1.0

    if (CR_init_ < 0.0) CR_init_ = 0.0;
    if (CR_init_ > 1.0) CR_init_ = 1.0;

    if (tau1_ < 0.0) tau1_ = 0.1;
    if (tau1_ > 1.0) tau1_ = 1.0;
    if (tau2_ < 0.0) tau2_ = 0.1;
    if (tau2_ > 1.0) tau2_ = 1.0;

    // In-run local search
    local_rate_   = mc.getDbl("local_rate", local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;
    local_method_ = mc.getStr("local_method", local_method_);

    // Final local (can also be set per method)
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
}

void jDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();

    // If a per-method population is defined, it is enforced.
    if (pop_cfg_ > 0) {
        Optimizer::setPopulation(pop_cfg_);
    }

    pop_init_ = std::max(4, Optimizer::population());
    setPopulation(pop_init_); // to be displayed correctly in the summary

    X_.clear();
    FX_.clear();
    F_i_.clear();
    CR_i_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    int N = static_cast<int>(X_.size());

    FX_.assign(N, std::numeric_limits<double>::infinity());
    F_i_.assign(N, 0.5 * (F_min_ + F_max_));
    CR_i_.assign(N, CR_init_);

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (int i = 0; i < N; ++i) {
        if ((int)X_[i].size() != D) {
            X_[i].assign(D, 0.0);
        }
        ensureBounds(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    updateStop(FX_);
    printBest();
}

void jDE::ensureBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int  D = prob_->dimension();
    const int  m = static_cast<int>(std::min<std::size_t>(D, x.size()));

    for (int j = 0; j < m; ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
    for (std::size_t j = m; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.0;
    }
}

void jDE::one_iteration()
{
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    if (X_.empty()) return;

    const int D = prob_->dimension();
    int       N = static_cast<int>(X_.size());
    if (N < 4 || D <= 0) return;

    if ((int)FX_.size()  != N) FX_.resize(N, std::numeric_limits<double>::infinity());
    if ((int)F_i_.size() != N) F_i_.assign(N, 0.5 * (F_min_ + F_max_));
    if ((int)CR_i_.size()!= N) CR_i_.assign(N, CR_init_);

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int>     Ui(0, N - 1);

    std::vector<Vec>    newPop(N);
    std::vector<double> newFit(N, std::numeric_limits<double>::quiet_NaN());

    auto randIndexExcl = [&](int avoid1, int avoid2, int avoid3) {
        int r;
        do {
            r = Ui(rng_);
        } while (r == avoid1 || r == avoid2 || r == avoid3);
        return r;
    };

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        // Self-adaptation of F_i and CR_i (Brest jDE)
        // FIX (logic): in canonical jDE the freshly sampled F_i/CR_i are part
        // of the individual and survive ONLY if the trial vector wins the
        // selection; otherwise the old values are kept. Previously F_i_[i] and
        // CR_i_[i] were overwritten unconditionally before the trial and never
        // reverted on failure, so the self-adaptation degenerated into a pure
        // random walk of the control parameters (no selection pressure on
        // F/CR), which noticeably hurts convergence.
        const double F_old  = F_i_[i];
        const double CR_old = CR_i_[i];

        double Fi  = F_old;
        double CRi = CR_old;

        double r1 = U01(rng_);
        if (r1 < tau1_) {
            Fi = F_min_ + U01(rng_) * (F_max_ - F_min_);
        }
        double r2 = U01(rng_);
        if (r2 < tau2_) {
            CRi = U01(rng_);
        }

        // Mutation: DE/rand/1
        int r1_idx = randIndexExcl(i, -1, -1);
        int r2_idx = randIndexExcl(i, r1_idx, -1);
        int r3_idx = randIndexExcl(i, r1_idx, r2_idx);

        Vec v(D);
        for (int j = 0; j < D; ++j) {
            v[j] = X_[r1_idx][j] + Fi * (X_[r2_idx][j] - X_[r3_idx][j]);
        }

        // Binomial crossover
        Vec u(D);
        std::uniform_int_distribution<int> UiD(0, D - 1);
        int jrand = UiD(rng_);
        for (int j = 0; j < D; ++j) {
            double r = U01(rng_);
            if (r <= CRi || j == jrand) u[j] = v[j];
            else                        u[j] = X_[i][j];
        }

        ensureBounds(u);
        double f_new = eval(u);
        if (prob_->calls() >= max_evals_) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
            break;
        }

        if (f_new <= FX_[i]) {
            newPop[i] = u;
            newFit[i] = f_new;
            // Trial won the selection: the sampled control parameters survive.
            F_i_[i]  = Fi;
            CR_i_[i] = CRi;
        } else {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
            // Trial lost: the individual keeps its previous F/CR (F_old/CR_old
            // are still stored in F_i_[i]/CR_i_[i], nothing to do).
        }

        if (newFit[i] < best_f_) {
            best_f_ = newFit[i];
            best_x_ = newPop[i];
        }
    }

    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(newFit[i])) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // In-run local search (GA-style as in PPSO)
    if (!local_method_.empty() && local_rate_ > 0.0) {
        std::uniform_real_distribution<double> Uloc(0.0, 1.0);
        for (int i = 0; i < N && prob_->calls() < max_evals_; ++i) {
            if (Uloc(rng_) < local_rate_) {
                auto res = localSearch(local_method_, X_[i]);
                const Vec& xloc = res.first;
                double     floc = res.second;
                if (!xloc.empty() &&
                    xloc.size() == static_cast<std::size_t>(D) &&
                    std::isfinite(floc) && floc < FX_[i]) {
                    X_[i]  = xloc;
                    FX_[i] = floc;
                    if (floc < best_f_) {
                        best_f_ = floc;
                        best_x_ = xloc;
                    }
                }
            }
        }
    }

    updateStop(FX_);
    printBest();
}

void jDE::end()
{
    if (!prob_) return;

    const int D = prob_->dimension();

    if (end_local_refine_ && !end_local_method_.empty() && !best_x_.empty()) {
        auto res = localSearch(end_local_method_, best_x_);
        const Vec& xloc = res.first;
        double     floc = res.second;
        if (!xloc.empty() &&
            xloc.size() == static_cast<std::size_t>(D) &&
            std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = xloc;
        }

        if (!X_.empty()) {
            int    worst = 0;
            double fw    = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) {
                    fw = FX_[i];
                    worst = i;
                }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }

        printBest();
    }

    updateStop(FX_);
}

} // namespace optimsolution
