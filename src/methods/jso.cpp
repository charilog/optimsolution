#include "jso.h"
#include "init.h"

#include <numeric>  // iota, accumulate
#include <cctype>

namespace optimsolution {

void JSO::configure(const MethodConfig& mc)
{
    // Base: current global population (from [global]).
    int basePop = population();
    if (basePop < 4) basePop = 4;

    // If a population is provided in [jso], it takes precedence.
    int pop_override = mc.getInt("population", 0);
    if (pop_override > 3) {
        pop_init_ = pop_override;
    } else {
        pop_init_ = basePop;
    }

    // The Optimizer is also configured to appear correctly in the summary.
    setPopulation(pop_init_);

    // np_min
    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    // H (memory size)
    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    // p_min / p_max as in mLSHADE_RL.
    pmin_ = mc.getDbl("p_min", pmin_);
    pmax_ = mc.getDbl("p_max", pmax_);
    if (pmin_ <= 0.0) pmin_ = 0.05;
    if (pmax_ <= pmin_) pmax_ = std::max(pmin_ + 0.05, 0.5);

    arc_rate_ = mc.getDbl("arc_rate", arc_rate_);
    if (arc_rate_ < 0.0) arc_rate_ = 0.0;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;

    // In-run local search
    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;
}

void JSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    // Safety: if a value is missing for any reason, the Optimizer value is used.
    if (pop_init_ <= 0) {
        pop_init_ = population();
        if (pop_init_ < 4) pop_init_ = 50;
        setPopulation(pop_init_);
    }

    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    archive_.clear();

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < X_.size(); ++i) {
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    // Initialize the F/CR memory.
    MF_.assign(H_, 0.5);
    MCR_.assign(H_, 0.8);
    mem_idx_ = 0;

    updateStop(FX_);
    printBest();
}

void JSO::ensureInBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

void JSO::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    int N = static_cast<int>(X_.size());
    if (N < 4) return;

    double maxFES = static_cast<double>(std::max<long long>(max_evals_, 1));
    double nfes   = static_cast<double>(prob_->calls());

    // --- Linear population size reduction (LPSR) ---
    int targetN = static_cast<int>(
        std::round(pop_init_ - (pop_init_ - pop_min_) * (nfes / maxFES))
    );
    if (targetN < pop_min_)  targetN = pop_min_;
    if (targetN > pop_init_) targetN = pop_init_;

    if (targetN < N) {
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return FX_[a] < FX_[b]; });

        std::vector<Vec>    newX;
        std::vector<double> newF;
        newX.reserve(targetN);
        newF.reserve(targetN);

        for (int k = 0; k < targetN; ++k) {
            newX.push_back(X_[idx[k]]);
            newF.push_back(FX_[idx[k]]);
        }
        X_.swap(newX);
        FX_.swap(newF);
        N = targetN;
    }

    if (N < 4) return;

    // --- p-best size ---
    nfes = static_cast<double>(prob_->calls());
    double fes_ratio = nfes / maxFES;
    if (fes_ratio < 0.0) fes_ratio = 0.0;
    if (fes_ratio > 1.0) fes_ratio = 1.0;

    double p = pmin_ + (pmax_ - pmin_) * fes_ratio;
    if (p < pmin_) p = pmin_;
    if (p > pmax_) p = pmax_;

    int p_best_size = static_cast<int>(std::round(p * N));
    if (p_best_size < 2) p_best_size = 2;
    if (p_best_size > N) p_best_size = N;

    // Indices are sorted by fitness.
    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    std::vector<Vec>    newPop(N, Vec(D));
    std::vector<double> newFit(N);

    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> dF;

    SF.reserve(N);
    SCR.reserve(N);
    dF.reserve(N);

    std::uniform_real_distribution<double> U01(0.0, 1.0);

    auto pickDistinct = [&](int count, int avoid1, int avoid2) {
        std::vector<int> pool;
        pool.reserve(N);
        for (int j = 0; j < N; ++j) {
            if (j == avoid1 || j == avoid2) continue;
            pool.push_back(j);
        }
        if (count > (int)pool.size()) count = (int)pool.size();
        std::shuffle(pool.begin(), pool.end(), rng_);
        pool.resize(count);
        return pool;
    };

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        // --- memory selection for F/CR ---
        std::uniform_int_distribution<int> Ui_mem(0, H_ - 1);
        int r_mem = Ui_mem(rng_);

        // jSO tweak: the last memory slot is pushed toward 0.9.
        if (H_ > 0 && r_mem == H_ - 1) {
            MF_[r_mem]  = 0.9;
            MCR_[r_mem] = 0.9;
        }

        double muF  = MF_[r_mem];
        double muCR = MCR_[r_mem];

        // --- F sampled from Cauchy(muF, scale) ---
        double Fi;
        {
            std::cauchy_distribution<double> cauchy(muF, cauchy_scale_F_);
            do {
                Fi = cauchy(rng_);
            } while (Fi <= 0.0);
            if (Fi > 1.0) Fi = 1.0;
        }

        // --- CR sampled from Normal(muCR, std) ---
        double CRi;
        if (muCR < 0.0) {
            CRi = 0.0;
        } else {
            std::normal_distribution<double> normal(muCR, normal_std_CR_);
            CRi = normal(rng_);
        }
        if (CRi < 0.0) CRi = 0.0;
        if (CRi > 1.0) CRi = 1.0;

        // --- Early-stage CR/F adjustments (jSO-like) ---
        double g_ratio = static_cast<double>(prob_->calls()) / maxFES;
        if (g_ratio < 0.25) {
            if (CRi < 0.7) CRi = 0.7;
        } else if (g_ratio < 0.5) {
            if (CRi < 0.6) CRi = 0.6;
        }
        if (g_ratio < 0.6 && Fi > 0.7) {
            Fi = 0.7;
        }

        // --- Weighted F_w ---
        double fes_ratio_now = static_cast<double>(prob_->calls()) / maxFES;
        double Fw;
        if (fes_ratio_now < 0.2) {
            Fw = 0.7 * Fi;
        } else if (fes_ratio_now < 0.4) {
            Fw = 0.8 * Fi;
        } else {
            Fw = 1.2 * Fi;
        }

        // --- p-best selection ---
        std::uniform_int_distribution<int> Ui_pbest(0, p_best_size - 1);
        int p_sorted_idx = Ui_pbest(rng_);
        int p_idx        = idx_sorted[p_sorted_idx];

        // --- r1, r2 selection ---
        auto r = pickDistinct(2, i, p_idx);
        int r1 = r[0];
        int r2 = r[1];

        const Vec& xi  = X_[i];
        const Vec& xp  = X_[p_idx];
        const Vec& xr1 = X_[r1];

        Vec xr2;
        if (!archive_.empty() && U01(rng_) < 0.5) {
            std::uniform_int_distribution<int> Ui_arc(0, (int)archive_.size() - 1);
            xr2 = archive_[Ui_arc(rng_)];
        } else {
            xr2 = X_[r2];
        }

        std::uniform_int_distribution<int> Ui_dim(0, D - 1);
        int jrand = Ui_dim(rng_);

        Vec ui(D);
        Vec vi(D);

        // current-to-pbest-w/1 mutation + binomial crossover
        for (int j = 0; j < D; ++j) {
            if (U01(rng_) < CRi || j == jrand) {
                vi[j] = xi[j]
                      + Fw * (xp[j] - xi[j])
                      + Fi * (xr1[j] - xr2[j]);
                ui[j] = vi[j];
            } else {
                ui[j] = xi[j];
            }
        }

        ensureInBounds(ui);

        double f_old = FX_[i];
        double f_new = eval(ui);

        bool improved = (f_new <= f_old);
        if (improved) {
            newPop[i] = ui;
            newFit[i] = f_new;

            // Archive update.
            if (arc_rate_ > 0.0) {
                archive_.push_back(xi);
                int max_arc = static_cast<int>(std::round(arc_rate_ * pop_init_));
                if (max_arc < 0) max_arc = 0;

                if (max_arc == 0) {
                    archive_.clear();
                } else {
                    while ((int)archive_.size() > max_arc) {
                        std::uniform_int_distribution<int> Ui_arc(0, (int)archive_.size() - 1);
                        int idx_remove = Ui_arc(rng_);
                        archive_.erase(archive_.begin() + idx_remove);
                    }
                }
            }

            double diff = f_old - f_new;
            if (diff < 0.0) diff = 0.0;
            SF.push_back(Fi);
            SCR.push_back(CRi);
            dF.push_back(diff);

            // In-run local search with probability local_rate_.
            if (local_rate_ > 0.0 && !local_method_.empty() && prob_->calls() < max_evals_) {
                double ru = U01(rng_);
                if (ru < local_rate_) {
                    auto res = localSearch(local_method_, newPop[i]);
                    if (!res.first.empty() &&
                        std::isfinite(res.second) &&
                        res.second < newFit[i]) {
                        newPop[i] = std::move(res.first);
                        newFit[i] = res.second;
                    }
                }
            }

            if (f_new < best_f_) {
                best_f_ = f_new;
                best_x_ = newPop[i];
            }
        } else {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    // Any remaining uninitialized entries are filled.
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(newFit[i])) {
            newPop[i] = X_[i];
            newFit[i] = FX_[i];
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    // --- MF/MCR memory update ---
    if (!SF.empty()) {
        double sum_dF = 0.0;
        for (double v : dF) sum_dF += v;
        if (sum_dF <= 0.0) sum_dF = 1.0;

        std::vector<double> w(SF.size());
        for (size_t k = 0; k < SF.size(); ++k) {
            w[k] = dF[k] / sum_dF;
        }

        double sum_wF  = 0.0;
        double sum_wF2 = 0.0;
        double sum_wCR = 0.0;
        double sum_w   = 0.0;

        for (size_t k = 0; k < SF.size(); ++k) {
            double wk  = w[k];
            double Fk  = SF[k];
            double CRk = SCR[k];

            sum_w   += wk;
            sum_wF  += wk * Fk;
            sum_wF2 += wk * Fk * Fk;
            sum_wCR += wk * CRk;
        }

        if (sum_w > 0.0 && sum_wF > 0.0) {
            double meanF_Lehmer = sum_wF2 / sum_wF;
            double meanCR       = sum_wCR / sum_w;

            MF_[mem_idx_]  = (1.0 - c_mem_) * MF_[mem_idx_]  + c_mem_ * meanF_Lehmer;
            MCR_[mem_idx_] = (1.0 - c_mem_) * MCR_[mem_idx_] + c_mem_ * meanCR;

            ++mem_idx_;
            if (mem_idx_ >= H_) mem_idx_ = 0;
        }
    }

    updateStop(FX_);
    printBest();
}

void JSO::end()
{
    if (!prob_) return;
    if (!end_local_refine_) return;
    if (end_local_method_.empty()) return;
    if (best_x_.empty()) return;

    auto res = localSearch(end_local_method_, best_x_);
    if (!res.first.empty() &&
        std::isfinite(res.second) &&
        res.second < best_f_) {

        best_x_ = std::move(res.first);
        best_f_ = res.second;

        // Replace the worst individual with the polished best.
        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < (int)FX_.size(); ++i) {
                if (FX_[i] > fw) { fw = FX_[i]; worst = i; }
            }
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
