#include "bjso.h"
#include "init.h"

#include <numeric>
#include <cctype>

namespace optimsolution {

namespace {

static inline std::string to_lower_copy(std::string s)
{
    for (auto& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static inline std::string trim_copy(const std::string& s)
{
    size_t a = 0;
    size_t b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

static inline bool parse_bool_like(const std::string& s, bool fallback)
{
    const std::string t = to_lower_copy(trim_copy(s));
    if (t == "1" || t == "true" || t == "on" || t == "yes") return true;
    if (t == "0" || t == "false" || t == "off" || t == "no") return false;
    return fallback;
}

} // namespace

void BJSO::configure(const MethodConfig& mc)
{
    int basePop = population();
    if (basePop < 4) basePop = 4;

    int pop_override = mc.getInt("population", 0);
    if (pop_override > 3) {
        pop_init_ = pop_override;
    } else {
        pop_init_ = basePop;
    }

    setPopulation(pop_init_);

    pop_min_ = mc.getInt("np_min", pop_min_);
    if (pop_min_ < 4) pop_min_ = 4;
    if (pop_min_ > pop_init_) pop_min_ = pop_init_;

    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    pmin_ = mc.getDbl("p_min", pmin_);
    pmax_ = mc.getDbl("p_max", pmax_);
    if (pmin_ <= 0.0) pmin_ = 0.05;
    if (pmax_ <= pmin_) pmax_ = std::max(2.0 * pmin_, 0.25);

    arc_rate_ = mc.getDbl("arc_rate", arc_rate_);
    if (arc_rate_ < 0.0) arc_rate_ = 0.0;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;

    local_method_ = mc.getStr("local_method", local_method_);
    for (auto& c : local_method_) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    double lr = mc.getDbl("local_rate", local_rate_);
    if (lr < 0.0) lr = 0.0;
    if (lr > 1.0) lr = 1.0;
    local_rate_ = lr;

    bj_enable_ = parse_bool_like(mc.getStr("bj_enable", bj_enable_ ? "1" : "0"), bj_enable_);

    bj_band_beta_ = mc.getDbl("bj_band_beta", bj_band_beta_);
    if (bj_band_beta_ < 0.0) bj_band_beta_ = 0.0;
    if (bj_band_beta_ > 10.0) bj_band_beta_ = 10.0;
}

void BJSO::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

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

    MF_.assign(H_, 0.3);
    MCR_.assign(H_, 0.8);
    if (H_ > 0) {
        MF_[H_ - 1]  = 0.9;
        MCR_[H_ - 1] = 0.9;
    }
    mem_idx_ = 0;

    updateStop(FX_);
    printBest();
}

void BJSO::ensureInBounds(Vec& x)
{
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) x[j] = 0.5 * (L[j] + U[j]);
        if (x[j] < L[j]) x[j] = L[j];
        if (x[j] > U[j]) x[j] = U[j];
    }
}

bool BJSO::isInBounds(const Vec& x) const
{
    if (!prob_) return false;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    if (x.size() != L.size() || x.size() != U.size()) return false;
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j])) return false;
        if (x[j] < L[j] || x[j] > U[j]) return false;
    }
    return true;
}

void BJSO::trimArchive(int max_size)
{
    if (max_size < 0) max_size = 0;
    while ((int)archive_.size() > max_size) {
        std::uniform_int_distribution<int> Ui_arc(0, (int)archive_.size() - 1);
        archive_.erase(archive_.begin() + Ui_arc(rng_));
    }
}

Vec BJSO::meanOfRange(const std::vector<int>& idx_sorted, int begin_idx, int end_idx) const
{
    const int D = prob_->dimension();
    Vec mean(D, 0.0);
    const int N = static_cast<int>(idx_sorted.size());
    if (N <= 0) return mean;

    if (begin_idx < 0) begin_idx = 0;
    if (end_idx > N) end_idx = N;
    if (begin_idx >= end_idx) return mean;

    const int count = end_idx - begin_idx;
    for (int k = begin_idx; k < end_idx; ++k) {
        const Vec& x = X_[idx_sorted[k]];
        for (int j = 0; j < D; ++j) mean[j] += x[j];
    }

    const double inv = 1.0 / static_cast<double>(count);
    for (int j = 0; j < D; ++j) mean[j] *= inv;
    return mean;
}

void BJSO::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int D = prob_->dimension();
    int N = static_cast<int>(X_.size());
    if (N < 4) return;

    const double maxFES = static_cast<double>(std::max<long long>(max_evals_, 1));
    double nfes = static_cast<double>(prob_->calls());

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

    trimArchive(static_cast<int>(std::round(arc_rate_ * N)));

    nfes = static_cast<double>(prob_->calls());
    double fes_ratio = nfes / maxFES;
    if (fes_ratio < 0.0) fes_ratio = 0.0;
    if (fes_ratio > 1.0) fes_ratio = 1.0;

    double p = pmax_ - (pmax_ - pmin_) * fes_ratio;
    if (p < pmin_) p = pmin_;
    if (p > pmax_) p = pmax_;

    int p_best_size = static_cast<int>(std::round(p * N));
    if (p_best_size < 2) p_best_size = 2;
    if (p_best_size > N) p_best_size = N;

    std::vector<int> idx_sorted(N);
    std::iota(idx_sorted.begin(), idx_sorted.end(), 0);
    std::sort(idx_sorted.begin(), idx_sorted.end(),
              [&](int a, int b) { return FX_[a] < FX_[b]; });

    Vec x_band2_10;
    if (bj_enable_) {
        int band2_begin  = static_cast<int>(std::floor(0.02 * N));
        int band10_end  = static_cast<int>(std::floor(0.10 * N));

        if (band2_begin < 0) band2_begin = 0;
        if (band2_begin >= N) band2_begin = std::max(0, N - 1);

        if (band10_end <= band2_begin) band10_end = std::min(N, band2_begin + 1);
        if (band10_end > N) band10_end = N;

        x_band2_10 = meanOfRange(idx_sorted, band2_begin, band10_end);
    }

    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    std::vector<double> SF;
    std::vector<double> SCR;
    std::vector<double> dF;
    SF.reserve(N);
    SCR.reserve(N);
    dF.reserve(N);

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Ui_dim(0, D - 1);
    std::uniform_int_distribution<int> Ui_mem(0, H_ - 1);

    auto samplePopIndex = [&](int avoid1, int avoid2, int avoid3) {
        std::uniform_int_distribution<int> Ui_pop(0, N - 1);
        int idx;
        do {
            idx = Ui_pop(rng_);
        } while (idx == avoid1 || idx == avoid2 || idx == avoid3);
        return idx;
    };

    auto sampleR2 = [&](int avoid1, int avoid2, int avoid3, Vec& xr2) {
        const int total = N + static_cast<int>(archive_.size());
        if (total <= 0) return false;

        std::uniform_int_distribution<int> Ui_union(0, total - 1);
        for (int tries = 0; tries < 128; ++tries) {
            int pick = Ui_union(rng_);
            if (pick < N) {
                if (pick == avoid1 || pick == avoid2 || pick == avoid3) continue;
                xr2 = X_[pick];
                return true;
            }
            xr2 = archive_[pick - N];
            return true;
        }

        int r2 = samplePopIndex(avoid1, avoid2, avoid3);
        xr2 = X_[r2];
        return true;
    };

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        const int r_mem = Ui_mem(rng_);
        const double muF  = (H_ > 0 && r_mem == H_ - 1) ? 0.9 : MF_[r_mem];
        const double muCR = (H_ > 0 && r_mem == H_ - 1) ? 0.9 : MCR_[r_mem];

        double Fi;
        {
            std::cauchy_distribution<double> cauchy(muF, cauchy_scale_F_);
            do {
                Fi = cauchy(rng_);
            } while (Fi <= 0.0);
            if (Fi > 1.0) Fi = 1.0;
        }

        double CRi;
        if (muCR < 0.0) {
            CRi = 0.0;
        } else {
            std::normal_distribution<double> normal(muCR, normal_std_CR_);
            CRi = normal(rng_);
        }
        if (CRi < 0.0) CRi = 0.0;
        if (CRi > 1.0) CRi = 1.0;

        const double g_ratio = static_cast<double>(prob_->calls()) / maxFES;
        if (g_ratio < 0.25) {
            if (CRi < 0.7) CRi = 0.7;
        } else if (g_ratio < 0.5) {
            if (CRi < 0.6) CRi = 0.6;
        }
        if (g_ratio < 0.6 && Fi > 0.7) {
            Fi = 0.7;
        }

        const double fes_ratio_now = static_cast<double>(prob_->calls()) / maxFES;
        double Fw;
        if (fes_ratio_now < 0.2) {
            Fw = 0.7 * Fi;
        } else if (fes_ratio_now < 0.4) {
            Fw = 0.8 * Fi;
        } else {
            Fw = 1.2 * Fi;
        }

        const Vec& xi = X_[i];

        Vec ui(D);
        bool feasible = false;
        for (int repeat = 0; repeat < 64 && !feasible; ++repeat) {
            std::uniform_int_distribution<int> Ui_pbest(0, p_best_size - 1);
            const int p_idx = idx_sorted[Ui_pbest(rng_)];
            const Vec& xp = X_[p_idx];

            const int r1 = samplePopIndex(i, p_idx, -1);
            const Vec& xr1 = X_[r1];

            Vec xr2;
            if (!sampleR2(i, p_idx, r1, xr2)) {
                break;
            }

            const int jrand = Ui_dim(rng_);
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < CRi || j == jrand) {
                    double donor = xi[j]
                                 + Fw * (xp[j] - xi[j])
                                 + Fi * (xr1[j] - xr2[j]);

                    if (bj_enable_) {
                        donor += (bj_band_beta_ * Fw) * (x_band2_10[j] - xi[j]);
                    }

                    ui[j] = donor;
                } else {
                    ui[j] = xi[j];
                }
            }
            feasible = isInBounds(ui);
        }

        if (!feasible) {
            ensureInBounds(ui);
        }

        const double f_old = FX_[i];
        const double f_new = eval(ui);
        const bool accepted = (f_new <= f_old);

        if (accepted) {
            newPop[i] = ui;
            newFit[i] = f_new;

            if (local_rate_ > 0.0 && !local_method_.empty() && prob_->calls() < max_evals_) {
                const double ru = U01(rng_);
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

            const double accepted_f = newFit[i];
            const bool strict_success = (accepted_f < f_old);
            if (strict_success) {
                if (arc_rate_ > 0.0) {
                    archive_.push_back(xi);
                }
                SF.push_back(Fi);
                SCR.push_back(CRi);
                dF.push_back(f_old - accepted_f);
            }

            if (accepted_f < best_f_) {
                best_f_ = accepted_f;
                best_x_ = newPop[i];
            }
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);
    trimArchive(static_cast<int>(std::round(arc_rate_ * static_cast<int>(X_.size()))));

    if (!SF.empty()) {
        double sum_dF = 0.0;
        for (double v : dF) sum_dF += v;
        if (sum_dF <= 0.0) sum_dF = 1.0;

        double sum_wF = 0.0;
        double sum_wF2 = 0.0;
        double sum_wCR = 0.0;
        double sum_w = 0.0;

        for (size_t k = 0; k < SF.size(); ++k) {
            const double wk = dF[k] / sum_dF;
            const double Fk = SF[k];
            const double CRk = SCR[k];
            sum_w += wk;
            sum_wF += wk * Fk;
            sum_wF2 += wk * Fk * Fk;
            sum_wCR += wk * CRk;
        }

        if (sum_w > 0.0 && sum_wF > 0.0) {
            const double meanF_Lehmer = sum_wF2 / sum_wF;
            const double meanCR = sum_wCR / sum_w;

            if (H_ > 1) {
                MF_[mem_idx_]  = 0.5 * (MF_[mem_idx_]  + meanF_Lehmer);
                MCR_[mem_idx_] = 0.5 * (MCR_[mem_idx_] + meanCR);
                ++mem_idx_;
                if (mem_idx_ >= H_ - 1) mem_idx_ = 0;
            }
        }
    }

    if (H_ > 0) {
        MF_[H_ - 1]  = 0.9;
        MCR_[H_ - 1] = 0.9;
    }

    updateStop(FX_);
    printBest();
}

void BJSO::end()
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

        if (!X_.empty()) {
            int worst = 0;
            double fw = FX_[0];
            for (int i = 1; i < static_cast<int>(FX_.size()); ++i) {
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
