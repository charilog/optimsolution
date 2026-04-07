#include "sfcde.h"
#include "init.h"

#include <numeric>
#include <cctype>

namespace optimsolution {

void SFCDE::configure(const MethodConfig& mc)
{
    int basePop = population();
    if (basePop < 4) basePop = 50;

    const int pop_override = mc.getInt("population", 0);
    if (pop_override >= 4) {
        pop_init_ = pop_override;
    } else {
        pop_init_ = basePop;
    }
    setPopulation(pop_init_);

    H_ = mc.getInt("H", H_);
    if (H_ < 1) H_ = 1;

    c_mem_ = mc.getDbl("c_mem", c_mem_);
    if (c_mem_ <= 0.0 || c_mem_ > 1.0) c_mem_ = 0.1;

    mu_f_init_ = mc.getDbl("mu_f_init", mu_f_init_);
    if (mu_f_init_ <= 0.0) mu_f_init_ = 0.5;
    if (mu_f_init_ > 1.0) mu_f_init_ = 1.0;

    mu_cr_init_ = mc.getDbl("mu_cr_init", mu_cr_init_);
    if (mu_cr_init_ < 0.0) mu_cr_init_ = 0.0;
    if (mu_cr_init_ > 1.0) mu_cr_init_ = 1.0;

    cauchy_scale_F_ = mc.getDbl("cauchy_scale_F", cauchy_scale_F_);
    if (cauchy_scale_F_ <= 0.0) cauchy_scale_F_ = 0.1;

    normal_std_CR_ = mc.getDbl("normal_std_CR", normal_std_CR_);
    if (normal_std_CR_ <= 0.0) normal_std_CR_ = 0.1;
}

double SFCDE::meanLehmer(const std::vector<double>& values) const
{
    double numer = 0.0;
    double denom = 0.0;
    for (double v : values) {
        numer += v * v;
        denom += v;
    }
    if (denom <= 0.0) return 0.0;
    return numer / denom;
}

double SFCDE::meanArithmetic(const std::vector<double>& values) const
{
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    return sum / static_cast<double>(values.size());
}

double SFCDE::sampleF(double mu)
{
    std::cauchy_distribution<double> dist(mu, cauchy_scale_F_);
    double F = mu;
    do {
        F = dist(rng_);
    } while (F <= 0.0);
    if (F > 1.0) F = 1.0;
    return F;
}

double SFCDE::sampleCR(double mu)
{
    std::normal_distribution<double> dist(mu, normal_std_CR_);
    double cr = dist(rng_);
    if (cr < 0.0) cr = 0.0;
    if (cr > 1.0) cr = 1.0;
    return cr;
}

void SFCDE::repairRandom(Vec& x)
{
    if (!prob_) return;
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (!std::isfinite(x[j]) || x[j] < L[j] || x[j] > U[j]) {
            std::uniform_real_distribution<double> Uj(L[j], U[j]);
            x[j] = Uj(rng_);
        }
    }
}

int SFCDE::bestIndex() const
{
    if (FX_.empty()) return -1;
    return static_cast<int>(std::min_element(FX_.begin(), FX_.end()) - FX_.begin());
}

void SFCDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    if (pop_init_ < 4) pop_init_ = 50;
    setPopulation(pop_init_);

    Initializer initSampler;
    initSampler.configure(initopt_);

    X_.clear();
    FX_.clear();
    fail_F_.clear();
    fail_CR_.clear();

    X_ = initSampler.samplePopulation(*prob_, rng_, pop_init_);
    FX_.assign(X_.size(), std::numeric_limits<double>::infinity());

    best_x_.assign(D, 0.0);
    best_f_ = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < X_.size(); ++i) {
        repairRandom(X_[i]);
        FX_[i] = eval(X_[i]);
        if (FX_[i] < best_f_) {
            best_f_ = FX_[i];
            best_x_ = X_[i];
        }
        if (prob_->calls() >= max_evals_) break;
    }

    MF_.assign(H_, mu_f_init_);
    MCR_.assign(H_, mu_cr_init_);

    updateStop(FX_);
    printBest();
}

void SFCDE::one_iteration()
{
    if (!prob_) return;
    if (X_.empty()) return;
    if (prob_->calls() >= max_evals_) return;

    const int N = static_cast<int>(X_.size());
    const int D = prob_->dimension();
    if (N < 4 || D <= 0) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> Ui_mem(0, H_ - 1);
    std::uniform_int_distribution<int> Ui_dim(0, D - 1);

    const int F_idx  = Ui_mem(rng_);
    const int CR_idx = Ui_mem(rng_);

    const bool has_fail_F  = !fail_F_.empty();
    const bool has_fail_CR = !fail_CR_.empty();
    const double mu_fail_F  = has_fail_F  ? meanArithmetic(fail_F_)  : 0.0;
    const double mu_fail_CR = has_fail_CR ? meanArithmetic(fail_CR_) : 0.0;

    fail_F_.clear();
    fail_CR_.clear();

    std::vector<double> success_F;
    std::vector<double> success_CR;
    success_F.reserve(N);
    success_CR.reserve(N);

    std::vector<Vec>    newPop = X_;
    std::vector<double> newFit = FX_;

    const int bidx = bestIndex();
    if (bidx < 0) return;
    const Vec xbest = X_[bidx];

    auto sampleDistinctIndex = [&](int avoid1, int avoid2, int avoid3) {
        std::uniform_int_distribution<int> Ui(0, N - 1);
        int idx = Ui(rng_);
        while (idx == avoid1 || idx == avoid2 || idx == avoid3) {
            idx = Ui(rng_);
        }
        return idx;
    };

    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;

        int idx = sampleDistinctIndex(i, -1, -1);
        int r1  = sampleDistinctIndex(i, idx, -1);
        int r2  = sampleDistinctIndex(i, idx, r1);

        double F  = sampleF(MF_[F_idx]);
        double CR = sampleCR(MCR_[CR_idx]);

        if (has_fail_F) {
            while (std::abs(F - mu_fail_F) < std::abs(F - MF_[F_idx])) {
                const double Dfail = std::abs(F - mu_fail_F);
                const double Dsucc = std::abs(F - MF_[F_idx]);
                const double prob  = (Dfail <= 1e-15)
                    ? 0.0
                    : std::exp(-std::sqrt(Dsucc / Dfail));

                if (U01(rng_) < prob) break;
                F = sampleF(MF_[F_idx]);
            }
        }

        if (has_fail_CR) {
            while (std::abs(CR - mu_fail_CR) < std::abs(CR - MCR_[CR_idx])) {
                const double Dfail = std::abs(CR - mu_fail_CR);
                const double Dsucc = std::abs(CR - MCR_[CR_idx]);
                const double prob  = (Dfail <= 1e-15)
                    ? 0.0
                    : std::exp(-std::sqrt(Dsucc / Dfail));

                if (U01(rng_) < prob) break;
                CR = sampleCR(MCR_[CR_idx]);
            }
        }

        const Vec& winner = (FX_[idx] < FX_[i]) ? X_[idx] : X_[i];

        Vec donor(D, 0.0);
        for (int j = 0; j < D; ++j) {
            donor[j] = winner[j]
                     + F * (xbest[j] - winner[j])
                     + F * (X_[r1][j] - X_[r2][j]);
        }

        Vec trial = donor;
        const int jrand = Ui_dim(rng_);
        for (int j = 0; j < D; ++j) {
            if (!(U01(rng_) < CR || j == jrand)) {
                trial[j] = X_[i][j];
            }
        }

        repairRandom(trial);

        const double f_trial = eval(trial);
        if (f_trial < FX_[i]) {
            newPop[i] = trial;
            newFit[i] = f_trial;
            success_F.push_back(F);
            success_CR.push_back(CR);

            if (f_trial < best_f_) {
                best_f_ = f_trial;
                best_x_ = trial;
            }
        } else {
            fail_F_.push_back(F);
            fail_CR_.push_back(CR);
        }
    }

    X_.swap(newPop);
    FX_.swap(newFit);

    if (!success_F.empty()) {
        MF_[F_idx] = (1.0 - c_mem_) * MF_[F_idx] + c_mem_ * meanLehmer(success_F);
    }
    if (!success_CR.empty()) {
        MCR_[CR_idx] = (1.0 - c_mem_) * MCR_[CR_idx] + c_mem_ * meanArithmetic(success_CR);
    }

    updateStop(FX_);
    printBest();
}

void SFCDE::end()
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

        if (!X_.empty() && !FX_.empty()) {
            const int worst = static_cast<int>(std::max_element(FX_.begin(), FX_.end()) - FX_.begin());
            X_[worst]  = best_x_;
            FX_[worst] = best_f_;
        }
    }

    updateStop(FX_);
    printBest();
}

} // namespace optimsolution
