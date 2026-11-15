#include "rdex.h"
#include "init.h"
#include <numeric>

namespace optimsolution {

// Line references "RDEx.cpp:NNN" point to the authors'
// RDEx_SOP/code/rdex-sop/RDEx.cpp (v1.0.0).

namespace {

struct Trio { const Vec* best; const Vec* mid; const Vec* worst; };
// Ordering of three donors with the tie-breaking of the authors' EB_order() (RDEx.cpp:310-349).
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

double RDEx::progress() const {
    return (max_evals_ > 0) ? double(prob_->calls()) / double(max_evals_) : 1.0;
}

double RDEx::meanWL(const std::vector<double>& v, const std::vector<double>& w) const {
    // RDEx.cpp:259-276; all-zero weights give 0/0 there, i.e. the 1.0 fallback.
    double sumW = 0.0;
    for (double x : w) sumW += x;
    if (!(sumW > 0.0)) return 1.0;
    double sumSq = 0.0, sum = 0.0;
    for (size_t t = 0; t < v.size(); ++t) {
        const double wt = w[t] / sumW;
        sumSq += wt * v[t] * v[t];
        sum   += wt * v[t];
    }
    return (std::fabs(sum) > 1e-8) ? sumSq / sum : 1.0;
}

void RDEx::init() {
    if (!prob_) return;
    const int D = prob_->dimension();

    N0_ = (fixed_population_ > 3)
        ? fixed_population_
        : std::max(Nmin_ + 1, static_cast<int>(Nmax_mult_ * double(D))); // RDEx.cpp:897,905
    pop_ = N0_;

    Initializer initSampler;
    initSampler.configure(initopt_);
    XC_ = initSampler.samplePopulation(*prob_, rng_, N0_);
    FC_.assign(XC_.size(), std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(D, 0.0);

    for (size_t i = 0; i < XC_.size(); ++i) {
        if (prob_->calls() >= max_evals_) break;
        const double f = eval(XC_[i]);
        FC_[i] = f;
        if (f < best_f_) { best_f_ = f; best_x_ = XC_[i]; }
    }

    // Front = current population sorted by fitness (RDEx.cpp:432-449).
    const std::vector<int> idx = ascendingOrder(FC_, N0_);
    XF_.resize(N0_);
    FF_.resize(N0_);
    for (int i = 0; i < N0_; ++i) { XF_[i] = XC_[idx[i]]; FF_[i] = FC_[idx[i]]; }
    fit_mass_ = FF_;

    mem_F_.assign(H_, mem0_);   // RDEx.cpp:240-243
    mem_CR_.assign(H_, mem0_);
    mem_pos_  = 0;
    rho_eb_   = rho_eb0_;
    sr_       = sr0_;
    pf_index_ = 0;
    fit_temp_.assign(N0_, std::numeric_limits<double>::infinity());
    eb_flag_.assign(N0_, 0);

    updateStop(FF_);
    printBest();
}

void RDEx::one_iteration() {
    if (!prob_) return;
    if (prob_->calls() >= max_evals_) return;
    const int D = prob_->dimension();
    const auto& L = prob_->lb();
    const auto& U = prob_->ub();
    const int N = static_cast<int>(XF_.size());
    if (N < 4 || static_cast<int>(XC_.size()) < N) return;

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::uniform_int_distribution<int> UN(0, N - 1);
    std::uniform_int_distribution<int> UH(0, H_ - 1);
    std::uniform_int_distribution<int> UHp1(0, H_);    // {0..H}: H = the fixed 0.9 entry
    std::uniform_int_distribution<int> Ujrand(0, D - 1);
    std::normal_distribution<double> N01(0.0, 1.0);
    std::cauchy_distribution<double> C01(0.0, 1.0);

    auto inBox = [&](double val, int j) {
        if (!std::isfinite(val) || val < L[j] || val > U[j]) {
            std::uniform_real_distribution<double> ud(L[j], U[j]);
            return ud(rng_);
        }
        return val;
    };

    // ---- per-generation rankings and selectors (RDEx.cpp:453-493) ----
    const double meanF = 0.4 + 0.25 * std::tanh(5.0 * sr_);
    const double prog0 = progress();
    const std::vector<int> idxC = ascendingOrder(FC_, N); // current population (first N)
    const std::vector<int> idxF = ascendingOrder(FF_, N); // front

    std::vector<double> wExp(N), wRsp(N);
    for (int i = 0; i < N; ++i) {
        wExp[i] = std::exp(-double(i) / double(N) * 3.0);
        wRsp[i] = 3.0 * double(N - i);
    }
    std::discrete_distribution<int> selFrontExp(wExp.begin(), wExp.end());
    std::discrete_distribution<int> selRspAll(wRsp.begin(), wRsp.end());

    int psizeval = std::max(2, static_cast<int>(double(N) * xi_ * std::exp(-sr_ * k_)));
    if (psizeval > N) psizeval = N;
    std::uniform_int_distribution<int> Upbest(0, psizeval - 1);

    int psizeval2 = static_cast<int>(double(N) * eb_pfrac_ * (1.0 - 0.5 * prog0));
    if (psizeval2 <= 1) psizeval2 = 2;
    if (psizeval2 > N)  psizeval2 = N;
    std::discrete_distribution<int> selRspTop(wRsp.begin(), wRsp.begin() + psizeval2);

    std::vector<double> succ_F, succ_CR, succ_dF;

    for (int it = 0; it < N; ++it) {
        const int t    = UN(rng_);   // target drawn with replacement (RDEx.cpp:497)
        const int mci  = UH(rng_);
        const int mci2 = UHp1(rng_);

        // Standard-branch donors (RDEx.cpp:500-508). As in the original, the
        // exclusion tests compare indices of the two populations directly.
        int prand = idxC[Upbest(rng_)];
        for (int g = 0; prand == t && g < 1000; ++g) prand = idxC[Upbest(rng_)];
        int r1 = idxF[selFrontExp(rng_)];
        for (int g = 0; r1 == prand && g < 1000; ++g) r1 = idxF[selFrontExp(rng_)];
        int r2 = idxC[UN(rng_)];
        for (int g = 0; (r2 == prand || r2 == r1) && g < 1000; ++g) r2 = idxC[UN(rng_)];

        double F_i = meanF + sigma_F_ * N01(rng_);
        for (int g = 0; (F_i < 0.0 || F_i > 1.0) && g < 1000; ++g) F_i = meanF + sigma_F_ * N01(rng_);
        F_i = std::min(std::max(F_i, 0.0), 1.0);
        double CR_i = mem_CR_[mci] + sigma_cr_ * N01(rng_);
        CR_i = std::min(std::max(CR_i, 0.0), 1.0);
        const int jrand = Ujrand(rng_);

        // EB branch only after eb_start of the budget (RDEx.cpp:517-521; the
        // factor (1 - NFEval/MaxFEval) there is an integer division, i.e. 1).
        const double prog = progress();
        const bool useEB = (prog >= eb_start_) && (U01(rng_) < rho_eb_);

        Vec u(D);
        double actualCr = 0.0;
        const bool perturbation = (U01(rng_) < p_r_);
        const double sigScale = sigma_loc_;

        if (useEB) {
            eb_flag_[t] = 1;
            prand = idxC[selRspTop(rng_)];
            for (int g = 0; prand == t && g < 1000; ++g) prand = idxC[selRspTop(rng_)];
            r1 = idxF[selRspAll(rng_)];
            for (int g = 0; r1 == prand && g < 1000; ++g) r1 = idxF[selRspAll(rng_)];
            r2 = idxC[selRspAll(rng_)];
            for (int g = 0; (r2 == prand || r2 == r1) && g < 1000; ++g) r2 = idxC[selRspAll(rng_)];
            const Trio tr = orderTrio(&XC_[prand], FC_[prand], &XF_[r1], FF_[r1], &XC_[r2], FC_[r2]);

            // F and Cr from the H+1 memory choices (RDEx.cpp:534-561).
            const double locF = (mci2 < H_) ? mem_F_[mci2] : 0.9;
            F_i = locF + 0.1 * C01(rng_);
            for (int g = 0; F_i < 0.0 && g < 1000; ++g) F_i = locF + 0.1 * C01(rng_);
            if (F_i < 0.0) F_i = 0.0;
            if (F_i > 1.0) F_i = 1.0;
            if (prog < 0.6 && F_i > 0.7) F_i = 0.7;   // kept from the original; inactive when eb_start >= 0.6
            if (mci2 < H_) CR_i = (mem_CR_[mci2] < 0.0) ? 0.0 : mem_CR_[mci2] + sigma_cr_eb_ * N01(rng_);
            else           CR_i = 0.9 + sigma_cr_eb_ * N01(rng_);
            CR_i = std::min(std::max(CR_i, 0.0), 1.0);
            if (prog < 0.25) CR_i = std::max(CR_i, 0.7); // kept from the original; inactive when eb_start >= 0.5
            if (prog < 0.5)  CR_i = std::max(CR_i, 0.6);

            const Vec& xt = XF_[t];
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < CR_i || j == jrand) {
                    u[j] = inBox(xt[j] + F_i * ((*tr.best)[j] - xt[j]) + F_i * ((*tr.mid)[j] - (*tr.worst)[j]), j);
                    actualCr += 1.0;
                } else {
                    u[j] = perturbation ? inBox(xt[j] + sigScale * (U[j] - L[j]) * C01(rng_), j) : xt[j];
                }
            }
        } else {
            eb_flag_[t] = 0;
            const Vec& xt = XF_[t];
            for (int j = 0; j < D; ++j) {
                if (U01(rng_) < CR_i || j == jrand) {
                    u[j] = inBox(xt[j] + F_i * (XC_[prand][j] - xt[j]) + F_i * (XF_[r1][j] - XC_[r2][j]), j);
                    actualCr += 1.0;
                } else {
                    u[j] = perturbation ? inBox(xt[j] + sigScale * (U[j] - L[j]) * C01(rng_), j) : xt[j];
                }
            }
        }
        actualCr /= double(D);

        const double fu = eval(u);
        fit_temp_[t] = fu;
        if (fu < best_f_) { best_f_ = fu; best_x_ = u; }

        // Selection against the target; the trial goes to slot pf_index_ of
        // the front and is appended to the current population (RDEx.cpp:614-629).
        if (fu <= FF_[t]) {
            XC_.push_back(u);
            FC_.push_back(fu);
            // After a front reduction pf_index_ can point one past the active
            // front; the original then writes into an inactive slot, i.e. the
            // trial does not enter the front (it is still in the current population).
            if (pf_index_ < N) {
                XF_[pf_index_] = std::move(u);
                FF_[pf_index_] = fu;
            }
            succ_CR.push_back(actualCr);
            succ_F.push_back(F_i);
            // As in the original, |FF[t] - fu| is read after the write above
            // (it is 0 when pf_index_ == t).
            succ_dF.push_back(std::fabs(FF_[t] - fu));
            pf_index_ = (pf_index_ + 1) % N;
        }

        if (local_rate_ > 0.0 && local_method_ != "none" && !local_method_.empty()
            && U01(rng_) < local_rate_ && prob_->calls() < max_evals_) {
            auto [xl, fl] = localSearch(local_method_, XF_[t]);
            if (fl < FF_[t]) { XF_[t] = std::move(xl); FF_[t] = fl; }
            if (FF_[t] < best_f_) { best_f_ = FF_[t]; best_x_ = XF_[t]; }
        }

        if (prob_->calls() >= max_evals_) break;
    }
    const int S = static_cast<int>(succ_dF.size());

    // ---- EB rate update (RDEx.cpp:386-416, 632-638) ----
    {
        double sumEB = 0.0, sumStd = 0.0;
        for (int c = 0; c < N; ++c) {
            if (fit_temp_[c] <= fit_mass_[c]) {
                if (eb_flag_[c]) sumEB += fit_mass_[c] - fit_temp_[c];
                else             sumStd += fit_mass_[c] - fit_temp_[c];
            }
        }
        if (sumEB != 0.0 && sumStd != 0.0) {
            rho_eb_ = sumEB / (sumEB + sumStd);
            rho_eb_ = std::min(std::max(rho_eb_, 0.0), 1.0);
        } else {
            rho_eb_ = rho_eb0_;
        }
        for (int c = 0; c < N; ++c) fit_mass_[c] = FF_[c];
    }

    sr_ = double(S) / double(N);

    // ---- linear front-size reduction, remove the worst (RDEx.cpp:640-642, 286-309) ----
    const double nfe   = double(prob_->calls());
    const double maxfe = double(std::max<long long>(1, (long long)max_evals_));
    int newN = static_cast<int>(double(Nmin_ - N0_) / maxfe * nfe + double(N0_));
    if (newN < Nmin_) newN = Nmin_;
    if (newN > N)     newN = N;
    while (static_cast<int>(XF_.size()) > newN) {
        size_t worst = 0;
        for (size_t k = 1; k < FF_.size(); ++k) if (FF_[k] > FF_[worst]) worst = k;
        XF_.erase(XF_.begin() + worst);
        FF_.erase(FF_.begin() + worst);
        fit_mass_.erase(fit_mass_.begin() + worst);
    }
    // pf_index_ is left unchanged, as in the original (see the selection step).

    // ---- memory update (RDEx.cpp:250-258) ----
    if (S > 0) {
        mem_CR_[mem_pos_] = 0.5 * (meanWL(succ_CR, succ_dF) + mem_CR_[mem_pos_]);
        mem_F_[mem_pos_]  = meanWL(succ_F, succ_dF);
        mem_pos_ = (mem_pos_ + 1) % H_;
    }

    // ---- current population <- best newN of its first newN+S entries (RDEx.cpp:644-666) ----
    // (Same as the original: the pool is the first newN+S stored entries, i.e.
    // when the front shrank, the last accepted trials of the generation may be
    // left out of the pool, exactly as in LSRTDE/RDEx.)
    {
        const int pool = std::min(newN + S, static_cast<int>(XC_.size()));
        std::vector<Vec>    Xk;
        std::vector<double> Fk;
        if (pool > newN) {
            const std::vector<int> idx = ascendingOrder(FC_, pool);
            Xk.reserve(newN); Fk.reserve(newN);
            for (int i = 0; i < newN; ++i) { Xk.push_back(XC_[idx[i]]); Fk.push_back(FC_[idx[i]]); }
        } else {
            Xk.assign(XC_.begin(), XC_.begin() + newN);
            Fk.assign(FC_.begin(), FC_.begin() + newN);
        }
        XC_ = std::move(Xk);
        FC_ = std::move(Fk);
    }

    pop_ = newN;
    updateStop(FF_);
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

    if (!XF_.empty() && !FF_.empty()) {
        size_t worst = 0; double fw = FF_[0];
        for (size_t k = 1; k < FF_.size(); ++k) {
            if (FF_[k] > fw) { fw = FF_[k]; worst = k; }
        }
        XF_[worst] = best_x_;
        FF_[worst] = best_f_;
    }
    printBest();
}

} // namespace optimsolution
