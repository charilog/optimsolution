#include "sioa.h"
#include <numeric>
#include <cmath>

namespace optimsolution {

// ---------------- configure ----------------
void SIOA::configure(const MethodConfig& mc)
{
    // 1) population (as in DE).
    pop_cfg_ = mc.getInt("population", pop_cfg_);
    if (pop_cfg_ == 0) pop_cfg_ = -1;
    if (pop_cfg_ > 0) this->setPopulation(pop_cfg_);

    // 2) stop parameters (aligned with bss defaults).
    eps_stop_      = mc.getDbl("eps_stop", eps_stop_);
    NM_            = std::max(1, mc.getInt("NM", NM_));
    plateau_iters_ = mc.getInt("plateau_iters", plateau_iters_);

    // Comment translated from Greek.
    local_rate_ = mc.getDbl("local_rate", local_rate_);
    if (local_rate_ < 0.0) local_rate_ = 0.0;
    if (local_rate_ > 1.0) local_rate_ = 1.0;

    local_method_ = mc.getStr("local_method", local_method_);
    for (auto &ch : local_method_) ch = (char)std::tolower((unsigned char)ch);
    if (local_method_ == "none" || local_method_ == "off" || local_method_ == "0") {
        local_method_.clear();          // Actual deactivation.
        local_rate_ = 0.0;
    }

    inrun_on_improve_ = mc.getBool("inrun_on_improve", inrun_on_improve_);

    // 4) final local at end (via method, not via the global hook).
    end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
    end_local_method_ = mc.getStr("end_local_method", end_local_method_);
    for (auto &ch : end_local_method_) ch = (char)std::tolower((unsigned char)ch);
    if (end_local_method_ == "none" || end_local_method_ == "off" || end_local_method_ == "0")
        end_local_method_.clear();

    // 5) SIOA params
    c1_  = mc.getDbl("c1", c1_);
    c2_  = mc.getDbl("c2", c2_);
    Rmin_ = std::max(1e-12, mc.getDbl("Rmin", Rmin_));
    Rmax_ = std::max(Rmin_,  mc.getDbl("Rmax", Rmax_));
    p_spor0_ = std::clamp(mc.getDbl("pspor0", p_spor0_), 0.0, 1.0);
    p_germ0_ = std::clamp(mc.getDbl("pgerm0", p_germ0_), 0.0, 1.0);
    p_zero_  = std::clamp(mc.getDbl("pzero",  p_zero_),  0.0, 1.0);
    adapt_R_kappa_    = std::clamp(mc.getDbl("adapt_R_kappa",    adapt_R_kappa_),    0.0, 1.0);
    adapt_prob_kappa_ = std::clamp(mc.getDbl("adapt_prob_kappa", adapt_prob_kappa_), 0.0, 1.0);
    crowding_metric_  = mc.getStr("crowding_metric", crowding_metric_);
    for (auto &ch : crowding_metric_) ch = (char)std::tolower((unsigned char)ch);
}

// --------------- utilities -----------------
void SIOA::ensureBounds(Vec& v)
{
    const auto &L = prob_->lb();
    const auto &U = prob_->ub();
    for (size_t j = 0; j < v.size(); ++j)
    {
        if (!std::isfinite(v[j])) v[j] = 0.5*(L[j] + U[j]);
        if (v[j] < L[j]) v[j] = L[j];
        if (v[j] > U[j]) v[j] = U[j];
    }
}

int SIOA::most_similar_index_(const Vec& s) const
{
    const int n = (int)X_.size();
    const int D = (int)s.size();
    const auto &L = prob_->lb();
    const auto &U = prob_->ub();

    int arg = 0;
    double bestd = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i)
    {
        double d2 = 0.0;
        if (crowding_metric_ == "bnorm") {
            for (int j = 0; j < D; ++j) {
                const double span = std::max(1e-32, U[j] - L[j]);
                const double z = (s[j] - X_[i][j]) / span;
                d2 += z*z;
            }
        } else {
            for (int j = 0; j < D; ++j) {
                const double z = s[j] - X_[i][j];
                d2 += z*z;
            }
        }
        if (d2 < bestd) { bestd = d2; arg = i; }
    }
    return arg;
}

void SIOA::adapt_controls_(double avg_f, double prev_avg)
{
    if (!std::isfinite(prev_avg)) {
        last_avg_f_ = avg_f;
        return;
    }

    const bool   improving = (avg_f < prev_avg - 1e-12);
    const double dir       = improving ? -1.0 : +1.0;

    R_      = std::clamp(R_ + dir * adapt_R_kappa_ * (Rmax_ - Rmin_), Rmin_, Rmax_);
    p_spor_ = std::clamp(p_spor_ + (improving ? -adapt_prob_kappa_ : +adapt_prob_kappa_), 0.0, 1.0);
    p_germ_ = std::clamp(p_germ_ + (improving ? +adapt_prob_kappa_ : -adapt_prob_kappa_), 0.0, 1.0);

    last_avg_f_ = avg_f;
}

bool SIOA::stopping_hold_(double fmin_k)
{
    // Keeps fmin_k and consecutive differences for NM steps.
    if (window_fmin_.empty()) {
        window_fmin_.push_back(fmin_k);
        return false;
    }
    double prev_anchor = window_fmin_.front();
    double delta       = std::fabs(fmin_k - prev_anchor);

    window_fmin_.push_back(delta);
    // Ensures the size remains <= 1+NM_.
    while ((int)window_fmin_.size() > (1 + NM_))
        window_fmin_.pop_back();

    // Updates the anchor at the front.
    window_fmin_.front() = fmin_k;

    if ((int)window_fmin_.size() < (1 + NM_)) return false;
    for (int i = 1; i < (int)window_fmin_.size(); ++i)
        if (window_fmin_[i] > eps_stop_) return false;
    return true;
}

// --------------- primitive ------------------
Vec SIOA::make_spore_(const Vec& xi)
{
    const auto &L = prob_->lb();
    const auto &U = prob_->ub();
    const int D   = (int)xi.size();

    std::uniform_real_distribution<double> U01(0.0, 1.0);
    auto Uni = [&](double a, double b) { return a + (b - a)*U01(rng_local_); };

    Vec s(D, 0.0);
    for (int j = 0; j < D; ++j)
    {
        const double u1 = Uni(-1.0,  1.0);
        const double u2 = Uni(-R_,   R_);
        s[j] = xi[j] + R_*u1*c1_ + c2_*(gbest_x_[j] - xi[j] + u2);
        if (U01(rng_local_) < p_zero_) s[j] = 0.0;
        if (s[j] < L[j]) s[j] = L[j];
        if (s[j] > U[j]) s[j] = U[j];
    }
    return s;
}

// ------------------ init --------------------
void SIOA::init()
{
    if (!prob_) return;

    // Synchronizes the population as in DE.
    const int globalN = this->population();
    final_population_ = std::max(4, (pop_cfg_ > 0 ? pop_cfg_ : globalN));
    this->setPopulation(final_population_);

    const int D = prob_->dimension();

    X_.assign(final_population_, Vec(D, 0.0));
    fX_.assign(final_population_, std::numeric_limits<double>::infinity());
    gbest_x_.assign(D, 0.0);
    gbest_f_      = std::numeric_limits<double>::infinity();
    window_fmin_.clear();
    last_avg_f_   = std::numeric_limits<double>::infinity();

    // Seeds the local RNG from the central RNG.
    rng_local_.seed( 0xA7F0D1BC4E12ULL ^ (uint64_t)rng_() );

    R_      = 0.5*(Rmin_ + Rmax_);
    p_spor_ = p_spor0_;
    p_germ_ = p_germ0_;

    // Comment translated from Greek.
    Initializer initSampler;
    initSampler.configure(initopt_);
    auto seeds = initSampler.samplePopulation(*prob_, rng_, final_population_);

    for (int i = 0; i < final_population_; ++i)
    {
        X_[i] = (i < (int)seeds.size() ? seeds[i] : X_[i]);
        ensureBounds(X_[i]);
        fX_[i] = eval(X_[i]);
        if (fX_[i] < gbest_f_) { gbest_f_ = fX_[i]; gbest_x_ = X_[i]; }
        if (prob_->calls() >= max_evals_) break;
    }

    best_x_ = gbest_x_;
    best_f_ = gbest_f_;

    // Does not print or update stop_ here, as in GA.
}

// --------------- one_iteration --------------
void SIOA::one_iteration()
{
    if (!prob_) return;

    const int N = (int)X_.size();

    std::uniform_real_distribution<double> U01_main(0.0, 1.0);
    std::uniform_real_distribution<double> U01_local(0.0, 1.0);

    // Stores the previous mean for adaptation.
    double prev_avg = last_avg_f_;

    for (int i = 0; i < N; ++i)
    {
        // sporulation
        if (U01_local(rng_local_) >= p_spor_)
            continue;

        Vec trial = make_spore_(X_[i]);
        double ftrial = eval(trial);

        if (prob_->calls() >= max_evals_) break;

        // Comment translated from Greek.
        if (U01_local(rng_local_) < p_germ_)
        {
            int m = most_similar_index_(trial);

            if (ftrial < fX_[m])
            {
                // 1) accepts the trial.
                X_[m]  = std::move(trial);
                fX_[m] = ftrial;

                // 2) in-run local search on the new individual, exactly as in GA.
                bool doLocal = (local_rate_ > 0.0 && !local_method_.empty());
                if (doLocal)
                {
                    bool fire = false;
                    if (!inrun_on_improve_) {
                        fire = (U01_main(rng_) < local_rate_);
                    } else {
                        // Comment translated from Greek.
                        fire = (U01_main(rng_) < local_rate_);
                    }

                    if (fire)
                    {
                        auto [xloc, floc] = localSearch(local_method_, X_[m]);
                        if (std::isfinite(floc) && floc < fX_[m]) {
                            X_[m]  = std::move(xloc);
                            fX_[m] = floc;
                        }
                    }
                }

                // 3) updates the global best.
                if (fX_[m] < gbest_f_) {
                    gbest_f_ = fX_[m];
                    gbest_x_ = X_[m];
                }
            }
        }

        if (prob_->calls() >= max_evals_) break;
    }

    // Comment translated from Greek.
    double avg_f = 0.0;
    for (double v : fX_) avg_f += v;
    avg_f /= std::max(1, N);

    // Comment translated from Greek.
    adapt_controls_(avg_f, prev_avg);

    // Updates best for the Optimizer base.
    if (gbest_f_ < best_f_) { best_f_ = gbest_f_; best_x_ = gbest_x_; }

    // Prints and updates BSS, as in GA/DE.
    printBest();
    updateStop(fX_);
}

// ------------------- end --------------------
void SIOA::end()
{
    if (!prob_) return;

    // Comment translated from Greek.
    if (end_local_refine_ && !end_local_method_.empty())
    {
        auto [xloc, floc] = localSearch(end_local_method_, best_x_);
        if (std::isfinite(floc) && floc < best_f_) {
            best_f_ = floc;
            best_x_ = std::move(xloc);
        }
    }

    // Comment translated from Greek.
    printBest();
}

} // namespace optimsolution
