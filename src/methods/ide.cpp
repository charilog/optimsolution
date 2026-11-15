// =============================================================================
//  iDE -- see ide.h
//
//  The core (configure keys, sampling, LPSR, two populations, CR memory) is
//  the L-SRTDE port of lsrtde.cpp, kept statement by statement so that with
//  every module switched off the random stream and the results are identical
//  to L-SRTDE. Module code is marked [EIG], [ARCH], [POLISH], [REJUV].
// =============================================================================

#include "ide.h"
#include "init.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <limits>
#include <cctype>
#include <iomanip>
#include <sstream>

namespace optimsolution {

namespace {

std::string ideLower(std::string s)
{
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string ideTrim(const std::string& s)
{
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

} // namespace

// -----------------------------------------------------------------------------
// configuration (keys are lower-case, Config::load lower-cases them)
// -----------------------------------------------------------------------------
void IDE::configure(const MethodConfig& mc)
{
    // ---- L-SRTDE core (same keys and defaults as [lsrtde]) ----
    pop_override_ = mc.getInt("population", pop_override_);
    if (pop_override_ < 0) pop_override_ = 0;
    np_init_mult_ = mc.getDbl("np_init_multiplier", np_init_mult_);
    if (!(np_init_mult_ > 0.0)) np_init_mult_ = 20.0;
    np_min_ = mc.getInt("np_min", np_min_);
    if (np_min_ < 4) np_min_ = 4;
    memory_size_ = mc.getInt("memory_size", memory_size_);
    if (memory_size_ < 1) memory_size_ = 1;
    mcr_init_ = std::min(1.0, std::max(0.0, mc.getDbl("mcr_init", mcr_init_)));
    memory_c_ = std::min(1.0, std::max(0.0, mc.getDbl("memory_c", memory_c_)));
    sigma_cr_ = mc.getDbl("sigma_cr", sigma_cr_);
    if (sigma_cr_ < 0.0) sigma_cr_ = 0.05;
    f_base_    = mc.getDbl("f_base", f_base_);
    f_amp_     = mc.getDbl("f_amp", f_amp_);
    f_sr_coef_ = mc.getDbl("f_sr_coef", f_sr_coef_);
    sigma_f_   = mc.getDbl("sigma_f", sigma_f_);
    if (sigma_f_ < 0.0) sigma_f_ = 0.02;
    pbest_rate_    = std::min(1.0, std::max(0.0, mc.getDbl("pbest_rate", pbest_rate_)));
    pbest_sr_coef_ = mc.getDbl("pbest_sr_coef", pbest_sr_coef_);
    pbest_min_     = mc.getInt("pbest_min", pbest_min_);
    if (pbest_min_ < 2) pbest_min_ = 2;
    rank_pressure_ = mc.getDbl("rank_pressure", rank_pressure_);
    if (rank_pressure_ < 0.0) rank_pressure_ = 0.0;
    sr_init_ = std::min(1.0, std::max(0.0, mc.getDbl("sr_init", sr_init_)));
    delta_from_parent_ = mc.getBool("delta_from_parent", delta_from_parent_);
    {
        const std::string b = ideLower(ideTrim(mc.getStr("bound_handling", "")));
        if (b == "random" || b == "reinit" || b == "uniform") bound_mode_ = BoundMode::Random;
        else if (b == "midpoint" || b == "midpoint_target")   bound_mode_ = BoundMode::Midpoint;
        else if (b == "clip" || b == "clamp")                 bound_mode_ = BoundMode::Clip;
    }

    // ---- modules ----
    eig_enable_  = mc.getBool("eig_enable", eig_enable_);
    eig_prob_    = std::min(1.0, std::max(0.0, mc.getDbl("eig_prob", eig_prob_)));
    eig_frac_    = std::min(1.0, std::max(0.0, mc.getDbl("eig_frac", eig_frac_)));
    eig_period_  = std::max(1, mc.getInt("eig_period", eig_period_));
    eig_max_dim_ = std::max(2, mc.getInt("eig_max_dim", eig_max_dim_));
    eig_adapt_   = mc.getBool("eig_adapt", eig_adapt_);
    eig_pmin_    = std::min(1.0, std::max(0.0, mc.getDbl("eig_pmin", eig_pmin_)));
    eig_pmax_    = std::min(1.0, std::max(eig_pmin_, mc.getDbl("eig_pmax", eig_pmax_)));

    arch_enable_ = mc.getBool("archive_enable", arch_enable_);
    arch_rate_   = std::max(0.0, mc.getDbl("archive_rate", arch_rate_));

    polish_enable_ = mc.getBool("polish_enable", polish_enable_);
    polish_start_  = std::min(1.0, std::max(0.0, mc.getDbl("polish_start", polish_start_)));
    polish_budget_ = std::min(1.0, std::max(0.0, mc.getDbl("polish_budget", polish_budget_)));
    polish_patience_ = std::max(1, mc.getInt("polish_patience", polish_patience_));

    rejuv_enable_  = mc.getBool("rejuv_enable", rejuv_enable_);
    rejuv_gens_    = std::max(0, mc.getInt("rejuv_gens", rejuv_gens_));
    rejuv_keep_    = std::min(1.0, std::max(0.0, mc.getDbl("rejuv_keep", rejuv_keep_)));
    rejuv_cutoff_  = std::min(1.0, std::max(0.0, mc.getDbl("rejuv_cutoff", rejuv_cutoff_)));

    scout_enable_      = mc.getBool("scout_enable", scout_enable_);
    scout_after_       = std::min(1.0, std::max(0.0, mc.getDbl("scout_after", scout_after_)));
    scout_window_      = std::min(1.0, std::max(0.001, mc.getDbl("scout_window", scout_window_)));
    scout_tol_         = std::max(0.0, mc.getDbl("scout_tol", scout_tol_));
    scout_max_         = std::min(0.95, std::max(0.0, mc.getDbl("scout_max", scout_max_)));
    scout_max_remain_  = std::min(1.0, std::max(0.0, mc.getDbl("scout_max_remain", scout_max_remain_)));
    scout_max_phases_  = std::max(0, mc.getInt("scout_max_phases", scout_max_phases_));
    scout_clpso_share_ = std::min(1.0, std::max(0.0, mc.getDbl("scout_clpso_share", scout_clpso_share_)));
    scout_adopt_frac_  = std::min(1.0, std::max(0.0, mc.getDbl("scout_adopt_frac", scout_adopt_frac_)));
    scout_min_dist_    = std::max(0.0, mc.getDbl("scout_min_dist", scout_min_dist_));
    scout_spread_      = std::max(0.0, mc.getDbl("scout_spread", scout_spread_));
    scout_gain_frac_   = std::max(0.0, mc.getDbl("scout_gain_frac", scout_gain_frac_));
    scout_fspread_     = std::max(0.0, mc.getDbl("scout_fspread", scout_fspread_));
    scout_ratio_       = std::max(0.0, mc.getDbl("scout_ratio", scout_ratio_));
    scout_cma_max_dim_ = std::max(1, mc.getInt("scout_cma_max_dim", scout_cma_max_dim_));

    debug_ide_ = mc.getInt("debug_ide", debug_ide_);

    {
        auto it = mc.kv.find("end_local_refine");
        if (it != mc.kv.end()) end_local_refine_ = mc.getBool("end_local_refine", end_local_refine_);
        auto im = mc.kv.find("end_local_method");
        if (im != mc.kv.end()) end_local_method_ = ideLower(ideTrim(im->second));
    }
    {
        const std::string m = ideLower(ideTrim(end_local_method_));
        if (m.empty() || m == "none" || m == "off" || m == "0") end_local_refine_ = false;
    }
}

// -----------------------------------------------------------------------------
// L-SRTDE helpers (identical to lsrtde.cpp)
// -----------------------------------------------------------------------------
double IDE::eval(const Vec& x)
{
    double f = prob_->evaluate(x);
    if (!std::isfinite(f)) f = 1e100;
    return f;
}

int IDE::randInt(int n)
{
    if (n <= 1) return 0;
    std::uniform_int_distribution<int> d(0, n - 1);
    return d(rng_);
}

double IDE::randU01()
{
    std::uniform_real_distribution<double> d(0.0, 1.0);
    return d(rng_);
}

double IDE::sampleF(double meanF)
{
    if (!(sigma_f_ > 0.0)) return std::min(1.0, std::max(0.0, meanF));
    std::normal_distribution<double> nd(meanF, sigma_f_);
    for (int t = 0; t < 1000; ++t) {
        const double F = nd(rng_);
        if (F >= 0.0 && F <= 1.0) return F;
    }
    return std::min(1.0, std::max(0.0, meanF));
}

double IDE::sampleCR(double mcr)
{
    if (!(sigma_cr_ > 0.0)) return std::min(1.0, std::max(0.0, mcr));
    std::normal_distribution<double> nd(mcr, sigma_cr_);
    double cr = nd(rng_);
    if (!std::isfinite(cr)) cr = mcr;
    return std::min(1.0, std::max(0.0, cr));
}

double IDE::repairComponent(int j, double v, double parent_j)
{
    const double lo = prob_->lb()[j];
    const double hi = prob_->ub()[j];
    if (std::isfinite(v) && v >= lo && v <= hi) return v;
    switch (bound_mode_) {
        case BoundMode::Midpoint:
            if (!std::isfinite(v)) return 0.5 * (parent_j + 0.5 * (lo + hi));
            return (v < lo) ? 0.5 * (parent_j + lo) : 0.5 * (parent_j + hi);
        case BoundMode::Clip:
            if (!std::isfinite(v)) return parent_j;
            return (v < lo) ? lo : hi;
        case BoundMode::Random:
        default: {
            std::uniform_real_distribution<double> d(lo, hi);
            return (hi > lo) ? d(rng_) : lo;
        }
    }
}

double IDE::weightedLehmerMean(const std::vector<double>& values,
                               const std::vector<double>& weights) const
{
    const size_t n = std::min(values.size(), weights.size());
    if (n == 0) return 1.0;
    double sumW = 0.0;
    for (size_t i = 0; i < n; ++i) sumW += weights[i];
    if (!(sumW > 0.0) || !std::isfinite(sumW)) return 1.0;
    double sumSq = 0.0, sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double w = weights[i] / sumW;
        sumSq += w * values[i] * values[i];
        sum   += w * values[i];
    }
    if (std::fabs(sum) > 1e-8 && std::isfinite(sumSq)) return sumSq / sum;
    return 1.0;
}

int IDE::targetPopulationSize() const
{
    const double nfe     = static_cast<double>(prob_->calls());
    const double nfe_max = static_cast<double>(std::max<long long>(1, max_evals_));
    const double v = static_cast<double>(np_min_ - n_init_) / nfe_max * nfe + static_cast<double>(n_init_);
    int n = static_cast<int>(v);
    const int cur = static_cast<int>(front_X_.size());
    if (n < np_min_) n = np_min_;
    if (n > cur)     n = cur;
    return n;
}

void IDE::removeWorstFromFront(int new_size)
{
    while (static_cast<int>(front_X_.size()) > new_size && !front_X_.empty()) {
        size_t worst = 0;
        for (size_t i = 1; i < front_F_.size(); ++i)
            if (front_F_[i] > front_F_[worst]) worst = i;
        front_X_.erase(front_X_.begin() + static_cast<std::ptrdiff_t>(worst));
        front_F_.erase(front_F_.begin() + static_cast<std::ptrdiff_t>(worst));
    }
}

void IDE::sortAndTruncateMain(int new_size)
{
    std::vector<int> idx;
    sortedOrder(main_F_, static_cast<int>(main_F_.size()), idx);
    const int keep = std::min(new_size, static_cast<int>(idx.size()));
    std::vector<Vec>    nx; nx.reserve(static_cast<size_t>(keep));
    std::vector<double> nf; nf.reserve(static_cast<size_t>(keep));
    for (int i = 0; i < keep; ++i) {
        nx.push_back(std::move(main_X_[static_cast<size_t>(idx[i])]));
        nf.push_back(main_F_[static_cast<size_t>(idx[i])]);
    }
    main_X_.swap(nx);
    main_F_.swap(nf);
}

void IDE::sortedOrder(const std::vector<double>& f, int n, std::vector<int>& idx) const
{
    idx.resize(static_cast<size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return f[static_cast<size_t>(a)] < f[static_cast<size_t>(b)]; });
}

void IDE::noteBest(const Vec& x, double f)
{
    if (f < best_f_) { best_f_ = f; best_x_ = x; }
}

// -----------------------------------------------------------------------------
// [EIG] eigen-basis of the covariance of the best part of the main population
// -----------------------------------------------------------------------------
void IDE::jacobiEigen(const Mat& Ain, Mat& V, Vec& w)
{
    const int n = static_cast<int>(Ain.size());
    Mat A = Ain;
    V.assign(static_cast<size_t>(n), Vec(static_cast<size_t>(n), 0.0));
    for (int i = 0; i < n; ++i) V[i][i] = 1.0;
    w.assign(static_cast<size_t>(n), 0.0);
    double scale = 0.0;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) scale += A[i][j] * A[i][j];
    const double tol = 1e-24 * std::max(scale, 1e-300);
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) off += A[p][q] * A[p][q];
        if (off <= tol) break;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double Apq = A[p][q];
                if (std::fabs(Apq) < 1e-300) continue;
                const double App = A[p][p], Aqq = A[q][q];
                const double theta = (Aqq - App) / (2.0 * Apq);
                double t;
                if (std::fabs(theta) > 1e15) t = 1.0 / (2.0 * theta);
                else t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                A[p][p] = App - t * Apq;
                A[q][q] = Aqq + t * Apq;
                A[p][q] = A[q][p] = 0.0;
                for (int r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double Arp = A[r][p], Arq = A[r][q];
                    A[r][p] = A[p][r] = c * Arp - s * Arq;
                    A[r][q] = A[q][r] = s * Arp + c * Arq;
                }
                for (int r = 0; r < n; ++r) {
                    const double Vrp = V[r][p], Vrq = V[r][q];
                    V[r][p] = c * Vrp - s * Vrq;
                    V[r][q] = s * Vrp + c * Vrq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) w[static_cast<size_t>(i)] = A[i][i];
}

void IDE::refreshEigenBasis()
{
    const int D = prob_->dimension();
    if (D < 2 || D > eig_max_dim_) { eig_valid_ = false; return; }
    const int N = static_cast<int>(main_X_.size());           // sorted (best first)
    int top = std::max(D + 2, static_cast<int>(std::lround(eig_frac_ * N)));
    if (top > N) return;                                       // too few: keep the last basis
    Vec mean(static_cast<size_t>(D), 0.0);
    for (int i = 0; i < top; ++i)
        for (int j = 0; j < D; ++j) mean[j] += main_X_[i][j];
    for (int j = 0; j < D; ++j) mean[j] /= top;
    Mat C(static_cast<size_t>(D), Vec(static_cast<size_t>(D), 0.0));
    for (int i = 0; i < top; ++i)
        for (int a = 0; a < D; ++a) {
            const double da = main_X_[i][a] - mean[a];
            for (int b = a; b < D; ++b) C[a][b] += da * (main_X_[i][b] - mean[b]);
        }
    double tr = 0.0;
    for (int a = 0; a < D; ++a) {
        for (int b = a; b < D; ++b) { C[a][b] /= (top - 1); C[b][a] = C[a][b]; }
        tr += C[a][a];
    }
    if (!(tr > 0.0) || !std::isfinite(tr)) return;             // collapsed: keep the last basis
    for (int a = 0; a < D; ++a) C[a][a] += 1e-12 * tr / D;
    jacobiEigen(C, B_, eig_val_);
    eig_valid_ = true;
    gens_since_eig_ = 0;
}

// -----------------------------------------------------------------------------
// [ARCH] archive of overwritten front members (random removal when full)
// -----------------------------------------------------------------------------
void IDE::archivePush(const Vec& x)
{
    const int cap = static_cast<int>(std::lround(arch_rate_ * static_cast<double>(front_X_.size())));
    if (cap <= 0) { archive_.clear(); return; }
    if (static_cast<int>(archive_.size()) < cap) archive_.push_back(x);
    else archive_[static_cast<size_t>(randInt(static_cast<int>(archive_.size())))] = x;
}

// -----------------------------------------------------------------------------
// [POLISH] (1+1)-CMA-ES, Igel, Suttorp & Hansen (2006), covariance version:
//   d = 1 + n/2, p_target = 2/11, c_p = 1/12, c_c = 2/(n+2),
//   c_cov = 2/(n^2+6), p_thresh = 0.44
// -----------------------------------------------------------------------------
bool IDE::choleskyLower(const Mat& C, Mat& A) const
{
    const size_t n = C.size();
    A.assign(n, Vec(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double s = C[i][j];
            for (size_t k = 0; k < j; ++k) s -= A[i][k] * A[j][k];
            if (i == j) {
                if (!(s > 0.0) || !std::isfinite(s)) return false;
                A[i][i] = std::sqrt(s);
            } else {
                A[i][j] = s / A[j][j];
            }
        }
    }
    return true;
}

void IDE::injectIntoPopulations(const Vec& x, double f)
{
    // main: replace its worst member; front: replace its worst member
    if (!main_F_.empty()) {
        size_t w = 0;
        for (size_t i = 1; i < main_F_.size(); ++i) if (main_F_[i] > main_F_[w]) w = i;
        if (f < main_F_[w]) { main_X_[w] = x; main_F_[w] = f; }
    }
    if (!front_F_.empty()) {
        size_t w = 0;
        for (size_t i = 1; i < front_F_.size(); ++i) if (front_F_[i] > front_F_[w]) w = i;
        if (f < front_F_[w]) {
            if (arch_enable_) archivePush(front_X_[w]);
            front_X_[w] = x; front_F_[w] = f;
        }
    }
}

void IDE::polishInit()
{
    const int D = prob_->dimension();
    pol_started_ = true;
    pol_x_ = best_x_;
    pol_f_ = best_f_;
    pol_psucc_ = 2.0 / 11.0;
    pol_pc_.assign(static_cast<size_t>(D), 0.0);
    pol_fail_ = 0;

    // step size: RMS spread of the current main population
    double var = 0.0;
    const int N = static_cast<int>(main_X_.size());
    if (N >= 2) {
        for (int j = 0; j < D; ++j) {
            double m = 0.0, s = 0.0;
            for (int i = 0; i < N; ++i) m += main_X_[i][j];
            m /= N;
            for (int i = 0; i < N; ++i) { const double d = main_X_[i][j] - m; s += d * d; }
            var += s / (N - 1);
        }
        var /= D;
    }
    double range = 0.0;
    for (int j = 0; j < D; ++j) range += prob_->ub()[j] - prob_->lb()[j];
    range /= D;
    pol_sigma_ = std::sqrt(var);
    if (!(pol_sigma_ > 1e-12 * range)) pol_sigma_ = 1e-3 * range;
    if (pol_sigma_ > 0.3 * range) pol_sigma_ = 0.3 * range;

    // shape: the population's eigen-basis (if known), normalised to trace n
    pol_C_.assign(static_cast<size_t>(D), Vec(static_cast<size_t>(D), 0.0));
    bool shaped = false;
    if (eig_valid_ && static_cast<int>(B_.size()) == D) {
        double lmax = 0.0;
        for (double v : eig_val_) lmax = std::max(lmax, v);
        if (lmax > 0.0) {
            Vec lam(static_cast<size_t>(D));
            double sum = 0.0;
            for (int k = 0; k < D; ++k) { lam[k] = std::max(eig_val_[k], 1e-8 * lmax); sum += lam[k]; }
            for (int k = 0; k < D; ++k) lam[k] *= D / sum;
            for (int a = 0; a < D; ++a)
                for (int b = 0; b < D; ++b) {
                    double s = 0.0;
                    for (int k = 0; k < D; ++k) s += B_[a][k] * lam[k] * B_[b][k];
                    pol_C_[a][b] = s;
                }
            shaped = choleskyLower(pol_C_, pol_A_);
        }
    }
    if (!shaped) {
        for (int a = 0; a < D; ++a) pol_C_[a][a] = 1.0;
        choleskyLower(pol_C_, pol_A_);
    }
}

void IDE::polishBurst()
{
    const int D = prob_->dimension();
    const long long cap = static_cast<long long>(polish_budget_ * static_cast<double>(max_evals_));
    if (pol_used_ >= cap) return;
    if (!pol_started_) polishInit();
    if (best_f_ < pol_f_) { pol_x_ = best_x_; pol_f_ = best_f_; }   // the DE found a better point

    double range = 0.0;
    for (int j = 0; j < D; ++j) range += prob_->ub()[j] - prob_->lb()[j];
    range /= D;

    const double n = static_cast<double>(D);
    const double d_s = 1.0 + n / 2.0, p_tgt = 2.0 / 11.0, c_p = 1.0 / 12.0;
    const double c_c = 2.0 / (n + 2.0), c_cov = 2.0 / (n * n + 6.0), p_thr = 0.44;

    std::normal_distribution<double> N01(0.0, 1.0);
    Vec z(static_cast<size_t>(D)), y(static_cast<size_t>(D)), step(static_cast<size_t>(D));
    const int burst = std::max(10, D);
    for (int t = 0; t < burst; ++t) {
        if (prob_->calls() >= max_evals_ || pol_used_ >= cap) break;
        if (!(pol_sigma_ > 1e-15 * range)) {                    // converged: restart the shape
            pol_started_ = false;
            if (pol_used_ < cap) polishInit();
            if (!(pol_sigma_ > 1e-15 * range)) { pol_dead_ = true; break; }
        }
        for (int j = 0; j < D; ++j) z[j] = N01(rng_);
        for (int i = 0; i < D; ++i) {
            double s = 0.0;
            for (int k = 0; k <= i; ++k) s += pol_A_[i][k] * z[k];
            const double lo = prob_->lb()[i], hi = prob_->ub()[i];
            y[i] = std::min(hi, std::max(lo, pol_x_[i] + pol_sigma_ * s));
        }
        const double fy = eval(y);
        ++pol_used_;
        const bool ok = (fy <= pol_f_);
        pol_psucc_ = (1.0 - c_p) * pol_psucc_ + c_p * (ok ? 1.0 : 0.0);
        const double sigma_old = pol_sigma_;
        pol_sigma_ *= std::exp((pol_psucc_ - p_tgt) / (d_s * (1.0 - p_tgt)));
        if (pol_sigma_ > 0.5 * range) pol_sigma_ = 0.5 * range;
        if (!ok) continue;

        for (int j = 0; j < D; ++j) step[j] = (y[j] - pol_x_[j]) / sigma_old;
        pol_x_ = y;
        pol_f_ = fy;
        if (fy < best_f_) { noteBest(y, fy); injectIntoPopulations(y, fy); }

        if (pol_psucc_ < p_thr) {
            const double a = std::sqrt(c_c * (2.0 - c_c));
            for (int j = 0; j < D; ++j) pol_pc_[j] = (1.0 - c_c) * pol_pc_[j] + a * step[j];
            for (int a1 = 0; a1 < D; ++a1)
                for (int b = 0; b < D; ++b)
                    pol_C_[a1][b] = (1.0 - c_cov) * pol_C_[a1][b] + c_cov * pol_pc_[a1] * pol_pc_[b];
        } else {
            for (int j = 0; j < D; ++j) pol_pc_[j] *= (1.0 - c_c);
            for (int a1 = 0; a1 < D; ++a1)
                for (int b = 0; b < D; ++b)
                    pol_C_[a1][b] = (1.0 - c_cov) * pol_C_[a1][b]
                                  + c_cov * (pol_pc_[a1] * pol_pc_[b] + c_c * (2.0 - c_c) * pol_C_[a1][b]);
        }
        if (!choleskyLower(pol_C_, pol_A_)) {                   // numerical breakdown: reset shape
            for (int a1 = 0; a1 < D; ++a1)
                for (int b = 0; b < D; ++b) pol_C_[a1][b] = (a1 == b) ? 1.0 : 0.0;
            choleskyLower(pol_C_, pol_A_);
        }
    }
}

// -----------------------------------------------------------------------------
// [REJUV] partial restart of the front after a long stagnation
// -----------------------------------------------------------------------------
void IDE::rejuvenate()
{
    const int D  = prob_->dimension();
    const int NF = static_cast<int>(front_X_.size());
    if (NF < 4) return;
    std::vector<int> idx;
    sortedOrder(front_F_, NF, idx);
    const int keep = std::max(1, static_cast<int>(std::lround(rejuv_keep_ * NF)));
    Initializer initSampler;
    initSampler.configure(initopt_);
    auto fresh = initSampler.samplePopulation(*prob_, rng_, NF - keep);
    for (int r = keep, q = 0; r < NF && q < static_cast<int>(fresh.size()); ++r, ++q) {
        if (prob_->calls() >= max_evals_) break;
        const int i = idx[static_cast<size_t>(r)];
        Vec& x = fresh[static_cast<size_t>(q)];
        if (static_cast<int>(x.size()) != D) continue;
        const double f = eval(x);
        if (arch_enable_) archivePush(front_X_[static_cast<size_t>(i)]);
        front_X_[static_cast<size_t>(i)] = x;
        front_F_[static_cast<size_t>(i)] = f;
        noteBest(x, f);
    }
    MCR_.assign(static_cast<size_t>(memory_size_), mcr_init_);
    success_rate_ = sr_init_;
    ++rejuv_count_;
}

// -----------------------------------------------------------------------------
// init
// -----------------------------------------------------------------------------
void IDE::init()
{
    if (!prob_) return;
    const int D = prob_->dimension();
    if (D <= 0) return;

    int N = (pop_override_ > 0) ? pop_override_
                                : static_cast<int>(std::llround(np_init_mult_ * static_cast<double>(D)));
    if (N < np_min_) N = np_min_;
    n_init_ = N;
    setPopulation(N);

    main_X_.clear(); main_F_.clear();
    front_X_.clear(); front_F_.clear();
    archive_.clear();

    Initializer initSampler;
    initSampler.configure(initopt_);
    main_X_ = initSampler.samplePopulation(*prob_, rng_, N);
    main_X_.resize(static_cast<size_t>(N), Vec(static_cast<size_t>(D), 0.0));
    main_F_.assign(static_cast<size_t>(N), std::numeric_limits<double>::infinity());

    best_f_ = std::numeric_limits<double>::infinity();
    best_x_.assign(static_cast<size_t>(D), 0.0);
    for (int i = 0; i < N; ++i) {
        if (prob_->calls() >= max_evals_) break;
        main_F_[static_cast<size_t>(i)] = eval(main_X_[static_cast<size_t>(i)]);
        noteBest(main_X_[static_cast<size_t>(i)], main_F_[static_cast<size_t>(i)]);
    }
    if (!std::isfinite(best_f_) && !main_X_.empty()) best_x_ = main_X_[0];

    std::vector<int> idx;
    sortedOrder(main_F_, N, idx);
    front_X_.reserve(static_cast<size_t>(N));
    front_F_.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        front_X_.push_back(main_X_[static_cast<size_t>(idx[i])]);
        front_F_.push_back(main_F_[static_cast<size_t>(idx[i])]);
    }
    // [EIG] the basis reads the best part of main, so main is sorted from the
    // start (only then: unsorted main = exact L-SRTDE random stream)
    if (eig_enable_) sortAndTruncateMain(N);

    MCR_.assign(static_cast<size_t>(memory_size_), mcr_init_);
    memory_iter_  = 0;
    pf_index_     = 0;
    success_rate_ = sr_init_;
    generation_   = 0;

    eig_valid_ = false; gens_since_eig_ = 0; B_.clear(); eig_val_.clear();
    eig_p_cur_ = eig_prob_; eig_ts_ = eig_tn_ = bin_ts_ = bin_tn_ = 0.0;
    pol_started_ = false; pol_dead_ = false; pol_used_ = 0; pol_bad_ = 0;
    last_best_ = best_f_; stall_gens_ = 0; rejuv_count_ = 0;
    if (rejuv_gens_ <= 0) rejuv_gens_ = std::max(50, 3 * D);
    clpso_ = ClpsoScout(); cma_ = CmaScout();
    scout_total_used_ = 0; scout_adoptions_ = 0; scout_phase_ = false; scout_phases_ = 0;
    best_hist_.clear();

    if (eig_enable_) refreshEigenBasis();

    if (debug_ide_) {
        std::ostringstream os;
        os << "[ide] init -> D=" << D << ", N=" << N << ", eig=" << eig_enable_
           << " (p=" << eig_prob_ << "), archive=" << arch_enable_
           << ", polish=" << polish_enable_ << " (start " << polish_start_ << ", budget "
           << polish_budget_ << "), rejuv=" << rejuv_enable_ << ", max_evals=" << max_evals_ << "\n";
        out() << os.str();
    }
    printBest();
}

// -----------------------------------------------------------------------------
// one generation
// -----------------------------------------------------------------------------
void IDE::one_iteration()
{
    if (!prob_ || front_X_.empty()) return;

    const int D  = prob_->dimension();
    const int NF = static_cast<int>(front_X_.size());
    if (NF < 4 || static_cast<int>(main_X_.size()) != NF) {
        printBest();
        updateStop(front_F_);
        return;
    }

    const double    gen_f0 = best_f_;                    // [POLISH] efficiency bookkeeping
    const long long gen_c0 = static_cast<long long>(prob_->calls());

    const double meanF = f_base_ + f_amp_ * std::tanh(success_rate_ * f_sr_coef_);

    std::vector<int> idx_main, idx_front;
    sortedOrder(main_F_,  NF, idx_main);
    sortedOrder(front_F_, NF, idx_front);

    std::vector<double> rank_w(static_cast<size_t>(NF));
    for (int i = 0; i < NF; ++i)
        rank_w[static_cast<size_t>(i)] = std::exp(-static_cast<double>(i) / static_cast<double>(NF) * rank_pressure_);
    std::discrete_distribution<int> rank_dist(rank_w.begin(), rank_w.end());

    int psize = static_cast<int>(static_cast<double>(NF) * pbest_rate_ * std::exp(-success_rate_ * pbest_sr_coef_));
    psize = std::max(pbest_min_, psize);
    psize = std::min(psize, NF);

    main_X_.reserve(static_cast<size_t>(2 * NF));
    main_F_.reserve(static_cast<size_t>(2 * NF));

    std::vector<double> S_cr;    S_cr.reserve(static_cast<size_t>(NF));
    std::vector<double> S_delta; S_delta.reserve(static_cast<size_t>(NF));
    int successes = 0;

    const bool use_eig  = eig_enable_ && eig_valid_ && static_cast<int>(B_.size()) == D;
    const bool use_arch = arch_enable_ && !archive_.empty();
    const int  NA       = use_arch ? static_cast<int>(archive_.size()) : 0;

    Vec trial(static_cast<size_t>(D)), mutant, xe, ve, ue;
    if (use_eig) { mutant.resize(static_cast<size_t>(D)); xe.resize(static_cast<size_t>(D));
                   ve.resize(static_cast<size_t>(D)); ue.resize(static_cast<size_t>(D)); }

    for (int it = 0; it < NF; ++it) {
        if (prob_->calls() >= max_evals_) break;

        const int c = randInt(NF);
        const int k = randInt(memory_size_);

        int prand, r1, r2;
        do { prand = idx_main[static_cast<size_t>(randInt(psize))]; } while (prand == c);
        do { r1 = idx_front[static_cast<size_t>(rank_dist(rng_))]; }  while (r1 == prand);
        const Vec* x2p = nullptr;
        if (use_arch) {
            // [ARCH] r2 from main + archive
            do { r2 = randInt(NF + NA); } while (r2 < NF && (idx_main[static_cast<size_t>(r2)] == prand ||
                                                            idx_main[static_cast<size_t>(r2)] == r1));
            x2p = (r2 < NF) ? &main_X_[static_cast<size_t>(idx_main[static_cast<size_t>(r2)])]
                            : &archive_[static_cast<size_t>(r2 - NF)];
        } else {
            do { r2 = idx_main[static_cast<size_t>(randInt(NF))]; } while (r2 == prand || r2 == r1);
            x2p = &main_X_[static_cast<size_t>(r2)];
        }

        const double F  = sampleF(meanF);
        const double Cr = sampleCR(MCR_[static_cast<size_t>(k)]);

        const Vec& xc = front_X_[static_cast<size_t>(c)];
        const Vec& xp = main_X_[static_cast<size_t>(prand)];
        const Vec& x1 = front_X_[static_cast<size_t>(r1)];
        const Vec& x2 = *x2p;

        int crossed = 0;
        const bool eig_now = use_eig && (randU01() < (eig_adapt_ ? eig_p_cur_ : eig_prob_));
        if (!eig_now) {
            const int jrand = randInt(D);
            for (int j = 0; j < D; ++j) {
                const size_t sj = static_cast<size_t>(j);
                if (randU01() < Cr || j == jrand) {
                    const double v = xc[sj] + F * (xp[sj] - xc[sj]) + F * (x1[sj] - x2[sj]);
                    trial[sj] = repairComponent(j, v, xc[sj]);
                    ++crossed;
                } else {
                    trial[sj] = xc[sj];
                }
            }
        } else {
            // [EIG] binomial crossover in the eigen coordinates: u = B * bin(B^T x, B^T v)
            for (int j = 0; j < D; ++j)
                mutant[j] = xc[j] + F * (xp[j] - xc[j]) + F * (x1[j] - x2[j]);
            for (int q = 0; q < D; ++q) {
                double sx = 0.0, sv = 0.0;
                for (int i = 0; i < D; ++i) { sx += B_[i][q] * xc[i]; sv += B_[i][q] * mutant[i]; }
                xe[q] = sx; ve[q] = sv;
            }
            const int jrand = randInt(D);
            for (int q = 0; q < D; ++q) {
                if (randU01() < Cr || q == jrand) { ue[q] = ve[q]; ++crossed; }
                else ue[q] = xe[q];
            }
            for (int i = 0; i < D; ++i) {
                double s = 0.0;
                for (int q = 0; q < D; ++q) s += B_[i][q] * ue[q];
                trial[i] = repairComponent(i, s, xc[i]);
            }
        }
        const double actualCr = static_cast<double>(crossed) / static_cast<double>(D);

        const double ft = eval(trial);
        const bool success = (ft <= front_F_[static_cast<size_t>(c)]);
        if (use_eig) {                                         // [EIG] success statistics per crossover
            if (eig_now) { eig_tn_ += 1.0; if (success) eig_ts_ += 1.0; }
            else         { bin_tn_ += 1.0; if (success) bin_ts_ += 1.0; }
        }

        if (success) {
            const double parent_f = front_F_[static_cast<size_t>(c)];
            main_X_.push_back(trial);
            main_F_.push_back(ft);
            if (arch_enable_) archivePush(front_X_[static_cast<size_t>(pf_index_)]);   // [ARCH]
            front_X_[static_cast<size_t>(pf_index_)] = trial;
            front_F_[static_cast<size_t>(pf_index_)] = ft;
            noteBest(trial, ft);
            S_cr.push_back(actualCr);
            const double delta = delta_from_parent_
                                   ? std::fabs(parent_f - ft)
                                   : std::fabs(front_F_[static_cast<size_t>(c)] - ft);
            S_delta.push_back(delta);
            ++successes;
            pf_index_ = (pf_index_ + 1) % NF;
        }
    }

    success_rate_ = static_cast<double>(successes) / static_cast<double>(NF);

    const int newN = targetPopulationSize();
    removeWorstFromFront(newN);
    if (pf_index_ >= newN) pf_index_ %= newN;

    if (successes > 0) {
        const double mwl = weightedLehmerMean(S_cr, S_delta);
        double& m = MCR_[static_cast<size_t>(memory_iter_)];
        m = memory_c_ * m + (1.0 - memory_c_) * mwl;
        m = std::min(1.0, std::max(0.0, m));
        memory_iter_ = (memory_iter_ + 1) % memory_size_;
    }

    sortAndTruncateMain(newN);
    if (arch_enable_) {                                        // [ARCH] shrink with the population
        const size_t cap = static_cast<size_t>(std::lround(arch_rate_ * newN));
        while (archive_.size() > cap) archive_.erase(archive_.begin() + randInt(static_cast<int>(archive_.size())));
    }
    ++generation_;

    // [EIG] adaptive share of the eigen crossover: p = SR_eig / (SR_eig + SR_bin),
    // success rates with a forgetting factor 0.9 per generation and a +1/+2
    // prior, clamped to [eig_pmin, eig_pmax]. On separable problems the
    // coordinate-wise crossover wins and the share drops; on rotated ones
    // the eigen crossover wins and the share grows.
    if (use_eig && eig_adapt_) {
        const double se = (eig_ts_ + 1.0) / (eig_tn_ + 2.0);
        const double sb = (bin_ts_ + 1.0) / (bin_tn_ + 2.0);
        eig_p_cur_ = std::min(eig_pmax_, std::max(eig_pmin_, se / (se + sb)));
        eig_ts_ *= 0.9; eig_tn_ *= 0.9; bin_ts_ *= 0.9; bin_tn_ *= 0.9;
    }

    // [EIG] periodic refresh of the basis
    if (eig_enable_ && ++gens_since_eig_ >= eig_period_) refreshEigenBasis();

    // [SCOUT] a running scouting phase: the scouts receive scout_ratio
    // evaluations per evaluation of this DE generation (the DE has stagnated,
    // so most of the budget goes to the scouts, but it keeps refining).
    if (scout_phase_) {
        const long long de_gen = std::max<long long>(1, static_cast<long long>(prob_->calls()) - gen_c0);
        const bool adopted = scoutsRun(static_cast<long long>(scout_ratio_ * static_cast<double>(de_gen)));
        if (adopted || !(clpso_.active || cma_.active) || prob_->calls() >= max_evals_) {
            scout_phase_ = false;
            clpso_.active = cma_.active = false;
            best_hist_.clear();                          // stagnation is measured afresh
        }
        if (debug_ide_ > 1) {
            std::ostringstream os;
            os << "[ide] scout phase: used=" << scout_total_used_ << " adoptions=" << scout_adoptions_
               << " clpso_best=" << (clpso_.started ? clpso_.bf : 0.0)
               << " cma_best=" << (cma_.started ? cma_.bf : 0.0) << " cma_restarts=" << cma_.restarts
               << " de_best=" << mainBest() << (scout_phase_ ? "" : " -> phase ended") << "\n";
            out() << os.str();
        }
    }
    // [SCOUT] stagnation test: the DE best improved by less than scout_tol
    // (relative, ftol-style) over the last scout_window of the budget AND the
    // front population has collapsed into one funnel (mean distance to its
    // centroid below scout_spread of the box diagonal) -> start a scouting
    // phase. A DE that is still descending, or that sits on a plateau with a
    // spread-out population (Rastrigin-type functions: plateaus of ~30% of the
    // budget at spread 0.05, then a final descent), does not trigger it.
    else if (scout_enable_ && scout_phases_ < scout_max_phases_ && prob_->calls() < max_evals_) {
        const double E = static_cast<double>(max_evals_);
        const long long now = prob_->calls();
        const double fb = mainBest();
        best_hist_.emplace_back(now, fb);
        const long long win = std::max<long long>(1, static_cast<long long>(scout_window_ * E));
        if (static_cast<double>(now) >= scout_after_ * E && now - best_hist_.front().first >= win) {
            // DE best 'win' evaluations ago (latest record not newer than now - win)
            double f_then = best_hist_.front().second;
            size_t cut = 0;
            for (size_t i = 0; i < best_hist_.size(); ++i) {
                if (best_hist_[i].first <= now - win) { f_then = best_hist_[i].second; cut = i; }
                else break;
            }
            if (cut > 0) best_hist_.erase(best_hist_.begin(), best_hist_.begin() + static_cast<std::ptrdiff_t>(cut));
            const double gain = f_then - fb;
            const double rel  = gain / std::max(1.0, std::fabs(fb));
            // the gain must also be small compared with the fitness spread of
            // the front (median - best): a population that is still CONVERGING
            // (e.g. Schaffer F7 refining from 1e-2 to 1e-8) moves its best by
            // about its own spread per window, a STALLED one (sitting in a
            // funnel, waiting for rare lucky trials) by much less.
            std::vector<double> ff = front_F_;
            std::nth_element(ff.begin(), ff.begin() + static_cast<std::ptrdiff_t>(ff.size() / 2), ff.end());
            const double fspread = ff[ff.size() / 2] - fb;
            // and the population must agree in fitness (median - best tiny compared
            // with |best|): Rastrigin-type plateaus (CEC 2017 F5/F7) keep a front
            // whose members sit in different local minima (spread ~5-10% of |f|).
            const double frel = fspread / std::max(1.0, std::fabs(fb));
            if (rel < scout_tol_ && gain <= scout_gain_frac_ * fspread && frel < scout_fspread_
                && frontSpread() < scout_spread_) {
                const long long remain = max_evals_ - now;
                const long long total = std::min(static_cast<long long>(scout_max_ * E),
                                                 static_cast<long long>(scout_max_remain_ * static_cast<double>(remain)));
                scoutsStart(total);
            }
        }
    }

    // [REJUV]
    if (rejuv_enable_) {
        if (best_f_ < last_best_) { last_best_ = best_f_; stall_gens_ = 0; }
        else ++stall_gens_;
        if (stall_gens_ >= rejuv_gens_ && progressFraction() < rejuv_cutoff_) {
            rejuvenate();
            stall_gens_ = 0;
        }
    }

    // [POLISH] with efficiency arbitration (SPARQ): a burst must gain at least
    // as much per evaluation as the DE generation before it; after
    // polish_patience consecutive bursts that do not, the polish is switched
    // off for the rest of the run and the budget flows back to the DE (on
    // rugged landscapes, e.g. Weierstrass, the local refinement only steals
    // evaluations from the DE end-phase).
    if (polish_enable_ && !pol_dead_ && progressFraction() >= polish_start_ && prob_->calls() < max_evals_) {
        const double de_gain  = std::max(0.0, gen_f0 - best_f_);
        const double de_evals = static_cast<double>(std::max<long long>(1, prob_->calls() - gen_c0));
        const double pf0 = best_f_;
        const long long pu0 = pol_used_;
        polishBurst();
        const double pol_gain  = std::max(0.0, pf0 - best_f_);
        const double pol_evals = static_cast<double>(std::max<long long>(1, pol_used_ - pu0));
        if (pol_used_ > pu0) {
            if (pol_gain / pol_evals < de_gain / de_evals || pol_gain <= 0.0) {
                if (++pol_bad_ >= polish_patience_) pol_dead_ = true;
            } else {
                pol_bad_ = 0;
            }
        }
    }

    setPopulation(static_cast<int>(front_X_.size()));

    if (debug_ide_ > 1) {
        std::ostringstream os;
        os << "[ide] gen " << generation_ << " N=" << NF << "->" << newN
           << " SR=" << std::setprecision(4) << success_rate_ << " meanF=" << meanF
           << " psize=" << psize << " eig=" << (use_eig ? 1 : 0) << " p_eig=" << eig_p_cur_ << " arch=" << archive_.size()
           << " polish_used=" << pol_used_ << " rejuv=" << rejuv_count_
           << " scout_used=" << scout_total_used_ << " adoptions=" << scout_adoptions_
           << " scout_phases=" << scout_phases_ << " spread=" << frontSpread() << " best=" << mainBest()
           << " fspread=" << [&]{ std::vector<double> ff = front_F_; std::nth_element(ff.begin(), ff.begin() + static_cast<std::ptrdiff_t>(ff.size() / 2), ff.end()); return ff[ff.size() / 2] - mainBest(); }() << "\n";
        out() << os.str();
    }

    printBest();
    updateStop(front_F_);
}

// -----------------------------------------------------------------------------
// [SCOUT] funnel scouting & adoption (iDE-FS)
//
// Diagnosis (CEC 2022, see ide.h): on multi-funnel functions (composition
// F10-F12, Rosenbrock D=20) the L-SRTDE/iDE dynamics converge to the SAME
// funnel from any random start (best-of-8 independent restarts changes
// nothing), while search dynamics of a different kind land in other funnels:
// CLPSO enters the global funnel of F10 (o1 = an unrotated, separable Schwefel
// component), short CMA-ES runs find the global valley of Rosenbrock D=20.
// The scouts below run on their own budget quota, never receive information
// from the DE (so they stay independent of its funnel), and their best point
// is ADOPTED by the DE whenever it is better than the DE's best: a point far
// from the DE's best (another funnel) brings a small cloud of samples with it
// and replaces the worst part of both populations, so that the DE, which is
// excellent at exploitation, descends to the bottom of the adopted funnel
// with the remaining budget. Unused quota flows back to the DE automatically
// (the LPSR schedule counts evaluations, not generations).
// -----------------------------------------------------------------------------
double IDE::scoutEval(const Vec& x)
{
    const double f = eval(x);
    ++scout_total_used_;
    noteBest(x, f);
    return f;
}

// mean Euclidean distance of the front members to their centroid, relative
// to the box diagonal (0 = collapsed population)
double IDE::frontSpread() const
{
    const size_t n = front_X_.size();
    if (n < 2 || !prob_) return 0.0;
    const size_t D = front_X_[0].size();
    Vec c(D, 0.0);
    for (const Vec& x : front_X_) for (size_t j = 0; j < D; ++j) c[j] += x[j];
    for (size_t j = 0; j < D; ++j) c[j] /= static_cast<double>(n);
    double s = 0.0, diag2 = 0.0;
    for (size_t j = 0; j < D; ++j) { const double r = prob_->ub()[j] - prob_->lb()[j]; diag2 += r * r; }
    for (const Vec& x : front_X_) {
        double d2 = 0.0;
        for (size_t j = 0; j < D; ++j) { const double d = x[j] - c[j]; d2 += d * d; }
        s += std::sqrt(d2);
    }
    return s / static_cast<double>(n) / std::sqrt(std::max(diag2, 1e-300));
}

double IDE::mainBest(Vec* x) const
{
    double mf = std::numeric_limits<double>::infinity();
    const Vec* bx = nullptr;
    for (size_t i = 0; i < main_F_.size(); ++i)  if (main_F_[i] < mf)  { mf = main_F_[i];  bx = &main_X_[i]; }
    for (size_t i = 0; i < front_F_.size(); ++i) if (front_F_[i] < mf) { mf = front_F_[i]; bx = &front_X_[i]; }
    if (x && bx) *x = *bx;
    return mf;
}

void IDE::scoutsStart(long long total)
{
    ++scout_phases_;
    clpso_ = ClpsoScout();
    cma_   = CmaScout();
    if (!prob_) return;
    const int D = prob_->dimension();
    double share = scout_clpso_share_;
    if (D > scout_cma_max_dim_) share = 1.0;             // CMA scout too expensive: all to CLPSO
    clpso_.quota = static_cast<long long>(share * static_cast<double>(total));
    cma_.quota   = total - clpso_.quota;
    clpso_.N = (D <= 10) ? 10 : 40;                      // Liang et al. (2006)
    clpso_.active = (clpso_.quota >= 2LL * clpso_.N);
    cma_.lambda0 = 4 + static_cast<int>(std::floor(3.0 * std::log(static_cast<double>(std::max(1, D)))));
    cma_.active  = (cma_.quota >= 4LL * cma_.lambda0);
    if (!clpso_.active) clpso_.quota = 0;
    if (!cma_.active)   cma_.quota = 0;
    scout_phase_ = clpso_.active || cma_.active;
}

bool IDE::scoutsRun(long long budget)
{
    while (budget > 0 && prob_->calls() < max_evals_) {
        const bool c_ok = clpso_.active && clpso_.used < clpso_.quota;
        const bool m_ok = cma_.active   && cma_.used   < cma_.quota;
        if (!c_ok) clpso_.active = false;
        if (!m_ok) cma_.active   = false;
        if (!c_ok && !m_ok) break;
        // step the scout that is furthest behind its own quota
        const double rc = c_ok ? static_cast<double>(clpso_.used) / static_cast<double>(std::max<long long>(1, clpso_.quota)) : 2.0;
        const double rm = m_ok ? static_cast<double>(cma_.used)   / static_cast<double>(std::max<long long>(1, cma_.quota))   : 2.0;
        const long long before = scout_total_used_;
        if (rc <= rm) clpsoStep(clpso_); else cmaStep(cma_);
        const long long spent = scout_total_used_ - before;
        if (spent <= 0) {                                // no evaluation possible: that scout is done
            if (rc <= rm) clpso_.active = false; else cma_.active = false;
            continue;
        }
        budget -= spent;
    }

    // adoption test against the best member of the DE populations
    double mf = mainBest();
    const int D = prob_->dimension();
    bool adopted = false;
    const double tol = 1e-12 * std::max(1.0, std::fabs(mf));
    if (clpso_.started && clpso_.bf < mf - tol) {
        Vec spread(static_cast<size_t>(D), 0.0);         // std of the scout's pbests
        for (int j = 0; j < D; ++j) {
            double m = 0.0, q = 0.0;
            for (int i = 0; i < clpso_.N; ++i) m += clpso_.P[i][j];
            m /= clpso_.N;
            for (int i = 0; i < clpso_.N; ++i) { const double d = clpso_.P[i][j] - m; q += d * d; }
            spread[j] = std::sqrt(q / std::max(1, clpso_.N - 1));
        }
        adopted = scoutAdopt(clpso_.bx, clpso_.bf, spread) || adopted;
        mf = std::min(mf, clpso_.bf);
    }
    if (cma_.started && cma_.bf < mf - tol) {
        Vec spread(static_cast<size_t>(D), 0.0);         // sigma * sqrt(diag C), back in x units
        for (int j = 0; j < D; ++j) {
            const double r = prob_->ub()[j] - prob_->lb()[j];
            spread[j] = cma_.sigma * std::sqrt(std::max(0.0, cma_.C[j][j])) * r;
        }
        adopted = scoutAdopt(cma_.bx, cma_.bf, spread) || adopted;
    }
    return adopted;
}

bool IDE::scoutAdopt(const Vec& sx, double sf, const Vec& spread)
{
    const int D  = prob_->dimension();
    const int NF = static_cast<int>(front_X_.size());
    if (NF < 1 || static_cast<int>(main_X_.size()) != NF) return false;

    // distance to the DE's best member, relative to the box diagonal
    Vec xb = sx;
    mainBest(&xb);
    double dist2 = 0.0, diag2 = 0.0;
    for (int j = 0; j < D; ++j) {
        const double r = prob_->ub()[j] - prob_->lb()[j];
        const double d = sx[j] - xb[j];
        dist2 += d * d; diag2 += r * r;
    }
    const bool other_funnel = std::sqrt(dist2) > scout_min_dist_ * std::sqrt(std::max(diag2, 1e-300));

    int k = 1;
    if (other_funnel) k = std::max(1, static_cast<int>(std::lround(scout_adopt_frac_ * NF)));
    k = std::min(k, NF);

    std::vector<Vec> cx; std::vector<double> cf;
    cx.push_back(sx); cf.push_back(sf);
    std::normal_distribution<double> N01(0.0, 1.0);
    for (int t = 1; t < k && prob_->calls() < max_evals_; ++t) {
        Vec x = sx;
        for (int j = 0; j < D; ++j) {
            const double r  = prob_->ub()[j] - prob_->lb()[j];
            const double sd = std::min(0.05 * r, std::max(1e-3 * r, spread[j]));
            x[j] = std::min(prob_->ub()[j], std::max(prob_->lb()[j], x[j] + sd * N01(rng_)));
        }
        const double f = eval(x);
        noteBest(x, f);
        cx.push_back(std::move(x)); cf.push_back(f);
    }

    // replace the worst members of the front and of the main population
    std::vector<int> fi, mi;
    sortedOrder(front_F_, NF, fi);
    sortedOrder(main_F_,  NF, mi);
    for (size_t t = 0; t < cx.size(); ++t) {
        const int a = fi[static_cast<size_t>(NF - 1 - static_cast<int>(t))];
        const int b = mi[static_cast<size_t>(NF - 1 - static_cast<int>(t))];
        if (cf[t] < front_F_[a]) {
            if (arch_enable_) archivePush(front_X_[a]);
            front_X_[a] = cx[t]; front_F_[a] = cf[t];
        }
        if (cf[t] < main_F_[b]) { main_X_[b] = cx[t]; main_F_[b] = cf[t]; }
    }
    if (other_funnel) ++scout_adoptions_;
    return other_funnel;
}

// ---- CLPSO scout (Liang, Qin, Suganthan & Baskar, IEEE TEVC 2006) ----------
//  v = w v + c r (pbest_{f_i(d)}(d) - x), c = 1.49445, w 0.9 -> 0.4 over the
//  scout's own quota, Pc_i = 0.05 + 0.45 (e^{10 i/(N-1)} - 1)/(e^{10} - 1),
//  exemplar refresh gap m = 7, |v| <= 0.2 (ub - lb), reflection at the bounds
//  (same operators as [clpso] of this framework).
void IDE::clpsoExemplars(ClpsoScout& s, int i)
{
    const int D = prob_->dimension();
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    int nonself = 0;
    for (int j = 0; j < D; ++j) {
        if (s.N <= 1 || U01(rng_) > s.Pc[i]) { s.ex[i][j] = i; continue; }
        int a = randInt(s.N), b = randInt(s.N), g = 0;
        while (a == i && g++ < 10) a = randInt(s.N);
        g = 0;
        while ((b == i || b == a) && g++ < 10) b = randInt(s.N);
        if (a == i) a = (i + 1) % s.N;
        if (b == i) b = (i + 2) % s.N;
        s.ex[i][j] = (s.FP[b] < s.FP[a]) ? b : a;
        if (s.ex[i][j] != i) ++nonself;
    }
    if (nonself == 0 && D > 0 && s.N > 1) {
        int other = randInt(s.N - 1);
        if (other >= i) ++other;
        s.ex[i][randInt(D)] = other;
    }
}

void IDE::clpsoStart(ClpsoScout& s)
{
    const int D = prob_->dimension();
    s.started = true;
    Initializer initSampler;
    initSampler.configure(initopt_);
    s.X = initSampler.samplePopulation(*prob_, rng_, s.N);
    s.X.resize(static_cast<size_t>(s.N), Vec(static_cast<size_t>(D), 0.0));
    s.V.assign(static_cast<size_t>(s.N), Vec(static_cast<size_t>(D), 0.0));
    s.vmax.assign(static_cast<size_t>(D), 1.0);
    for (int j = 0; j < D; ++j) {
        const double r = prob_->ub()[j] - prob_->lb()[j];
        s.vmax[j] = (std::isfinite(r) && r > 0.0) ? 0.2 * r : 1.0;
    }
    s.FP.assign(static_cast<size_t>(s.N), std::numeric_limits<double>::infinity());
    s.bf = std::numeric_limits<double>::infinity();
    s.bx = s.X[0];
    for (int i = 0; i < s.N; ++i) {
        if (prob_->calls() >= max_evals_ || s.used >= s.quota) break;
        s.FP[i] = scoutEval(s.X[i]);
        ++s.used;
        if (s.FP[i] < s.bf) { s.bf = s.FP[i]; s.bx = s.X[i]; }
    }
    s.P = s.X;
    s.Pc.assign(static_cast<size_t>(s.N), 0.05);
    const double den = std::exp(10.0) - 1.0;
    for (int i = 0; i < s.N && s.N > 1; ++i)
        s.Pc[i] = 0.05 + 0.45 * (std::exp(10.0 * i / (s.N - 1)) - 1.0) / den;
    s.ex.assign(static_cast<size_t>(s.N), std::vector<int>(static_cast<size_t>(D), 0));
    s.noimp.assign(static_cast<size_t>(s.N), 0);
    for (int i = 0; i < s.N; ++i) clpsoExemplars(s, i);
}

void IDE::clpsoStep(ClpsoScout& s)
{
    if (!s.started) { clpsoStart(s); return; }
    const int D = prob_->dimension();
    const double t = std::min(1.0, static_cast<double>(s.used) / static_cast<double>(std::max<long long>(1, s.quota)));
    const double w = 0.9 - 0.5 * t;
    const double c = 1.49445;
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    const Vec& L = prob_->lb();
    const Vec& U = prob_->ub();
    for (int i = 0; i < s.N; ++i) {
        if (s.used >= s.quota || prob_->calls() >= max_evals_) return;
        if (s.noimp[i] >= 7) {
            clpsoExemplars(s, i);
            std::fill(s.V[i].begin(), s.V[i].end(), 0.0);
            s.noimp[i] = 0;
        }
        Vec& v = s.V[i];
        Vec& x = s.X[i];
        for (int j = 0; j < D; ++j) {
            v[j] = w * v[j] + c * U01(rng_) * (s.P[s.ex[i][j]][j] - x[j]);
            if (v[j] >  s.vmax[j]) v[j] =  s.vmax[j];
            if (v[j] < -s.vmax[j]) v[j] = -s.vmax[j];
            x[j] += v[j];
            if (!std::isfinite(x[j])) { x[j] = 0.5 * (L[j] + U[j]); v[j] = 0.0; }
            if (x[j] < L[j])      { x[j] = L[j] + (L[j] - x[j]); v[j] *= -0.5; }
            else if (x[j] > U[j]) { x[j] = U[j] - (x[j] - U[j]); v[j] *= -0.5; }
            x[j] = std::min(U[j], std::max(L[j], x[j]));
        }
        const double f = scoutEval(x);
        ++s.used;
        if (f < s.FP[i]) { s.FP[i] = f; s.P[i] = x; s.noimp[i] = 0; }
        else ++s.noimp[i];
        if (f < s.bf) { s.bf = f; s.bx = x; }
    }
}

// ---- CMA-ES scout (Hansen 2016 tutorial defaults) with IPOP restarts -------
//  Works in the box normalised to [0,1]^D (sigma0 = 0.3, mean uniform), the
//  sampled point is clipped to the box and the clipped point is used in the
//  update (Lamarckian repair). Restart when sigma*max(d) < 1e-12, the run
//  stalls for 10 + 30 n / lambda generations, cond(C) > 1e14, or sigma
//  diverges; lambda doubles at every restart.
void IDE::cmaRestart(CmaScout& s)
{
    const int n = prob_->dimension();
    if (!s.started) { s.started = true; s.lambda = s.lambda0; s.bf = std::numeric_limits<double>::infinity(); }
    else { ++s.restarts; s.lambda = std::min(s.lambda * 2, 64 * s.lambda0); }
    s.mu = s.lambda / 2;
    s.w.assign(static_cast<size_t>(s.mu), 0.0);
    double sw = 0.0, sw2 = 0.0;
    for (int i = 0; i < s.mu; ++i) { s.w[i] = std::log(s.mu + 0.5) - std::log(i + 1.0); sw += s.w[i]; }
    for (int i = 0; i < s.mu; ++i) { s.w[i] /= sw; sw2 += s.w[i] * s.w[i]; }
    s.mueff = 1.0 / sw2;
    const double nn = static_cast<double>(n);
    s.cc    = (4.0 + s.mueff / nn) / (nn + 4.0 + 2.0 * s.mueff / nn);
    s.cs    = (s.mueff + 2.0) / (nn + s.mueff + 5.0);
    s.c1    = 2.0 / ((nn + 1.3) * (nn + 1.3) + s.mueff);
    s.cmu   = std::min(1.0 - s.c1, 2.0 * (s.mueff - 2.0 + 1.0 / s.mueff) / ((nn + 2.0) * (nn + 2.0) + s.mueff));
    s.damps = 1.0 + 2.0 * std::max(0.0, std::sqrt((s.mueff - 1.0) / (nn + 1.0)) - 1.0) + s.cs;
    s.chiN  = std::sqrt(nn) * (1.0 - 1.0 / (4.0 * nn) + 1.0 / (21.0 * nn * nn));
    s.m.assign(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) s.m[j] = randU01();
    s.pc.assign(static_cast<size_t>(n), 0.0);
    s.ps.assign(static_cast<size_t>(n), 0.0);
    s.d.assign(static_cast<size_t>(n), 1.0);
    s.C.assign(static_cast<size_t>(n), Vec(static_cast<size_t>(n), 0.0));
    s.B.assign(static_cast<size_t>(n), Vec(static_cast<size_t>(n), 0.0));
    for (int j = 0; j < n; ++j) { s.C[j][j] = 1.0; s.B[j][j] = 1.0; }
    s.sigma = 0.3;
    s.gen = 0; s.eigen_gen = 0; s.stall = 0;
    s.run_best = std::numeric_limits<double>::infinity();
}

void IDE::cmaEigen(CmaScout& s)
{
    const int n = prob_->dimension();
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b) { const double v = 0.5 * (s.C[a][b] + s.C[b][a]); s.C[a][b] = s.C[b][a] = v; }
    Vec ev;
    jacobiEigen(s.C, s.B, ev);
    for (int j = 0; j < n; ++j) s.d[j] = std::sqrt(std::max(ev[j], 1e-300));
    s.eigen_gen = s.gen;
}

void IDE::cmaStep(CmaScout& s)
{
    const int n = prob_->dimension();
    if (!s.started) cmaRestart(s);
    if (prob_->calls() + s.lambda > max_evals_) { s.active = false; return; }
    if (s.used + s.lambda > s.quota) { s.used = s.quota; return; }   // rest of the quota is too small for a generation

    std::normal_distribution<double> N01(0.0, 1.0);
    std::vector<Vec> Y(static_cast<size_t>(s.lambda), Vec(static_cast<size_t>(n)));
    std::vector<double> F(static_cast<size_t>(s.lambda));
    Vec z(static_cast<size_t>(n)), x(static_cast<size_t>(n));
    for (int k = 0; k < s.lambda; ++k) {
        for (int j = 0; j < n; ++j) z[j] = s.d[j] * N01(rng_);
        Vec& y = Y[static_cast<size_t>(k)];
        for (int i = 0; i < n; ++i) {
            double v = 0.0;
            for (int j = 0; j < n; ++j) v += s.B[i][j] * z[j];
            double u = s.m[i] + s.sigma * v;
            if (!std::isfinite(u)) u = 0.5;
            u = std::min(1.0, std::max(0.0, u));                       // clip to the unit box
            y[i] = (u - s.m[i]) / s.sigma;                             // repaired step
            x[i] = prob_->lb()[i] + u * (prob_->ub()[i] - prob_->lb()[i]);
        }
        F[static_cast<size_t>(k)] = scoutEval(x);
        ++s.used;
        if (F[static_cast<size_t>(k)] < s.bf) { s.bf = F[static_cast<size_t>(k)]; s.bx = x; }
    }

    std::vector<int> idx;
    sortedOrder(F, s.lambda, idx);
    Vec yw(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < s.mu; ++i)
        for (int j = 0; j < n; ++j) yw[j] += s.w[i] * Y[static_cast<size_t>(idx[i])][j];
    for (int j = 0; j < n; ++j) s.m[j] += s.sigma * yw[j];

    // C^{-1/2} yw = B diag(1/d) B^T yw
    Vec t1(static_cast<size_t>(n), 0.0), cinv(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) { double v = 0.0; for (int i = 0; i < n; ++i) v += s.B[i][j] * yw[i]; t1[j] = v / s.d[j]; }
    for (int i = 0; i < n; ++i) { double v = 0.0; for (int j = 0; j < n; ++j) v += s.B[i][j] * t1[j]; cinv[i] = v; }
    double psn = 0.0;
    const double a_s = std::sqrt(s.cs * (2.0 - s.cs) * s.mueff);
    for (int j = 0; j < n; ++j) { s.ps[j] = (1.0 - s.cs) * s.ps[j] + a_s * cinv[j]; psn += s.ps[j] * s.ps[j]; }
    psn = std::sqrt(psn);
    const double hden = std::sqrt(1.0 - std::pow(1.0 - s.cs, 2.0 * (s.gen + 1)));
    const bool hsig = (psn / std::max(hden, 1e-300) / s.chiN) < (1.4 + 2.0 / (n + 1.0));
    const double a_c = std::sqrt(s.cc * (2.0 - s.cc) * s.mueff);
    for (int j = 0; j < n; ++j) s.pc[j] = (1.0 - s.cc) * s.pc[j] + (hsig ? a_c * yw[j] : 0.0);
    const double dh = hsig ? 0.0 : s.cc * (2.0 - s.cc);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            double rmu = 0.0;
            for (int i = 0; i < s.mu; ++i) {
                const Vec& y = Y[static_cast<size_t>(idx[i])];
                rmu += s.w[i] * y[a] * y[b];
            }
            const double v = (1.0 - s.c1 - s.cmu) * s.C[a][b]
                           + s.c1 * (s.pc[a] * s.pc[b] + dh * s.C[a][b])
                           + s.cmu * rmu;
            s.C[a][b] = s.C[b][a] = v;
        }
    s.sigma *= std::exp((s.cs / s.damps) * (psn / s.chiN - 1.0));
    ++s.gen;
    if (static_cast<double>(s.gen - s.eigen_gen) > 1.0 / ((s.c1 + s.cmu) * n * 10.0)) cmaEigen(s);

    const double fgen = F[static_cast<size_t>(idx[0])];
    if (fgen < s.run_best - 1e-12 * std::max(1.0, std::fabs(s.run_best))) { s.run_best = fgen; s.stall = 0; }
    else ++s.stall;

    double dmax = 0.0, dmin = std::numeric_limits<double>::infinity();
    for (double v : s.d) { dmax = std::max(dmax, v); dmin = std::min(dmin, v); }
    const bool tiny    = !(s.sigma * dmax > 1e-12);
    const bool stalled = s.stall > 10 + static_cast<int>(std::ceil(30.0 * n / s.lambda));
    const bool illcond = (dmax / std::max(dmin, 1e-300)) > 1e7;
    const bool diverge = !std::isfinite(s.sigma) || s.sigma * dmax > 2.0;
    if (tiny || stalled || illcond || diverge) cmaRestart(s);
}

void IDE::end()
{
    if (!end_local_refine_ || !prob_ || end_local_method_.empty() || best_x_.empty()) return;
    std::pair<Vec, double> loc = localSearch(end_local_method_, best_x_);
    if (std::isfinite(loc.second) && loc.second < best_f_) {
        best_x_ = loc.first;
        best_f_ = loc.second;
    }
    printBest();
}

} // namespace optimsolution
