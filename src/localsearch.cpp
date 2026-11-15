#include "optimizer.h"
#include "problem.h"
#include "config.h"
#include "localsearch.h"   // brings in inline dot / l2norm / project_to_bounds / backtrackingArmijo

#include <vector>
#include <deque>           // FIX #7: deque for O(1) pop_front in L-BFGS history
#include <random>
#include <utility>
#include <limits>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <sstream>
#include <mutex>
#include <set>
#include <cstdio>

namespace optimsolution {

// ===== Config helpers =====
// BUGFIX: the settings file used to be opened with the RELATIVE path
// "optimsolution.cfg", i.e. relative to the current working directory. When
// the program is started from another directory (GUI, batch scripts) the
// file was not found and every local-search setting silently fell back to
// its compiled default, while the rest of the program (which locates the
// file with findOptimsolutionCfg()) did read it. The path is now set once by
// main() through setLocalSearchConfigPath(). The file was also re-read and
// re-parsed on EVERY local-search call; sections are now cached
// (thread-safe for the OpenMP parallel-runs mode).
namespace {
std::mutex& lsMutex() { static std::mutex m; return m; }
std::string& lsCfgPath() { static std::string p = "optimsolution.cfg"; return p; }
std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& lsCache() {
    static std::unordered_map<std::string, std::unordered_map<std::string, std::string>> c;
    return c;
}
} // namespace

void setLocalSearchConfigPath(const std::string& path)
{
    std::lock_guard<std::mutex> lk(lsMutex());
    if (!path.empty()) lsCfgPath() = path;
    lsCache().clear();
}

namespace {
// Per-thread section override (see ScopedLocalSearchOverride in localsearch.h)
thread_local const std::unordered_map<std::string, std::string>* tlsOverrideKV = nullptr;
thread_local std::string tlsOverrideSection;
} // namespace

ScopedLocalSearchOverride::ScopedLocalSearchOverride(
    const std::string& section, const std::unordered_map<std::string, std::string>& kv)
    : prev_kv_(tlsOverrideKV), prev_section_(tlsOverrideSection)
{
    std::string s = section;
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    tlsOverrideKV = &kv;
    tlsOverrideSection = s;
}

ScopedLocalSearchOverride::~ScopedLocalSearchOverride()
{
    tlsOverrideKV = static_cast<const std::unordered_map<std::string, std::string>*>(prev_kv_);
    tlsOverrideSection = prev_section_;
}

static std::unordered_map<std::string, std::string>
loadKV(const std::string& method_section)
{
    std::unordered_map<std::string, std::string> kv;
    {
        std::lock_guard<std::mutex> lk(lsMutex());
        auto& cache = lsCache();
        auto it = cache.find(method_section);
        if (it != cache.end()) {
            kv = it->second;
        } else {
            auto cfg = Config::load(lsCfgPath(), method_section);
            cache[method_section] = cfg.methodKV.kv;
            kv = cfg.methodKV.kv;
        }
    }
    if (tlsOverrideKV && tlsOverrideSection == method_section) {
        for (const auto& p : *tlsOverrideKV) {
            std::string k = p.first;
            for (char& c : k) c = (char)std::tolower((unsigned char)c);
            kv[k] = p.second;
        }
    }
    return kv;
}

static double getDouble(const std::unordered_map<std::string, std::string>& kv,
                        const std::string& key, double defv)
{
    auto it = kv.find(key);
    if (it == kv.end()) return defv;
    try   { return std::stod(it->second); }
    catch (...) { return defv; }
}

static int getInt(const std::unordered_map<std::string, std::string>& kv,
                  const std::string& key, int defv)
{
    auto it = kv.find(key);
    if (it == kv.end()) return defv;
    try   { return std::stoi(it->second); }
    catch (...) { return defv; }
}

// ===== Gradient Descent (Armijo backtracking) =====
std::pair<Vec, double> localGD(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0, const std::string& section,
                                int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);
    const int MAX_ITERS = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    const double GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    const double alpha0     = getDouble(kv, "alpha0",     1.0);
    const double c1         = getDouble(kv, "c1",         1e-4);
    const int    backtracks = getInt   (kv, "backtracks", 20);

    Vec    x = x0;
    project_to_bounds(x, prob);
    double f = prob->evaluate(x);
    // Gradient computed once here and then forwarded from ls.g_new each step,
    // avoiding the double evaluation (backtrackingArmijo already computes it).
    Vec    g = prob->gradient(x);

    // FIX (step scaling): with the raw gradient as direction and a fixed
    // alpha0 = 1, the first trial step had length |g| (e.g. 1e8 on badly
    // scaled problems) and 'backtracks' halvings (2^-20 ~ 1e-6) could not
    // bring it down, so GD stopped at the starting point. The first step now
    // has length <= alpha0 (alpha = alpha0 / max(1, |g|), Nocedal & Wright
    // sec. 3.5), and each later line search starts from twice the previously
    // accepted step (capped at alpha0).
    double alpha_next = alpha0 / std::max(1.0, l2norm(g));
    for (int k = 0; k < MAX_ITERS; ++k) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        if (l2norm(g) <= GRAD_TOL) break;

        Vec d(x.size());
        for (size_t j = 0; j < x.size(); ++j) d[j] = -g[j];

        auto ls = backtrackingArmijo(prob, x, g, d, alpha_next, c1, backtracks, f, maxEvalsCap);
        if (ls.x_new == x) break;  // line search made no progress
        alpha_next = std::min(alpha0, 2.0 * ls.alpha);

        x = std::move(ls.x_new);
        f = ls.f_new;
        g = std::move(ls.g_new);   // reuse — no second gradient call needed
    }

    return { x, f };
}

// ===== L-BFGS (two-loop recursion) =====
static Vec lbfgs_direction(const Vec&               grad,
                           const std::deque<Vec>&   s_hist,
                           const std::deque<Vec>&   y_hist,
                           const std::deque<double>& rho_hist)
{
    int  m = (int)s_hist.size();
    Vec  q = grad;
    std::vector<double> alpha(m, 0.0);

    for (int i = m - 1; i >= 0; --i) {
        alpha[i] = rho_hist[i] * dot(s_hist[i], q);
        for (size_t j = 0; j < q.size(); ++j) q[j] -= alpha[i] * y_hist[i][j];
    }

    double scale = 1.0;
    if (m > 0) {
        double sy = dot(s_hist.back(), y_hist.back());
        double yy = dot(y_hist.back(), y_hist.back());
        if (yy > 0.0) scale = sy / yy;
    }
    for (size_t j = 0; j < q.size(); ++j) q[j] *= scale;

    for (int i = 0; i < m; ++i) {
        double beta = rho_hist[i] * dot(y_hist[i], q);
        for (size_t j = 0; j < q.size(); ++j)
            q[j] += (alpha[i] - beta) * s_hist[i][j];
    }

    for (size_t j = 0; j < q.size(); ++j) q[j] = -q[j];
    return q;
}

std::pair<Vec, double> localLBFGS(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0, const std::string& section,
                                   int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);
    const int MAX_ITERS = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    const double GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    const int    m_history  = getInt   (kv, "m_history",  10);
    const double alpha0     = getDouble(kv, "alpha0",     1.0);
    const double c1         = getDouble(kv, "c1",         1e-4);
    const int    backtracks = getInt   (kv, "backtracks", 20);

    Vec    x = x0;
    project_to_bounds(x, prob);
    double f = prob->evaluate(x);
    Vec    g = prob->gradient(x);

    // FIX #7: deque gives O(1) pop_front instead of O(m) erase(begin())
    std::deque<Vec>    s_hist, y_hist;
    std::deque<double> rho_hist;

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        if (l2norm(g) <= GRAD_TOL) break;

        Vec d = s_hist.empty()
            ? [&]{ Vec neg(x.size()); for (size_t j=0;j<x.size();++j) neg[j]=-g[j]; return neg; }()
            : lbfgs_direction(g, s_hist, y_hist, rho_hist);

        // FIX (step scaling): without curvature information the direction is
        // the raw -g; its first trial step is limited to length alpha0
        // (Nocedal & Wright sec. 3.5). With history the scaled two-loop
        // direction makes alpha0 = 1 the natural first trial.
        const double a_try = s_hist.empty() ? alpha0 / std::max(1.0, l2norm(g)) : alpha0;
        auto ls = backtrackingArmijo(prob, x, g, d, a_try, c1, backtracks, f, maxEvalsCap);
        if (ls.x_new == x) break;

        // Build s and y for the history update
        Vec s(x.size()), y(x.size());
        for (size_t j = 0; j < x.size(); ++j) {
            s[j] = ls.x_new[j] - x[j];
            y[j] = ls.g_new[j] - g[j];
        }

        double ys = dot(y, s);
        if (ys > 1e-12) {
            if ((int)s_hist.size() == m_history) {
                s_hist.pop_front();     // FIX #7: O(1)
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            s_hist.push_back(std::move(s));
            y_hist.push_back(std::move(y));
            rho_hist.push_back(1.0 / ys);
        }

        x = std::move(ls.x_new);
        f = ls.f_new;
        g = std::move(ls.g_new);
    }

    return { x, f };
}

// ===== BFGS (full dense inverse-Hessian) =====
std::pair<Vec, double> localBFGS(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0, const std::string& section,
                                  int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);

    const int MAX_ITERS = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    const double GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    const double alpha0     = getDouble(kv, "alpha0",     1.0);
    const double c1         = getDouble(kv, "c1",         1e-4);
    const int    backtracks = getInt   (kv, "backtracks", 20);

    Vec x = x0;
    project_to_bounds(x, prob);
    const int D = (int)x.size();

    // H starts as the identity (approximate inverse Hessian)
    std::vector<double> H(D * D, 0.0);
    for (int i = 0; i < D; ++i) H[i * D + i] = 1.0;

    double f = prob->evaluate(x);
    Vec    g = prob->gradient(x);
    bool   H_is_identity = true;

    auto H_mul = [&](const Vec& v) {
        Vec r(D, 0.0);
        for (int i = 0; i < D; ++i) {
            double s = 0.0;
            for (int j = 0; j < D; ++j) s += H[i * D + j] * v[j];
            r[i] = s;
        }
        return r;
    };

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        if (l2norm(g) <= GRAD_TOL) break;

        Vec d = H_mul(g);
        for (int j = 0; j < D; ++j) d[j] = -d[j];

        // Guard: if d is not a descent direction (numerical drift in H),
        // reset H to identity and fall back to steepest-descent step.
        if (dot(d, g) >= 0.0) {
            std::fill(H.begin(), H.end(), 0.0);
            for (int i = 0; i < D; ++i) H[i * D + i] = 1.0;
            for (int j = 0; j < D; ++j) d[j] = -g[j];
            H_is_identity = true;
        }

        // FIX (step scaling): while H is still the identity the direction is
        // the raw -g; limit that first trial step to length alpha0.
        const double a_try = H_is_identity ? alpha0 / std::max(1.0, l2norm(g)) : alpha0;
        auto ls = backtrackingArmijo(prob, x, g, d, a_try, c1, backtracks, f, maxEvalsCap);
        if (ls.x_new == x) break;

        Vec s(D), y(D);
        for (int j = 0; j < D; ++j) {
            s[j] = ls.x_new[j] - x[j];
            y[j] = ls.g_new[j] - g[j];
        }

        double ys = dot(y, s);
        if (ys > 1e-12) {
            // FIX: scale the initial inverse Hessian H0 = (y's / y'y) I before the
            // first update (Nocedal & Wright eq. 6.20); unscaled H0 = I is badly
            // mis-sized whenever the curvature is far from 1.
            if (H_is_identity) {
                const double yy = dot(y, y);
                if (yy > 0.0) {
                    const double gamma = ys / yy;
                    for (int i = 0; i < D; ++i) H[i * D + i] = gamma;
                }
                H_is_identity = false;
            }
            Vec Hy(D, 0.0);
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j)
                    Hy[i] += H[i * D + j] * y[j];

            double yHy = dot(y, Hy);

            // Standard BFGS inverse-Hessian update:
            // H+ = H + (1 + y'Hy/y's)(ss')/y's - (Hys' + sy'H)/y's
            for (int i = 0; i < D; ++i) {
                for (int j = 0; j < D; ++j) {
                    double term1 = (1.0 + yHy / ys) * (s[i] * s[j]) / ys;
                    double term2 = (Hy[i] * s[j] + s[i] * Hy[j]) / ys;
                    H[i * D + j] += term1 - term2;
                }
            }
        }

        x = std::move(ls.x_new);
        f = ls.f_new;
        g = std::move(ls.g_new);
    }

    return { x, f };
}

// ===== Nelder–Mead simplex (derivative-free) =====

// FIX #8: renamed to simplex_radius for clarity (it's max distance from centroid,
//         not the diameter / longest edge of the simplex).
static double simplex_radius(const std::vector<Vec>& S)
{
    if (S.empty()) return 0.0;
    const size_t D = S[0].size();
    const size_t N = S.size();

    Vec c(D, 0.0);
    for (const auto& v : S)
        for (size_t j = 0; j < D; ++j) c[j] += v[j];
    for (size_t j = 0; j < D; ++j) c[j] /= (double)N;

    double md = 0.0;
    for (const auto& v : S) {
        double dist = 0.0;
        for (size_t j = 0; j < D; ++j) {
            double d = std::abs(v[j] - c[j]);
            if (d > dist) dist = d;
        }
        if (dist > md) md = dist;
    }
    return md;
}

std::pair<Vec, double> localNM(Problem* prob, std::mt19937_64& rng, const Vec& x0, const std::string& section,
                                int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);

    const int MAX_ITERS = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    const double simplex_step = getDouble(kv, "simplex_delta", 1e-2);
    // Optional: initial simplex edge as a FRACTION of each coordinate's box
    // width (0 = off -> absolute simplex_delta). An absolute 1e-2 is fine for
    // refining a nearly converged point but makes a stand-alone Nelder-Mead
    // run (method "nm", random start in e.g. [-100,100]) crawl.
    const double simplex_rel  = getDouble(kv, "simplex_rel", 0.0);
    // FIX #6: termination tolerance from config (was hardcoded 1e-6)
    const double SIMPLEX_TOL  = getDouble(kv, "tol",           1e-6);

    double alpha_nm = getDouble(kv, "alpha", 1.0);   // reflection
    double gamma_nm = getDouble(kv, "gamma", 2.0);   // expansion
    // FIX #5: config keys corrected from "beta"/"delta" to "rho"/"sigma"
    double rho_nm   = getDouble(kv, "rho",   0.5);   // contraction
    double sigma_nm = getDouble(kv, "sigma", 0.5);   // shrink

    const int D = (int)x0.size();

    // BUGFIX: Nelder-Mead used to evaluate reflected/expanded points outside
    // the search box and could return a point outside it; every vertex is
    // now projected onto the bounds. The evaluation cap is also checked
    // before EVERY evaluation (it was only checked once per iteration, and
    // the shrink step alone costs D evaluations); a vertex that can no longer
    // be evaluated gets f = +inf and the search stops.
    bool capHit = false;
    auto ev = [&](Vec& v) -> double {
        project_to_bounds(v, prob);
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) {
            capHit = true;
            return std::numeric_limits<double>::infinity();
        }
        return prob->evaluate(v);
    };

    std::vector<Vec> S(D + 1, x0);
    std::uniform_real_distribution<double> U(-1.0, 1.0);
    for (int i = 1; i <= D; ++i) {
        for (int j = 0; j < D; ++j) {
            double step = simplex_step;
            if (simplex_rel > 0.0 && j < (int)prob->lb().size() && j < (int)prob->ub().size())
                step = simplex_rel * (prob->ub()[j] - prob->lb()[j]);
            S[i][j] += (i == j + 1 ? step : 0.0) + step * U(rng);
        }
    }

    std::vector<double> F(D + 1, std::numeric_limits<double>::infinity());
    for (int i = 0; i <= D; ++i) F[i] = ev(S[i]);

    auto order = [&]() {
        std::vector<int> idx(D + 1);
        for (int i = 0; i <= D; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return F[a] < F[b]; });
        return idx;
    };

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (capHit || (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap)) break;
        auto idx = order();
        int lo = idx[0], hi = idx[D], nh = idx[D - 1];

        // FIX #6: use configurable tolerance
        if (simplex_radius(S) <= SIMPLEX_TOL) break;

        // Centroid of all vertices except worst
        Vec xc(D, 0.0);
        for (int t = 0; t < D; ++t) {
            double s = 0.0;
            for (int i = 0; i < D; ++i) s += S[idx[i]][t];
            xc[t] = s / (double)D;
        }

        // Reflection
        Vec xr(D);
        for (int j = 0; j < D; ++j)
            xr[j] = xc[j] + alpha_nm * (xc[j] - S[hi][j]);
        double fr = ev(xr);

        if (fr < F[lo]) {
            // Expansion
            Vec xe(D);
            for (int j = 0; j < D; ++j)
                xe[j] = xc[j] + gamma_nm * (xr[j] - xc[j]);
            double fe = ev(xe);
            if (fe < fr) { S[hi] = std::move(xe); F[hi] = fe; }
            else          { S[hi] = std::move(xr); F[hi] = fr; }
        } else if (fr < F[nh]) {
            // Accepted reflection
            S[hi] = std::move(xr); F[hi] = fr;
        } else {
            // Contraction
            bool outside = (fr < F[hi]);
            Vec  xcand(D);
            if (outside)
                for (int j = 0; j < D; ++j)
                    xcand[j] = xc[j] + rho_nm * (xr[j] - xc[j]);
            else
                for (int j = 0; j < D; ++j)
                    xcand[j] = xc[j] + rho_nm * (S[hi][j] - xc[j]);

            double fc = ev(xcand);
            if (fc < (outside ? fr : F[hi])) {
                S[hi] = std::move(xcand); F[hi] = fc;
            } else {
                // Shrink
                for (int i = 1; i <= D; ++i) {
                    for (int j = 0; j < D; ++j)
                        S[idx[i]][j] = S[lo][j] + sigma_nm * (S[idx[i]][j] - S[lo][j]);
                    F[idx[i]] = ev(S[idx[i]]);
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i <= D; ++i) if (F[i] < F[best]) best = i;
    return { S[best], F[best] };
}

// ===========================================================================
// Shared helpers for cg / ncg / adam
// ===========================================================================

static std::string getStr(const std::unordered_map<std::string, std::string>& kv,
                          const std::string& key, const std::string& defv)
{
    auto it = kv.find(key);
    if (it == kv.end() || it->second.empty()) return defv;
    std::string s = it->second;
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// A coordinate counts as lying ON a bound when it is within
// 1e-10 * max(1, ub_j - lb_j) of it (epsilon-active set, Bertsekas 1982).
// With an exact test (x_j <= lb_j) a coordinate left 1e-17 away from a bound
// by round-off was "free", although no measurable move toward that bound is
// possible: the CG feasible step amax then became ~0 on every iteration and
// the method silently degraded to projected steepest descent.
static inline double boundTol(const Problem* prob, size_t j)
{
    const Vec& lb = prob->lb();
    const Vec& ub = prob->ub();
    double w = 1.0;
    if (j < lb.size() && j < ub.size() && std::isfinite(lb[j]) && std::isfinite(ub[j]))
        w = std::max(1.0, ub[j] - lb[j]);
    return 1e-10 * w;
}
static inline bool atLower(const Vec& x, const Problem* prob, size_t j)
{
    const Vec& lb = prob->lb();
    return j < lb.size() && std::isfinite(lb[j]) && x[j] - lb[j] <= boundTol(prob, j);
}
static inline bool atUpper(const Vec& x, const Problem* prob, size_t j)
{
    const Vec& ub = prob->ub();
    return j < ub.size() && std::isfinite(ub[j]) && ub[j] - x[j] <= boundTol(prob, j);
}

// Parameter validation for cg / ncg / adam. An out-of-range value (which
// would otherwise give NaN steps, e.g. beta1 = 1 in Adam, or a strong-Wolfe
// search with c2 <= c1 that has no solution) is replaced by the documented
// default and reported ONCE per section/key on stderr.
static void warnBadParam(const std::string& section, const std::string& key,
                         const std::string& given, const std::string& used)
{
    static std::mutex m;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(m);
    if (!seen.insert(section + "." + key).second) return;
    std::fprintf(stderr, "[%s] %s = %s is out of range; using %s\n",
                 section.c_str(), key.c_str(), given.c_str(), used.c_str());
}
static std::string numStr(double v) { char b[64]; std::snprintf(b, sizeof b, "%g", v); return b; }

// v must satisfy ok(v), otherwise it is reset to def (with a warning).
template <class T, class Ok>
static void checkParam(const std::string& section, const char* key, T& v, T def, Ok ok)
{
    if (ok(v)) return;
    warnBadParam(section, key, numStr((double)v), numStr((double)def));
    v = def;
}

// Coordinate j is "blocked" when x_j sits on a bound and the steepest-descent
// direction -g_j points out of the box: no feasible descent move exists along
// that coordinate, so it is held fixed for the current iteration.
static inline bool blockedAtBound(const Vec& x, const Vec& g, const Problem* prob, size_t j)
{
    if (atLower(x, prob, j) && g[j] > 0.0) return true;
    if (atUpper(x, prob, j) && g[j] < 0.0) return true;
    return false;
}

// Zero the components of a search direction that point out of the box at a
// coordinate lying on a bound. Needed for directions that are not -P(g)
// (CG's -g + beta*d, the Newton direction): a coordinate on a bound whose
// gradient points inward is not blocked, but beta*d may still push it
// outward. For a descent direction this never makes g'd larger (the dropped
// terms g_j d_j are >= 0 there), and it keeps the feasible step amax > 0.
static void dropOutwardAtBounds(Vec& d, const Vec& x, const Problem* prob)
{
    for (size_t j = 0; j < d.size(); ++j) {
        if ((d[j] < 0.0 && atLower(x, prob, j)) || (d[j] > 0.0 && atUpper(x, prob, j))) d[j] = 0.0;
    }
}

// Projected gradient: g with the blocked components set to zero. Its norm is
// zero exactly at a first-order (KKT) point of the box-constrained problem.
static Vec projectedGradient(const Vec& x, const Vec& g, const Problem* prob)
{
    Vec pg = g;
    for (size_t j = 0; j < pg.size(); ++j) {
        if (!std::isfinite(pg[j]) || blockedAtBound(x, g, prob, j)) pg[j] = 0.0;
    }
    return pg;
}

static std::vector<char> blockedMask(const Vec& x, const Vec& g, const Problem* prob)
{
    std::vector<char> m(x.size(), 0);
    for (size_t j = 0; j < x.size(); ++j) m[j] = blockedAtBound(x, g, prob, j) ? 1 : 0;
    return m;
}

static inline bool allFinite(const Vec& v)
{
    for (double a : v) if (!std::isfinite(a)) return false;
    return true;
}

// Length of the box diagonal, or a large fallback if any bound is missing or
// infinite. Used only to cap an over-long Newton step.
static double boxDiagonal(const Problem* prob, const Vec& x)
{
    const Vec& lb = prob->lb();
    const Vec& ub = prob->ub();
    double s = 0.0;
    for (size_t j = 0; j < x.size(); ++j) {
        if (j >= lb.size() || j >= ub.size() || !std::isfinite(lb[j]) || !std::isfinite(ub[j]))
            return 1e3 * (1.0 + l2norm(x));
        const double w = ub[j] - lb[j];
        s += w * w;
    }
    return std::sqrt(s);
}

static inline bool insideBox(const Vec& x, const Problem* prob)
{
    const Vec& lb = prob->lb();
    const Vec& ub = prob->ub();
    for (size_t j = 0; j < x.size(); ++j) {
        if (j < lb.size() && x[j] < lb[j]) return false;
        if (j < ub.size() && x[j] > ub[j]) return false;
    }
    return true;
}

// Largest alpha >= 0 with x + alpha*d inside the box (infinity if no bound
// is ever reached along d).
static double maxFeasibleStep(const Vec& x, const Vec& d, const Problem* prob)
{
    const Vec& lb = prob->lb();
    const Vec& ub = prob->ub();
    double amax = std::numeric_limits<double>::infinity();
    for (size_t j = 0; j < x.size(); ++j) {
        if (d[j] > 0.0 && j < ub.size() && std::isfinite(ub[j]))
            amax = std::min(amax, std::max(0.0, (ub[j] - x[j]) / d[j]));
        else if (d[j] < 0.0 && j < lb.size() && std::isfinite(lb[j]))
            amax = std::min(amax, std::max(0.0, (lb[j] - x[j]) / d[j]));
    }
    return amax;
}

// Strong-Wolfe line search (Nocedal & Wright Alg. 3.5 + zoom Alg. 3.6) on
// phi(a) = f(x + a d), restricted to 0 < a <= amax so that every trial point
// is feasible (no projection needed, the Wolfe theory applies unchanged):
//   f(x+ad) <= f0 + c1 a g'd      and      |g(x+ad)'d| <= c2 |g'd|.
// If the bracket is hit at a = amax with sufficient decrease, that point is
// accepted (the next iterate then lies on a bound). On failure, or when the
// evaluation cap is reached, the best sufficient-decrease point found is
// returned; alpha = 0 means no acceptable point at all.
static LineSearchResult strongWolfe(Problem* prob, const Vec& x, double f0, const Vec& g0, const Vec& d,
                                    double a_init, double amax, double c1, double c2, int max_evals_ls,
                                    long long maxEvalsCap)
{
    const int D = (int)x.size();
    const double dphi0 = dot(g0, d);
    LineSearchResult fail{ x, f0, g0, 0.0 };
    if (!(dphi0 < 0.0) || !(amax > 0.0)) return fail;

    // Bracket width below which trial points are indistinguishable in x.
    const double x_res = 1e-14 * std::max(1.0, l2norm(x)) / std::max(l2norm(d), 1e-300);

    int used = 0;
    auto capHit = [&]() {
        return used >= max_evals_ls || (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap);
    };
    auto point = [&](double a) {
        Vec xa(D);
        for (int j = 0; j < D; ++j) xa[j] = x[j] + a * d[j];
        project_to_bounds(xa, prob);           // removes round-off only (a <= amax)
        return xa;
    };

    // Best sufficient-decrease point seen so far (returned on failure).
    LineSearchResult best = fail;
    auto remember = [&](double a, const Vec& xa, double fa, const Vec& ga) {
        if (fa < best.f_new) best = { xa, fa, ga, a };
    };

    // Safeguarded interpolation inside [lo, hi] (either order): cubic when
    // both slopes are known, quadratic otherwise, bisection as last resort.
    auto interpolate = [&](double alo, double flo, double dlo, double ahi, double fhi, double dhi, bool haveDhi) {
        const double a1 = std::min(alo, ahi), a2 = std::max(alo, ahi);
        const double w  = a2 - a1;
        double a = std::numeric_limits<double>::quiet_NaN();
        if (haveDhi) {
            const double d1 = dlo + dhi - 3.0 * (flo - fhi) / (alo - ahi);
            const double rad = d1 * d1 - dlo * dhi;
            if (rad >= 0.0) {
                const double d2 = std::copysign(std::sqrt(rad), ahi - alo);
                const double den = dhi - dlo + 2.0 * d2;
                if (den != 0.0) a = ahi - (ahi - alo) * (dhi + d2 - d1) / den;
            }
        }
        if (!std::isfinite(a)) {
            const double h = ahi - alo;
            const double den = 2.0 * (fhi - flo - dlo * h);
            if (den > 0.0) a = alo - dlo * h * h / den;
        }
        if (!std::isfinite(a) || a < a1 + 0.1 * w || a > a2 - 0.1 * w) a = 0.5 * (a1 + a2);
        return a;
    };

    auto zoom = [&](double alo, double flo, double dlo,
                    double ahi, double fhi, double dhi, bool haveDhi) -> LineSearchResult {
        for (int it = 0; it < 30; ++it) {
            if (capHit() || std::fabs(ahi - alo) <= x_res) break;
            const double a = interpolate(alo, flo, dlo, ahi, fhi, dhi, haveDhi);
            Vec xa = point(a);
            const double fa = prob->evaluate(xa); ++used;
            if (!std::isfinite(fa) || fa > f0 + c1 * a * dphi0 || fa >= flo) {
                ahi = a; fhi = std::isfinite(fa) ? fa : std::numeric_limits<double>::max(); haveDhi = false;
                continue;
            }
            Vec ga = prob->gradient(xa);
            const double dpa = dot(ga, d);
            remember(a, xa, fa, ga);
            if (std::fabs(dpa) <= -c2 * dphi0) return { std::move(xa), fa, std::move(ga), a };
            if (dpa * (ahi - alo) >= 0.0) { ahi = alo; fhi = flo; dhi = dlo; haveDhi = true; }
            alo = a; flo = fa; dlo = dpa;
        }
        return best;
    };

    double a_prev = 0.0, f_prev = f0, dphi_prev = dphi0;
    double a = std::min(a_init, amax);
    if (!(a > 0.0)) a = amax;

    for (int i = 0; i < 30; ++i) {
        if (capHit()) break;
        Vec xa = point(a);
        const double fa = prob->evaluate(xa); ++used;
        if (!std::isfinite(fa) || fa > f0 + c1 * a * dphi0 || (i > 0 && fa >= f_prev)) {
            const double fh = std::isfinite(fa) ? fa : std::numeric_limits<double>::max();
            return zoom(a_prev, f_prev, dphi_prev, a, fh, 0.0, false);
        }
        Vec ga = prob->gradient(xa);
        const double dpa = dot(ga, d);
        remember(a, xa, fa, ga);
        if (std::fabs(dpa) <= -c2 * dphi0) return { std::move(xa), fa, std::move(ga), a };
        if (dpa >= 0.0)
            return zoom(a, fa, dpa, a_prev, f_prev, dphi_prev, true);
        if (a >= amax) return best;            // still descending at the bound: stop on it
        a_prev = a; f_prev = fa; dphi_prev = dpa;
        a = std::min(2.0 * a, amax);           // extrapolate
    }
    return best;
}

// ===== Nonlinear Conjugate Gradient =====
//
// Directions d_{k+1} = -P(g_{k+1}) + beta_k d_k with the beta rule chosen by
// the "beta" key:
//   fr  : Fletcher-Reeves     |g+|^2 / |g|^2
//   pr  : Polak-Ribiere       g+'y / |g|^2                      (y = g+ - g)
//   pr+ : max(0, PR)  -- default, Gilbert & Nocedal (1992)
//   hs  : Hestenes-Stiefel    g+'y / d'y
//   dy  : Dai-Yuan            |g+|^2 / d'y
//   hz  : Hager-Zhang (2005)  (y - 2 d |y|^2/d'y)' g+ / d'y, bounded below by
//         -1 / (|d| min(0.01, |g|))
// Line search: strong Wolfe (c1, c2 = 0.1 as recommended for CG by Nocedal &
// Wright) restricted to the feasible segment of the direction, falling back
// to the framework's projected Armijo backtracking when the direction hits a
// bound at once or Wolfe fails. Since an inexact search does not guarantee
// that the next CG direction is a descent direction, every direction is
// checked and replaced by -P(g) when it is not. Components that would leave
// the box at a coordinate already on a bound are dropped before the search.
// Additional restarts (d = -P(g)): every restart_every iterations (0 = n),
// Powell's test |g+'g| >= powell_nu |g+|^2 (powell_nu = 0 disables), a change
// of the set of blocked bound coordinates, a step taken by the projected-
// Armijo fallback, and a non-finite or numerically degenerate beta. If the line search fails along a CG direction
// it is retried once along -P(g) before the method stops. It also stops after
// stall_iters consecutive iterations whose decrease of f is below
// ftol * max(1,|f|) (finite-difference gradients rarely reach |P(g)| <= tol).
std::pair<Vec, double> localCG(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0, const std::string& section,
                                int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);
    const int    MAX_ITERS     = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    double       GRAD_TOL      = getDouble(kv, "tol",           1e-8);
    double       alpha0        = getDouble(kv, "alpha0",        1.0);
    double       c1            = getDouble(kv, "c1",            1e-4);
    double       c2            = getDouble(kv, "c2",            0.1);
    int          ls_evals      = getInt   (kv, "ls_max_evals",  40);
    int          backtracks    = getInt   (kv, "backtracks",    30);
    double       powell_nu     = getDouble(kv, "powell_nu",     0.2);
    double       ftol          = getDouble(kv, "ftol",          1e-15);
    int          stall_iters   = getInt   (kv, "stall_iters",   3);
    int          restart_every = getInt   (kv, "restart_every", 0);
    std::string  rule          = getStr   (kv, "beta",          "pr+");
    if (rule != "fr" && rule != "pr" && rule != "pr+" && rule != "hs" && rule != "dy" && rule != "hz") {
        warnBadParam(section, "beta", rule, "pr+");
        rule = "pr+";
    }
    // Strong Wolfe needs 0 < c1 < c2 < 1 (Nocedal & Wright Lemma 3.1).
    checkParam(section, "c1", c1, 1e-4, [](double v) { return v > 0.0 && v < 1.0; });
    checkParam(section, "c2", c2, (c1 < 0.1) ? 0.1 : 0.5 * (1.0 + c1),   // always in (c1, 1)
               [c1](double v) { return v > c1 && v < 1.0; });
    checkParam(section, "tol",          GRAD_TOL,    1e-8,  [](double v) { return v >= 0.0; });
    checkParam(section, "alpha0",       alpha0,      1.0,   [](double v) { return v > 0.0 && std::isfinite(v); });
    checkParam(section, "ls_max_evals", ls_evals,    40,    [](int v)    { return v >= 1; });
    checkParam(section, "backtracks",   backtracks,  30,    [](int v)    { return v >= 1; });
    checkParam(section, "powell_nu",    powell_nu,   0.2,   [](double v) { return v >= 0.0; });
    checkParam(section, "ftol",         ftol,        1e-15, [](double v) { return v >= 0.0; });
    checkParam(section, "stall_iters",  stall_iters, 3,     [](int v)    { return v >= 0; });

    Vec x = x0;
    project_to_bounds(x, prob);
    const int D = (int)x.size();
    if (restart_every <= 0) restart_every = std::max(1, D);
    const double box_diag = boxDiagonal(prob, x);

    double f  = prob->evaluate(x);
    Vec    g  = prob->gradient(x);
    Vec    pg = projectedGradient(x, g, prob);
    auto   mask = blockedMask(x, g, prob);

    Vec d(D);
    for (int j = 0; j < D; ++j) d[j] = -pg[j];

    double alpha_prev = 0.0;   // accepted step of the previous iteration
    double dg_prev    = 0.0;   // directional derivative used in that iteration
    int    since_restart = 0;
    int    stalled = 0;        // consecutive iterations with a negligible decrease

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        const double pgnorm = l2norm(pg);
        if (pgnorm <= GRAD_TOL) break;

        // Blocked coordinates never move; no component may leave the box at
        // a coordinate that is already on a bound.
        for (int j = 0; j < D; ++j) if (mask[j]) d[j] = 0.0;
        dropOutwardAtBounds(d, x, prob);

        double dg = dot(pg, d);
        bool steepest = false;
        if (!(dg < 0.0) || !allFinite(d)) {        // not a descent direction: restart
            for (int j = 0; j < D; ++j) d[j] = -pg[j];
            dg = -pgnorm * pgnorm;
            steepest = true;
            since_restart = 0;
        }

        // Initial trial step. First iteration: |step| <= alpha0 (same rule as
        // gd/lbfgs). Later: Nocedal & Wright eq. (3.60), alpha_prev * dg_prev / dg,
        // i.e. keep the first-order change of f the same as last time.
        double a_try = alpha0 / std::max(1.0, l2norm(d));
        if (alpha_prev > 0.0 && dg_prev < 0.0) {
            const double guess = alpha_prev * dg_prev / dg;
            if (std::isfinite(guess) && guess > 0.0) a_try = guess;
        }
        // Never try a first step longer than the box diagonal.
        const double dnorm = l2norm(d);
        if (dnorm > 0.0 && a_try * dnorm > box_diag) a_try = box_diag / dnorm;

        // Strong-Wolfe search inside the box; projected Armijo if the direction
        // immediately hits a bound or Wolfe finds no acceptable point.
        bool usedProjected = false;   // step taken by projected Armijo (not along d)
        auto search = [&](const Vec& dir, double a0) {
            const double amax = maxFeasibleStep(x, dir, prob);
            LineSearchResult r{ x, f, g, 0.0 };
            // Skip Wolfe only if the longest feasible MOVE is negligible. The
            // test is in x-space: comparing amax itself with a constant is
            // not scale-invariant (|g| ~ 1e20 makes every useful alpha ~ 1e-20).
            if (amax * l2norm(dir) > 1e-12 * std::max(1.0, l2norm(x)))
                r = strongWolfe(prob, x, f, g, dir, a0, amax, c1, c2, ls_evals, maxEvalsCap);
            if (r.alpha <= 0.0) {
                r = backtrackingArmijo(prob, x, g, dir, a0, c1, backtracks, f, maxEvalsCap);
                usedProjected = (r.alpha > 0.0);
            }
            return r;
        };

        auto ls = search(d, a_try);
        if (ls.alpha <= 0.0 && !steepest) {
            // CG direction failed: one retry along the projected steepest descent.
            for (int j = 0; j < D; ++j) d[j] = -pg[j];
            dg = -pgnorm * pgnorm;
            steepest = true;
            since_restart = 0;
            a_try = alpha0 / std::max(1.0, pgnorm);
            ls = search(d, a_try);
        }
        if (ls.alpha <= 0.0) break;               // no progress possible

        Vec  pgn     = projectedGradient(ls.x_new, ls.g_new, prob);
        auto mask_n  = blockedMask(ls.x_new, ls.g_new, prob);
        const bool activeChanged = (mask_n != mask);

        // ---- beta ----
        // A projected-Armijo step is not along d, so the conjugacy relations
        // (d'y in HS/DY/HZ, the Powell test) no longer refer to the step that
        // was actually taken: restart.
        double beta = 0.0;
        bool   restart = activeChanged || usedProjected || (++since_restart >= restart_every);
        const double gg_old = pgnorm * pgnorm;
        const double gg_new = dot(pgn, pgn);
        if (!restart && powell_nu > 0.0 && std::fabs(dot(pgn, pg)) >= powell_nu * gg_new)
            restart = true;

        if (!restart) {
            Vec y(D);
            for (int j = 0; j < D; ++j) y[j] = pgn[j] - pg[j];
            const double dy_ = dot(d, y);
            const double tiny = 1e-300;
            if (rule == "fr") {
                beta = gg_new / gg_old;
            } else if (rule == "pr" || rule == "pr+") {
                beta = dot(pgn, y) / gg_old;
                if (rule == "pr+") beta = std::max(0.0, beta);
            } else if (rule == "hs") {
                beta = (std::fabs(dy_) > tiny) ? dot(pgn, y) / dy_ : 0.0;
            } else if (rule == "dy") {
                beta = (std::fabs(dy_) > tiny) ? gg_new / dy_ : 0.0;
            } else { // hz
                if (std::fabs(dy_) > tiny) {
                    const double yy = dot(y, y);
                    beta = (dot(y, pgn) - 2.0 * yy * dot(d, pgn) / dy_) / dy_;
                    const double dn = l2norm(d);
                    if (dn > 0.0) {
                        const double eta_k = -1.0 / (dn * std::min(0.01, pgnorm));
                        beta = std::max(beta, eta_k);
                    }
                } else {
                    beta = 0.0;
                }
            }
            if (!std::isfinite(beta)) beta = 0.0;
        }
        if (restart) since_restart = 0;

        alpha_prev = ls.alpha;
        dg_prev    = dg;

        // Stagnation: with a finite-difference gradient |P(g)| may never reach
        // tol near a minimizer, while every line search still "succeeds" with
        // a negligible decrease. Stop after stall_iters such iterations.
        const bool negligible = (f - ls.f_new) <= ftol * std::max(1.0, std::fabs(f));

        Vec dn(D);
        for (int j = 0; j < D; ++j) dn[j] = -pgn[j] + beta * d[j];

        x    = std::move(ls.x_new);
        f    = ls.f_new;
        g    = std::move(ls.g_new);
        pg   = std::move(pgn);
        mask = std::move(mask_n);
        d    = std::move(dn);

        if (negligible) { if (stall_iters > 0 && ++stalled >= stall_iters) break; }
        else            { stalled = 0; }
    }

    return { x, f };
}

// ===== Truncated Newton (line-search Newton-CG) =====
//
// Outer iteration: solve H p = -P(g) approximately with linear CG restricted
// to the free (non-blocked) coordinates, then a projected Armijo line search
// starting at the full Newton step alpha = 1 (shortened if |p| exceeds
// max_step; 0 = box diagonal).
// Inner CG (Nocedal & Wright Alg. 7.1):
//   * forcing term  |r_j| <= eta_k |P(g)|, eta_k = min(0.5, sqrt|P(g)|)
//     (superlinear convergence);
//   * on non-positive curvature d'Hd <= curv_eps |d|^2 it stops and returns
//     the current iterate (or -P(g) on the very first inner step);
//   * at most max_cg inner steps (0 = n).
// Hessian-vector products come from forward differences of the gradient with
// the relative step hv_eps * max(1, |x|_inf), switching to a backward
// difference if the forward point would leave the box. Each product costs one
// gradient call (reported in gradCalls(), not in calls()). Stops on
// |P(g)| <= tol, max_iters, the evaluation cap, a failed line search, or
// stall_iters consecutive iterations with a decrease <= ftol * max(1,|f|).
std::pair<Vec, double> localNCG(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0, const std::string& section,
                                 int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);
    const int    MAX_ITERS  = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    double       GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    double       alpha0     = getDouble(kv, "alpha0",     1.0);
    double       c1         = getDouble(kv, "c1",         1e-4);
    int          backtracks = getInt   (kv, "backtracks", 30);
    int          max_cg     = getInt   (kv, "max_cg",     20);
    double       hv_eps     = getDouble(kv, "hv_eps",     1e-5);
    double       curv_eps   = getDouble(kv, "curv_eps",   1e-12);
    double       max_step   = getDouble(kv, "max_step",   0.0);
    double       ftol       = getDouble(kv, "ftol",       1e-15);
    int          stall_iters= getInt   (kv, "stall_iters",3);
    checkParam(section, "tol",         GRAD_TOL,    1e-8,  [](double v) { return v >= 0.0; });
    checkParam(section, "alpha0",      alpha0,      1.0,   [](double v) { return v > 0.0 && std::isfinite(v); });
    checkParam(section, "c1",          c1,          1e-4,  [](double v) { return v > 0.0 && v < 1.0; });
    checkParam(section, "backtracks",  backtracks,  30,    [](int v)    { return v >= 1; });
    checkParam(section, "hv_eps",      hv_eps,      1e-5,  [](double v) { return v > 0.0 && v < 1.0; });
    checkParam(section, "curv_eps",    curv_eps,    1e-12, [](double v) { return v >= 0.0; });
    checkParam(section, "max_step",    max_step,    0.0,   [](double v) { return v >= 0.0; });
    checkParam(section, "ftol",        ftol,        1e-15, [](double v) { return v >= 0.0; });
    checkParam(section, "stall_iters", stall_iters, 3,     [](int v)    { return v >= 0; });

    Vec x = x0;
    project_to_bounds(x, prob);
    const int D = (int)x.size();
    if (max_cg <= 0 || max_cg > D) max_cg = std::max(1, D);
    if (!(max_step > 0.0)) max_step = boxDiagonal(prob, x);

    double f = prob->evaluate(x);
    Vec    g = prob->gradient(x);
    int    stalled = 0;

    // H v at the current x (gradient gx already known). Returns false if no
    // difference point inside the box could be found.
    auto hessVec = [&](const Vec& xc, const Vec& gx, const Vec& v,
                       const std::vector<char>& blocked, Vec& Hv) -> bool {
        Hv.assign(D, 0.0);
        const double nv = l2norm(v);
        if (!(nv > 0.0)) return true;
        double xinf = 1.0;
        for (double a : xc) xinf = std::max(xinf, std::fabs(a));
        double delta = hv_eps * xinf;
        for (int attempt = 0; attempt < 8; ++attempt, delta *= 0.25) {
            for (int sgn : {+1, -1}) {
                Vec xt(D);
                for (int j = 0; j < D; ++j) xt[j] = xc[j] + sgn * delta * v[j] / nv;
                if (!insideBox(xt, prob)) continue;
                Vec gt = prob->gradient(xt);
                for (int j = 0; j < D; ++j) {
                    Hv[j] = blocked[j] ? 0.0 : sgn * (gt[j] - gx[j]) * nv / delta;
                    if (!std::isfinite(Hv[j])) return false;
                }
                return true;
            }
        }
        return false;
    };

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        const Vec  pg      = projectedGradient(x, g, prob);
        const auto blocked = blockedMask(x, g, prob);
        const double pgnorm = l2norm(pg);
        if (pgnorm <= GRAD_TOL) break;

        // ---- inner CG on H p = -pg ----
        const double eta_k = std::min(0.5, std::sqrt(pgnorm));
        const double r_tol = eta_k * pgnorm;
        Vec z(D, 0.0), r = pg, dcg(D), Hd;
        for (int j = 0; j < D; ++j) dcg[j] = -r[j];
        double rr = dot(r, r);
        bool   zIsZero = true;

        for (int it = 0; it < max_cg; ++it) {
            if (!hessVec(x, g, dcg, blocked, Hd)) break;
            const double dHd = dot(dcg, Hd);
            if (dHd <= curv_eps * dot(dcg, dcg)) break;   // non-positive curvature
            const double a = rr / dHd;
            for (int j = 0; j < D; ++j) { z[j] += a * dcg[j]; r[j] += a * Hd[j]; }
            zIsZero = false;
            const double rr_new = dot(r, r);
            if (std::sqrt(rr_new) <= r_tol) break;
            const double b = rr_new / rr;
            for (int j = 0; j < D; ++j) dcg[j] = -r[j] + b * dcg[j];
            rr = rr_new;
        }

        Vec p(D);
        if (!zIsZero && allFinite(z)) dropOutwardAtBounds(z, x, prob);
        bool steepest = zIsZero || !allFinite(z) || !(dot(z, pg) < 0.0);
        if (steepest) { for (int j = 0; j < D; ++j) p[j] = -pg[j]; }
        else          { p = z; }

        // Newton step: natural trial alpha = 1, shortened if |p| > max_step.
        // Steepest fallback: |first step| <= alpha0, as in gd.
        const double pn = l2norm(p);
        double a_try = steepest ? alpha0 / std::max(1.0, pn)
                                : ((pn > max_step) ? max_step / pn : 1.0);

        auto ls = backtrackingArmijo(prob, x, g, p, a_try, c1, backtracks, f, maxEvalsCap);
        if (ls.alpha <= 0.0 && !steepest) {
            for (int j = 0; j < D; ++j) p[j] = -pg[j];
            a_try = alpha0 / std::max(1.0, pgnorm);
            ls = backtrackingArmijo(prob, x, g, p, a_try, c1, backtracks, f, maxEvalsCap);
        }
        if (ls.alpha <= 0.0) break;

        // Stagnation stop (see localCG): negligible decrease for stall_iters
        // consecutive Newton iterations.
        const bool negligible = (f - ls.f_new) <= ftol * std::max(1.0, std::fabs(f));

        x = std::move(ls.x_new);
        f = ls.f_new;
        g = std::move(ls.g_new);

        if (negligible) { if (stall_iters > 0 && ++stalled >= stall_iters) break; }
        else            { stalled = 0; }
    }

    return { x, f };
}

// ===== Adam / AMSGrad =====
//
//   m_t = b1 m + (1-b1) g,  v_t = b2 v + (1-b2) g^2
//   x  -= lr_t * mhat / (sqrt(vhat) + eps),  mhat = m/(1-b1^t), vhat = v/(1-b2^t)
// with g the box-projected gradient and every iterate projected onto the box.
// amsgrad = 1 uses vhat = max_{s<=t} v_s / (1-b2^t) (Reddi et al. 2018).
// Step size: lr, or (if lr_rel > 0) lr_rel * (ub_j - lb_j) per coordinate,
// which makes the step independent of the problem's scale. lr_decay:
// none | invsqrt (lr/sqrt(t)) | cosine (0.5 lr (1+cos(pi (t-1)/T))).
// Stops on |P(g)| <= tol, max_iters, the evaluation cap, a step that the
// projection removes completely, or (patience > 0) no improvement of the best
// f for `patience` consecutive iterations. Returns the best point visited.
std::pair<Vec, double> localAdam(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0, const std::string& section,
                                  int maxItersOverride, long long maxEvalsCap)
{
    auto kv = loadKV(section);
    const int    MAX_ITERS = (maxItersOverride > 0) ? maxItersOverride : getInt(kv, "max_iters", 500);
    double       GRAD_TOL  = getDouble(kv, "tol",      1e-8);
    double       lr        = getDouble(kv, "lr",       1e-3);
    double       lr_rel    = getDouble(kv, "lr_rel",   0.0);
    double       b1        = getDouble(kv, "beta1",    0.9);
    double       b2        = getDouble(kv, "beta2",    0.999);
    double       eps       = getDouble(kv, "eps",      1e-8);
    const bool   amsgrad   = getInt   (kv, "amsgrad",  0) != 0;
    int          patience  = getInt   (kv, "patience", 0);
    std::string  decay     = getStr   (kv, "lr_decay", "none");
    if (decay != "none" && decay != "invsqrt" && decay != "cosine") {
        warnBadParam(section, "lr_decay", decay, "none");
        decay = "none";
    }
    // beta1, beta2 must be < 1, otherwise the bias corrections 1 - beta^t are 0.
    checkParam(section, "tol",      GRAD_TOL, 1e-8,  [](double v) { return v >= 0.0; });
    checkParam(section, "lr",       lr,       1e-3,  [](double v) { return v > 0.0 && std::isfinite(v); });
    checkParam(section, "lr_rel",   lr_rel,   0.0,   [](double v) { return v >= 0.0 && std::isfinite(v); });
    checkParam(section, "beta1",    b1,       0.9,   [](double v) { return v >= 0.0 && v < 1.0; });
    checkParam(section, "beta2",    b2,       0.999, [](double v) { return v >= 0.0 && v < 1.0; });
    checkParam(section, "eps",      eps,      1e-8,  [](double v) { return v > 0.0; });
    checkParam(section, "patience", patience, 0,     [](int v)    { return v >= 0; });

    Vec x = x0;
    project_to_bounds(x, prob);
    const int D = (int)x.size();

    // Per-coordinate step size.
    Vec lrv(D, lr);
    if (lr_rel > 0.0) {
        const Vec& lb = prob->lb();
        const Vec& ub = prob->ub();
        for (int j = 0; j < D; ++j) {
            if (j < (int)lb.size() && j < (int)ub.size() &&
                std::isfinite(lb[j]) && std::isfinite(ub[j]) && ub[j] > lb[j])
                lrv[j] = lr_rel * (ub[j] - lb[j]);
        }
    }

    double f = prob->evaluate(x);
    Vec    g = prob->gradient(x);

    Vec    best_x = x;
    double best_f = std::isfinite(f) ? f : std::numeric_limits<double>::infinity();

    Vec m(D, 0.0), v(D, 0.0), vmax(D, 0.0);
    double b1t = 1.0, b2t = 1.0;
    int    since_improve = 0;
    const double PI = 3.14159265358979323846;

    for (int t = 1; t <= MAX_ITERS; ++t) {
        if (maxEvalsCap > 0 && (long long)prob->calls() >= maxEvalsCap) break;
        const Vec pg = projectedGradient(x, g, prob);   // also maps non-finite components to 0
        if (l2norm(pg) <= GRAD_TOL) break;

        b1t *= b1;
        b2t *= b2;
        double sched = 1.0;
        if (decay == "invsqrt")     sched = 1.0 / std::sqrt((double)t);
        else if (decay == "cosine") sched = 0.5 * (1.0 + std::cos(PI * (double)(t - 1) / (double)MAX_ITERS));

        const Vec xold = x;
        for (int j = 0; j < D; ++j) {
            m[j] = b1 * m[j] + (1.0 - b1) * pg[j];
            v[j] = b2 * v[j] + (1.0 - b2) * pg[j] * pg[j];
            double vj = v[j];
            if (amsgrad) { vmax[j] = std::max(vmax[j], v[j]); vj = vmax[j]; }
            const double mhat = m[j] / (1.0 - b1t);
            const double vhat = vj   / (1.0 - b2t);
            x[j] -= sched * lrv[j] * mhat / (std::sqrt(vhat) + eps);
        }
        project_to_bounds(x, prob);
        if (x == xold) break;                     // projection removed the whole step

        f = prob->evaluate(x);
        g = prob->gradient(x);
        if (std::isfinite(f) && f < best_f) {
            best_f = f;
            best_x = x;
            since_improve = 0;
        } else if (patience > 0 && ++since_improve >= patience) {
            break;
        }
    }

    if (!std::isfinite(best_f)) return { x, f };   // nothing finite was ever seen
    return { best_x, best_f };
}

// ===== runLocalSearch (dispatcher) =====
std::pair<Vec, double> runLocalSearch(const std::string& method,
                                      Problem*           prob,
                                      std::mt19937_64&   rng,
                                      const Vec&         x0,
                                      int                maxItersOverride,
                                      long long          maxEvalsCap)
{
    std::string m = method;
    for (char& c : m) c = (char)std::tolower((unsigned char)c);

    if (m == "gd")    return localGD   (prob, rng, x0, "gd",    maxItersOverride, maxEvalsCap);
    if (m == "lbfgs") return localLBFGS(prob, rng, x0, "lbfgs", maxItersOverride, maxEvalsCap);
    if (m == "bfgs")  return localBFGS (prob, rng, x0, "bfgs",  maxItersOverride, maxEvalsCap);
    if (m == "nm")    return localNM   (prob, rng, x0, "nm",    maxItersOverride, maxEvalsCap);
    if (m == "cg")    return localCG   (prob, rng, x0, "cg",    maxItersOverride, maxEvalsCap);
    if (m == "ncg")   return localNCG  (prob, rng, x0, "ncg",   maxItersOverride, maxEvalsCap);
    if (m == "adam")  return localAdam (prob, rng, x0, "adam",  maxItersOverride, maxEvalsCap);

    // Unknown method: fallback, no-op local search
    return { x0, prob->evaluate(x0) };
}

} // namespace optimsolution
