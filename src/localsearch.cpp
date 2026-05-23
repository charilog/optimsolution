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

namespace optimsolution {

// ===== Config helpers =====
static std::unordered_map<std::string, std::string>
loadKV(const std::string& method_section)
{
    auto cfg = Config::load("optimsolution.cfg", method_section);
    return cfg.methodKV.kv;
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
std::pair<Vec, double> localGD(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0)
{
    auto kv = loadKV("gd");

    const int    MAX_ITERS  = getInt   (kv, "max_iters",  500);
    const double GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    const double alpha0     = getDouble(kv, "alpha0",     1.0);
    const double c1         = getDouble(kv, "c1",         1e-4);
    const int    backtracks = getInt   (kv, "backtracks", 20);

    Vec    x = x0;
    double f = prob->evaluate(x);
    // Gradient computed once here and then forwarded from ls.g_new each step,
    // avoiding the double evaluation (backtrackingArmijo already computes it).
    Vec    g = prob->gradient(x);

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (l2norm(g) <= GRAD_TOL) break;

        Vec d(x.size());
        for (size_t j = 0; j < x.size(); ++j) d[j] = -g[j];

        auto ls = backtrackingArmijo(prob, x, g, d, alpha0, c1, backtracks);
        if (ls.x_new == x) break;  // line search made no progress

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

std::pair<Vec, double> localLBFGS(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0)
{
    auto kv = loadKV("lbfgs");

    const int    MAX_ITERS  = getInt   (kv, "max_iters",  500);
    const double GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    const int    m_history  = getInt   (kv, "m_history",  10);
    const double alpha0     = getDouble(kv, "alpha0",     1.0);
    const double c1         = getDouble(kv, "c1",         1e-4);
    const int    backtracks = getInt   (kv, "backtracks", 20);

    Vec    x = x0;
    double f = prob->evaluate(x);
    Vec    g = prob->gradient(x);

    // FIX #7: deque gives O(1) pop_front instead of O(m) erase(begin())
    std::deque<Vec>    s_hist, y_hist;
    std::deque<double> rho_hist;

    for (int k = 0; k < MAX_ITERS; ++k) {
        if (l2norm(g) <= GRAD_TOL) break;

        Vec d = s_hist.empty()
            ? [&]{ Vec neg(x.size()); for (size_t j=0;j<x.size();++j) neg[j]=-g[j]; return neg; }()
            : lbfgs_direction(g, s_hist, y_hist, rho_hist);

        auto ls = backtrackingArmijo(prob, x, g, d, alpha0, c1, backtracks);
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
std::pair<Vec, double> localBFGS(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0)
{
    auto kv = loadKV("bfgs");

    const int    MAX_ITERS  = getInt   (kv, "max_iters",  500);
    const double GRAD_TOL   = getDouble(kv, "tol",        1e-8);
    const double alpha0     = getDouble(kv, "alpha0",     1.0);
    const double c1         = getDouble(kv, "c1",         1e-4);
    const int    backtracks = getInt   (kv, "backtracks", 20);

    Vec x = x0;
    const int D = (int)x.size();

    // H starts as the identity (approximate inverse Hessian)
    std::vector<double> H(D * D, 0.0);
    for (int i = 0; i < D; ++i) H[i * D + i] = 1.0;

    double f = prob->evaluate(x);
    Vec    g = prob->gradient(x);

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
        if (l2norm(g) <= GRAD_TOL) break;

        Vec d = H_mul(g);
        for (int j = 0; j < D; ++j) d[j] = -d[j];

        // Guard: if d is not a descent direction (numerical drift in H),
        // reset H to identity and fall back to steepest-descent step.
        if (dot(d, g) >= 0.0) {
            std::fill(H.begin(), H.end(), 0.0);
            for (int i = 0; i < D; ++i) H[i * D + i] = 1.0;
            for (int j = 0; j < D; ++j) d[j] = -g[j];
        }

        auto ls = backtrackingArmijo(prob, x, g, d, alpha0, c1, backtracks);
        if (ls.x_new == x) break;

        Vec s(D), y(D);
        for (int j = 0; j < D; ++j) {
            s[j] = ls.x_new[j] - x[j];
            y[j] = ls.g_new[j] - g[j];
        }

        double ys = dot(y, s);
        if (ys > 1e-12) {
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

std::pair<Vec, double> localNM(Problem* prob, std::mt19937_64& rng, const Vec& x0)
{
    auto kv = loadKV("nm");

    const int    MAX_ITERS    = getInt   (kv, "max_iters",    500);
    const double simplex_step = getDouble(kv, "simplex_delta", 1e-2);
    // FIX #6: termination tolerance from config (was hardcoded 1e-6)
    const double SIMPLEX_TOL  = getDouble(kv, "tol",           1e-6);

    double alpha_nm = getDouble(kv, "alpha", 1.0);   // reflection
    double gamma_nm = getDouble(kv, "gamma", 2.0);   // expansion
    // FIX #5: config keys corrected from "beta"/"delta" to "rho"/"sigma"
    double rho_nm   = getDouble(kv, "rho",   0.5);   // contraction
    double sigma_nm = getDouble(kv, "sigma", 0.5);   // shrink

    const int D = (int)x0.size();

    std::vector<Vec> S(D + 1, x0);
    std::uniform_real_distribution<double> U(-simplex_step, simplex_step);
    for (int i = 1; i <= D; ++i) {
        for (int j = 0; j < D; ++j)
            S[i][j] += (i == j + 1 ? simplex_step : 0.0) + U(rng);
    }

    std::vector<double> F(D + 1, std::numeric_limits<double>::infinity());
    for (int i = 0; i <= D; ++i) F[i] = prob->evaluate(S[i]);

    auto order = [&]() {
        std::vector<int> idx(D + 1);
        for (int i = 0; i <= D; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return F[a] < F[b]; });
        return idx;
    };

    for (int k = 0; k < MAX_ITERS; ++k) {
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
        double fr = prob->evaluate(xr);

        if (fr < F[lo]) {
            // Expansion
            Vec xe(D);
            for (int j = 0; j < D; ++j)
                xe[j] = xc[j] + gamma_nm * (xr[j] - xc[j]);
            double fe = prob->evaluate(xe);
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

            double fc = prob->evaluate(xcand);
            if (fc < (outside ? fr : F[hi])) {
                S[hi] = std::move(xcand); F[hi] = fc;
            } else {
                // Shrink
                for (int i = 1; i <= D; ++i) {
                    for (int j = 0; j < D; ++j)
                        S[idx[i]][j] = S[lo][j] + sigma_nm * (S[idx[i]][j] - S[lo][j]);
                    F[idx[i]] = prob->evaluate(S[idx[i]]);
                }
            }
        }
    }

    int best = 0;
    for (int i = 1; i <= D; ++i) if (F[i] < F[best]) best = i;
    return { S[best], F[best] };
}

// ===== runLocalSearch (dispatcher) =====
std::pair<Vec, double> runLocalSearch(const std::string& method,
                                      Problem*           prob,
                                      std::mt19937_64&   rng,
                                      const Vec&         x0)
{
    std::string m = method;
    for (char& c : m) c = (char)std::tolower((unsigned char)c);

    if (m == "gd")    return localGD   (prob, rng, x0);
    if (m == "lbfgs") return localLBFGS(prob, rng, x0);
    if (m == "bfgs")  return localBFGS (prob, rng, x0);
    if (m == "nm")    return localNM   (prob, rng, x0);

    // Unknown method: fallback, no-op local search
    return { x0, prob->evaluate(x0) };
}

} // namespace optimsolution
