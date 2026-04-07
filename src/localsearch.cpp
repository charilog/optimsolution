#include "optimizer.h"
#include "problem.h"
#include "config.h"

#include <vector>
#include <random>
#include <utility>
#include <limits>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <sstream>

namespace optimsolution {

// ===== Helpers =====
double dot(const Vec& a, const Vec& b){
    double s=0.0; for (size_t i=0;i<a.size();++i) s+=a[i]*b[i]; return s;
}
static inline double norm2(const Vec& g){ return std::sqrt(dot(g,g)); }

static std::unordered_map<std::string,std::string>
loadKV(const std::string& method_section)
{
    auto cfg = Config::load("optimsolution.cfg", method_section);
    return cfg.methodKV.kv;
}

static double getDouble(const std::unordered_map<std::string,std::string>& kv,
                        const std::string& key, double defv)
{
    auto it = kv.find(key);
    if (it==kv.end()) return defv;
    try { return std::stod(it->second); }
    catch(...) { return defv; }
}

static int getInt(const std::unordered_map<std::string,std::string>& kv,
                  const std::string& key, int defv)
{
    auto it = kv.find(key);
    if (it==kv.end()) return defv;
    try { return std::stoi(it->second); }
    catch(...) { return defv; }
}

// ===== Gradient Descent (with Armijo) =====
std::pair<Vec,double> localGD(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0)
{
    auto kv = loadKV("gd");

    const int    MAX_ITERS = getInt(kv, "max_iters", 500);
    const double GRAD_TOL  = getDouble(kv, "tol", 1e-8);

    double alpha0     = getDouble(kv, "alpha0",     1.0);
    double c1         = getDouble(kv, "c1",         1e-4);
    int    backtracks = getInt   (kv, "backtracks", 20);

    Vec x = x0;
    const int D = (int)x.size();

    double f = prob->evaluate(x);

    for (int k=0; k<MAX_ITERS; ++k){
        Vec g = prob->gradient(x);
        if (norm2(g) <= GRAD_TOL) break;

        Vec p(D,0.0); for (int j=0;j<D;++j) p[j] = -g[j];

        double alpha = alpha0;
        const double f0 = f;
        const double dg = dot(g, p);
        bool accepted = false;

        for (int bt=0; bt<backtracks; ++bt){
            Vec xn = x;
            for (int j=0;j<D;++j) xn[j] += alpha*p[j];
            double fn = prob->evaluate(xn);
            if (fn <= f0 + c1*alpha*dg){
                x = std::move(xn);
                f = fn;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) break;
    }

    return {x, f};
}

// ===== L-BFGS =====
static Vec lbfgs_direction(const Vec& grad,
                           const std::vector<Vec>& s_hist,
                           const std::vector<Vec>& y_hist,
                           const std::vector<double>& rho_hist)
{
    int m = (int)s_hist.size();
    std::vector<double> alpha(m, 0.0);
    Vec q = grad;

    for (int i = m-1; i >= 0; --i){
        alpha[i] = rho_hist[i] * dot(s_hist[i], q);
        for (size_t j=0;j<q.size();++j) q[j] -= alpha[i]*y_hist[i][j];
    }
    double scale = 1.0;
    if (m>0){
        double sy = dot(s_hist.back(), y_hist.back());
        double yy = dot(y_hist.back(), y_hist.back());
        if (yy > 0.0) scale = sy/yy;
    }
    for (size_t j=0;j<q.size();++j) q[j] *= scale;

    for (int i = 0; i < m; ++i){
        double beta = rho_hist[i] * dot(y_hist[i], q);
        for (size_t j=0;j<q.size();++j) q[j] += (alpha[i]-beta) * s_hist[i][j];
    }

    for (size_t j=0;j<q.size();++j) q[j] = -q[j];
    return q;
}

std::pair<Vec,double> localLBFGS(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0)
{
    auto kv = loadKV("lbfgs");

    const int    MAX_ITERS = getInt(kv, "max_iters", 500);
    const double GRAD_TOL  = getDouble(kv, "tol", 1e-8);
    const int    m_history = getInt   (kv, "m_history", 10);
    double alpha0          = getDouble(kv, "alpha0",     1.0);
    double c1              = getDouble(kv, "c1",         1e-4);
    int    backtracks      = getInt   (kv, "backtracks", 20);

    Vec x = x0;
    const int D = (int)x.size();

    double f = prob->evaluate(x);
    Vec g = prob->gradient(x);

    std::vector<Vec> s_hist, y_hist;
    std::vector<double> rho_hist;

    for (int k=0; k<MAX_ITERS; ++k){
        if (norm2(g) <= GRAD_TOL) break;

        Vec p;
        if (s_hist.empty()){
            p = Vec(D,0.0); for (int j=0;j<D;++j) p[j] = -g[j];
        } else {
            p = lbfgs_direction(g, s_hist, y_hist, rho_hist);
        }

        double alpha = alpha0;
        const double f0 = f;
        const double dg = dot(g, p);
        bool accepted = false;

        for (int bt=0; bt<backtracks; ++bt){
            Vec xn = x;
            for (int j=0;j<D;++j) xn[j] += alpha*p[j];
            double fn = prob->evaluate(xn);
            if (fn <= f0 + c1*alpha*dg){
                Vec s(D,0.0); for (int j=0;j<D;++j) s[j] = xn[j] - x[j];
                Vec g_new = prob->gradient(xn);
                Vec y(D,0.0); for (int j=0;j<D;++j) y[j] = g_new[j] - g[j];

                double ys = dot(y, s);
                if (ys > 1e-12) {
                    double rho = 1.0 / ys;
                    if ((int)s_hist.size() == m_history){
                        s_hist.erase(s_hist.begin());
                        y_hist.erase(y_hist.begin());
                        rho_hist.erase(rho_hist.begin());
                    }
                    s_hist.push_back(std::move(s));
                    y_hist.push_back(std::move(y));
                    rho_hist.push_back(rho);
                }

                x = std::move(xn);
                f = fn;
                g = std::move(g_new);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) break;
    }

    return {x, f};
}

// ===== BFGS (full) =====
std::pair<Vec,double> localBFGS(Problem* prob, std::mt19937_64& /*rng*/, const Vec& x0)
{
    auto kv = loadKV("bfgs");

    const int    MAX_ITERS = getInt(kv, "max_iters", 500);
    const double GRAD_TOL  = getDouble(kv, "tol", 1e-8);

    double alpha0     = getDouble(kv, "alpha0",     1.0);
    double c1         = getDouble(kv, "c1",         1e-4);
    int    backtracks = getInt   (kv, "backtracks", 20);

    Vec x = x0;
    const int D = (int)x.size();

    std::vector<double> H(D*D, 0.0);
    for (int i=0;i<D;++i) H[i*D+i] = 1.0;

    double f = prob->evaluate(x);
    Vec g = prob->gradient(x);

    auto H_mul = [&](const Vec& v){
        Vec r(D,0.0);
        for (int i=0;i<D;++i){
            double s=0.0;
            for (int j=0;j<D;++j) s += H[i*D+j]*v[j];
            r[i]=s;
        }
        return r;
    };

    for (int k=0; k<MAX_ITERS; ++k){
        if (norm2(g) <= GRAD_TOL) break;

        Vec p = H_mul(g);
        for (int j=0;j<D;++j) p[j] = -p[j];

        double alpha = alpha0;
        const double f0 = f;
        const double dg = dot(g, p);
        bool accepted = false;

        Vec xn; xn.reserve(D);
        for (int bt=0; bt<backtracks; ++bt){
            xn = x;
            for (int j=0;j<D;++j) xn[j] += alpha*p[j];
            double fn = prob->evaluate(xn);
            if (fn <= f0 + c1*alpha*dg){
                Vec s(D,0.0); for (int j=0;j<D;++j) s[j] = xn[j] - x[j];
                Vec g_new = prob->gradient(xn);
                Vec y(D,0.0); for (int j=0;j<D;++j) y[j] = g_new[j] - g[j];

                double ys = dot(y, s);
                if (ys > 1e-12){
                    Vec Hy(D,0.0);
                    for (int i=0;i<D;++i){
                        double sum=0.0;
                        for (int j=0;j<D;++j) sum += H[i*D+j]*y[j];
                        Hy[i]=sum;
                    }
                    double yHy=0.0; for (int j=0;j<D;++j) yHy += y[j]*Hy[j];

                    for (int i=0;i<D;++i){
                        for (int j=0;j<D;++j){
                            double term1 = (1.0 + yHy/ys) * (s[i]*s[j]) / ys;
                            double term2 = (Hy[i]*s[j] + s[i]*Hy[j]) / ys;
                            H[i*D+j] = H[i*D+j] + term1 - term2;
                        }
                    }
                }

                x = std::move(xn);
                f = fn;
                g = std::move(g_new);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) break;
    }

    return {x, f};
}

// ===== Nelder–Mead =====
static inline double distLinf(const Vec& a, const Vec& b){
    double m=0.0; for (size_t i=0;i<a.size();++i) m = std::max(m, std::abs(a[i]-b[i])); return m;
}
static inline double simplex_diameter(const std::vector<Vec>& S){
    if (S.empty()) return 0.0;
    const size_t D = S[0].size();
    Vec c(D,0.0);
    for (const auto& v : S) for (size_t j=0;j<D;++j) c[j]+=v[j];
    for (size_t j=0;j<D;++j) c[j]/=(double)S.size();
    double md = 0.0;
    for (const auto& v : S) md = std::max(md, distLinf(v,c));
    return md;
}

std::pair<Vec,double> localNM(Problem* prob, std::mt19937_64& rng, const Vec& x0)
{
    auto kv = loadKV("nm");

    const int    MAX_ITERS    = getInt   (kv, "max_iters",    500);
    const double simplex_step = getDouble(kv, "simplex_delta",1e-2);

    double alpha = getDouble(kv, "alpha", 1.0);
    double gamma = getDouble(kv, "gamma", 2.0);
    double rho   = getDouble(kv, "beta",  0.5);
    double sigma = getDouble(kv, "delta", 0.5);

    const int D = (int)x0.size();

    std::vector<Vec> S(D+1, x0);
    std::uniform_real_distribution<double> U(-simplex_step, simplex_step);
    for (int i=1;i<=D;++i){
        for (int j=0;j<D;++j){
            S[i][j] += (i==j+1 ? simplex_step : 0.0) + U(rng);
        }
    }

    std::vector<double> F(D+1, std::numeric_limits<double>::infinity());
    for (int i=0;i<=D;++i) F[i] = prob->evaluate(S[i]);

    auto order = [&](){
        std::vector<int> idx(D+1);
        for (int i=0;i<=D;++i) idx[i]=i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return F[a] < F[b]; });
        return idx;
    };

    for (int k=0; k<MAX_ITERS; ++k){
        auto idx = order();
        int lo = idx[0], hi = idx[D], nh = idx[D-1];

        if (simplex_diameter(S) <= 1e-6) break;

        Vec xc(D,0.0);
        for (int t=0;t<D;++t){
            double s=0.0;
            for (int i=0;i<D;++i) s += S[idx[i]][t];
            xc[t] = s / (double)D;
        }

        Vec xr(D,0.0);
        for (int j=0;j<D;++j) xr[j] = xc[j] + alpha * (xc[j] - S[hi][j]);
        double fr = prob->evaluate(xr);

        if (fr < F[lo]){
            Vec xe(D,0.0);
            for (int j=0;j<D;++j) xe[j] = xc[j] + gamma * (xr[j] - xc[j]);
            double fe = prob->evaluate(xe);
            if (fe < fr){
                S[hi] = std::move(xe); F[hi]=fe;
            } else {
                S[hi] = std::move(xr); F[hi]=fr;
            }
        } else if (fr < F[nh]){
            S[hi] = std::move(xr); F[hi]=fr;
        } else {
            bool outside = (fr < F[hi]);
            Vec xcand(D,0.0);
            if (outside){
                for (int j=0;j<D;++j) xcand[j] = xc[j] + rho * (xr[j] - xc[j]);
            } else {
                for (int j=0;j<D;++j) xcand[j] = xc[j] + rho * (S[hi][j] - xc[j]);
            }
            double fc = prob->evaluate(xcand);
            if (fc < (outside ? fr : F[hi])){
                S[hi] = std::move(xcand); F[hi]=fc;
            } else {
                for (int i=1;i<=D;++i){
                    for (int j=0;j<D;++j){
                        S[idx[i]][j] = S[lo][j] + sigma*(S[idx[i]][j] - S[lo][j]);
                    }
                    F[idx[i]] = prob->evaluate(S[idx[i]]);
                }
            }
        }
    }

    int best = 0;
    for (int i=1;i<=D;++i) if (F[i]<F[best]) best=i;
    return { S[best], F[best] };
}

// ===== runLocalSearch (dispatch) =====
std::pair<Vec,double> runLocalSearch(const std::string& method,
                                     Problem* prob,
                                     std::mt19937_64& rng,
                                     const Vec& x0)
{
    std::string m = method;
    for (char& c : m) c = (char)std::tolower((unsigned char)c);

    if (m == "gd")      return localGD(prob, rng, x0);
    if (m == "lbfgs")   return localLBFGS(prob, rng, x0);
    if (m == "bfgs")    return localBFGS(prob, rng, x0);
    if (m == "nm")      return localNM(prob, rng, x0);

    // unknown method: απλό fallback
    Vec x = x0;
    double f = prob->evaluate(x);
    return {x, f};
}

} // namespace optimsolution
