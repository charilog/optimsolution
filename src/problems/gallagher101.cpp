#include "gallagher101.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace optimsolution {

Gallagher101::Gallagher101()
{
    setName("gallagher101");
    setFullName("Gallagher's Gaussian 101-peaks function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Gallagher101::init(int dim)
{
    if (dim < 1)
        dim = 1;

    D_ = dim;
    Problem::init(D_);

    Vec lo(D_, -5.0), hi(D_, 5.0);
    setBounds(lo, hi);

    build_instance();

    // global best peak
    if (!Y_.empty() && (int)Y_[0].size() == D_)
        setKnownGlobalOptimum(0.0, Y_[0]);
}

void Gallagher101::build_instance()
{
    std::mt19937 rng(seed_);

    // ---------- 1) Rotation ----------
    make_rotation(R_);

    // ---------- 2) Centres Y_i ----------
    Y_.assign(K_, Vec(D_, 0.0));

    std::uniform_real_distribution<double> Uy1(-4.0,  4.0);
    std::uniform_real_distribution<double> Uy2(-5.0,  5.0);

    for (int j = 0; j < D_; ++j)
        Y_[0][j] = Uy1(rng);

    for (int i = 1; i < K_; ++i)
        for (int j = 0; j < D_; ++j)
            Y_[i][j] = Uy2(rng);

    // ---------- 3) Weights ----------
    w_.assign(K_, 0.0);
    w_[0] = 10.0;

    for (int i = 1; i < K_; ++i)
        w_[i] = 1.1 + 8.0 * double(i - 1) / 99.0;

    // ---------- 4) Alpha values ----------
    std::vector<double> alpha_pool(100);
    for (int j = 0; j < 100; ++j)
        alpha_pool[j] = std::pow(1000.0, 2.0 * double(j) / 99.0);

    std::shuffle(alpha_pool.begin(), alpha_pool.end(), rng);

    std::vector<double> alpha(K_, 1000.0);
    for (int i = 1; i < K_; ++i)
        alpha[i] = alpha_pool[i - 1];

    // ---------- 5) diag(C_i) ----------
    diagCi_.assign(K_, Vec(D_, 1.0));

    for (int i = 0; i < K_; ++i)
    {
        const double a = alpha[i];
        Vec base(D_, 1.0);

        if (D_ > 1)
            for (int k = 0; k < D_; ++k)
                base[k] = std::pow(a, 0.5 * double(k) / double(D_ - 1));

        std::vector<int> perm(D_);
        for (int k = 0; k < D_; ++k)
            perm[k] = k;

        std::shuffle(perm.begin(), perm.end(), rng);

        double scale = std::pow(a, 0.25);

        for (int k = 0; k < D_; ++k)
            diagCi_[i][k] = base[perm[k]] / scale;
    }
}

void Gallagher101::make_rotation(std::vector<double>& M)
{
    std::mt19937 rng(seed_ + 1337u);
    std::normal_distribution<double> N01(0.0, 1.0);

    M.assign((size_t)D_ * (size_t)D_, 0.0);

    for (int j = 0; j < D_; ++j)
        for (int i = 0; i < D_; ++i)
            M[i * D_ + j] = N01(rng);

    // Gram–Schmidt
    for (int j = 0; j < D_; ++j)
    {
        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < D_; ++i)
                dot += M[i * D_ + j] * M[i * D_ + k];
            for (int i = 0; i < D_; ++i)
                M[i * D_ + j] -= dot * M[i * D_ + k];
        }

        double norm = 0.0;
        for (int i = 0; i < D_; ++i)
            norm += M[i * D_ + j] * M[i * D_ + j];

        norm = std::sqrt(std::max(norm, 1e-300));

        for (int i = 0; i < D_; ++i)
            M[i * D_ + j] /= norm;
    }
}

double Gallagher101::f_pen(const Vec& x)
{
    double s = 0.0;
    for (double xi : x) {
        double a = std::abs(xi) - 5.0;
        if (a > 0.0) s += a * a;
    }
    return s;
}

double Gallagher101::tosz_scalar(double x)
{
    if (x == 0.0)
        return 0.0;

    double sgn = (x > 0.0 ? 1.0 : -1.0);
    double h   = std::log(std::abs(x));
    double c1  = (x > 0.0 ? 10.0 : 5.5);
    double c2  = (x > 0.0 ? 7.9  : 3.1);

    return sgn * std::exp(h + 0.049 * (std::sin(c1 * h) + std::sin(c2 * h)));
}

double Gallagher101::evaluate_core(const Vec& x)
{
    double best = -1e300;

    Vec t(D_, 0.0);
    Vec z(D_, 0.0);

    for (int i = 0; i < K_; ++i)
    {
        for (int d = 0; d < D_; ++d)
            t[d] = x[d] - Y_[i][d];

        for (int r = 0; r < D_; ++r) {
            double acc = 0.0;
            const double* Rrow = &R_[r * D_];
            for (int c = 0; c < D_; ++c)
                acc += Rrow[c] * t[c];
            z[r] = acc;
        }

        double quad = 0.0;
        for (int d = 0; d < D_; ++d)
            quad += diagCi_[i][d] * z[d] * z[d];

        quad *= (1.0 / (2.0 * double(D_)));

        double val = w_[i] * std::exp(-quad);
        if (val > best)
            best = val;
    }

    double outer = 10.0 - best;
    double fosz  = tosz_scalar(outer);
    double fval  = fosz * fosz + f_pen(x);

    if (!(fval >= 0.0) || std::isnan(fval) || std::isinf(fval))
        return 1e12;

    return fval;
}

void Gallagher101::gradient_core(const Vec& x, Vec& g)
{
    g.assign(D_, 0.0);

    double f0 = evaluate_core(x);
    Vec xt = x;

    for (int k = 0; k < D_; ++k)
    {
        double h = std::max(1e-6, std::abs(x[k]) * 1e-6);

        if (x[k] + h > 5.0)
            h = std::min(h, 5.0 - x[k]);

        if (h <= 0.0) {
            g[k] = 0.0;
            continue;
        }

        xt[k] = x[k] + h;
        double f1 = evaluate_core(xt);
        g[k] = (f1 - f0) / h;

        xt[k] = x[k];
    }
}

} // namespace optimsolution
