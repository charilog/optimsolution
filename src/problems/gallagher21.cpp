#include "gallagher21.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace optimsolution {

Gallagher21::Gallagher21()
{
    setName("gallagher21");
    setFullName("Gallagher's Gaussian 21-me Peaks");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    // Will be refined in init(dim) once the instance is built.
    setKnownGlobalOptimum(0.0);
}

void Gallagher21::init(int dim)
{
    if (dim < 1)
        dim = 1;

    D_ = dim;
    Problem::init(D_);

    // Bounds [-5, 5]^D
    Vec lo(D_, -5.0), hi(D_, 5.0);
    setBounds(lo, hi);

    // Build BBOB instance
    build_instance();

    // Global optimum: for this construction, the best basin is at y_0 with f = 0.
    if (!Y_.empty() && static_cast<int>(Y_[0].size()) == D_) {
        setKnownGlobalOptimum(0.0, Y_[0]);
    } else {
        // Fallback: just keep f* = 0 without explicit x*
        setKnownGlobalOptimum(0.0);
    }
}

void Gallagher21::build_instance()
{
    std::mt19937 rng(seed_);
    std::uniform_real_distribution<double> U01(0.0, 1.0);

    // 1) Rotation matrix R (D x D) via Gram–Schmidt on N(0,1)
    make_rotation(R_);

    // 2) Centers y_i
    Y_.assign(K_, Vec(D_, 0.0));

    // y1 in [-4,4]^D (global basin)
    std::uniform_real_distribution<double> U_y1(-4.0, 4.0);
    for (int j = 0; j < D_; ++j)
        Y_[0][j] = U_y1(rng);

    // y2..yK in [-5,5]^D
    std::uniform_real_distribution<double> U_y(-5.0, 5.0);
    for (int i = 1; i < K_; ++i)
        for (int j = 0; j < D_; ++j)
            Y_[i][j] = U_y(rng);

    // 3) Weights: w1 = 10; for i = 2..21: w_i = 1.1 + 8*(i-2)/19 (range [1.1,9.1])
    w_.assign(K_, 0.0);
    w_[0] = 10.0;
    for (int i = 1; i < K_; ++i) {
        w_[i] = 1.1 + 8.0 * static_cast<double>(i - 1) / 19.0;
    }

    // 4) Alpha_i: alpha1=1000; for i>=2 sample without replacement from {1000^(2j/19), j=0..19}
    std::vector<double> alpha_pool(20);
    for (int j = 0; j < 20; ++j)
        alpha_pool[j] = std::pow(1000.0, 2.0 * static_cast<double>(j) / 19.0);
    std::shuffle(alpha_pool.begin(), alpha_pool.end(), rng);

    std::vector<double> alpha(K_, 1000.0);
    for (int i = 1; i < K_; ++i)
        alpha[i] = alpha_pool[i - 1];

    // 5) diagCi_: for each i, construct diag from powers of alpha_i, permuted, scaled by alpha^{1/4}
    diagCi_.assign(K_, Vec(D_, 1.0));
    for (int i = 0; i < K_; ++i) {
        const double a = alpha[i];

        // base diagonal: a^{0.5 * (k/(D-1))}, k=0..D-1  (if D=1 → all ones)
        Vec base(D_, 1.0);
        if (D_ > 1) {
            for (int k = 0; k < D_; ++k) {
                base[k] = std::pow(a, 0.5 * static_cast<double>(k) / static_cast<double>(D_ - 1));
            }
        }

        // random permutation of diagonal entries
        std::vector<int> perm(D_);
        for (int k = 0; k < D_; ++k)
            perm[k] = k;
        std::shuffle(perm.begin(), perm.end(), rng);

        const double scale = std::pow(a, 0.25);
        for (int k = 0; k < D_; ++k) {
            diagCi_[i][k] = base[perm[k]] / scale;
        }
    }
}

void Gallagher21::make_rotation(std::vector<double>& M)
{
    std::mt19937 rng(seed_ + 1337u);
    std::normal_distribution<double> N01(0.0, 1.0);

    M.assign(static_cast<size_t>(D_) * static_cast<size_t>(D_), 0.0);

    // Fill columns with N(0,1)
    for (int j = 0; j < D_; ++j)
        for (int i = 0; i < D_; ++i)
            M[i * D_ + j] = N01(rng);

    // Orthonormalize columns (classical Gram–Schmidt)
    for (int j = 0; j < D_; ++j) {
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

double Gallagher21::f_pen(const Vec& x)
{
    double s = 0.0;
    for (double xi : x) {
        const double a = std::abs(xi) - 5.0;
        if (a > 0.0)
            s += a * a;
    }
    return s;
}

double Gallagher21::tosz_scalar(double x)
{
    if (x == 0.0)
        return 0.0;

    const double sgn = (x > 0.0 ? 1.0 : -1.0);
    const double h   = std::log(std::abs(x));
    const double c1  = (x > 0.0 ? 10.0 : 5.5);
    const double c2  = (x > 0.0 ? 7.9  : 3.1);
    return sgn * std::exp(h + 0.049 * (std::sin(c1 * h) + std::sin(c2 * h)));
}

double Gallagher21::evaluate_core(const Vec& x)
{
    // f(x) = Tosz(10 - max_i w_i exp(-0.5/D * (R(x-y_i))^T Ci (R(x-y_i))))^2 + f_pen(x)
    double best = -1e300;
    Vec t(D_, 0.0), z(D_, 0.0);

    for (int i = 0; i < K_; ++i) {
        // t = x - y_i
        for (int d = 0; d < D_; ++d)
            t[d] = x[d] - Y_[i][d];

        // z = R * t  (R row-major)
        for (int r = 0; r < D_; ++r) {
            const double* Rrow = &R_[r * D_];
            double acc = 0.0;
            for (int c = 0; c < D_; ++c)
                acc += Rrow[c] * t[c];
            z[r] = acc;
        }

        // quad = (1/(2D)) * z^T Ci z, where Ci is diagonal (perm)
        double quad = 0.0;
        for (int d = 0; d < D_; ++d)
            quad += diagCi_[i][d] * z[d] * z[d];
        quad *= (1.0 / (2.0 * static_cast<double>(D_)));

        const double val = w_[i] * std::exp(-quad);
        if (val > best)
            best = val;
    }

    const double outer = 10.0 - best;
    const double fosz  = tosz_scalar(outer);
    const double fval  = fosz * fosz + f_pen(x); // f_opt = 0

    if (!(fval >= 0.0) || std::isnan(fval) || std::isinf(fval))
        return 1e12;

    return fval;
}

void Gallagher21::gradient_core(const Vec& x, Vec& g)
{
    // Numeric forward differences (ίδιο στιλ με τα υπόλοιπα προβλήματα)
    g.assign(D_, 0.0);
    const double f0 = evaluate_core(x);
    Vec xt = x;

    for (int k = 0; k < D_; ++k) {
        double h = std::max(1e-6, std::abs(x[k]) * 1e-6);

        // keep step within box [-5,5]
        if (x[k] + h > 5.0)
            h = std::min(h, 5.0 - x[k]);
        if (h <= 0.0) {
            g[k] = 0.0;
            continue;
        }

        xt[k] = x[k] + h;
        const double f1 = evaluate_core(xt);
        g[k] = (f1 - f0) / h;
        xt[k] = x[k];
    }
}

} // namespace optimsolution
