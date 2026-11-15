#include "attractivesector.h"
#include <random>
#include <cmath>
#include <algorithm>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Attractive Sector (BBOB F6) — faithful implementation.
//
// f(x) = T_osz( sum_i (s_i * z_i)^2 )^0.9
//   z     = Lambda^10 * R * x                  (x_opt = 0)
//   s_i   = 100 if z_i > 0, else 1
//   Lambda^10_ii = 10^(0.5 * i / (D-1))          (condition number 10)
//   R     = random orthogonal matrix (fixed per instance via seed_)
//
// This is the genuinely rotated, conditioned, sign-asymmetric function BBOB
// F6 documents — see attractivesector.h for why the previous version (a
// purely separable |x_i|^(2+sgn(x_i)) placeholder) was wrong.
// -----------------------------------------------------------------------------

AttractiveSector::AttractiveSector()
{
    setName("attractivesector");
    setFullName("Attractive Sector benchmark function (BBOB F6)");
    setModality("unimodal");
    setSeparability("non-separable");   // true here: R mixes all coordinates
    setCategory("BBOB synthetic benchmark");

    setKnownGlobalOptimum(0.0);
}

void AttractiveSector::init(int dim)
{
    if (dim < 1) dim = 1;
    D_ = dim;

    Problem::init(D_);

    Vec l(D_, -100.0);
    Vec u(D_,  100.0);
    setBounds(l, u);

    build_instance();

    Vec xopt(D_, 0.0);
    setKnownGlobalOptimum(0.0, xopt);
}

void AttractiveSector::build_instance()
{
    make_rotation(R_);

    diag_.assign(D_, 1.0);
    for (int i = 0; i < D_; ++i) {
        diag_[i] = (D_ > 1)
            ? std::pow(10.0, 0.5 * double(i) / double(D_ - 1))
            : 1.0;
    }
}

void AttractiveSector::make_rotation(std::vector<double>& M)
{
    // Same construction as gallagher101.cpp's make_rotation: Gaussian
    // columns orthonormalised via Gram-Schmidt, giving a uniformly-random
    // orthogonal matrix (up to numerical precision), fixed per instance.
    std::mt19937 rng(seed_ + 7919u);
    std::normal_distribution<double> N01(0.0, 1.0);

    M.assign((size_t)D_ * (size_t)D_, 0.0);

    for (int j = 0; j < D_; ++j)
        for (int i = 0; i < D_; ++i)
            M[i * D_ + j] = N01(rng);

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

std::vector<double> AttractiveSector::rotatedScaled(const Vec& x) const
{
    Vec z(D_, 0.0);
    for (int r = 0; r < D_; ++r) {
        double acc = 0.0;
        const double* Rrow = &R_[r * D_];
        for (int c = 0; c < D_; ++c)
            acc += Rrow[c] * x[c];
        z[r] = diag_[r] * acc;
    }
    return z;
}

double AttractiveSector::tosz_scalar(double x)
{
    if (x == 0.0)
        return 0.0;

    double sgn = (x > 0.0 ? 1.0 : -1.0);
    double h   = std::log(std::abs(x));
    double c1  = (x > 0.0 ? 10.0 : 5.5);
    double c2  = (x > 0.0 ? 7.9  : 3.1);

    return sgn * std::exp(h + 0.049 * (std::sin(c1 * h) + std::sin(c2 * h)));
}

double AttractiveSector::evaluate_core(const Vec& x)
{
    Vec z = rotatedScaled(x);

    double s = 0.0;
    for (int i = 0; i < D_; ++i) {
        const double si = (z[i] > 0.0) ? 100.0 : 1.0;
        const double t  = si * z[i];
        s += t * t;
    }

    // T_osz is applied to the scalar sum itself (not element-wise) per the
    // BBOB F6 definition, and only on the positive branch makes sense here
    // since s >= 0 always (sum of squares) — tosz_scalar handles s=0 too.
    const double tosz = tosz_scalar(s);
    double fval = std::pow(std::max(0.0, tosz), 0.9);

    if (!(fval >= 0.0) || std::isnan(fval) || std::isinf(fval))
        return 1e12;

    return fval;
}

void AttractiveSector::gradient_core(const Vec& x, Vec& g)
{
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
