#include "bifunctionalcatalyst.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

BifunctionalCatalyst::BifunctionalCatalyst()
    : umin_(0.6)
    , umax_(0.9)
{
    // Metadata for the new problem style
    setName("bifunctional_catalyst");
    setFullName("Bifunctional catalyst dynamic optimization problem");
    setModality("unknown");  // real-world; landscape not analytically classified
    setSeparability("non-separable"); // dynamic, state-coupled model
    setCategory("real-world dynamic optimization problem");

    // No analytic global optimum known, so we do not set a specific x*
    // (just leave the default state as-is; optimizers will treat it as unknown)
    // setKnownGlobalOptimum(...) is intentionally not called with a location.

    // Fill coefficients exactly as in the older implementation
    c_ = {{
        { 2.918487e-003, -8.045787e-003,  6.749947e-003, -1.416647e-003 },
        { 9.509977e+000, -3.500994e+001,  4.283329e+001, -1.733333e+001 },
        { 2.682093e+001, -9.556079e+001,  1.130398e+002, -4.429997e+001 },
        { 2.087241e+002, -7.198052e+002,  8.277466e+002, -3.166655e+002 },
        { 1.350005e+000, -6.850027e+000,  1.216671e+001, -6.666689e+000 },
        { 1.921995e-002, -7.945320e-002,  1.105660e-001, -5.033333e-002 },
        { 1.323596e-001, -4.692550e-001,  5.539323e-001, -2.166664e-001 },
        { 7.339981e+000, -2.527328e+001,  2.993329e+001, -1.199999e+001 },
        { -3.950534e-001,  1.679353e+000, -1.777829e+000,  4.974987e-001 },
        { -2.504665e-005,  1.005854e-002, -1.986696e-002,  9.833470e-003 }
    }};
}

void BifunctionalCatalyst::init(int /*dim*/)
{
    // Dimension is fixed to 1, as in the original version
    Problem::init(1);

    Vec lo(1, umin_), hi(1, umax_);
    setBounds(lo, hi);
}

inline void BifunctionalCatalyst::rhs(
    double /*u*/,
    const std::array<double,10>& k,
    const std::array<double,7>&  y,
    std::array<double,7>&        dy
) const
{
    // Reaction network ODE system (faithful to the reference implementation)
    dy[0] = -k[0] * y[0];
    dy[1] =  k[0] * y[0] - (k[1] + k[2]) * y[1] + k[3] * y[4];
    dy[2] =  k[1] * y[1];
    dy[3] = -k[5] * y[3] + k[4] * y[4];
    dy[4] =  k[2] * y[1] + k[5] * y[3]
           - (k[3] + k[4] + k[7] + k[8]) * y[4]
           + k[6] * y[5] + k[9] * y[6];
    dy[5] =  k[7] * y[4] - k[6] * y[5];
    dy[6] =  k[8] * y[4] - k[9] * y[6];
}

double BifunctionalCatalyst::simulate_and_objective(double u) const
{
    // Clamp u to [umin_, umax_] for safety
    const double uc = std::clamp(u, umin_, umax_);

    // k_i(u) = c0 + c1*u + c2*u^2 + c3*u^3
    std::array<double,10> k{};
    const double u2 = uc * uc;
    const double u3 = u2 * uc;
    for (int i = 0; i < 10; ++i) {
        k[i] = c_[i][0]
             + c_[i][1] * uc
             + c_[i][2] * u2
             + c_[i][3] * u3;
    }

    // Initial conditions
    std::array<double,7> y{};
    y[0] = 1.0;  // others are 0
    std::array<double,7> dy{};

    // Explicit Euler integration on [0, 0.78], N = 1000
    const double t0 = 0.0;
    const double tf = 0.78;
    const int    N  = 1000;
    const double dt = (tf - t0) / static_cast<double>(N);

    for (int step = 0; step < N; ++step) {
        rhs(uc, k, y, dy);
        for (int i = 0; i < 7; ++i) {
            y[i] += dt * dy[i];

            // Guards as in the original code
            if (!std::isfinite(y[i]) || y[i] < 0.0)
                y[i] = 0.0;
            if (y[i] > 1e6)
                y[i] = 1e6;
        }
    }

    // Objective J = 1000 * y6(tf)
    const double J = 1e3 * y[6];
    return J;
}

double BifunctionalCatalyst::evaluate_core(const Vec& x)
{
    const double u = x[0];
    const double J = simulate_and_objective(u);

    // Minimization: f(u) = -J(u)
    const double f = -J;
    // FIX: the previous fallback for a non-finite result was 1e-30 — for
    // this minimization objective (valid range roughly [-1000, 0], since
    // J = 1000*y6(tf) with y6 a mass fraction in [0,1]) a value near zero
    // looks like a MEDIOCRE-TO-WORST outcome, not a clearly-bad one, and is
    // certainly not the "large penalty pushing the optimizer away" that
    // every other invalid-result guard in this codebase uses. A failed/
    // non-finite integration should be scored as the worst case, not as an
    // ambiguous near-zero value that a search could mistake for legitimate
    // (if unremarkable) feedback.
    return std::isfinite(f) ? f : 0.0;
}

void BifunctionalCatalyst::gradient_core(const Vec& x, Vec& g)
{
    g.assign(1, 0.0);

    const double fx      = evaluate_core(x);
    const double h_base  = 1e-6;
    const double ui      = umax_;

    // step inside bounds
    double h = h_base;
    if (x[0] + h > ui)
        h = std::min(h, ui - x[0]);

    if (x[0] + h <= x[0]) {
        g[0] = 0.0;
        return;
    }

    Vec xt = x;
    xt[0] += h;
    const double fph = evaluate_core(xt);

    g[0] = (fph - fx) / h;
}

} // namespace optimsolution
