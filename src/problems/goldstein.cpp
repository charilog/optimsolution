#include "goldstein.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Goldstein–Price function (2D)
//
// Extremely rugged 2D function with many local minima.
// Global minimum: f* = 3 at (0, -1)
// Domain: [-2, 2]^2
//
// Properties:
//   - multimodal
//   - non-separable
// -----------------------------------------------------------------------------

Goldstein::Goldstein()
{
    setName("goldstein");
    setFullName("Goldstein–Price function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous 2D benchmark test function");

    setKnownGlobalOptimum(3.0);
}

void Goldstein::init(int dim)
{
    Problem::init(2);

    Vec l = { -2.0, -2.0 };
    Vec u = {  2.0,  2.0 };
    setBounds(l, u);

    Vec xopt = { 0.0, -1.0 };
    setKnownGlobalOptimum(3.0, xopt);
}

double Goldstein::evaluate_core(const Vec& x)
{
    const double X = x[0];
    const double Y = x[1];

    double A = X + Y + 1.0;
    double B = 19.0 - 14.0*X + 3.0*X*X - 14.0*Y + 6.0*X*Y + 3.0*Y*Y;

    double C = 2.0*X - 3.0*Y;
    double D = 18.0 - 32.0*X + 12.0*X*X + 48.0*Y - 36.0*X*Y + 27.0*Y*Y;

    return (1.0 + A*A * B) * (30.0 + C*C * D);
}

void Goldstein::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double X = x[0];
    const double Y = x[1];

    // Precompute common terms
    double A = X + Y + 1.0;
    double B = 19.0 - 14.0*X + 3.0*X*X - 14.0*Y + 6.0*X*Y + 3.0*Y*Y;

    double C = 2.0*X - 3.0*Y;
    double D = 18.0 - 32.0*X + 12.0*X*X + 48.0*Y - 36.0*X*Y + 27.0*Y*Y;

    double term1 = 1.0 + A*A * B;
    double term2 = 30.0 + C*C * D;

    // Partial derivatives computed analytically (long but standard)
    double dA_dx = 1.0;
    double dA_dy = 1.0;

    double dB_dx = -14.0 + 6.0*X + 6.0*Y;
    double dB_dy = -14.0 + 6.0*Y + 6.0*X;

    double dC_dx = 2.0;
    double dC_dy = -3.0;

    double dD_dx = -32.0 + 24.0*X - 36.0*Y;
    double dD_dy = 48.0 - 36.0*X + 54.0*Y;

    double df_dx =
        (2.0*A*dA_dx*B + A*A*dB_dx) * term2
        + term1 * (2.0*C*dC_dx * D + C*C * dD_dx);

    double df_dy =
        (2.0*A*dA_dy*B + A*A*dB_dy) * term2
        + term1 * (2.0*C*dC_dy * D + C*C * dD_dy);

    g[0] = df_dx;
    g[1] = df_dy;
}

} // namespace optimsolution
