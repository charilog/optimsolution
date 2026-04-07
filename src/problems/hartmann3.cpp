#include "hartmann3.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Hartmann 3D function
//
// Domain: [0,1]^3
// Global minimum: f* ≈ -3.86278214782076
// At x* ≈ (0.114614, 0.555649, 0.852547)
//
// Properties:
//   - multimodal
//   - non-separable
// -----------------------------------------------------------------------------

namespace {
    // Alpha (4)
    constexpr double ALPHA[4] = { 1.0, 1.2, 3.0, 3.2 };

    // A (4x3) coefficients
    constexpr double A[4][3] = {
        { 3.0, 10.0, 30.0 },
        { 0.1, 10.0, 35.0 },
        { 3.0, 10.0, 30.0 },
        { 0.1, 10.0, 35.0 }
    };

    // P (4x3) coefficients, scaled by 1e-4 in the formula
    constexpr double P[4][3] = {
        { 3689.0, 1170.0, 2673.0 },
        { 4699.0, 4387.0, 7470.0 },
        { 1091.0, 8732.0, 5547.0 },
        {  381.0, 5743.0, 8828.0 }
    };
}

Hartmann3::Hartmann3()
{
    setName("hartmann3");
    setFullName("Hartmann 3D function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous test function");

    setKnownGlobalOptimum(-3.86278214782076);
}

void Hartmann3::init(int dim)
{
    Problem::init(3);

    Vec l = { 0.0, 0.0, 0.0 };
    Vec u = { 1.0, 1.0, 1.0 };
    setBounds(l, u);

    Vec xopt = { 0.114614, 0.555649, 0.852547 };
    setKnownGlobalOptimum(-3.86278214782076, xopt);
}

double Hartmann3::evaluate_core(const Vec& x)
{
    double sum = 0.0;

    for (int i = 0; i < 4; ++i)
    {
        double inner = 0.0;
        for (int j = 0; j < 3; ++j)
        {
            double pj = P[i][j] * 0.0001;
            double dx = x[j] - pj;
            inner += A[i][j] * dx * dx;
        }
        sum += ALPHA[i] * std::exp(-inner);
    }

    return -sum;
}

void Hartmann3::gradient_core(const Vec& x, Vec& g)
{
    g.assign(3, 0.0);

    for (int i = 0; i < 4; ++i)
    {
        double inner = 0.0;
        double dx[3];

        for (int j = 0; j < 3; ++j)
        {
            double pj = P[i][j] * 0.0001;
            dx[j] = x[j] - pj;
            inner += A[i][j] * dx[j] * dx[j];
        }

        double e = std::exp(-inner);

        for (int j = 0; j < 3; ++j)
        {
            g[j] += (ALPHA[i] * e) * (2.0 * A[i][j] * dx[j]);
        }
    }
}

} // namespace optimsolution
