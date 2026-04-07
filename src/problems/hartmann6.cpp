#include "hartmann6.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Hartmann 6D function
//
// Domain: [0,1]^6
// Global minimum: f* ≈ -3.32236801141551
// At x* ≈ (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
//
// Properties:
//   - multimodal
//   - non-separable
// -----------------------------------------------------------------------------

namespace {
    // Alpha (4)
    constexpr double ALPHA[4] = { 1.0, 1.2, 3.0, 3.2 };

    // A (4x6)
    constexpr double A[4][6] = {
        {10.0,  3.0, 17.0, 3.5, 1.7,  8.0},
        { 0.05,10.0, 17.0, 0.1, 8.0, 14.0},
        { 3.0,  3.5, 1.7, 10.0,17.0,  8.0},
        {17.0,  8.0, 0.05,10.0,0.1, 14.0}
    };

    // P (4x6), scaled by 1e-4
    constexpr double P[4][6] = {
        {1312.0, 1696.0, 5569.0,  124.0, 8283.0, 5886.0},
        {2329.0, 4135.0, 8307.0, 3736.0, 1004.0, 9991.0},
        {2348.0, 1451.0, 3522.0, 2883.0, 3047.0, 6650.0},
        {4047.0, 8828.0, 8732.0, 5743.0, 1091.0,  381.0}
    };
}

Hartmann6::Hartmann6()
{
    setName("hartmann6");
    setFullName("Hartmann 6D function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous test function");

    setKnownGlobalOptimum(-3.32236801141551);
}

void Hartmann6::init(int dim)
{
    Problem::init(6);

    Vec l(6, 0.0);
    Vec u(6, 1.0);
    setBounds(l, u);

    Vec xopt = {0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573};
    setKnownGlobalOptimum(-3.32236801141551, xopt);
}

double Hartmann6::evaluate_core(const Vec& x)
{
    double sum = 0.0;

    for (int i = 0; i < 4; ++i)
    {
        double inner = 0.0;
        for (int j = 0; j < 6; ++j)
        {
            double pj = P[i][j] * 0.0001;
            double dx = x[j] - pj;
            inner += A[i][j] * dx * dx;
        }
        sum += (ALPHA[i] * std::exp(-inner));
    }

    return -sum;
}

void Hartmann6::gradient_core(const Vec& x, Vec& g)
{
    g.assign(6, 0.0);

    for (int i = 0; i < 4; ++i)
    {
        double inner = 0.0;
        double dx[6];

        for (int j = 0; j < 6; ++j)
        {
            double pj = P[i][j] * 0.0001;
            dx[j] = x[j] - pj;
            inner += A[i][j] * dx[j] * dx[j];
        }

        double e = std::exp(-inner);

        for (int j = 0; j < 6; ++j)
        {
            g[j] += (ALPHA[i] * e) * (2.0 * A[i][j] * dx[j]);
        }
    }
}

} // namespace optimsolution
