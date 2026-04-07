#include "hansen.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Hansen function (2D)
//
// Reference formulation (one common benchmark definition):
//   f(x,y) = A(x) * B(y)
//   A(x) = Σ_{i=0..4} (i+1) * cos(i*x + i + 1)
//   B(y) = Σ_{j=0..4} (j+1) * cos((j+2)*y + j + 1)
//
// Domain: [-10,10]^2
// Global minimum: f* = -176.5417931 (multiple global minimizers exist)
// One minimizer (example): x* ≈ (-1.3067077036, 4.8580568793)
//
// Properties:
//   - multimodal
//   - non-separable
// -----------------------------------------------------------------------------

Hansen::Hansen()
{
    setName("hansen");
    setFullName("Hansen function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-176.54179313674572);
}

void Hansen::init(int dim)
{
    Problem::init(2);

    Vec l = { -10.0, -10.0 };
    Vec u = {  10.0,  10.0 };
    setBounds(l, u);

    Vec xopt = { -1.306707703621301, 4.85805687926125 };
    setKnownGlobalOptimum(-176.54179313674572, xopt);
}

double Hansen::evaluate_core(const Vec& x)
{
    const double X = x[0];
    const double Y = x[1];

    double A = 0.0;
    double B = 0.0;

    // A(x) = sum_{i=0..4} (i+1) * cos(i*x + i + 1)
    for (int i = 0; i <= 4; ++i)
    {
        A += (i + 1) * std::cos(i * X + i + 1);
    }

    // B(y) = sum_{j=0..4} (j+1) * cos((j+2)*y + j + 1)
    for (int j = 0; j <= 4; ++j)
    {
        B += (j + 1) * std::cos((j + 2) * Y + j + 1);
    }

    return A * B;
}

void Hansen::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);

    const double X = x[0];
    const double Y = x[1];

    double A  = 0.0, dA = 0.0;
    double B  = 0.0, dB = 0.0;

    // A(x) = sum_{i=0..4} (i+1) * cos(i*x + i + 1)
    // dA/dx = sum_{i=0..4} (i+1) * (-sin(i*x + i + 1)) * i
    for (int i = 0; i <= 4; ++i)
    {
        const double t = i * X + i + 1;
        A  += (i + 1) * std::cos(t);
        dA += (i + 1) * (-std::sin(t)) * i;
    }

    // B(y) = sum_{j=0..4} (j+1) * cos((j+2)*y + j + 1)
    // dB/dy = sum_{j=0..4} (j+1) * (-sin((j+2)*y + j + 1)) * (j+2)
    for (int j = 0; j <= 4; ++j)
    {
        const double t = (j + 2) * Y + j + 1;
        B  += (j + 1) * std::cos(t);
        dB += (j + 1) * (-std::sin(t)) * (j + 2);
    }

    g[0] = dA * B;
    g[1] = A * dB;
}

} // namespace optimsolution
