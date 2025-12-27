#include "hansen.h"
#include <cmath>

namespace optimsolution {

// -----------------------------------------------------------------------------
// Hansen function (2D)
// f(x,y) = A(x) * B(y)
// A(x) = Σ i * cos((i+1)x + i), i=0..4
// B(y) = Σ i * cos((i+1)y + i), i=0..4
//
// Domain: [-10,10]^2
// Global min: ~ -176.541 at x=y≈ -7.58989
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

    setKnownGlobalOptimum(-176.541);
}

void Hansen::init(int dim)
{
    Problem::init(2);

    Vec l = { -10.0, -10.0 };
    Vec u = {  10.0,  10.0 };
    setBounds(l, u);

    Vec xopt = { -7.58989, -7.58989 };
    setKnownGlobalOptimum(-176.541, xopt);
}

double Hansen::evaluate_core(const Vec& x)
{
    double X = x[0];
    double Y = x[1];

    double A = 0.0;
    double B = 0.0;

    for (int i = 0; i <= 4; ++i)
    {
        A += i * std::cos((i+1)*X + i);
        B += i * std::cos((i+1)*Y + i);
    }

    return A * B;
}

void Hansen::gradient_core(const Vec& x, Vec& g)
{
    g.resize(2);
    double X = x[0];
    double Y = x[1];

    double A = 0.0, dA = 0.0;
    double B = 0.0, dB = 0.0;

    for (int i = 0; i <= 4; ++i)
    {
        A += i * std::cos((i+1)*X + i);
        dA += -i * (i+1) * std::sin((i+1)*X + i);

        B += i * std::cos((i+1)*Y + i);
        dB += -i * (i+1) * std::sin((i+1)*Y + i);
    }

    g[0] = dA * B;
    g[1] = A * dB;
}

} // namespace optimsolution
