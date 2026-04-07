#include "shubert.h"
#include <cmath>

namespace optimsolution {

Shubert::Shubert()
{
    setName("shubert");
    setFullName("Shubert function (2D, highly multimodal)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(-186.7309);
}

void Shubert::init(int /*dim*/)
{

    Problem::init(2);


    Vec lo(2, -10.0), hi(2, 10.0);
    setBounds(lo, hi);
}

// S(t) = Σ_{i=1..5} i cos((i+1)t + i)
// f(x,y) = S(x) * S(y)
double Shubert::evaluate_core(const Vec& x)
{
    const double X = x[0];
    const double Y = x[1];

    double Sx = 0.0, Sy = 0.0;
    for (int i = 1; i <= 5; ++i) {
        Sx += i * std::cos((i + 1) * X + i);
        Sy += i * std::cos((i + 1) * Y + i);
    }
    return Sx * Sy;
}

// ∇f = (S'(x) * S(y), S(x) * S'(y))
// with S'(t) = - Σ i(i+1) sin((i+1)t + i)
void Shubert::gradient_core(const Vec& x, Vec& g)
{
    g.assign(2, 0.0);

    const double X = x[0];
    const double Y = x[1];

    double Sx = 0.0, Sy = 0.0;
    double dSx = 0.0, dSy = 0.0;

    for (int i = 1; i <= 5; ++i) {
        const double phi_x = (i + 1) * X + i;
        const double phi_y = (i + 1) * Y + i;

        const double cos_phi_x = std::cos(phi_x);
        const double cos_phi_y = std::cos(phi_y);
        const double sin_phi_x = std::sin(phi_x);
        const double sin_phi_y = std::sin(phi_y);

        Sx  += i * cos_phi_x;
        Sy  += i * cos_phi_y;

        dSx += -static_cast<double>(i) * (i + 1) * sin_phi_x;
        dSy += -static_cast<double>(i) * (i + 1) * sin_phi_y;
    }

    g[0] = dSx * Sy;  // ∂f/∂x
    g[1] = Sx  * dSy; // ∂f/∂y
}

} // namespace optimsolution
