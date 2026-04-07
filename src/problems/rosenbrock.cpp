#include "rosenbrock.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double A = 1.0;
    constexpr double B = 100.0;
}

// -----------------------------------------------------
// Rosenbrock metadata:
// - f(x) = sum_{i=1}^{D-1} [ 100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2 ]
// - Known global optimum: f* = 0 at x* = (1,...,1)
// - Modality:   unimodal (παραδοσιακά ταξινομημένη έτσι, αν και με "valley")
// - Separability: non-separable
// - Type/category: continuous benchmark
// - Typical bounds: [-2.048, 2.048]^D
// -----------------------------------------------------

Rosenbrock::Rosenbrock()
{
    setName("rosenbrock");
    setFullName("Rosenbrock benchmark function");
    setModality("unimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Rosenbrock::init(int dim)
{
    Problem::init(dim);


    Vec l(dim, -2.048), u(dim, 2.048);
    setBounds(l, u);

    // Global minimizer x* = (1,...,1)
    Vec xopt(dim, 1.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Rosenbrock::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double s = 0.0;
    for (int i = 0; i < D - 1; ++i) {
        double xi  = x[i];
        double xip = x[i+1];
        double t1  = xip - xi*xi;
        double t2  = xi - A;
        s += B * t1*t1 + t2*t2;
    }
    return s;
}

// Gradient:
// ∂f/∂x_i for i = 1..D:
//  i = 0:  df/dx0 = -400 x0 (x1 - x0^2) - 2 (1 - x0)
//  0 < i < D-1:  
//  df/dx_i = 200 (x_i - x_{i-1}^2) - 400 x_i (x_{i+1} - x_i^2) - 2 (1 - x_i)
//  i = D-1: df/dx_{D-1} = 200 (x_{D-1} - x_{D-2}^2)
void Rosenbrock::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.assign(D, 0.0);
    if (D < 2) return;

    // i = 0
    {
        double x0  = x[0];
        double x1  = x[1];
        double t1  = x1 - x0*x0;
        double t2  = x0 - A;
        g[0] = -400.0 * x0 * t1 + 2.0 * t2;
    }

    // 0 < i < D-1
    for (int i = 1; i < D-1; ++i) {
        double xim1 = x[i-1];
        double xi   = x[i];
        double xip1 = x[i+1];

        double t_prev = xi   - xim1*xim1;  
        double t_curr = xip1 - xi*xi;      

        double d_prev = 200.0 * t_prev;
        double d_curr = -400.0 * xi * t_curr + 2.0 * (xi - A);

        g[i] = d_prev + d_curr;
    }

    // i = D-1
    {
        int i = D-1;
        double xim1 = x[i-1];
        double xi   = x[i];
        double t_prev = xi - xim1*xim1;
        g[i] = 200.0 * t_prev;
    }
}

} // namespace optimsolution
