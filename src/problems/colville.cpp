#include "colville.h"

namespace optimsolution {

Colville::Colville()
{
    setName("colville");
    setFullName("Colville function");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void Colville::init(int /*dim*/)
{
    Problem::init(4);

    Vec lo(4, -10.0), hi(4, 10.0);
    setBounds(lo, hi);

    Vec xopt(4, 1.0);
    setKnownGlobalOptimum(0.0, xopt);
}

double Colville::evaluate_core(const Vec& x)
{
    const double x1 = x[0];
    const double x2 = x[1];
    const double x3 = x[2];
    const double x4 = x[3];

    const double t1 = x1 * x1 - x2;
    const double t2 = x1 - 1.0;
    const double t3 = x3 - 1.0;
    const double t4 = x3 * x3 - x4;
    const double t5 = x2 - 1.0;
    const double t6 = x4 - 1.0;

    return 100.0 * t1 * t1 + t2 * t2 + t3 * t3
         + 90.0 * t4 * t4
         + 10.1 * (t5 * t5 + t6 * t6)
         + 19.8 * t5 * t6;
}

void Colville::gradient_core(const Vec& x, Vec& g)
{
    g.assign(4, 0.0);

    const double x1 = x[0];
    const double x2 = x[1];
    const double x3 = x[2];
    const double x4 = x[3];

    const double t1 = x1 * x1 - x2;   // (x1^2 - x2)
    const double t4 = x3 * x3 - x4;   // (x3^2 - x4)
    const double t5 = x2 - 1.0;
    const double t6 = x4 - 1.0;

    // df/dx1 = 400*x1*(x1^2-x2) + 2*(x1-1)
    g[0] = 400.0 * x1 * t1 + 2.0 * (x1 - 1.0);
    // df/dx2 = -200*(x1^2-x2) + 20.2*(x2-1) + 19.8*(x4-1)
    g[1] = -200.0 * t1 + 20.2 * t5 + 19.8 * t6;
    // df/dx3 = 2*(x3-1) + 360*x3*(x3^2-x4)
    g[2] = 2.0 * (x3 - 1.0) + 360.0 * x3 * t4;
    // df/dx4 = -180*(x3^2-x4) + 20.2*(x4-1) + 19.8*(x2-1)
    g[3] = -180.0 * t4 + 20.2 * t6 + 19.8 * t5;
}

} // namespace optimsolution
