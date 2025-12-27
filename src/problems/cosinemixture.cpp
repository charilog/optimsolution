#include "cosinemixture.h"
#include <cmath>

namespace optimsolution {

namespace {
    constexpr double PI = 3.14159265358979323846;
}

CosineMixture::CosineMixture()
{
    setName("cm");
    setFullName("Cosine Mixture function");
    setModality("multimodal");
    setSeparability("separable");
    setCategory("continuous benchmark test function");

    setKnownGlobalOptimum(0.0);
}

void CosineMixture::init(int dim)
{
    Problem::init(dim);

    Vec l(dim, -1.0);
    Vec u(dim,  1.0);
    setBounds(l, u);

    Vec xopt(dim, 0.0);
    setKnownGlobalOptimum(0.1 * dim, xopt); // maximum value at x=0
}

double CosineMixture::evaluate_core(const Vec& x)
{
    const int D = dimension();
    double sum = 0.0;

    for (int i = 0; i < D; ++i)
        sum += 0.1 * std::cos(5.0 * PI * x[i]) - x[i] * x[i];

    return sum;
}

void CosineMixture::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);

    for (int i = 0; i < D; ++i)
    {
        g[i] = -0.1 * 5.0 * PI * std::sin(5.0 * PI * x[i]) - 2.0 * x[i];
    }
}

} // namespace optimsolution
