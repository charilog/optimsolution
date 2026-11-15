#include "himmelblau.h"
#include <cmath>
#include <algorithm>

namespace optimsolution {

Himmelblau::Himmelblau()
{
    setName("himmelblau");
    setFullName("Himmelblau (maximize-style variant)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function");

  
}

void Himmelblau::init(int /*dim*/)
{
    // Faithful: treat it as 2-D (the original uses x[0], x[1])
    Problem::init(2);

    Vec lo(2, -6.0), hi(2, 6.0);
    setBounds(lo, hi);  // same bounds as reference
}

double Himmelblau::evaluate_core(const Vec& x)
{
    // f(x) = 200 - (x1^2 + x2 - 11)^2 - (x1 + x2^2 - 7)^2
    const double x1 = x[0];
    const double x2 = x[1];

    const double t1 = (x1 * x1 + x2 - 11.0);
    const double t2 = (x1 + x2 * x2 - 7.0);
    const double f  = 200.0 - t1 * t1 - t2 * t2;

    // keep behavior stable if NaN/Inf
    if (!(f >= -1e300) || std::isnan(f) || std::isinf(f))
        return 1e30;

    return f;
}

void Himmelblau::gradient_core(const Vec& x, Vec& g)
{
    // Central finite differences with a relative, bounds-aware step (see
    // Problem::finiteDifferenceGradient()). The previous hand-written
    // difference used a fixed step, which gave wrong gradients when |f| is
    // large (round-off ~ eps*|f|/h, e.g. ~3 percent error on CEC 2017 F11).
    finiteDifferenceGradient(x, g);
}

} // namespace optimsolution
