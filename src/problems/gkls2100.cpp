#include "gkls2100.h"
#include "gkls.h"
#include <stdexcept>

namespace optimsolution {

namespace {
    constexpr int          GKLS2100_DIM     = 2;
    constexpr unsigned int GKLS2100_MINIMA  = 100;
    constexpr unsigned int GKLS2100_FUNC_ID = 2;
}

// -----------------------------------------------------
// GKLS 2D, 100 minima (D-type)
// Domain: [-1, 1]^2
// -----------------------------------------------------

Gkls2100::Gkls2100()
{
    setName("gkls2100");
    setFullName("GKLS 2D, 100 minima (D-type)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function (GKLS)");

    setKnownGlobalOptimum(0.0);
}

void Gkls2100::init(int /*dim*/)
{
    const int D = GKLS2100_DIM;
    Problem::init(D);

    Vec l(D, -1.0), u(D, 1.0);
    setBounds(l, u);

 
    GKLS_free();
    GKLS_domain_free();

    GKLS_dim        = static_cast<unsigned int>(D);
    GKLS_num_minima = GKLS2100_MINIMA;

    int error = GKLS_domain_alloc();
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_domain_alloc() failed for Gkls2100");
    }

    double min_side = GKLS_domain_right[0] - GKLS_domain_left[0];
    for (unsigned int i = 1; i < GKLS_dim; ++i) {
        double tmp = GKLS_domain_right[i] - GKLS_domain_left[i];
        if (tmp < min_side) min_side = tmp;
    }

    GKLS_global_dist   = min_side / 3.0;
    GKLS_global_radius = 0.5 * GKLS_global_dist;
    GKLS_global_value  = GKLS_GLOBAL_MIN_VALUE;

    error = GKLS_parameters_check();
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_parameters_check() failed for Gkls2100");
    }

    error = GKLS_arg_generate(GKLS2100_FUNC_ID);
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_arg_generate() failed for Gkls2100");
    }

    if (GKLS_glob.num_global_minima > 0) {
        unsigned int idx = GKLS_glob.gm_index[0];
        Vec xopt(D);
        for (int j = 0; j < D; ++j) {
            xopt[j] = GKLS_minima.local_min[idx][j];
        }
        double fopt = GKLS_D_func(xopt.data());
        setKnownGlobalOptimum(fopt, xopt);
    } else {
        setKnownGlobalOptimum(GKLS_global_value);
    }
}

double Gkls2100::evaluate_core(const Vec& x)
{
    return GKLS_D_func(const_cast<double*>(x.data()));
}

void Gkls2100::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);
    GKLS_D_gradient(const_cast<double*>(x.data()), g.data());
}

} // namespace optimsolution
