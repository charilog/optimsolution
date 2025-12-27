#include "gkls350.h"
#include "gkls.h"
#include <stdexcept>

namespace optimsolution {

namespace {
    constexpr int          GKLS350_DIM     = 3;
    constexpr unsigned int GKLS350_MINIMA  = 50;
    constexpr unsigned int GKLS350_FUNC_ID = 2;   // function number (1..100)
}

// -----------------------------------------------------
// GKLS 3D, 50 minima (D-type)
// Domain: [-1, 1]^3
// -----------------------------------------------------

Gkls350::Gkls350()
{
    setName("gkls350");
    setFullName("GKLS 3D, 50 minima (D-type)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function (GKLS)");

    setKnownGlobalOptimum(0.0);
}

void Gkls350::init(int /*dim*/)
{
    const int D = GKLS350_DIM;
    Problem::init(D);

    Vec l(D, -1.0), u(D, 1.0);
    setBounds(l, u);


    GKLS_free();
    GKLS_domain_free();

    GKLS_dim        = static_cast<unsigned int>(D);
    GKLS_num_minima = GKLS350_MINIMA;

    int error = GKLS_domain_alloc();
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_domain_alloc() failed for Gkls350");
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
        throw std::runtime_error("GKLS_parameters_check() failed for Gkls350");
    }

    error = GKLS_arg_generate(GKLS350_FUNC_ID);
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_arg_generate() failed for Gkls350");
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

double Gkls350::evaluate_core(const Vec& x)
{
    return GKLS_D_func(const_cast<double*>(x.data()));
}

void Gkls350::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);
    GKLS_D_gradient(const_cast<double*>(x.data()), g.data());
}

} // namespace optimsolution
