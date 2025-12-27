#include "gkls250.h"
#include "gkls.h"
#include <stdexcept>

namespace optimsolution {

namespace {
    constexpr int          GKLS250_DIM     = 2;
    constexpr unsigned int GKLS250_MINIMA  = 50;
    constexpr unsigned int GKLS250_FUNC_ID = 2;   // function number (1..100)
}

// -----------------------------------------------------
// GKLS 2D, 50 minima (D-type) for optimsolution
// Domain: [-1, 1]^2
// -----------------------------------------------------

Gkls250::Gkls250()
{
    setName("gkls250");
    setFullName("GKLS 2D, 50 minima (D-type)");
    setModality("multimodal");
    setSeparability("non-separable");
    setCategory("continuous benchmark test function (GKLS)");

    
    setKnownGlobalOptimum(0.0);
}

void Gkls250::init(int /*dim*/)
{
    const int D = GKLS250_DIM;
    Problem::init(D);

    // --------------------------------------------
    // 1. Bounds in optimsolution
    // --------------------------------------------
    Vec l(D, -1.0), u(D, 1.0);
    setBounds(l, u);


    GKLS_free();
    GKLS_domain_free();


    GKLS_dim        = static_cast<unsigned int>(D);
    GKLS_num_minima = GKLS250_MINIMA;

    // Domain [-1,1]^D
    int error = GKLS_domain_alloc();
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_domain_alloc() failed for Gkls250");
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
        throw std::runtime_error("GKLS_parameters_check() failed for Gkls250");
    }


    error = GKLS_arg_generate(GKLS250_FUNC_ID);
    if (error != GKLS_OK) {
        throw std::runtime_error("GKLS_arg_generate() failed for Gkls250");
    }


    if (GKLS_glob.num_global_minima > 0) {
        unsigned int idx = GKLS_glob.gm_index[0];  // πρώτο global minimum
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

double Gkls250::evaluate_core(const Vec& x)
{
  
    return GKLS_D_func(const_cast<double*>(x.data()));
}

void Gkls250::gradient_core(const Vec& x, Vec& g)
{
    const int D = dimension();
    g.resize(D);
    GKLS_D_gradient(const_cast<double*>(x.data()), g.data());
}

} // namespace optimsolution
