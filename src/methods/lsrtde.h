#pragma once
// =============================================================================
//  L-SRTDE -- Linear population size reduction Success RaTe-based
//             adaptive Differential Evolution
//
//  Winner of the IEEE CEC 2024 competition on single-objective bound-
//  constrained numerical optimization.
//
//  Reference:
//    V. Stanovov, E. Semenkin, "Success Rate-based Adaptive Differential
//    Evolution L-SRTDE for CEC 2024 Competition", 2024 IEEE Congress on
//    Evolutionary Computation (CEC), Yokohama, Japan, 2024, pp. 1-8.
//    doi:10.1109/CEC60901.2024.10611907
//  Official source code (ported here line by line):
//    https://github.com/VladimirStanovov/L-SRTDE_CEC-2024  (L-SRTDE.cpp)
//
//  Algorithm outline (two populations, L-NTADE scheme):
//    * "front" population  (PopulFront in the reference code): the parents.
//      Successful trials do NOT replace their own parent; they overwrite
//      front slots in round-robin (FIFO) order through a moving pointer.
//    * "main" population   (Popul in the reference code): the best N
//      solutions found so far. Successful trials are appended during the
//      generation; at the end of the generation the union is sorted and
//      truncated to the new size N.
//    * mutation  r-new-to-ptop/n/t :
//        v = x_c + F*(m_pbest - x_c) + F*(x_r1 - m_r2)
//      x_c, x_r1 from the front (r1 rank-selected, weights exp(-3*rank/N));
//      m_pbest from the top-p of the main pop, m_r2 uniform from the main pop.
//    * F ~ N(0.4 + 0.25*tanh(5*SR), 0.02), resampled until it lies in [0,1],
//      where SR = success rate of the previous generation (initially 0.5).
//    * p-best size = max(2, int(0.7 * N * exp(-7*SR))).
//    * CR ~ N(M_CR[k], 0.05) clamped to [0,1]; the memory (size 5, init 1.0)
//      stores the weighted Lehmer mean of the ACTUAL crossover rate (fraction
//      of components really taken from the mutant), weights = |delta f|.
//    * linear population size reduction N: 20*D -> 4 (worst removed).
//    * out-of-bounds components are re-sampled uniformly within the bounds.
// =============================================================================

#include "optimizer.h"

#include <vector>
#include <string>
#include <random>
#include <cmath>

namespace optimsolution {

class LSRTDE : public Optimizer {
public:
    LSRTDE() = default;
    ~LSRTDE() override = default;

    std::string methodShortName() const override { return "L-SRTDE"; }
    std::string methodFullName()  const override {
        return "L-SRTDE (Success Rate-based adaptive DE with LPSR, CEC 2024 winner)";
    }

    void setEndLocalFromGlobal(bool enable, const std::string& method) override {
        Optimizer::setEndLocalFromGlobal(enable, method);   // normalises none/off/0
        end_local_refine_ = finalLocalEnabled();
        // keep the [global] method even when refinement is globally off, so
        // "end_local_refine = 1" alone in [lsrtde] falls back to it
        end_local_method_ = method;
    }

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override;
    void end() override;

private:
    using Vec = std::vector<double>;

    enum class BoundMode { Random, Midpoint, Clip };

    double eval(const Vec& x);
    int    randInt(int n);                 // uniform in [0, n-1]
    double randU01();                      // uniform in [0, 1)
    double sampleF(double meanF);          // N(meanF, sigma_f) restricted to [0,1]
    double sampleCR(double mcr);           // N(mcr, sigma_cr) clamped to [0,1]
    double repairComponent(int j, double v, double parent_j);
    double weightedLehmerMean(const std::vector<double>& values,
                              const std::vector<double>& weights) const;
    int    targetPopulationSize() const;   // LPSR schedule
    void   removeWorstFromFront(int new_size);
    void   sortAndTruncateMain(int new_size);
    void   sortedOrder(const std::vector<double>& f, int n, std::vector<int>& idx) const;

    // ---- populations -------------------------------------------------------
    std::vector<Vec>    main_X_;    // "Popul"      (best-so-far population)
    std::vector<double> main_F_;    // "FitArr"
    std::vector<Vec>    front_X_;   // "PopulFront" (parents)
    std::vector<double> front_F_;   // "FitArrFront"

    // ---- adaptation state --------------------------------------------------
    std::vector<double> MCR_;           // CR memory ("MemoryCr")
    int    memory_iter_    = 0;         // "MemoryIter"
    int    pf_index_       = 0;         // "PFIndex" (round-robin front pointer)
    double success_rate_   = 0.5;       // "SuccessRate"
    int    n_init_         = 0;         // "NIndsFrontMax"
    int    generation_     = 0;

    // ---- configuration ([lsrtde] section of optimsolution.cfg) -------------
    int    pop_override_       = 0;     // population (0 = np_init_multiplier*D)
    double np_init_mult_       = 20.0;  // N_init = np_init_multiplier * D
    int    np_min_             = 4;     // final population size of LPSR
    int    memory_size_        = 5;     // H
    double mcr_init_           = 1.0;   // initial value of the CR memory
    double memory_c_           = 0.5;   // M_CR <- c*M_CR + (1-c)*meanWL
    double sigma_cr_           = 0.05;  // std of the CR normal sampling
    double f_base_             = 0.4;   // meanF = f_base + f_amp*tanh(f_sr_coef*SR)
    double f_amp_              = 0.25;
    double f_sr_coef_          = 5.0;
    double sigma_f_            = 0.02;  // std of the F normal sampling
    double pbest_rate_         = 0.7;   // p-best = max(pbest_min, int(rate*N*exp(-coef*SR)))
    double pbest_sr_coef_      = 7.0;
    int    pbest_min_          = 2;
    double rank_pressure_      = 3.0;   // r1 weights: exp(-rank_pressure*rank/N)
    double sr_init_            = 0.5;   // success rate before the first generation
    bool   delta_from_parent_  = false; // see lsrtde.cpp, success handling
    BoundMode bound_mode_      = BoundMode::Random;

    bool        end_local_refine_ = false;
    std::string end_local_method_;

    int debug_lsrtde_ = 0;
};

} // namespace optimsolution
