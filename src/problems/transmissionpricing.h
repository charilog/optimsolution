#pragma once
#include "problem.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * TransmissionPricing — PTDF-based DC power-flow pricing
 *
 * Decision vector x: flattened GD_{i,j} (power sent from generator i to load j).
 * Dimension D = NG * ND (ND inferred from init(dim) if dim%NG==0, otherwise default).
 *
 * Objective:
 *   cost_usage + pen_congestion + pen_generators + pen_load + pen_system + small_reg
 *
 * Grid: embedded small 6-bus toy system (lines with x, Fmax, alpha; 3 generators; 3 base load buses).
 * No static members. No external deps.
 */
class TransmissionPricing : public Problem {
public:
    TransmissionPricing();
    void init(int dim) override;                  // set D = NG*ND, bounds, build ND loads

protected:
    double evaluate_core(const Vec& x) override;  // objective value
    void   gradient_core(const Vec& x, Vec& g) override; // numeric forward diffs

private:
    // ------- Grid definition -------
    int NB_ = 6;           // buses
    int slack_ = 0;        // slack bus index

    struct Line { int a, b; double x, Fmax, alpha; };
    std::vector<Line> lines_;
    int NL_ = 0;

    // Generators
    std::vector<int>    gen_bus_;                 // size NG_
    std::vector<double> Pg_min_, Pg_max_;         // MW
    int NG_ = 0;

    // Base load buses & base demands (will be expanded to ND_)
    std::vector<int>    load_bus_base_;           // candidate load buses
    std::vector<double> Pd_base_;                 // MW on each base bus

    // After init:
    int ND_ = 0;                                  // #loads actually used
    int Dv_ = 0;                                  // = NG_ * ND_
    std::vector<int>    load_bus_;                // size ND_
    std::vector<double> Pd_;                      // size ND_

    // Penalty weights
    double W_cong_ = 50.0;                        // congestion (quadratic, normalized)
    double W_gen_  = 30.0;                        // generator bounds
    double W_load_ = 40.0;                        // load service
    double W_sys_  = 20.0;                        // system balance
    double reg_w_  = 1e-5;                        // tiny L1-like regularizer

    // ---- Helpers (no static) ----
    void build_Bbus(std::vector<std::vector<double>>& Bbus) const;
    bool invert_dense(std::vector<std::vector<double>> A,
                      std::vector<std::vector<double>>& Ainv) const;
    bool reduced_inverse(const std::vector<std::vector<double>>& M,
                         int skip_row_col,
                         std::vector<std::vector<double>>& Minv_red) const;
    void ptdf_for_pair(int s_bus, int r_bus,
                       const std::vector<std::vector<double>>& Binv_red,
                       std::vector<double>& ptdf_lr) const;
    void expand_loads_by_ND();                    // fill load_bus_, Pd_ from base

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
};

} // namespace optimsolution
