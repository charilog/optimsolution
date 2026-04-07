#pragma once
#include "problem.h"
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

namespace optimsolution {

/**
 * ELD1 – Economic Load Dispatch (single period)
 */
class ELD1 : public Problem {
public:
    ELD1();
    void init(int dim) override;                 // set NG (=dim or default 6) and bounds [Pmin,Pmax]

protected:
    double evaluate_core(const Vec& x) override; // fuel + penalties (balance, bounds, POZ, losses*)
    void   gradient_core(const Vec& x, Vec& g) override; // forward finite-diff

private:
    
    int NG;              

    
    double PD;           
    bool   use_losses;   
    std::vector<std::vector<double>> B;  // NGxNG
    std::vector<double> B0;              // NG
    double B00;                          

    
    // fuel: f_i(P) = a_i P^2 + b_i P + c_i
    std::vector<double> a, b, c;
    std::vector<double> Pmin, Pmax;

    // valve-point: e_i * |sin(f_i*(Pmin_i - P_i))|
    bool   use_valve;
    std::vector<double> e_vp, f_vp;

    // prohibited operating zones: list vectors [L,U] per unit
    std::vector<std::vector<std::pair<double,double>>> poz;

    
    double w_balance;    // (sumP - (PD+PL))^2
    double w_bounds;     // soft bounds
    double w_poz;        // POZ

    // ---- helpers ----
    void set_defaults();     
    void expand_to_NG();     
    void build_bounds();     

    static inline double clamp(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    static inline double smooth_abs(double z) {
        const double eps = 1e-12;                
        return std::sqrt(z*z + eps*eps);
    }
};

} // namespace optimsolution
