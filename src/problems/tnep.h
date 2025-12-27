#pragma once
#include "problem.h"
#include <vector>
#include <string>
#include <iostream>   // για εκτύπωση

namespace optimsolution {
/**
* TNEP – Transmission Network Expansion Planning (DC-OPF surrogate).
*
* Design variables:
* x[l] (l = 0..L-1): continuous encoding of the number of new circuits
* in line l. In the kernel, rounding is done
* to the nearest integer and restriction to [0, nmax_l].
*
* Internal representation:
* - from_[l], to_[l] : node pairs for each line
* - xreact_[l] : reactance per circuit (p.u.)
* - bper_[l] : susceptance per circuit (= 1/xreact)
* - fmax_[l] : thermal capacity per circuit (MW)
* - cost_[l] : cost per new circuit
* - n0_[l], nmax_[l] : initial & maximum number of circuits
* - Pinj_[k] : injections into DC power flow
*
* Objective:
* f(x) = investment_cost(x)
* + W_over_ * overload(x)
* + W_shed_ * load_shedding(x)
*
* where overshoots and shedding arise from DC power flow
* (solve_dc_pf) in a 6-node network / 11 lines.
*/
class TNEP : public Problem {
public:
    TNEP();
    void init(int dim) override;

protected:
    double evaluate_core(const Vec& x) override;
    void   gradient_core(const Vec& x, Vec& g) override;

private:
    int N_ = 6;   
    int L_ = 11;  

    std::vector<int>    from_, to_;
    std::vector<double> xreact_, bper_;
    std::vector<double> fmax_;
    std::vector<double> cost_;
    std::vector<int>    n0_, nmax_;
    std::vector<double> Pinj_;

    // weights του penalty
    double W_over_ = 1e6;
    double W_shed_ = 5e6;

    bool solve_dc_pf(const std::vector<int>& n_tot,
                     std::vector<double>& theta,
                     std::vector<double>& flow,
                     double& shed) const;

    static inline double clampd(double v, double lo, double hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }

    
    double evaluate_from_integer(const std::vector<int>& n_add,
                                 double& out_over,
                                 double& out_cost) const;

    void print_reference_solution_once() const; 
};

} // namespace optimsolution
