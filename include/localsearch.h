#pragma once
#include <vector>
#include <random>
#include <string>
#include <utility>
#include <limits>
#include <cmath>
#include "problem.h"
#include "utils.h"

namespace optimsolution {


using Vec = std::vector<double>;


struct LineSearchResult {
    Vec x_new;
    double f_new;
    Vec g_new; 
};


double dot(const Vec& a, const Vec& b);

// ||a||_2
double l2norm(const Vec& a);


void project_to_bounds(Vec& x, const Problem* prob);

// Armijo backtracking d

LineSearchResult backtrackingArmijo(
    Problem* prob,
    const Vec& x,
    const Vec& g,
    const Vec& d,
    double alpha0,
    double c1,
    int max_backtracks);


// Gradient Descent (momentum=0, GD with backtracking)
std::pair<Vec,double> localGD(
    Problem* prob,
    std::mt19937_64& rng,
    const Vec& x0);

// Nelder–Mead simplex (derivative-free)
std::pair<Vec,double> localNM(
    Problem* prob,
    std::mt19937_64& rng,
    const Vec& x0);

// Limited-memory BFGS (quasi-Newton with two-loop recursion)
std::pair<Vec,double> localLBFGS(
    Problem* prob,
    std::mt19937_64& rng,
    const Vec& x0);

std::pair<Vec,double> runLocalSearch(
    const std::string& name,
    Problem* prob,
    std::mt19937_64& rng,
    const Vec& x0);

} // namespace optimsolution
