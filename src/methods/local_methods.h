#pragma once
// ----------------------------------------------------------------------------
// Local methods as stand-alone optimizers: gd, lbfgs, bfgs, nm, cg, ncg, adam.
//
// One registered name per algorithm. The SAME name is used
//   * as local_method / end_local_method inside the stochastic methods
//     (Optimizer::localSearch() -> runLocalSearch(), unchanged), and
//   * as a method of its own in every run mode (single, batch, sensitivity,
//     portfolio, complexity, ...), through the standard makeMethod()+run()
//     path like every other method.
// Both uses read the same settings section ([gd], [lbfgs], [bfgs], [nm],
// [cg], [ncg], [adam]); there are no separate "*_run" variants any more.
//
// A stand-alone run is ONE local search:
//   * it starts from a point drawn with the [init] options (a different
//     start for every run seed, like the initial population of the
//     stochastic methods -- previously the plain names started every run
//     from the all-zero vector, the optimum of many benchmark problems);
//   * it is capped by the run budget max_evals (including "<N>xD"), and
//     otherwise stops by the section's own criteria (max_iters, tol, ...);
//   * parameter values passed through configure() (e.g. a sensitivity sweep
//     over [adam] lr) override the settings-file section for this run;
//   * its best-so-far trajectory is recorded and printed as the usual
//     "[Run r | iter k | evals e] best_f = ..." progress lines, so the
//     convergence CSV, the GUI plots, the batch statistics and the 2-D
//     landscape log are filled in exactly as for the other methods;
//   * population is reported as 1.
// ----------------------------------------------------------------------------
#include "optimizer.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace optimsolution {

class LocalMethod : public Optimizer {
public:
    explicit LocalMethod(std::string name);

    // The seven local-method names, in registration order.
    static const std::vector<std::string>& names();
    static bool isLocalName(const std::string& name);

    std::string methodShortName() const override { return name_; }
    std::string methodFullName()  const override;

    void configure(const MethodConfig& mc) override;

protected:
    void init() override;
    void one_iteration() override { /* never reached: single local search in init() */ }

private:
    void writeProgressLine(int iter, long long evals, double f, const Vec& x) const;

    std::string name_;
    std::unordered_map<std::string, std::string> overrides_;
};

} // namespace optimsolution
