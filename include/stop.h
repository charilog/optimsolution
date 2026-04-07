#pragma once
#include <vector>
#include <limits>
#include "options.h"

namespace optimsolution {

class StopController {
public:
    StopController() = default;
    void configure(const TerminationOptions& to, int population);
    void reset();
    
    bool updateAndCheck(const std::vector<double>& fx);

private:
    // helpers 
    double variance_(const std::vector<double>& fx) const;

    TerminationOptions opt_;
    int NP_{0};

    
    int    iter_{0};
    int    stableCount_{0};
    double last_best_{0}, last_worst_{0}, last_sumTop_{0}, last_sumBot_{0}, last_range_{0};
    double last_impr_best_{0}, last_impr_worst_{0};
    double last_var_{0}, last_var_at_improvement_{0};
    double global_best_{std::numeric_limits<double>::infinity()};

    bool checkBSS(double cur_best);
    bool checkWSS(double cur_worst);
    bool checkTSS(const std::vector<double>& sorted);
    bool checkBOSS(const std::vector<double>& sorted);
    bool checkSRS(double cur_range);
    bool checkIRS(double impr_best, double impr_worst);
    bool checkDoublebox(const std::vector<double>& fx);
};

} // namespace optimsolution
