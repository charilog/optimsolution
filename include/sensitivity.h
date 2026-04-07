#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include "config.h"

namespace optimsolution {

// Run sensitivity analysis if enabled in cfg.sens.
// Sweeps over user-provided parameter grids (per-method keys like "F", "CR", "pbest", etc.).
// For each combination, runs cfg.g.runs experiments and summarizes to CSV.
int run_sensitivity(const std::string& method,
                    const std::string& problem,
                    int dim,
                    const Config& cfg);

} // namespace optimsolution
