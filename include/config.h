#pragma once
#include <string>
#include <vector>
#include <unordered_map>

#include "options.h"

namespace optimsolution {

struct SensitivityOptions {
    bool enabled = false;
    std::string mode = "grid";         // "grid" or "one-at-a-time"
    std::string output_csv = "sensitivity_results.csv";
    std::vector<std::string> params;   // order of parameters (e.g., {"F","CR"})
    // grid of values per parameter (key is method-specific option name, value list is doubles)
    std::unordered_map<std::string, std::vector<double>> values;
};


struct Config {
    GeneralOptions     g;
    TerminationOptions t;
    MethodConfig       methodKV;
    InitOptions        init;
    SensitivityOptions sens;    

    static Config load(const std::string& path, const std::string& methodName);
};

} // namespace optimsolution
