#pragma once
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>

namespace optimsolution {

/// Return fixed dimension for known fixed-D problems, or 0 if the problem
/// is not strictly fixed-dimension.
/// Names are matched case-insensitively.
inline int getFixedDimOrZero(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

    // ONLY problems whose *definition* is in a fixed dimension:
    // - Classic 2D/3D/6D analytical test functions (the standard form is in a fixed D)
    // - CEC real-world problems with clearly specified D (mainly CEC 2011)

    static const std::unordered_map<std::string,int> kFixed = {
        // -------------------------------------------------
        // 1. Classic low-D analytical benchmarks
        //    (not the generalized CEC scalable versions,
        //     but the original, fixed-dimension definitions)
        // -------------------------------------------------
        {"bohachevsky1", 2},
        {"bohachevsky2", 2},
        {"bohachevsky3", 2},
        {"branin",       2},
        {"camel",        2},  // six-hump camelback
        {"easom",        2},
        {"goldstein",    2},  // Goldstein–Price
        {"hansen",       2},
        {"himmelblau",   2},
        {"schaffer",     2},  // Schaffer F6/F7 in the classic 2D form

        // Hartmann family (fixed by definition)
        {"hartmann3",    3},
        {"hartman3",     3},
        {"hartmann",     6},  // if the problem is named plain "hartmann" => 6D
        {"hartman6",     6},
        {"hartmann6",    6},

        // Shekel test functions – dimension = 4
        {"shekel5",      4},
        {"shekel7",      4},
        {"shekel10",     4},

        // Dedicated 2D Rastrigin instance (if present in the project)
        {"rastrigin2",   2},
		{"gkls250",   2},
		{"gkls350",   3},
		{"gkls2100",   2},

        // -------------------------------------------------
        // 2. CEC 2011 Real-World Optimization Problems
        //    (dimensions from the original tech report and
        //     follow-up CEC 2011/2022 synthetic studies)
        // -------------------------------------------------

        // Parameter Estimation for Frequency-Modulated Sound Waves
        // CEC 2011 real-world: D = 6
        {"fmsynth",      6},

        // Bifunctional Catalyst Blend Optimal Control
        // CEC 2011: D = 1
        {"bifunctionalcatalyst", 1},

        // Tersoff Potential for model Si (B) / (C)
        // NOTE: this code uses D = 24 for both instances.
        {"tersoffb",     24},
        {"tersoffc",     24},

        // Spread Spectrum Radar Polyphase Code Design
        // Original CEC 2011 instance: 20 design parameters
        {"polyphase",    20},

        // Transmission Network Expansion Planning (Garver system)
        // CEC 2011: 7 binary decision variables
        {"tnep",         7},

        // Transmission Network Pricing problem
        // CEC 2011: D = 126 continuous decision variables
        {"transmissionpricing", 126},

        // Circular Antenna Array Design (6 amplitudes + 6 phases)
        // CEC 2011: D = 12
        {"antennaarray", 12},
		
        // Static Economic Load Dispatch (ELD1–ELD5)
        // CEC 2011: {6, 13, 15, 40, 140}
        {"eld1",         6},
        {"eld2",         13},
        {"eld3",         15},
        {"eld4",         40},
        {"eld5",         140},

        // Dynamic Economic Dispatch (DED1–DED2)
        // CEC 2011: {120, 216}
        {"ded1",         120},
        {"ded2",         216},

        // Hydrothermal Scheduling (CEC 2011 instances)
        // All CEC 2011 implementations use D = 96,
        // so the generic "hydrothermal" is fixed at 96.
        {"hydrothermal", 96},

        // Spacecraft trajectory problems (CEC 2011)
        // Messenger (P29): D = 26
        // Cassini 2 (P30): D = 22
        {"messenger",    26},
        {"cassini",      22},

        // Heat Exchanger Network design (CEC-style real-world): D = 22
        {"heatexchanger",      22},

        // Wireless coverage planning (CEC-style real-world): D = 6
        {"wirelesscoverage",   6},

        // Uniform Linear Antenna Array (amplitude taper): default D = 10
        {"antennaula",         10},

        // Alternate Polyphase code design instance (higher-D variant): D = 25
        {"polyphase",          25},

        // Mean–Variance Portfolio Optimization (Markowitz): D = 10 assets
        {"portfoliomv",        10},

        // Tandem (MGA-1DSM spacecraft trajectory): D = 18
        {"tandem",             18},

        // Vibrating platform design (mass-spring-damper isolator): D = 2
        {"vibratingplatform",   5},
		
        // PUMA 560 6-DOF Inverse Kinematics: D = 6
        {"ik6dof",   6},		
		
		
		// Idealized gas cycle efficiency (Brayton-type): D = 4
        {"gascycle",		    4}
    };

    auto it = kFixed.find(name);
    return (it == kFixed.end()) ? 0 : it->second;
}

} // namespace optimsolution
