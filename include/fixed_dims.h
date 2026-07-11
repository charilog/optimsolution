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
        {"booth",        2},  // Booth::init() ignores dim, always D=2
        {"beale",        2},  // Beale::init() ignores dim, always D=2
        {"matyas",       2},  // Matyas::init() ignores dim, always D=2
        {"mccormick",    2},  // McCormick::init() ignores dim, always D=2
        {"colville",     4},  // Colville::init() ignores dim, always D=4

        // Additional CEC/GTOP-style spacecraft trajectory surrogates
        // (see cassini1.cpp / sagas.cpp / gtoc1.cpp / rosetta.cpp for the
        // "surrogate ΔV model" disclaimer, same pattern as messenger/tandem)
        {"cassini1",     6},   // Cassini1::init() ignores dim, always D=6
        {"sagas",       12},   // Sagas::init() ignores dim, always D=12
        {"gtoc1",        8},   // GTOC1::init() ignores dim, always D=8
        {"rosetta",     22},   // Rosetta::init() ignores dim, always D=22

        // Optimal control of a stirred tank reactor (CEC 2011 T04-style):
        // D = 2*Nstages_ = 20 (10 stages, 2 controls per stage)
        {"stirredtankreactor", 20},

        // Classic constrained mechanical engineering design benchmarks
        // (all fixed-dim; each init() ignores the requested dim)
        {"weldedbeam",       4},  // WeldedBeam::init() ignores dim, always D=4
        {"speedreducer",     7},  // SpeedReducer::init() ignores dim, always D=7
        {"pressurevessel",   4},  // PressureVessel::init() ignores dim, always D=4
        {"springdesign",     3},  // SpringDesign::init() ignores dim, always D=3
        {"cantileverbeam",   5},  // CantileverBeam::init() ignores dim, always D=5
        {"threebartruss",    2},  // ThreeBarTruss::init() ignores dim, always D=2
        {"geartrain",        4},  // GearTrain::init() ignores dim, always D=4
        {"schaffer",     2},  // Schaffer F6/F7 in the classic 2D form
        {"shubert",      2},  // Shubert function (2D); Shubert::init() ignores dim, always D=2

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
        // CEC 2011 RWP3/RWP4: D = 30 (N = 12 atoms, D = 3N-6).
        {"tersoffb",     30},
        {"tersoffc",     30},

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

        // Hydrothermal Scheduling (smooth penalty model; see problems/hydrothermal.cpp)
        // Hydrothermal::init() ignores the requested dim and always builds
        // D = NG*T + NH*T with NG=3 thermal units, NH=2 hydro units, T=24
        // hours => D = 3*24 + 2*24 = 120. (Earlier this table said 96, based
        // on the classic CEC 2011 hydro-only instance, which does not match
        // this implementation's decision vector — it also includes the
        // thermal powers P_{i,t} as explicit decision variables.)
        {"hydrothermal", 120},

        // Spacecraft trajectory problems (CEC 2011)
        // NOTE: Messenger here is a simplified surrogate ΔV model (see
        // problems/messenger.cpp), not the full CEC 2011 P29 MGA-1DSM
        // trajectory model. Messenger::init() ignores the requested dim and
        // always calls Problem::init(14) — its decision vector is t0 + 5
        // leg durations + 5 DSM fractions + rp + k1 + k2 = 14, not the
        // literature's D=26. (Earlier this table said 26, which mismatched
        // the actual object and would have misreported this problem's
        // dimension anywhere fixedDimForProblem() is consulted.)
        {"messenger",    14},
        {"cassini",      22},

        // Heat Exchanger Network design (CEC-style real-world): D = 22
        {"heatexchanger",      22},

        // Wireless coverage planning (CEC-style real-world): D = 6
        {"wirelesscoverage",   6},

        // Uniform Linear Antenna Array (amplitude taper): default D = 10
        {"antennaula",         10},

        // NOTE: an "Alternate Polyphase code design instance (higher-D
        // variant, D=25)" entry used to live here with the SAME key
        // "polyphase" as the entry above. std::unordered_map's
        // initializer-list constructor silently keeps only the first
        // inserted value for a duplicate key, so that second entry never
        // actually took effect (dead code) and has been removed. Polyphase
        // is registered only once in the factory, and its own init()
        // already accepts any requested dimension (see polyphase.cpp), so
        // there is no separate D=25 instance to register here.

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
				// Weather-aware irrigation scheduling problem: D = 24
        //{"weatherirrigation",		    24}
    };

    auto it = kFixed.find(name);
    return (it == kFixed.end()) ? 0 : it->second;
}

} // namespace optimsolution
