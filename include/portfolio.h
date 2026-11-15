#pragma once
#include <string>

namespace optimsolution {

// Mode 8: mid-run dynamic algorithm switching ("--portfolio").
// See src/mode_portfolio.cpp for the full design writeup.
int runPortfolioCli(int argc, char** argv);

} // namespace optimsolution
