#pragma once
#include <memory>
#include <string>
#include <vector>
#include "problem.h"
#include "optimizer.h"


namespace optimsolution {


std::unique_ptr<Problem> makeProblem(const std::string& name);


std::unique_ptr<Optimizer> makeMethod(const std::string& name);

// -----------------------------------------------------------------------------
// Factory introspection
// -----------------------------------------------------------------------------
// The GUI (and other tooling) needs to enumerate available methods/problems at
// runtime. Parsing source files (e.g., src/factory.cpp) is fragile because it
// breaks when only the build directory is deployed or moved.
//
// These APIs provide a portable way to populate dropdowns and validate inputs.
// Returned names are the short, machine-friendly identifiers accepted by the
// CLI/GUI (e.g., "jso", "rastrigin").

std::vector<std::string> listProblemNames();
std::vector<std::string> listMethodNames();

} // namespace optimsolution
