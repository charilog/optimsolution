#pragma once
#include <memory>
#include <string>
#include <vector>
#include "problem.h"
#include "optimizer.h"
#include "moo_optimizer.h"


namespace optimsolution {


std::unique_ptr<Problem> makeProblem(const std::string& name);


std::unique_ptr<Optimizer> makeMethod(const std::string& name);

// -----------------------------------------------------------------------------
// Multi-objective (optional; additive, does not touch single-objective paths)
// -----------------------------------------------------------------------------
// A problem "is" multi-objective if Problem::numObjectives() >= 2 once
// constructed (most problems don't need a real dimension to answer this, so
// dim=2 is used as a cheap probe). Everything below is purely additive: it
// never affects makeProblem()/makeMethod() or how Single/Batch/Sensitivity
// modes evaluate a problem.
bool isMultiObjectiveProblem(const std::string& name);
std::vector<std::string> listMultiObjectiveProblemNames();
std::vector<std::string> listMultiObjectiveMethodNames();
std::unique_ptr<MOOOptimizer> makeMultiObjectiveMethod(const std::string& name);

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
