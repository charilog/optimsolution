#pragma once
#include <memory>
#include <string>
#include "problem.h"
#include "optimizer.h"


namespace optimsolution {


std::unique_ptr<Problem> makeProblem(const std::string& name);


std::unique_ptr<Optimizer> makeMethod(const std::string& name);

} // namespace optimsolution
