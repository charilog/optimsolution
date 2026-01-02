#pragma once
#include <QString>
#include <QStringList>

namespace optimsolution_gui {

struct FactoryLists {
  QStringList methods;
  QStringList problems;
};

// Extracts method/problem names from src/factory.cpp by parsing makeMethod/makeProblem bodies.
// This avoids mixing names and does not require any changes to the core factory.
FactoryLists readFactoryLists(const QString& factoryCppPath);

} // namespace optimsolution_gui
