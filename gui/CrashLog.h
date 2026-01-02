#pragma once
#include <QString>

namespace optimsolution_gui {

class CrashLog final {
public:
  static QString logPath();
  static void append(const QString& line);
  static void installQtMessageHandler();
  static void installTerminateHandler();
};

} // namespace optimsolution_gui
