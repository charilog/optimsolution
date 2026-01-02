#pragma once
#include <QString>

namespace optimsolution_gui {

class PathUtils final {
public:
  static QString findProjectRootFrom(const QString& startDir);
  static QString findExistingFileUpwards(const QString& startDir, const QString& relativePath);
  static QString normalizePath(const QString& p);
};

} // namespace optimsolution_gui
