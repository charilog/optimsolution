#include "PathUtils.h"
#include <QDir>
#include <QFileInfo>

namespace optimsolution_gui {

QString PathUtils::normalizePath(const QString& p) {
  return QDir::cleanPath(QDir(p).absolutePath());
}

QString PathUtils::findExistingFileUpwards(const QString& startDir, const QString& relativePath) {
  QDir dir(startDir);
  for (int i = 0; i < 12; ++i) {
    QString candidate = dir.filePath(relativePath);
    if (QFileInfo::exists(candidate)) return QDir::cleanPath(candidate);
    if (!dir.cdUp()) break;
  }
  return QString();
}

QString PathUtils::findProjectRootFrom(const QString& startDir) {
  // Project root is where CMakeLists.txt and src/ exist.
  QDir dir(startDir);
  for (int i = 0; i < 12; ++i) {
    if (QFileInfo::exists(dir.filePath("CMakeLists.txt")) && QDir(dir.filePath("src")).exists())
      return QDir::cleanPath(dir.absolutePath());
    if (!dir.cdUp()) break;
  }
  return QString();
}

} // namespace optimsolution_gui
