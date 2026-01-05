#include "MainWindow.h"
#include "CrashLog.h"

#include <QApplication>
#include <QIcon>
#include <QMessageBox>

int main(int argc, char** argv) {
  try {
    QApplication app(argc, argv);

    // Prefer a PNG from Qt resources for runtime window/taskbar icons.
    // The .ico is still embedded into the .exe via optimsolution_gui.rc for Explorer file icon.
    const QIcon icon(":/icons/optimsolution.png");
    if (!icon.isNull()) {
      QApplication::setWindowIcon(icon);
    } else {
      optimsolution_gui::CrashLog::append("Icon resource not found: :/icons/optimsolution.png (using default icon).");
    }

    optimsolution_gui::CrashLog::installQtMessageHandler();
    optimsolution_gui::CrashLog::installTerminateHandler();
    optimsolution_gui::CrashLog::append("optimsolution GUI starting.");

    optimsolution_gui::MainWindow w;
    if (!icon.isNull()) {
      w.setWindowIcon(icon);
    }
    w.resize(1100, 750);
    w.show();

    return app.exec();
  } catch (const std::exception& e) {
    optimsolution_gui::CrashLog::append(QString("Unhandled std::exception: %1").arg(e.what()));
    QMessageBox::critical(nullptr, "Fatal error", QString("Unhandled exception:\n%1").arg(e.what()));
  } catch (...) {
    optimsolution_gui::CrashLog::append("Unhandled non-std exception.");
    QMessageBox::critical(nullptr, "Fatal error", "Unhandled unknown exception.");
  }
  return 1;
}
