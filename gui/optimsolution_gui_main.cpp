#include "MainWindow.h"
#include "CrashLog.h"
#include <QApplication>
#include <QMessageBox>

int main(int argc, char** argv) {
  try {
    QApplication app(argc, argv);

    optimsolution_gui::CrashLog::installQtMessageHandler();
    optimsolution_gui::CrashLog::installTerminateHandler();
    optimsolution_gui::CrashLog::append("optimsolution GUI (v30) starting.");

    optimsolution_gui::MainWindow w;
    w.resize(1100, 750);
    w.show();

    return app.exec();
  } catch (const std::exception& e) {
    // Last-resort user-visible error + log.
    optimsolution_gui::CrashLog::append(QString("Unhandled std::exception: %1").arg(e.what()));
    QMessageBox::critical(nullptr, "Fatal error", QString("Unhandled exception:\n%1").arg(e.what()));
  } catch (...) {
    optimsolution_gui::CrashLog::append("Unhandled non-std exception.");
    QMessageBox::critical(nullptr, "Fatal error", "Unhandled unknown exception.");
  }
  return 1;
}
