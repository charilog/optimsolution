#include "CrashLog.h"
#include <QCoreApplication>
#include <QDateTime>
#include <QFile>
#include <QTextStream>
#include <QtGlobal>
#include <exception>

namespace optimsolution_gui {

static QString timestamp() {
  return QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
}

QString CrashLog::logPath() {
  // Keep in application directory to avoid needing write access elsewhere.
  const QString dir = QCoreApplication::applicationDirPath();
  return dir + "/optimsolution_gui.log";
}

void CrashLog::append(const QString& line) {
      const QString msg = "[" + timestamp() + "] " + line + "\n";

// Primary: application directory
  {
    QFile f(logPath());
    if (f.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
      QTextStream out(&f);
      out << msg;
    }
  }

  // Secondary: current working directory (helps when running from build trees)
  {
    QFile f2("optimsolution_gui.log");
    if (f2.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
      QTextStream out2(&f2);
      out2 << msg;
    }
  }
}

static void qtMessageHandler(QtMsgType type, const QMessageLogContext&, const QString& msg) {
  QString pfx;
  switch (type) {
    case QtDebugMsg:    pfx = "DEBUG"; break;
    case QtInfoMsg:     pfx = "INFO";  break;
    case QtWarningMsg:  pfx = "WARN";  break;
    case QtCriticalMsg: pfx = "CRIT";  break;
    case QtFatalMsg:    pfx = "FATAL"; break;
  }
  CrashLog::append(pfx + ": " + msg);
  if (type == QtFatalMsg) std::abort();
}

void CrashLog::installQtMessageHandler() {
  qInstallMessageHandler(qtMessageHandler);
  CrashLog::append("Qt message handler installed.");
}

static void terminateHandler() {
  try {
    auto eptr = std::current_exception();
    if (eptr) std::rethrow_exception(eptr);
    CrashLog::append("std::terminate called with no active exception.");
  } catch (const std::exception& e) {
    CrashLog::append(QString("std::terminate due to std::exception: %1").arg(e.what()));
  } catch (...) {
    CrashLog::append("std::terminate due to unknown exception.");
  }
  std::abort();
}

void CrashLog::installTerminateHandler() {
  std::set_terminate(terminateHandler);
  CrashLog::append("Terminate handler installed.");
}

} // namespace optimsolution_gui
