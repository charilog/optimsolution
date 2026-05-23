#include "ConfigFile.h"
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>

namespace optimsolution_gui {

static const QString SEC_GLOBAL = QStringLiteral("global"); // convention: global settings section
static const QString SEC_STOP   = QStringLiteral("stop");
static const QString SEC_INIT   = QStringLiteral("init");
static const QString SEC_SENS  = QStringLiteral("sensitivity");

QString ConfigFile::normalizeSection(const QString& s) {
  return s.trimmed();
}
QString ConfigFile::normalizeKey(const QString& k) {
  return k.trimmed();
}

bool ConfigFile::load(const QString& path) {
  QFile f(path);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
    loaded_ = false;
    return false;
  }

  data_.clear();
  path_ = path;

  QTextStream in(&f);
  in.setEncoding(QStringConverter::Utf8);

  QString currentSection = QStringLiteral("global"); // default
  if (!data_.contains(currentSection)) data_[currentSection] = Entry{};

  QRegularExpression reSection(QStringLiteral(R"(^\s*\[\s*([^\]]+)\s*\]\s*$)"));
  QRegularExpression reKV(QStringLiteral(R"(^\s*([^#;=\s][^=]*?)\s*=\s*(.*?)\s*$)"));

  while (!in.atEnd()) {
    QString line = in.readLine();

    auto ms = reSection.match(line);
    if (ms.hasMatch()) {
      currentSection = normalizeSection(ms.captured(1));
      if (!data_.contains(currentSection)) data_[currentSection] = Entry{};
      continue;
    }

    // ignore comments / empty
    QString trimmed = line.trimmed();
    if (trimmed.isEmpty() || trimmed.startsWith('#') || trimmed.startsWith(';')) continue;

    auto mk = reKV.match(line);
    if (mk.hasMatch()) {
      QString key = normalizeKey(mk.captured(1));
      QString val = mk.captured(2).trimmed();
      Entry& e = data_[currentSection];
      if (!e.kv.contains(key)) e.keyOrder << key;
      e.kv[key] = val;
    }
  }

  loaded_ = true;
  return true;
}

bool ConfigFile::save(const QString& path) const {
  if (!loaded_) return false;

  QFile f(path);
  if (!f.open(QIODevice::WriteOnly | QIODevice::Text | QIODevice::Truncate)) return false;

  QTextStream out(&f);
  out.setEncoding(QStringConverter::Utf8);

  // Write sections in a stable order: run, stop, init, then others alphabetical.
  QStringList secs = data_.keys();
  auto emitSection = [&](const QString& sec) {
    out << "[" << sec << "]\n";
    const Entry& e = data_[sec];
    QStringList keys = e.keyOrder;
    // include any keys not in order
    for (const auto& k : e.kv.keys()) if (!keys.contains(k)) keys << k;
    for (const auto& k : keys) {
      if (!e.kv.contains(k)) continue;
      out << k << " = " << e.kv.value(k) << "\n";
    }
    out << "\n";
  };

  if (secs.contains(SEC_GLOBAL)) { emitSection(SEC_GLOBAL); secs.removeAll(SEC_GLOBAL); }
  if (secs.contains(SEC_STOP))   { emitSection(SEC_STOP);   secs.removeAll(SEC_STOP); }
  if (secs.contains(SEC_INIT))   { emitSection(SEC_INIT);   secs.removeAll(SEC_INIT); }
  if (secs.contains(SEC_SENS))  { emitSection(SEC_SENS);  secs.removeAll(SEC_SENS); }

  secs.sort(Qt::CaseInsensitive);
  for (const auto& sec : secs) emitSection(sec);

  return true;
}

QStringList ConfigFile::sections() const {
  QStringList out = data_.keys();
  out.sort(Qt::CaseInsensitive);
  return out;
}

QMap<QString, QString> ConfigFile::sectionMap(const QString& section) const {
  QString sec = normalizeSection(section);
  if (!data_.contains(sec)) return {};
  return data_[sec].kv;
}

QString ConfigFile::value(const QString& section, const QString& key, const QString& def) const {
  QString sec = normalizeSection(section);
  QString k = normalizeKey(key);
  if (!data_.contains(sec)) return def;
  return data_[sec].kv.value(k, def);
}

void ConfigFile::setValue(const QString& section, const QString& key, const QString& value) {
  QString sec = normalizeSection(section);
  QString k = normalizeKey(key);
  if (!data_.contains(sec)) data_[sec] = Entry{};
  Entry& e = data_[sec];
  if (!e.kv.contains(k)) e.keyOrder << k;
  e.kv[k] = value.trimmed();
  loaded_ = true;
}

bool ConfigFile::removeKey(const QString& section, const QString& key) {
  QString sec = normalizeSection(section);
  QString k = normalizeKey(key);
  auto itS = data_.find(sec);
  if (itS == data_.end()) return false;
  Entry& e = itS.value();
  if (!e.kv.contains(k)) return false;
  e.kv.remove(k);
  e.keyOrder.removeAll(k);
  loaded_ = true;
  return true;
}

bool ConfigFile::hasSection(const QString& section) const {
  const QString sec = normalizeSection(section);
  return data_.contains(sec);
}

bool ConfigFile::removeSection(const QString& section) {
  const QString sec = normalizeSection(section);
  if (!data_.contains(sec)) return false;
  data_.remove(sec);
  loaded_ = true;
  return true;
}


QMap<QString, QString> ConfigFile::effectiveRunMap(const QString& methodSection) const {
  QMap<QString, QString> eff;

  // Base: [global]
  if (auto itRun = data_.constFind(SEC_GLOBAL); itRun != data_.constEnd()) {
    eff = itRun.value().kv;
  }

  // Overlay: [methodSection]
  const QString m = normalizeSection(methodSection);
  if (auto itM = data_.constFind(m); itM != data_.constEnd()) {
    const auto& kv = itM.value().kv;
    for (auto it = kv.constBegin(); it != kv.constEnd(); ++it) {
      eff[it.key()] = it.value();
    }
  }

  return eff;
}

QString ConfigFile::sourceOfEffectiveKey(const QString& methodSection, const QString& key) const {
  const QString k = normalizeKey(key);
  const QString m = normalizeSection(methodSection);

  if (auto itM = data_.constFind(m); itM != data_.constEnd()) {
    if (itM.value().kv.contains(k)) return QStringLiteral("method");
  }
  if (auto itRun = data_.constFind(SEC_GLOBAL); itRun != data_.constEnd()) {
    if (itRun.value().kv.contains(k)) return QStringLiteral("global");
  }
  return QStringLiteral("-");
}

} // namespace optimsolution_gui
