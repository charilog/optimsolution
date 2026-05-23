#pragma once
#include <QString>
#include <QMap>
#include <QStringList>

namespace optimsolution_gui {

class ConfigFile {
public:
  bool load(const QString& path);
  bool save(const QString& path) const;

  bool isLoaded() const { return loaded_; }

  QStringList sections() const;
  QMap<QString, QString> sectionMap(const QString& section) const;

  QString value(const QString& section, const QString& key, const QString& def = QString()) const;
  void setValue(const QString& section, const QString& key, const QString& value);
  bool removeKey(const QString& section, const QString& key);

  bool hasSection(const QString& section) const;
  bool removeSection(const QString& section);

  // Effective view: global + overrides
  QMap<QString, QString> effectiveRunMap(const QString& methodSection) const;
  QString sourceOfEffectiveKey(const QString& methodSection, const QString& key) const;

private:
  struct Entry {
    // key/value within section
    QMap<QString, QString> kv;
    // original ordering of keys as they appeared
    QStringList keyOrder;
  };

  bool loaded_ = false;
  QString path_;
  QMap<QString, Entry> data_;

  static QString normalizeSection(const QString& s);
  static QString normalizeKey(const QString& k);
};

} // namespace optimsolution_gui
