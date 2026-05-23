#include "FactoryIntrospect.h"
#include <QFile>
#include <QSet>

namespace optimsolution_gui {

static QStringList uniqueSorted(const QStringList& in) {
  QSet<QString> s;
  for (const auto& x : in) {
    const QString t = x.trimmed();
    if (!t.isEmpty()) s.insert(t);
  }
  QStringList out = s.values();
  out.sort(Qt::CaseInsensitive);
  return out;
}

static int findMatchingBrace(const QString& txt, int openBracePos) {
  // Assumes txt[openBracePos] == '{'
  int depth = 0;
  bool inStr = false;
  bool inChar = false;
  bool inLineComment = false;
  bool inBlockComment = false;
  bool escape = false;

  for (int i = openBracePos; i < txt.size(); ++i) {
    const QChar c = txt[i];
    const QChar n = (i + 1 < txt.size()) ? txt[i + 1] : QChar();

    if (inLineComment) {
      if (c == '\n') inLineComment = false;
      continue;
    }
    if (inBlockComment) {
      if (c == '*' && n == '/') { inBlockComment = false; ++i; }
      continue;
    }
    if (!inStr && !inChar) {
      if (c == '/' && n == '/') { inLineComment = true; ++i; continue; }
      if (c == '/' && n == '*') { inBlockComment = true; ++i; continue; }
    }

    if (inStr) {
      if (escape) { escape = false; continue; }
      if (c == '\\') { escape = true; continue; }
      if (c == '"') { inStr = false; continue; }
      continue;
    }
    if (inChar) {
      if (escape) { escape = false; continue; }
      if (c == '\\') { escape = true; continue; }
      if (c == '\'') { inChar = false; continue; }
      continue;
    }

    if (c == '"') { inStr = true; continue; }
    if (c == '\'') { inChar = true; continue; }

    if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0) return i;
    }
  }
  return -1;
}

static QString extractFunctionBody(const QString& txt, const QString& funcName) {
  // Find "funcName(" then the first '{' that follows, then match braces.
  const int p = txt.indexOf(funcName);
  if (p < 0) return QString();

  const int brace = txt.indexOf('{', p);
  if (brace < 0) return QString();

  const int end = findMatchingBrace(txt, brace);
  if (end < 0) return QString();

  return txt.mid(brace + 1, end - brace - 1);
}

static QStringList extractNamesFromBody(const QString& body) {
  // Capture occurrences of: == "name"
  // Note: factory.cpp uses std::string comparisons like if (name == "rastrigin")
  QStringList out;
  int i = 0;
  while (i < body.size()) {
    const int eq = body.indexOf("==", i);
    if (eq < 0) break;

    int j = eq + 2;
    while (j < body.size() && body[j].isSpace()) ++j;
    if (j >= body.size() || body[j] != '"') { i = eq + 2; continue; }

    ++j;
    QString s;
    while (j < body.size()) {
      const QChar c = body[j];
      if (c == '\\') { // skip escapes
        if (j + 1 < body.size()) { s.append(body[j + 1]); j += 2; continue; }
      }
      if (c == '"') break;
      s.append(c);
      ++j;
    }
    if (!s.isEmpty()) out << s;
    i = j + 1;
  }
  return out;
}

FactoryLists readFactoryLists(const QString& factoryCppPath) {
  FactoryLists out;

  QFile f(factoryCppPath);
  if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) return out;

  const QString txt = QString::fromUtf8(f.readAll());

  const QString probBody = extractFunctionBody(txt, "makeProblem");
  const QString methBody = extractFunctionBody(txt, "makeMethod");

  out.problems = uniqueSorted(extractNamesFromBody(probBody));
  out.methods  = uniqueSorted(extractNamesFromBody(methBody));
  return out;
}

} // namespace optimsolution_gui
