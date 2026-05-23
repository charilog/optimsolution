#pragma once
#include <QString>

namespace optimsolution_gui {

class AnsiStrip final {
public:
  // Removes ANSI/VT100 escape sequences and control characters from a text buffer.
  // Preserves \n and \t; normalizes \r to \n.
  static QString strip(const QString& in);

  // Streaming variant: appends chunk, returns cleaned text, keeps any trailing incomplete ESC sequence.
  static QString stripStreaming(const QString& chunk, QString& carry);
};

} // namespace optimsolution_gui
