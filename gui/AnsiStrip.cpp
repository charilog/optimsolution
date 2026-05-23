#include "AnsiStrip.h"

namespace optimsolution_gui {

static inline bool isControl(ushort u) {
  return (u < 0x20) || (u == 0x7F);
}

static inline bool isFinalCsiByte(ushort u) {
  return (u >= '@' && u <= '~');
}

static inline bool isOscDcsTerminatorAt(const QString& s, int i) {
  // Terminator for OSC/DCS/PM/APC can be BEL or ESC '\'
  const ushort u = s[i].unicode();
  if (u == 0x07) return true; // BEL
  if (u == 0x1B && i + 1 < s.size() && s[i + 1].unicode() == '\\') return true;
  return false;
}

static QString stripImpl(const QString& s, QString* carryOut) {
  QString out;
  out.reserve(s.size());

  int i = 0;
  while (i < s.size()) {
    const ushort u = s[i].unicode();

    if (u == '\r') {
      out.append('\n');
      ++i;
      continue;
    }

    if (u == 0x1B) { // ESC
      if (i + 1 >= s.size()) {
        if (carryOut) *carryOut = s.mid(i);
        break;
      }
      const ushort n = s[i + 1].unicode();

      // CSI: ESC [
      if (n == '[') {
        int j = i + 2;
        while (j < s.size() && !isFinalCsiByte(s[j].unicode())) ++j;
        if (j >= s.size()) {
          if (carryOut) *carryOut = s.mid(i);
          break;
        }
        i = j + 1; // skip final byte too
        continue;
      }

      // OSC/DCS/PM/APC: ESC ] / P / ^ / _
      if (n == ']' || n == 'P' || n == '^' || n == '_') {
        int j = i + 2;
        while (j < s.size() && !isOscDcsTerminatorAt(s, j)) ++j;
        if (j >= s.size()) {
          if (carryOut) *carryOut = s.mid(i);
          break;
        }
        // If terminator is ESC '\', consume both.
        if (s[j].unicode() == 0x1B && j + 1 < s.size() && s[j + 1].unicode() == '\\') j += 2;
        else j += 1; // BEL
        i = j;
        continue;
      }

      // Single-character ESC sequence: ESC <final>
      // Skip ESC + next char.
      i += 2;
      continue;
    }

    // Other control characters: drop except \n and \t
    if (u == '\n' || u == '\t') {
      out.append(s[i]);
      ++i;
      continue;
    }

    if (isControl(u)) {
      ++i;
      continue;
    }

    out.append(s[i]);
    ++i;
  }

  return out;
}

QString AnsiStrip::strip(const QString& in) {
  return stripImpl(in, nullptr);
}

QString AnsiStrip::stripStreaming(const QString& chunk, QString& carry) {
  const QString combined = carry + chunk;
  carry.clear();
  QString newCarry;
  const QString cleaned = stripImpl(combined, &newCarry);
  carry = newCarry;
  return cleaned;
}

} // namespace optimsolution_gui
