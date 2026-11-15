#pragma once
// A toolbar row that wraps onto further rows instead of demanding the sum of
// all its buttons as minimum width.
//
// Used for the button row at the top of the Output area: with a plain
// QHBoxLayout the row needs ~900 px in the sensitivity / batch modes (Run,
// Load experiment CSV, Refresh plots, Delete output files, Open output
// folder, Maximize), which kept the whole window from becoming narrower. Here
// the minimum width is only that of the widest single button; when the
// window is wide enough everything is on one line, exactly like before
// (items before addStretch() on the left, the rest right-aligned).
//
// Header-only, no Q_OBJECT.

#include <QLayout>
#include <QLayoutItem>
#include <QList>
#include <QRect>
#include <QSize>
#include <QWidget>
#include <algorithm>

class WrapToolbarLayout : public QLayout {
public:
  explicit WrapToolbarLayout(int hSpacing = 8, int vSpacing = 4)
    : hSpacing_(hSpacing), vSpacing_(vSpacing) {}
  ~WrapToolbarLayout() override {
    while (!items_.isEmpty()) delete items_.takeFirst();
  }

  // Same call names the old QHBoxLayout code used.
  void addStretch(int = 0) { splitIndex_ = items_.size(); }   // what follows is right-aligned
  void addSpacing(int) {}                                      // gaps come from the layout spacing

  void addItem(QLayoutItem* item) override { items_.append(item); }
  int count() const override { return items_.size(); }
  QLayoutItem* itemAt(int index) const override { return items_.value(index, nullptr); }
  QLayoutItem* takeAt(int index) override {
    if (index < 0 || index >= items_.size()) return nullptr;
    if (index < splitIndex_) --splitIndex_;
    return items_.takeAt(index);
  }

  Qt::Orientations expandingDirections() const override { return Qt::Orientations(); }
  bool hasHeightForWidth() const override { return true; }
  int heightForWidth(int width) const override { return doLayout(QRect(0, 0, width, 0), true); }

  void setGeometry(const QRect& rect) override {
    QLayout::setGeometry(rect);
    doLayout(rect, false);
  }

  QSize sizeHint() const override {
    // Everything on one line.
    int w = 0, h = 0, n = 0;
    for (QLayoutItem* it : items_) {
      if (it->isEmpty()) continue;
      const QSize s = it->sizeHint();
      w += s.width() + (n++ ? hSpacing_ : 0);
      h = std::max(h, s.height());
    }
    return QSize(w, h);
  }

  QSize minimumSize() const override {
    // Only the widest single item has to fit.
    int w = 0, h = 0;
    for (QLayoutItem* it : items_) {
      if (it->isEmpty()) continue;
      const QSize s = it->minimumSize();
      w = std::max(w, s.width());
      h = std::max(h, s.height());
    }
    return QSize(w, h);
  }

private:
  struct Entry { QLayoutItem* item; bool right; QSize size; };

  // Lays the visible items out in rows no wider than rect.width(); returns the
  // total height. With testOnly the items are not moved.
  int doLayout(const QRect& rect, bool testOnly) const {
    QList<Entry> entries;
    for (int i = 0; i < items_.size(); ++i) {
      QLayoutItem* it = items_[i];
      if (it->isEmpty()) continue;
      entries.append({it, i >= splitIndex_, it->sizeHint()});
    }

    const int avail = rect.width();
    int y = rect.y();
    int e = 0;
    bool firstRow = true;
    while (e < entries.size()) {
      // Take as many entries as fit on this row (at least one).
      int rowW = 0, rowH = 0;
      const int rowStart = e;
      while (e < entries.size()) {
        const int add = entries[e].size.width() + (e > rowStart ? hSpacing_ : 0);
        if (e > rowStart && rowW + add > avail) break;
        rowW += add;
        rowH = std::max(rowH, entries[e].size.height());
        ++e;
      }
      if (!firstRow) y += vSpacing_;
      firstRow = false;

      if (!testOnly) {
        // Left-group entries from the left edge, right-group entries from the
        // right edge (in their original order).
        int rightW = 0, nRight = 0;
        for (int k = rowStart; k < e; ++k)
          if (entries[k].right) rightW += entries[k].size.width() + (nRight++ ? hSpacing_ : 0);
        int xl = rect.x();
        int xr = rect.x() + std::max(0, avail - rightW);
        // keep the right group clear of the left one
        int leftW = 0, nLeft = 0;
        for (int k = rowStart; k < e; ++k)
          if (!entries[k].right) leftW += entries[k].size.width() + (nLeft++ ? hSpacing_ : 0);
        if (nLeft && nRight) xr = std::max(xr, rect.x() + leftW + hSpacing_);
        for (int k = rowStart; k < e; ++k) {
          const Entry& en = entries[k];
          const int yy = y + (rowH - en.size.height()) / 2;
          int& x = en.right ? xr : xl;
          const int w = std::min(en.size.width(), std::max(1, avail));
          en.item->setGeometry(QRect(x, yy, w, en.size.height()));
          x += w + hSpacing_;
        }
      }
      y += rowH;
    }
    return y - rect.y();
  }

  QList<QLayoutItem*> items_;
  int splitIndex_ = 1 << 30;   // index of the first right-aligned item
  int hSpacing_;
  int vSpacing_;
};
