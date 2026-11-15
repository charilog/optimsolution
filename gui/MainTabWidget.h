#pragma once
// Main window tab strip.
//
// A QTabWidget whose tab bar visually separates the "tool" tabs (Code Wizard,
// Settings) from the run-mode tabs: a gap + thin divider before the first tool
// tab (no custom fills, strips or text colours, so it is correct in light and
// dark mode). Pure presentation -- no tab is added, removed or
// re-indexed, so every index used by the rest of the GUI stays the same.
//
// Header-only and without Q_OBJECT (no moc needed).

#include <QTabBar>
#include <QTabWidget>
#include <QStylePainter>
#include <QStyleOptionTab>
#include <QPaintEvent>
#include <QColor>
#include <QPalette>

class MainTabBar : public QTabBar {
public:
  explicit MainTabBar(QWidget* parent = nullptr) : QTabBar(parent) {}

  // Tabs with index >= firstToolTab are drawn as "tool" tabs.
  void setFirstToolTab(int index) { firstTool_ = index; updateGeometry(); update(); }

  // Give the strip a little more room than the tabs strictly need, so the
  // scroll arrows only appear when the window really is too narrow.
  QSize sizeHint() const override {
    QSize s = QTabBar::sizeHint();
    s.rwidth() += 48;
    return s;
  }

protected:
  QSize tabSizeHint(int index) const override {
    QSize s = QTabBar::tabSizeHint(index);
    if (index == firstTool_) s.rwidth() += kGap;
    return s;
  }
  QSize minimumTabSizeHint(int index) const override {
    QSize s = QTabBar::minimumTabSizeHint(index);
    if (index == firstTool_) s.rwidth() += kGap;
    return s;
  }

  void paintEvent(QPaintEvent*) override {
    QStylePainter p(this);
    const int cur = currentIndex();

    auto drawTab = [&](int i) {
      QStyleOptionTab opt;
      initStyleOption(&opt, i);
      const bool tool = firstTool_ >= 0 && i >= firstTool_;
      if (tool && i == firstTool_) opt.rect.adjust(kGap, 0, 0, 0);   // keep the gap empty
      p.drawControl(QStyle::CE_TabBarTab, opt);
    };

    // Same order as QTabBar: unselected tabs first, the selected one on top.
    for (int i = 0; i < count(); ++i) if (i != cur) drawTab(i);
    if (cur >= 0) drawTab(cur);

    // Thin divider in the gap between mode tabs and tool tabs.
    if (firstTool_ > 0 && firstTool_ < count()) {
      const QRect r = tabRect(firstTool_);
      const int x = r.left() + kGap / 2;
      QColor line = palette().color(QPalette::Mid);
      p.setPen(line);
      p.drawLine(x, r.top() + 4, x, r.bottom() - 4);
    }
  }

private:
  static constexpr int kGap = 18;
  int    firstTool_ = -1;
};

class MainTabWidget : public QTabWidget {
public:
  explicit MainTabWidget(QWidget* parent = nullptr) : QTabWidget(parent) {
    setTabBar(new MainTabBar(this));
  }
  MainTabBar* mainTabBar() const { return static_cast<MainTabBar*>(tabBar()); }
};
