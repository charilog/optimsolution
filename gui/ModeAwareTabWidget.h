// ModeAwareTabWidget.h
// The Output area's run tabs (MainWindow::outputTabs_) are shared by all run
// modes, but each run-mode tab of the main window shows only the runs that
// belong to it. The other modes' run tabs are hidden (never removed, so every
// tab index -- and with it the index-paired outputRuns_ bookkeeping -- stays
// exactly as it was).
//
// This QTabWidget adds exactly two things (no signals/slots, so no moc needed):
//   * onTabInserted: called for every inserted tab, so MainWindow can tag the
//     tab with the run mode that created it;
//   * currentIndex()/currentWidget() report "no current tab" (-1 / nullptr)
//     while Qt's own current tab is one that is hidden. A mode that has no run
//     tab of its own therefore looks exactly like the original empty Output
//     area to all existing code (e.g. activeOutputTab()/appendLog() never
//     see another mode's tab). Qt itself cannot select "no tab" (-1) while
//     other tabs exist, hence this shadowing.
#pragma once
#include <QTabWidget>
#include <functional>

namespace optimsolution_gui {

class ModeAwareTabWidget : public QTabWidget {
public:
  explicit ModeAwareTabWidget(QWidget* parent = nullptr) : QTabWidget(parent) {}

  std::function<void(int)> onTabInserted;

  // Deliberately NOT virtual overrides (QTabWidget's are not virtual): they
  // shadow the base versions for callers that use this static type.
  int currentIndex() const {
    const int i = QTabWidget::currentIndex();
    return (i >= 0 && !isTabVisible(i)) ? -1 : i;
  }
  QWidget* currentWidget() const {
    const int i = currentIndex();
    return i >= 0 ? widget(i) : nullptr;
  }

protected:
  void tabInserted(int index) override {
    QTabWidget::tabInserted(index);
    if (onTabInserted) onTabInserted(index);
  }
};

} // namespace optimsolution_gui
