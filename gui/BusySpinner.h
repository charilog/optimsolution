#pragma once

#include <QWidget>
#include <QTimer>

namespace optimsolution_gui {

/**
 * A lightweight animated spinner widget to indicate a running task.
 * It draws a rotating arc using a timer-driven repaint.
 */
class BusySpinner final : public QWidget {
  Q_OBJECT
public:
  explicit BusySpinner(QWidget* parent = nullptr);

  void start();
  void stop();

  bool isRunning() const noexcept { return running_; }

protected:
  void paintEvent(QPaintEvent* e) override;

private:
  QTimer timer_;
  bool   running_ = false;
  int    angleDeg_ = 0;   // rotation angle
};

} // namespace optimsolution_gui
