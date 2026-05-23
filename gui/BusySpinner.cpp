#include "BusySpinner.h"

#include <QPainter>
#include <QPaintEvent>
#include <QtMath>

namespace optimsolution_gui {

BusySpinner::BusySpinner(QWidget* parent)
    : QWidget(parent) {
  setAttribute(Qt::WA_TransparentForMouseEvents, true);
  setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

  // ~30 FPS
  timer_.setInterval(33);
  connect(&timer_, &QTimer::timeout, this, [this]() {
    angleDeg_ = (angleDeg_ + 20) % 360;
    update();
  });
}

void BusySpinner::start() {
  if (running_) return;
  running_ = true;
  angleDeg_ = 0;
  timer_.start();
  update();
}

void BusySpinner::stop() {
  if (!running_) return;
  running_ = false;
  timer_.stop();
  update();
}

void BusySpinner::paintEvent(QPaintEvent* /*e*/) {
  QPainter p(this);
  p.setRenderHint(QPainter::Antialiasing, true);

  const int w = width();
  const int h = height();
  const int side = qMin(w, h);
  const int penW = qMax(2, side / 10);

  QRectF r((w - side) / 2.0 + penW,
           (h - side) / 2.0 + penW,
           side - 2.0 * penW,
           side - 2.0 * penW);

  // Palette-driven color (supports light/dark themes).
  QPen pen(palette().color(QPalette::WindowText));
  pen.setWidth(penW);
  pen.setCapStyle(Qt::RoundCap);
  p.setPen(pen);
  p.setBrush(Qt::NoBrush);

  if (!running_) {
    p.setOpacity(0.25);
    p.drawEllipse(QPointF(w / 2.0, h / 2.0), penW * 0.8, penW * 0.8);
    return;
  }

  // Draw a 270-degree arc, rotated by angleDeg_.
  const int span = 270 * 16;
  const int start = (90 - angleDeg_) * 16; // near 12 o'clock
  p.drawArc(r, start, -span);
}

} // namespace optimsolution_gui
