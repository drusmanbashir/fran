#pragma once
#include "plansmodel.h"

#include <QObject>

namespace Ui {
class MainWindow;
}
class TrainingController : public QObject {
  Q_OBJECT
public:
  explicit TrainingController(Ui::MainWindow *ui, QObject *parent = nullptr);
  void populatePlansCB(PlansModel *plansModel);

private:
  void initGPUSelector();
  Ui::MainWindow *ui_;
};
