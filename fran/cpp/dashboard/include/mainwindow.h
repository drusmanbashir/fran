#pragma once
#include "plansmodel.h"
#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT
public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

  void populateProjects();
  void loadProject();
  void showProps();
  void populatePlans();
  void onRowAction(int row);
  void onPlanAnalyzed();

private:
  Ui::MainWindow *ui;
  PlansModel *m_plansModel;
};

void analyze_resample(int plan_id, int n_procs, bool overwrite = false);
