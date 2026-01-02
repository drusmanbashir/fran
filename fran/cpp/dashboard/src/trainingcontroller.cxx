#include <QCheckBox>
#include "trainingcontroller.h"
#include "cuda_utils.h"
#include "ui_mainwindow.h"

TrainingController::TrainingController (Ui::MainWindow *ui,  QObject *parent): ui_(ui), QObject(parent) {
  initGPUSelector();
  ui_->kSpinBox->setValue(-1);
  ui_->kSpinBox->setToolTip("Default 0 means, every blob is kept.");

}

void TrainingController::initGPUSelector () {
  int numDevices = getGPUCount();

  for (int i = 0; i < numDevices; i++) {
    QCheckBox* gpuChkBx = new QCheckBox(QString("GPU %1").arg(i));
    ui_->GPULayout->addWidget(gpuChkBx);
    if (i==1 ) {
      gpuChkBx->setChecked(true);
    }
  }
    
}


void TrainingController::populatePlansCB (PlansModel* plansModel) {
  ui_->planTrainCB->clear();
  ui_->planTrainCB->addItems(plansModel->getPlanIDs());
    
}

