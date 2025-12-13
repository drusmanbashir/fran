#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDir>
#include <QString>
#include <iostream>

#include "plansmodel.h"
#include <qmessagebox.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  (void) FranProject::instance();
  ui->setupUi(this);
  populateProjects();

  connect(ui->loadProjectBtn, &QPushButton::clicked, this,
          &MainWindow::loadProject);
  connect(ui->showPropsBtn, &QPushButton::clicked, this,
          &MainWindow::showProps);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::populateProjects() {
  std::string franStorage = std::getenv("FRAN_COMMON_PATHS");
  std::string conf_fname = franStorage + "/config.yaml";
  YAML::Node cfg = YAML::LoadFile(conf_fname);
  std::string projects_folder = cfg["projects_folder"].as<std::string>();
  // std::cout << projects_folder << std::endl;
  QDir dir(projects_folder.c_str());
  QStringList folders =
      dir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);

  for (const QString &f : folders) {
    ui->projectsMenu->addItem(f);
  }
}

void MainWindow::loadProject() {
  QString projectName = ui->projectsMenu->currentText();
  try {
    FranProject::instance().loadProject(projectName.toStdString());
    ui->showPropsBtn->setEnabled(true);
  }

  catch (const std::exception &e) {
    QMessageBox::critical(this, "Error", e.what());
    ui->showPropsBtn->setEnabled(false);
  }
  populatePlans();
}

void MainWindow::showProps() {
  std::string dici = FranProject::instance().parseDict();
  QMessageBox::information(nullptr, "Properties", dici.c_str());
}

void MainWindow::populatePlans() {
  if (!FranProject::instance().plan_loaded()) {
    QMessageBox::warning(this, "Error", "No project loaded");
    return;
  }


  PlansModel *model = new PlansModel(FranProject::instance().getPlansDF(), this);
  ui->tableView->setModel(model);

}
