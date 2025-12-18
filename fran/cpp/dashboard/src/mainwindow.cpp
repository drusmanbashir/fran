#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDir>
#include <QMessageBox>
#include <QProcess>
#include <QString>
#include <iostream>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  (void)FranProject::instance();
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

void MainWindow::onRowAction(int row) {
  QModelIndex idx = m_plansModel->index(row, 0);
  QVariant plan_id = m_plansModel->data(idx, Qt::DisplayRole);
  ;
  QString message =
      QString("Row: %1\nPlanID: %2").arg(row).arg(plan_id.toString());
  QMessageBox::warning(this, "Message", message);
  std::cout << row;
  FranProject::instance().run_analyze_resample(plan_id.toInt(), 8, false);
};

void MainWindow::populatePlans() {
  if (!FranProject::instance().plan_loaded()) {
    QMessageBox::warning(this, "Error", "No project loaded");
    return;
  }
  m_plansModel = new PlansModel(FranProject::instance().getPlansDF(), this);
  ui->tableView->setModel(m_plansModel);
  const int actionCol = m_plansModel->columnCount(QModelIndex()) - 1;
  auto *delegate = new AnalyzeButtonDelegate(ui->tableView, actionCol);
  ui->tableView->setItemDelegateForColumn(actionCol, delegate);

  connect(delegate, &AnalyzeButtonDelegate::clickedRow, this,
          [this](int row) { onRowAction(row); });

  ui->tableView->setMouseTracking(true);
  ui->tableView->setColumnWidth(actionCol, 110);
}

void analyze_resample( int plan_id, int n_procs,
                      bool overwrite ) {

  std::string project_title_c = FranProject::instance().project_title();
  QString project_title = QString::fromStdString(project_title_c);

  QString python = "/home/ub/mambaforge/envs/dl/bin/python";
  QProcess *proc = new QProcess;
  QStringList args = {"/home/ub/code/fran/fran/run/analyze_resample.py",
                      "-t",
                      project_title,
                      "-p",
                      QString::number(plan_id),
                      "-n",
                      QString::number(n_procs),
                      "-o"};
  if (overwrite)
    args << "-o";
  QProcess::startDetached(python, args);
}
