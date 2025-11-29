#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QString>
#include <QDir>
#include <iostream>
#include "python_env.h"

#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {

  ui->setupUi(this);
  populateProjects();

  auto& env = PythonEnv::instance();

}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::populateProjects() {

  std::string franStorage = std::getenv("FRAN_COMMON_PATHS");
  std::string conf_fname = franStorage + "/config.yaml";
  YAML::Node cfg = YAML::LoadFile(conf_fname);
  std::string projects_folder = cfg["projects_folder"].as<std::string>();
  // std::cout << projects_folder << std::endl;
  QDir dir (projects_folder.c_str());
  QStringList folders = dir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);

  for (const QString &f : folders){
    ui->projectsMenu->addItem (f);
  }

}
