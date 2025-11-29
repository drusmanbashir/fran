#pragma once
#include <QMainWindow>
// #include "python_env.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void populateProjects();
    void loadProject();

private:
    Ui::MainWindow *ui;
};
