// -----------------------------------------------------------------------------
// file: bboxviewer.h  (patched)
// -----------------------------------------------------------------------------
#pragma once
#include "bboxes_info.h"
#include "ui_bboxviewer.h"
#include <QStandardItemModel>
#include <QWidget>

class BBoxViewer : public QWidget {
  Q_OBJECT
public:
  explicit BBoxViewer(QWidget *parent = nullptr);
  ~BBoxViewer() = default;

private slots:
  void onLoad();
  void onBrowse();

private:
  void populateTree();

  Ui::BBoxViewer *ui;
  QStandardItemModel *m_model;
  CaseBBoxes<3> m_bbi;
};


