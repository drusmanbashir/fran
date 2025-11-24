
// -----------------------------------------------------------------------------
// file: bboxviewer.cxx  (patched)
// -----------------------------------------------------------------------------
#include "bboxviewer.h"
#include "bboxes_info.h"
#include "utils.h" // for readLabelImage<3>()
#include "vtk_labelmap_view.h"

#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <iostream>

BBoxViewer::BBoxViewer(QWidget *parent)
    : QWidget(parent), ui(new Ui::BBoxViewer),
      m_model(new QStandardItemModel(this)) {
  ui->setupUi(this);

  m_model->setHorizontalHeaderLabels({"Label", "Index"});
  ui->treeBBoxes->setModel(m_model);

  connect(ui->btnBrowse, &QPushButton::clicked, this, &BBoxViewer::onBrowse);
  connect(ui->btnLoad, &QPushButton::clicked, this, &BBoxViewer::onLoad);
}

void BBoxViewer::onBrowse() {
  QSettings settings("MyQtWork", "bboxviewer");
  QString startDir = ui->lineEditPath->text();
  if (startDir.isEmpty()) {
    startDir = settings.value("lastDir", QDir::homePath()).toString();
  }

  QString filters = tr("NIfTI (*.nii *.nii.gz *.gz);; All files (*)");
  QString fn = QFileDialog::getOpenFileName(
      this, "Select label image (nifti, nrrd)", startDir, filters, nullptr);
  if (!fn.isEmpty()) {
    ui->lineEditPath->setText(fn);
    QFileInfo fi(fn);
    settings.setValue("lastDir", fi.absolutePath());
  }
}

void BBoxViewer::onLoad() {
  const QString fn = ui->lineEditPath->text();
  if (fn.isEmpty()) {
    QMessageBox::warning(this, "Error", "No file selected");
    return;
  }

  try {
    m_bbi = process_file<3>(fn.toStdString());
  } catch (const std::exception &e) {
    QMessageBox::critical(this, "Error", e.what());
    return;
  }

  ui->caseID->setText(QString::fromStdString(m_bbi.case_id));
  populateTree();

  // Show labelmap as 3D volume in VTK view
  try {
    auto img = readLabelImage<3>(fn.toStdString());
    if (ui->viewPort) { // name in .ui (promoted widget)
      ui->viewPort->setLabelImage(img);
    }
  } catch (const std::exception &e) {
    std::cerr << "VTK view error: " << e.what() << "\n";
  }
}

void BBoxViewer::populateTree() {
  if (!m_model) {
    return;
  }

  QStringList header;
  header << "Label" << "Obj#" << "IdxX" << "IdxY" << "IdxZ" << "SizeX"
         << "SizeY" << "SizeZ" << "Cx" << "Cy" << "Cz";

  m_model->clear();
  m_model->setHorizontalHeaderLabels(header);
  const int numStatObjects = m_bbi.bbox_stats.size();

  for (int s = 0; s < numStatObjects; ++s) {
    const BBoxStats<3> &stats = m_bbi.bbox_stats[s];

    const int numBBoxes = stats.bounding_boxes.size();
    QStandardItem *labelItem = new QStandardItem(QString::number(stats.label));
    QStandardItem *countItem = new QStandardItem(QString::number(numBBoxes));

    QList<QStandardItem *> parentRow;
    parentRow << labelItem << countItem;
    for (int c = 0; c < 9; ++c) {
      QStandardItem *emptyItem = new QStandardItem();
      parentRow << emptyItem;
    }

    m_model->appendRow(parentRow);

    // Child rows: one per bounding box
    const int numBoxes = static_cast<int>(stats.bounding_boxes.size());
    const int numCentroids = static_cast<int>(stats.centroids.size());
    const int numEntries = (numBoxes < numCentroids) ? numBoxes : numCentroids;

    for (int b = 0; b < numEntries; ++b) {
      const std::array<int, 6> &bbox =
          stats.bounding_boxes[static_cast<std::size_t>(b)];
      const std::array<float, 3> &centroid =
          stats.centroids[static_cast<std::size_t>(b)];

      QStandardItem *childLabelItem =
          new QStandardItem(QString::number(stats.label));
      QStandardItem *objIndexItem = new QStandardItem(QString::number(b));

      QStandardItem *idxXItem = new QStandardItem(QString::number(bbox[0]));
      QStandardItem *idxYItem = new QStandardItem(QString::number(bbox[1]));
      QStandardItem *idxZItem = new QStandardItem(QString::number(bbox[2]));

      QStandardItem *sizeXItem = new QStandardItem(QString::number(bbox[3]));
      QStandardItem *sizeYItem = new QStandardItem(QString::number(bbox[4]));
      QStandardItem *sizeZItem = new QStandardItem(QString::number(bbox[5]));

      QStandardItem *cxItem = new QStandardItem(QString::number(centroid[0]));
      QStandardItem *cyItem = new QStandardItem(QString::number(centroid[1]));
      QStandardItem *czItem = new QStandardItem(QString::number(centroid[2]));

      QList<QStandardItem *> childRow;
      childRow << childLabelItem << objIndexItem << idxXItem << idxYItem
               << idxZItem << sizeXItem << sizeYItem << sizeZItem << cxItem
               << cyItem << czItem;

      // Attach childRow under the parent label item
      labelItem->appendRow(childRow);
    }
  }
  ui->treeBBoxes->expandAll();
}
