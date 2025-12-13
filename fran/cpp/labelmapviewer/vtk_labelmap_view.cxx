// -----------------------------------------------------------------------------
// file: vtk_labelmap_view.cxx
// -----------------------------------------------------------------------------
#include "vtk_labelmap_view.h"
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkImageImport.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>
#include <vtkPiecewiseFunction.h>
#include <vtkColorTransferFunction.h>
#include <vtkCamera.h>
#include <vtkSmartPointer.h>

VTKLabelmapView::VTKLabelmapView(QWidget *parent)
    : QVTKOpenGLNativeWidget(parent) {
  m_renderWindow =
      vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
  this->setRenderWindow(m_renderWindow);

  m_renderer = vtkSmartPointer<vtkRenderer>::New();
  m_renderWindow->AddRenderer(m_renderer);

  auto interactor = m_renderWindow->GetInteractor();
  if (!interactor) {
    // QVTKOpenGLNativeWidget creates it lazily; make sure style is set once.
    this->renderWindow()->InitializeFromCurrentContext();
    interactor = this->renderWindow()->GetInteractor();
  }

  if (interactor) {
    auto style =
        vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    interactor->SetInteractorStyle(style);
  }

  m_renderer->SetBackground(0.05, 0.05, 0.08);
}

void VTKLabelmapView::setLabelImage(LabelImage<3>::Pointer img) {
  if (!img) {
    return;
  }

  const auto &region = img->GetLargestPossibleRegion();
  const auto &size   = region.GetSize();
  const auto &sp     = img->GetSpacing();

  const int nx = static_cast<int>(size[0]);
  const int ny = static_cast<int>(size[1]);
  const int nz = static_cast<int>(size[2]);

  const auto *buffer = img->GetBufferPointer();
  const std::size_t nvox =
      static_cast<std::size_t>(nx) * ny * nz;

  auto importer = vtkSmartPointer<vtkImageImport>::New();
  importer->SetDataScalarTypeToUnsignedShort();
  importer->SetNumberOfScalarComponents(1);

  // Deep copy ITK buffer into VTK-managed memory
  importer->CopyImportVoidPointer(
      const_cast<LabelT *>(buffer),
      static_cast<vtkIdType>(nvox * sizeof(LabelT)));

  importer->SetWholeExtent(0, nx - 1,
                           0, ny - 1,
                           0, nz - 1);
  importer->SetDataExtentToWholeExtent();
  importer->SetDataSpacing(sp[0], sp[1], sp[2]);
  importer->SetDataOrigin(0.0, 0.0, 0.0);
  importer->Update();

  auto mapper = vtkSmartPointer<vtkSmartVolumeMapper>::New();
  mapper->SetInputConnection(importer->GetOutputPort());

  // Simple labelmap transfer functions:
  auto opacity = vtkSmartPointer<vtkPiecewiseFunction>::New();
  // 0 -> fully transparent, >0 -> opaque-ish
  opacity->AddPoint(0.0,   0.0);
  opacity->AddPoint(1.0,   0.6);
  opacity->AddPoint(255.0, 0.9);

  auto colors = vtkSmartPointer<vtkColorTransferFunction>::New();
  colors->AddRGBPoint(0.0,   0.0, 0.0, 0.0);
  colors->AddRGBPoint(1.0,   0.8, 0.2, 0.2);
  colors->AddRGBPoint(255.0, 1.0, 0.9, 0.6);

  auto volProp = vtkSmartPointer<vtkVolumeProperty>::New();
  volProp->SetScalarOpacity(opacity);
  volProp->SetColor(colors);
  volProp->ShadeOff();
  volProp->SetInterpolationTypeToLinear();

  auto volume = vtkSmartPointer<vtkVolume>::New();
  volume->SetMapper(mapper);
  volume->SetProperty(volProp);

  m_renderer->RemoveAllViewProps();
  m_renderer->AddVolume(volume);
  m_renderer->ResetCamera();
  m_renderWindow->Render();
}


// #include "bboxes_info.h"
//
// // -----------------------------------------------------------------------------
// // file: bboxviewer.cxx  (patched)
// // -----------------------------------------------------------------------------
// #include "bboxviewer.h"
// #include "vtk_labelmap_view.h"
// #include "utils.h"  // for readLabelImage<3>()
//
// #include <QFileDialog>
// #include <QMessageBox>
// #include <QSettings>
// #include <QDir>
// #include <iostream>
//
// BBoxViewer::BBoxViewer(QWidget *parent)
//     : QWidget(parent),
//       ui(new Ui::BBoxViewer),
//       m_model(new QStandardItemModel(this)) {
//   ui->setupUi(this);
//
//   m_model->setHorizontalHeaderLabels({"Label", "Index"});
//   ui->treeBBoxes->setModel(m_model);
//
//   connect(ui->btnBrowse, &QPushButton::clicked,
//           this, &BBoxViewer::onBrowse);
//   connect(ui->btnLoad, &QPushButton::clicked,
//           this, &BBoxViewer::onLoad);
// }
//
// void BBoxViewer::onBrowse() {
//   QSettings settings("MyQtWork", "bboxviewer");
//   QString startDir = ui->lineEditPath->text();
//   if (startDir.isEmpty()) {
//     startDir = settings.value("lastDir", QDir::homePath()).toString();
//   }
//
//   QString filters =
//       tr("NIfTI (*.nii *.nii.gz *.gz);; All files (*)");
//   QString fn = QFileDialog::getOpenFileName(
//       this,
//       "Select label image (nifti, nrrd)",
//       startDir,
//       filters,
//       nullptr);
//   if (!fn.isEmpty()) {
//     ui->lineEditPath->setText(fn);
//     QFileInfo fi(fn);
//     settings.setValue("lastDir", fi.absolutePath());
//   }
// }
//
// void BBoxViewer::onLoad() {
//   const QString fn = ui->lineEditPath->text();
//   if (fn.isEmpty()) {
//     QMessageBox::warning(this, "Error", "No file selected");
//     return;
//   }
//
//   try {
//     m_bbi = process_file<3>(fn.toStdString());
//   } catch (const std::exception &e) {
//     QMessageBox::critical(this, "Error", e.what());
//     return;
//   }
//
//   ui->caseID->setText(QString::fromStdString(m_bbi.case_id));
//   populateTree();
//
//   // Show labelmap as 3D volume in VTK view
//   try {
//     auto img = readLabelImage<3>(fn.toStdString());
//     if (ui->viewPort) {  // name in .ui (promoted widget)
//       ui->viewPort->setLabelImage(img);
//     }
//   } catch (const std::exception &e) {
//     std::cerr << "VTK view error: " << e.what() << "\n";
//   }
// }
//
// void BBoxViewer::populateTree() {
//   if (!m_model) {
//     return;
//   }
//
//   m_model->clear();
//   m_model->setHorizontalHeaderLabels({
//       "Label", "Obj#", "IdxX", "IdxY", "IdxZ",
//       "SizeX", "SizeY", "SizeZ",
//       "Cx", "Cy", "Cz"});
//
//   for (const auto &ls : m_bbi.bbox_stats) {
//     auto *labelItem =
//         new QStandardItem(QString::number(ls.label));
//     auto *countItem =
//         new QStandardItem(QString::number(ls.bounding_boxes.size()));
//
//     QList<QStandardItem *> parentRow;
//     parentRow << labelItem << countItem;
//     for (int i = 0; i < 9; ++i) {
//       parentRow << new QStandardItem();
//     }
//     m_model->appendRow(parentRow);
//   }
// }
