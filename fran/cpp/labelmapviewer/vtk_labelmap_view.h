// -----------------------------------------------------------------------------
#pragma once
#include <QVTKOpenGLNativeWidget.h>
#include <QWidget>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include "bboxes_info.h" // for LabelImage<Dim>, LabelT
#include <itkImage.h>

class VTKLabelmapView : public QVTKOpenGLNativeWidget {
  Q_OBJECT
public:
  explicit VTKLabelmapView(QWidget *parent = nullptr);
  // ITK LabelImage<3> from your pipeline
  void setLabelImage(LabelImage<3>::Pointer img);

private:
  vtkSmartPointer<vtkGenericOpenGLRenderWindow> m_renderWindow;
  vtkSmartPointer<vtkRenderer> m_renderer;
};
