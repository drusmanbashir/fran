// -----------------------------------------------------------------------------
#pragma once

#include <QWidget>
#include <QVTKOpenGLNativeWidget.h>

#include <vtkSmartPointer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>

#include <itkImage.h>
#include "bboxes_info.h"  // for LabelImage<Dim>, LabelT

class VTKLabelmapView : public QVTKOpenGLNativeWidget {
  Q_OBJECT
public:
  explicit VTKLabelmapView(QWidget *parent = nullptr);

  // ITK LabelImage<3> from your pipeline
  void setLabelImage(LabelImage<3>::Pointer img);

private:
  vtkSmartPointer<vtkGenericOpenGLRenderWindow> m_renderWindow;
  vtkSmartPointer<vtkRenderer>                 m_renderer;
};

