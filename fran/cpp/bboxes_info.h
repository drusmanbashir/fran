#pragma once
#include "utils.h"
#include <cstdint>
#include <iostream>
#include <itkBinaryImageToShapeLabelMapFilter.h>
#include <itkImage.h>
#include <itkImageRegion.h>
#include <itkLabelImageToShapeLabelMapFilter.h>
#include <itkShapeLabelObject.h>
#include <itkThresholdImageFilter.h>
#include <string>
#include <vector>
using LabelT = uint16_t;
using LabelVectorType = std::vector<LabelT>;
using RegionType = itk::ImageRegion<3>;
// constexpr unsigned Dim = 3;

std::array<float, 3> itk_point_to_array(itk::Point<double, 3> p);
template <unsigned int Dim> using LabelImage = itk::Image<LabelT, Dim>;

template <unsigned int Dim> using UCharImage = itk::Image<unsigned char, Dim>;

template <unsigned int Dim> struct BBoxStats {
  unsigned int label;
  // std::vector<std::array<unsigned int, Dim>> bounding_boxes;
  std::vector<std::array<float, 3>> centroids;
  std::vector<std::array<int, 6>> bounding_boxes;
};

template <unsigned int Dim> struct CaseBBoxes {
  std::vector<uint16_t> labels;
  std::string case_id;
  std::string file_name;
  std::vector<BBoxStats<Dim>> bbox_stats;
  std::vector<BBoxStats<Dim>> get_bboxes(typename LabelImage<Dim>::Pointer img);
};

// Function declarations
template <unsigned int Dim>
CaseBBoxes<Dim> process_image(LabelImage<3>::Pointer img,
                              const std::string &file_name = "");

template <unsigned int Dim>
CaseBBoxes<Dim> process_file(std::string file_name);


template <unsigned int Dim>
CaseBBoxes<Dim> openFile(std::string file_name);
