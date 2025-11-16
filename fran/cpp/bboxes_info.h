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

std::array<float, 3> itk_point_to_array(itk::Point<double, 3> p) {
  std::array<float, 3> a;
  a[0] = p[0];
  a[1] = p[1];
  a[2] = p[2];
  return a;
}

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

template <unsigned int Dim>
std::vector<BBoxStats<Dim>>
CaseBBoxes<Dim>::get_bboxes(typename LabelImage<Dim>::Pointer img) {
  std::vector<BBoxStats<Dim>> bboxes;
  for (LabelT label : this->labels) {
    std::cout << label;
    BBoxStats<Dim> bbox;
    bbox.label = label;
    bboxes.push_back(bbox);
  }
  return bboxes;
}

template <unsigned int Dim>
CaseBBoxes<Dim> process_image(LabelImage<3>::Pointer img,
                              const std::string &file_name = "") {
  CaseBBoxes<3> bbi;
  bbi.file_name = file_name;
  bbi.labels = getUniqueLabels<3>(img);
  bbi.case_id = "crc_CRC004_20190425_CAP1p5";
  // bbi.bbox_stats = bbi.get_bboxes(img);
  std::cout << "\nDone";

  auto th = itk::ThresholdImageFilter<itk::Image<uint16_t, 3>>::New();
  th->SetInput(img);
  for (const auto &lab : bbi.labels) {
    BBoxStats<3> bb{};
    bb.label = lab;
    // th->SetInsideValue(1);

    LabelImagePtr<3> lm_bin = create_binary_image<3>(*img, lab, lab);
    auto fil = itk::BinaryImageToShapeLabelMapFilter<LabelImage<3>,
                                                     LabelMapType<3>>::New();
    fil->SetInput(lm_bin);
    fil->Update();
    auto lmap = fil->GetOutput();

    // working on LabelMap
    auto N = lmap->GetNumberOfLabelObjects();

    if (N == 0) {
      std::cout << "No objects found for label " << lab << "\n";
      continue;
    } else {
      for (unsigned int i = 0; i < N; ++i) {
        auto obj = lmap->GetNthLabelObject(i);
        const RegionType &bbox = obj->GetBoundingBox();
        itk::Point<double, 3> centroid = obj->GetCentroid();
        std::array<float, 3> centroid_arr = itk_point_to_array(centroid);
        std::array<int, 6> bbox_arr;

        const auto indx = bbox.GetIndex();
        const auto size = bbox.GetSize();
        for (int j = 0; j < 3; j++) {
          int indi = static_cast<int>(bbox.GetIndex()[j]);
          bbox_arr[j] = indi;
        }
        for (int j = 0; j < 3; j++) {
          int sz = static_cast<int>(bbox.GetSize()[j]);
          bbox_arr[j + 3] = sz;
        }

        bb.bounding_boxes.push_back(bbox_arr);
        bb.centroids.push_back(centroid_arr);
        // std::cout << "Printing bbox\n";
        // std::cout << bbox;
      }
    }
    bbi.bbox_stats.push_back(bb);
  }
  return bbi;
}

template <unsigned int Dim>
CaseBBoxes<Dim> process_file(std::string file_name) {
  LabelImage<3>::Pointer img = readLabelImage<3>(file_name);
  CaseBBoxes<3> bbi = process_image<3>(img, file_name);
  return bbi;
}


