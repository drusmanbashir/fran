#include "itkImage.h"
#include "protot/utils.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "bboxes_info.h"
#include "torch_to_itk.h"
#include <cstdint>
#include <cstdlib>

#include <set>
#include <vector>
int main() {
  // std::string file_name =
  // "/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/lidc2_0021.pt";
  std::string file_name =
      "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz";


  // LabelImage<3>::Pointer img = readLabelImage<3>(file_name);
  CaseBBoxes<3> bbi =  process_file<3>(file_name);
  // bbi.file_name = file_name;
  // bbi.labels = getUniqueLabels<3>(img);
  // bbi.case_id = "crc_CRC004_20190425_CAP1p5";
  // bbi.bbox_stats = bbi.get_bboxes(img);
  // std::cout << "\nDone";
  //
  // auto th = itk::ThresholdImageFilter<itk::Image<uint16_t, 3>>::New();
  // th->SetInput(img);
  // for (const auto &lab : bbi.labels) {
  //   BBoxStats<3> bb{};
  //   bb.label = lab;
  //   // th->SetInsideValue(1);
  //
  //   LabelImagePtr<3> lm_bin = create_binary_image<3>(*img, lab, lab);
  //   auto fil = itk::BinaryImageToShapeLabelMapFilter<LabelImage<3>,
  //                                                    LabelMapType<3>>::New();
  //   fil->SetInput(lm_bin);
  //   fil->Update();
  //   auto lmap = fil->GetOutput();
  //
  //   // working on LabelMap
  //   auto N = lmap->GetNumberOfLabelObjects();
  //
  //   if (N == 0) {
  //     std::cout << "No objects found for label " << lab << "\n";
  //     continue;
  //   } else {
  //     for (unsigned int i = 0; i < N; ++i) {
  //       auto obj = lmap->GetNthLabelObject(i);
  //       const RegionType &bbox = obj->GetBoundingBox();
  //       itk::Point<double, 3> centroid = obj->GetCentroid();
  //       std::array<int, 6> bbox_arr;
  //       std::array<float, 3> centroid_arr;
  //       const auto indx = bbox.GetIndex();
  //       const auto size = bbox.GetSize();
  //       for (int j = 0; j < 3; j++) {
  //         int indi = static_cast<int>(bbox.GetIndex()[j]);
  //         bbox_arr[j] = indi;
  //       }
  //       for (int j = 0; j < 3; j++) {
  //         int sz = static_cast<int>(bbox.GetSize()[j]);
  //         bbox_arr[j + 3] = sz;
  //       }
  //       bb.bounding_boxes.push_back(bbox_arr);
  //       std::cout << "Printing bbox\n";
  //       std::cout << bbox;
  //     }
  //   }
  //   bbi.bbox_stats.push_back(bb);
  // }
  std::cout << "Done\n\nDone\n\n\n";
}

// save_bboxes(bbi);
// std::cout<<img->GetSpacing();

// std::cout<<imgt>GetSpacing();
// runGeometry(img);

// LabelImage::Pointer li = TorchToItk<LabelT, Dim>(img);
// LabelImage::Pointer label_image = TorchToItk<LabelT, Dim>(img);
// std::string fileName2 =
// "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz"
// std::cout<<label_image->GetSpacing();
