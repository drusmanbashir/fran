#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkRelabelComponentImageFilter.h"
#include "itkLabelGeometryImageFilter.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "protot/utils.h"

#include <iostream>
#include <vector>
#include <set>
#include <cstdlib>
#include "torch_to_itk.h"

using LabelT = uint16_t;
// constexpr unsigned Dim = 3;
using LabelImage = itk::Image<LabelT, Dim>;
using UCharImage = itk::Image<unsigned char, Dim>;

int main() {
    // std::string file_name = "/home/ub/code/fran/fran/cpp/files/sample.pt";
    // std::string file_name = "/s/fran_storage/datasets/raw_data/lidc/lms/lidc_0030.nii.gz";
    std::string file_name = "/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/lidc2_0021.pt";
    // torch::jit::script::Module container = torch::jit::load(file_name);
    // torch::Tensor img = container.attr("tnsr").toTensor();
    torch::Tensor img;
    torch::load(img, file_name);
    // 
    std::cout<<img.sizes();
    // auto img = readLabelImage(file_name);
    std::cout<<"\n\nYO add\n\n";
    // std::cout<<imgt>GetSpacing();
    // runGeometry(img);

  
      
    // LabelImage::Pointer li = TorchToItk<LabelT, Dim>(img);
    // LabelImage::Pointer label_image = TorchToItk<LabelT, Dim>(img);
    // std::string fileName2 =  "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz"
    // std::cout<<label_image->GetSpacing();

    std::cout<<"\n\n\n\n\n";
}
