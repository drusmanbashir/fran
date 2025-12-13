#include "itkImage.h"
#include "protot/utils.h"     // brings in ITK + Slicer stuff (and Qt) :contentReference[oaicite:1]{index=1}

#include <QApplication>
#include <set>
#include <vector>

// Qt defines 'slots' as a macro, which breaks LibTorch's IValue::slots()
// #ifdef slots
// #undef slots
// #endif

// #include "torch/script.h"
// #include "torch/torch.h"

#include "bboxes_info.h"
// #include "torch_to_itk.h"
#include "bboxviewer.h"
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    // std::string file_name =
    //   "/r/datasets/preprocessed/lidc2/lbd/spc_080_080_150/lms/lidc2_0021.pt";
    std::string file_name =
        "/s/xnat_shadow/crc/lms/crc_CRC004_20190425_CAP1p5.nii.gz";

    // Run your ITK label processing
    CaseBBoxes<3> bbi = process_file<3>(file_name);

    // TODO: pass 'bbi' into the viewer when you add a setter
    BBoxViewer viewer;
    viewer.show();

    return app.exec();
}

