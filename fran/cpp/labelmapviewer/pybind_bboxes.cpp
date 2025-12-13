#include "bboxes_info.h"
#include "numpy_to_itk.h"
#include <cstdint>
#include <itkBinaryImageToShapeLabelMapFilter.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

static py::dict image_to_bboxes(const itk::Image<uint16_t, 3>::Pointer img,
                                const std::string &file_name = "") {
  CaseBBoxes<3> bbi = process_image<3>(img, file_name);
  py::dict out;
  out["file_name"] = bbi.file_name;
  out["labels"] = bbi.labels;

  py::list rows; // <<---- final flat table

  int cc_id = 0;

  for (const auto &bb : bbi.bbox_stats) {
    const int label = bb.label;

    const auto &centroids = bb.centroids;
    const auto &boxes = bb.bounding_boxes;

    const size_t n = std::min(centroids.size(), boxes.size());

    for (size_t i = 0; i < n; ++i) {
      ++cc_id;

      py::dict row;
      row["label"] = label;
      row["label_cc"] = static_cast<int>(cc_id);
      row["centroid"] = centroids[i]; // vector<float,3>
      row["bbox"] = boxes[i];         // array<int,6>

      rows.append(row);
    }
  }

  out["rows"] = rows;
  return out;
}

// torch.Tensor -> ITK label image -> bboxes

static py::dict process_file_py(const std::string &file_name) {
  auto img = itk::ReadImage<itk::Image<uint16_t, 3>>(file_name);
  return image_to_bboxes(img, file_name);
}

static py::dict numpy_to_bboxes(
    py::array_t<uint16_t, py::array::c_style | py::array::forcecast> arr,
    py::object spacing_obj = py::none(), py::object origin_obj = py::none(),
    const std::string &file_name = ""

) {

  auto img = numpy_to_itk<uint16_t, 3>(arr);
  if (!spacing_obj.is_none()) {
    auto seq = spacing_obj.cast<py::sequence>();
    if (py::len(seq) != 3) {
      throw std::runtime_error("Spacing must be a sequence of length 3");
    }

    itk::Image<uint16_t, 3>::SpacingType sp;
    for (unsigned d = 0; d < 3; ++d) {
      sp[d] = seq[d].cast<double>();
    }
    img->SetSpacing(sp);
  }

  // optional origin
  if (!origin_obj.is_none()) {
    auto seq = origin_obj.cast<py::sequence>();
    if (py::len(seq) != 3) {
      throw std::runtime_error("origin must be length-3 sequence");
    }
    itk::Image<uint16_t, 3>::PointType org;
    for (unsigned int d = 0; d < 3; ++d) {
      org[d] = seq[d].cast<double>();
    }
    img->SetOrigin(org);
  }
  return image_to_bboxes(img, file_name);
}

PYBIND11_MODULE(fran_hello, m) {
  m.doc() = "minimal bbox extractor";
  m.def("process_file_py", &process_file_py);
  m.def("image_to_bboxes", &image_to_bboxes);
  m.def("numpy_to_bboxes", &numpy_to_bboxes, py::arg("arr"),
        py::arg("spacing") = py::none(), py::arg("origin") = py::none(),
        py::arg("file_name") = "");
}
