#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <itkImage.h>
#include <itkImageRegionIterator.h>

namespace py = pybind11;

template <typename PixelT, unsigned int Dim>
typename itk::Image<PixelT, Dim>::Pointer
numpy_to_itk(py::array_t<PixelT, py::array::c_style | py::array::forcecast> arr)
{
    if (arr.ndim() != Dim) {
        throw std::runtime_error("Expected " + std::to_string(Dim) +
                                 "D array, got " + std::to_string(arr.ndim()));
    }

    using ImageType = itk::Image<PixelT, Dim>;
    using SizeType  = typename ImageType::SizeType;
    using IndexType = typename ImageType::IndexType;
    using RegionType= typename ImageType::RegionType;

    auto buf = arr.request();  // contiguous buffer
    const auto* shape = buf.shape.data();

    // Assume NumPy shape = (z, y, x) for 3D, (y, x) for 2D, etc.
    SizeType size;
    for (unsigned int d = 0; d < Dim; ++d) {
        // ITK index order [x,y,z], NumPy [z,y,x] if Dim==3
        size[d] = static_cast<typename SizeType::SizeValueType>(
            shape[Dim - 1 - d]
        );
    }

    IndexType start;
    start.Fill(0);

    RegionType region;
    region.SetIndex(start);
    region.SetSize(size);

    auto image = ImageType::New();
    image->SetRegions(region);
    image->Allocate();

    auto* data = static_cast<PixelT*>(buf.ptr);
    const std::size_t N = static_cast<std::size_t>(buf.size);

    itk::ImageRegionIterator<ImageType> it(image, region);
    std::size_t i = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++i) {
        if (i >= N) {
            throw std::runtime_error("Buffer size smaller than region size");
        }
        it.Set(data[i]);
    }

    return image;  // smart pointer
}
