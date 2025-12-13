#include <torch/torch.h>
#include <itkImportImageFilter.h>
#include <itkImage.h>

template<typename PixelT = float, unsigned int Dim = 3>
typename itk::Image<PixelT, Dim>::Pointer TorchToItk(torch::Tensor t)
{
    using ImageType = itk::Image<PixelT, Dim>;
    using ImportFilterType = itk::ImportImageFilter<PixelT, Dim>;

    // Sanity: tensor should be on CPU and contiguous
    TORCH_CHECK(!t.is_cuda(), "Tensor must be on CPU");
    t = t.contiguous();

    // Expected shape: [H, W, D]  or [1, H, W, D]
    if (t.dim() == 4 && t.size(0) == 1)
        t = t.squeeze(0);
    TORCH_CHECK(t.dim() == 3, "Expected 3D tensor (H,W,D) or (1,H,W,D)");

    const auto H = static_cast<size_t>(t.size(0));
    const auto W = static_cast<size_t>(t.size(1));
    const auto D = static_cast<size_t>(t.size(2));

    // Copy to host numeric buffer (convert dtype if needed)
    torch::Tensor t_cast;
    if (t.dtype() == torch::CppTypeToScalarType<PixelT>())
        t_cast = t;
    else
        t_cast = t.to(torch::CppTypeToScalarType<PixelT>());

    auto* src = t_cast.data_ptr<PixelT>();
    const size_t total = H * W * D;

    // Allocate an ImportImageFilter (takes ownership of a buffer)
    auto importer = ImportFilterType::New();
    typename ImageType::SizeType size = {{W, H, D}};  // ITK uses x,y,z = W,H,D
    typename ImageType::IndexType start; start.Fill(0);
    typename ImageType::RegionType region;
    region.SetIndex(start);
    region.SetSize(size);
    importer->SetRegion(region);

    double spacing[Dim]  = {1.0, 1.0, 1.0};
    double origin[Dim]   = {0.0, 0.0, 0.0};
    importer->SetSpacing(spacing);
    importer->SetOrigin(origin);

    // Allocate a new buffer and copy tensor data into it
    auto* buffer = static_cast<PixelT*>(malloc(total * sizeof(PixelT)));
    std::memcpy(buffer, src, total * sizeof(PixelT));

    importer->SetImportPointer(buffer, total, true); // ITK takes ownership
    importer->Update();
    return importer->GetOutput();
}
