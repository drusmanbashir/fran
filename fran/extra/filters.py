# %%
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from utilz.fileio import maybe_makedirs
from utilz.imageviewers import ImageMaskViewer

from fran.transforms.totensor import ToTensorT


def create_z_average_kernel(kernel_size_z=3):
    """Create a 3D averaging kernel that only averages in z-direction while preserving x,y.

    Args:
        kernel_size_z (int): Size of the kernel in z dimension

    Returns:
        torch.Tensor: Kernel of shape [1, 1, k, 1, 1] that averages only in z-direction
    """
    # Create a kernel that's only active in z direction
    # Shape: [1, 1, kernel_size_z, 1, 1]
    kernel = torch.ones(1, 1, kernel_size_z, 1, 1)
    kernel = kernel / kernel_size_z  # Normalize only by z dimension

    return kernel


class ZAverage3D(nn.Module):
    def __init__(self, kernel_size_z=3, padding="same"):
        super().__init__()
        self.kernel_size_z = kernel_size_z
        self.padding = padding
        # Create and register the z-averaging kernel
        kernel = create_z_average_kernel(kernel_size_z)
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        # Handle each channel separately to maintain independence
        B, C = x.shape[:2]
        x_reshaped = x.view(B * C, 1, *x.shape[2:])
        # Compute padding if 'same' - only pad in z direction
        if self.padding == "same":
            pad_z = self.kernel_size_z // 2
            padding = (0, 0, pad_z)  # (W, H, D) format
        else:
            padding = 0
        # Apply 3D convolution with the z-averaging kernel
        out = F.conv3d(x_reshaped, self.kernel, padding=padding)
        # Reshape back to original dimensions
        out = out.view(B, C, *out.shape[2:])
        return out


class ZStrideAvgPool3d(nn.Module):
    def __init__(self, stride_z=2):
        super().__init__()
        self.stride_z = stride_z
        # Create a 3D kernel that averages only in z direction
        # kernel shape: [1, 1, stride_z, 1, 1]
        kernel = torch.ones(1, 1, stride_z, 1, 1) / stride_z
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        # x shape: [B, C, D, H, W]
        # Apply convolution with stride only in z direction
        return F.conv3d(x, self.kernel, stride=(self.stride_z, 1, 1), padding=(0, 0, 0))


# %%
if __name__ == "__main__":
    # Create a sample 3D tensor
    sample = torch.randn(1, 1, 16, 32, 32)  # [B, C, D, H, W]
    img_fn = Path(
        "/s/xnat_shadow/crc/images_more/images/crc_CRC333_20150117_ABDOMEN.nii.gz"
    )
    tmpfldr = Path("/s/xnat_shadow/tmp")
    maybe_makedirs(tmpfldr)

    img = sitk.ReadImage(img_fn)
    original_spacing = img.GetSpacing()
    original_origin = img.GetOrigin()
    original_direction = img.GetDirection()

    # Convert SimpleITK image to tensor using ToTensorT transform
    totensor = ToTensorT()
    img_tn = totensor.encodes(img)
    img_tn = img_tn.float()
    img_tn2 = img_tn.unsqueeze(0).unsqueeze(0)
# %%
    nslice = img_tn2.shape[2]
    B = 1

    inslice = 512
    img_tn2 = img_tn2.permute(0, 3, 4, 1, 2)
# %%
    img_1d = img_tn2.reshape(-1, 1, nslice)
    C_out = 1
    kernel_size = 5
    stride = 2
    avg_filter2 = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        bias=False,
        padding=1,
        stride=stride,
    )
    conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
    conv.weight.data.fill_(1.0 / kernel_size)
    out = conv(img_1d)
    out2 = out.reshape(B, inslice, inslice, 1, -1)
    out2 = out2.detach()
    out3 = out2.squeeze(0, 3)
    out3 = out3.permute(2, 0, 1).contiguous()  # out3.shape
    print(out3.shape)
    ImageMaskViewer([img_tn, out3], dtypes="ii")
# %%
    new_spacing = list(original_spacing)
    new_spacing[2] *= stride  # z-spacing
    out_img = sitk.GetImageFromArray(out3)
    out_img.SetSpacing(new_spacing)
    out_img.SetOrigin(original_origin)
    out_img.SetDirection(original_direction)
    out_fn = tmpfldr / (img_fn.name)
    out_fn = out_fn.str_replace(".nii.gz", "thicker.nii.gz")

    sitk.WriteImage(out_img, out_fn)
    # 7. Save or view
    sitk.WriteImage(out_img, "downsampled_output.nii.gz")
    print("Saved downsampled image with spacing:", out_img.GetSpacing())
# %%
    # Add batch dimension if needed
    if len(img_tn2.shape) == 3:
        img_tn2 = img_tn2.unsqueeze(0).unsqueeze(0)  # [B, C, D, H, W]

    # Test Z-stride averaging
    z_pool = ZStrideAvgPool3d(stride_z=2)
    z_output = z_pool(img_tn2)
    print("\nZ-stride averaging:")
    print(f"Input shape: {img_tn2.shape}")
    print(f"Output shape: {z_output.shape}")
# %%

    # Test Z-only average filtering
    z_avg = ZAverage3D(kernel_size_z=3, padding="same")
    z_avg_output = z_avg(img_tn2)
    print("\nZ-only average filtering:")
    print(f"Input shape: {img_tn2.shape}")
    print(f"Output shape: {z_avg_output.shape}")

    # Verify that x,y dimensions are unchanged
    print("\nVerifying x,y preservation:")
    test_slice = torch.ones(1, 1, 5, 3, 3)  # Small tensor for easy verification
    z_avg_test = ZAverage3D(kernel_size_z=3, padding="valid")(test_slice)
    print("Single slice values (should be identical across z):")
    print(z_avg_test[0, 0, 0, :, :])  # Show one z-slice
# %%
