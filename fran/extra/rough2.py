# %%

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# pyright: reportAttributeAccessIssue=false
from skimage import data, img_as_float


# Load the "camera" image from scikit-image.
image = img_as_float(data.camera())
H, W = image.shape  # e.g., typically 512x512

# Convert the image to a PyTorch tensor with shape (1, 1, H, W)
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

##############################
# 1D Horizontal Convolution
##############################
# For a horizontal 1D convolution, treat each row as a 1D signal.
# Reshape the image to (H, 1, W): H samples, 1 channel, W-length sequence.
img_for_conv1d = image_tensor.squeeze(0).squeeze(0).unsqueeze(1)  # Shape: (H, 1, W)

# Define a 1D convolution layer with kernel size 3 and padding=1.
conv1d_horizontal = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
# Set the kernel to [-1, 0, 1] for horizontal edge detection.
with torch.no_grad():
    conv1d_horizontal.weight[:] = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32)

# Apply the horizontal 1D convolution
result1d_horizontal = conv1d_horizontal(img_for_conv1d)  # Shape: (H, 1, W)

##############################
# 1D Vertical Convolution
##############################
# To perform a vertical convolution, treat each column as a 1D signal.
# First, transpose the image so that columns become rows.
# Starting from shape (H, W), transposing gives (W, H), then add a channel dimension.
img_for_conv1d_vertical = image_tensor.squeeze(0).squeeze(0).transpose(0, 1).unsqueeze(1)  # Shape: (W, 1, H)

# Define a 1D convolution layer for vertical processing.
conv1d_vertical = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
# Use the same kernel [-1, 0, 1].
with torch.no_grad():
    conv1d_vertical.weight[:] = torch.tensor([[[-1, 0, 1]]], dtype=torch.float32)

# Apply the vertical 1D convolution.
result1d_vertical = conv1d_vertical(img_for_conv1d_vertical)  # Shape: (W, 1, H)
# Transpose the result back to (H, W)
result1d_vertical = result1d_vertical.squeeze(1).transpose(0, 1)

##############################
# 2D Convolution
##############################
# Define a 2D convolution layer with kernel size 3 and padding=1.
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
# Create a 2D kernel that applies the horizontal filter along the center row.
kernel_2d = torch.zeros((1, 1, 3, 3))
kernel_2d[0, 0, 1, :] = torch.tensor([-1, 0, 1], dtype=torch.float32)
with torch.no_grad():
    conv2d.weight[:] = kernel_2d

# Apply the 2D convolution on the original image tensor.
result2d = conv2d(image_tensor)  # Shape: (1, 1, H, W)

##############################
# Display the Results
##############################
plt.figure(figsize=(18, 5))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Camera Image")
plt.axis("off")

# 1D Horizontal Convolution
plt.subplot(1, 4, 2)
plt.imshow(result1d_horizontal.squeeze(1).detach().numpy(), cmap='gray')
plt.title("1D Convolution (Horizontal)")
plt.axis("off")

# 1D Vertical Convolution
plt.subplot(1, 4, 3)
plt.imshow(result1d_vertical.detach().numpy(), cmap='gray')
plt.title("1D Convolution (Vertical)")
plt.axis("off")

# 2D Convolution
plt.subplot(1, 4, 4)
plt.imshow(result2d.squeeze().detach().numpy(), cmap='gray')
plt.title("2D Convolution")
plt.axis("off")

plt.tight_layout()
plt.show()
# %%
