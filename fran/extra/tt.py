
# %%
import ipdb
tr = ipdb.set_trace

import torch
import torch.nn.functional as F

# Softmax derivative (Jacobian matrix of softmax)
def softmax_derivative(softmax_output):
    """
    Compute the derivative (Jacobian matrix) of the softmax function.

    Args:
    - softmax_output (torch.Tensor): The output from the softmax function.

    Returns:
    - torch.Tensor: The Jacobian of the softmax output.
    """
    B, C, H, W = softmax_output.shape  # Batch size, number of classes, height, width
    softmax_output = softmax_output.view(B, C, -1)  # Reshape to [B, C, H*W] for easier manipulation

    jacobian = torch.zeros(B, C, C, H * W, device=softmax_output.device)  # Jacobian [B, C, C, H*W]

    for i in range(C):
        for j in range(C):
            L = softmax_output[:, i, :]  # Select class i
            R = softmax_output[:, j, :]  # Select class j
            if i == j:
                jacobian[:, i, j, :] = L * (1 - R)  # Diagonal term
            else:
                jacobian[:, i, j, :] = -L * R  # Off-diagonal term

    return jacobian.view(B, C, C, H, W)  # Return to [B, C, C, H, W] shape

# Function to compute gradient of the loss with respect to softmax input
def compute_gradients(predictions, target, loss_func):
    """
    Compute the gradients of the loss function and the Jacobian of softmax.

    Args:
    - predictions (torch.Tensor): Random predictions from the model (after softmax).
    - target (torch.Tensor): One-hot encoded target labels.
    - loss_func (Callable): Loss function to compute loss.

    Returns:
    - grad_L_z (torch.Tensor): Gradients of the loss with respect to predictions.
    """
    # Apply softmax to predictions
    softmax_output = F.softmax(predictions, dim=1)

    # Compute total loss
    total_loss = loss_func(softmax_output, target)

    # Compute softmax Jacobian
    jacobian = softmax_derivative(softmax_output)

    # Compute gradients of the loss with respect to softmax input
    grad_L_sigma = torch.autograd.grad(total_loss, softmax_output, retain_graph=True)[0]

    # Multiply gradients by softmax Jacobian
    grad_L_sigma = grad_L_sigma.unsqueeze(2)  # Add dimension for matrix multiplication
    tr()
    grad_L_z = torch.matmul(jacobian, grad_L_sigma)

    return grad_L_z

# Test the implementation
if __name__ == "__main__":
    # Randomly generated input: a batch of grayscale 2D images (B=5, H=32, W=32, C=3 classes)
    input_data = torch.randn(5, 3, 32, 32, requires_grad=True)  # Batch of 5, 3-class predictions for 32x32 images

    # Randomly generated one-hot encoded target data (B=5, H=32, W=32, C=3)
    target_data = torch.randint(0, 3, (5, 32, 32))
    target_onehot = F.one_hot(target_data, num_classes=3).permute(0, 3, 1, 2).float()  # Convert to one-hot format

    # Define a loss function (CrossEntropy)
    loss_func = torch.nn.CrossEntropyLoss()

    # Perform the forward and backward passes
    grad_L_z = compute_gradients(input_data, target_onehot, loss_func)

    # Output the gradients
    print("Gradients with respect to softmax input:\n", grad_L_z)
# %%
