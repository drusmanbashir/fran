import torch
import torch.nn as nn
import numpy as np

def compute_G_i(model, inputs, targets, loss_fn, rho=1):
    # Forward pass
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    
    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    # Extract the gradient of the loss w.r.t pre-activation outputs
    pre_activation_grad = model.pre_activation.grad

    # Compute the derivative of the activation function (ReLU in this case)
    activation_deriv = (model.pre_activation > 0).float()  # ReLU derivative

    # Compute the term Σ'_L(z^{(L)}_i) * ∇_{x^{(L)}_i} L
    gradient_term = activation_deriv * pre_activation_grad

    # Compute the upper bound G_i = L * ρ * ||Σ'_L(z^{(L)}_i) ∇_{x^{(L)}_i} L||^2
    G_i = rho * torch.norm(gradient_term, p=2).item() ** 2
    
    return G_i

# Example usage
model = SimpleModel()
inputs = torch.randn(1, 10)  # Example input
targets = torch.randn(1, 1)  # Example target

# Define a loss function
loss_fn = nn.MSELoss()

# Compute G_i
G_i = compute_G_i(model, inputs, targets, loss_fn, rho=1.0)
# Example neural network class
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 1)
        self.pre_activation = None  # Store pre-activation outputs
        self.pre_activation_grad = None  # Store gradient w.r.t pre-activation

    def forward(self, x):
        x = self.fc1(x)
        self.pre_activation = x  # Capture pre-activation before ReLU
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Forward hook to capture pre-activation values
def forward_hook(module, input, output):
    module.pre_activation = output  # Store the pre-activation values

# Backward hook to capture gradients w.r.t pre-activation
def backward_hook(module, grad_input, grad_output):
    module.pre_activation_grad = grad_output[0]  # Store gradient w.r.t pre-activation

# Function to compute importance score
def compute_importance_score(model, loss, rho=1.0):
    # Calculate the term Σ'_L(z^{(L)}_i) * ∇_{x^{(L)}_i} L
    activation_deriv = (model.pre_activation > 0).float()  # Derivative of ReLU (0 or 1)
    gradient_term = activation_deriv * model.pre_activation_grad

    # Compute the upper bound G_i = L * ρ * ||Σ'_L(z^{(L)}_i) ∇_{x^{(L)}_i} L||^2
    G_i = rho * torch.norm(gradient_term, p=2).item() ** 2
    return G_i

# Training function with hooks and importance sampling
def train_with_importance_sampling(model, data_loader, loss_fn, epochs, B, b, tau_th, alpha_tau):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tau = 0
    
    # Register hooks on the layer where pre-activation and gradients are needed
    hook_handle_fwd = model.fc1.register_forward_hook(forward_hook)
    hook_handle_bwd = model.fc1.register_backward_hook(backward_hook)

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            
            if tau > tau_th:
                # Uniformly sample B data points
                U = [next(iter(data_loader)) for _ in range(B)]
                importance_scores = []
                
                for inp, targ in U:
                    # Perform forward and backward passes to compute importance score
                    outputs = model(inp)
                    loss = loss_fn(outputs, targ)
                    model.zero_grad()
                    loss.backward(retain_graph=True)
                    
                    G_i = compute_importance_score(model, loss)
                    importance_scores.append(G_i)

                # Normalize importance scores
                importance_scores = np.array(importance_scores)
                importance_scores /= importance_scores.sum()

                # Sample b data points based on the computed importance scores
                sampled_indices = np.random.choice(B, size=b, p=importance_scores)
                G = [U[i] for i in sampled_indices]
                weights = [1 / (B * importance_scores[i]) for i in sampled_indices]

                # Perform SGD step with importance sampling
                for (inp, targ), w in zip(G, weights):
                    optimizer.zero_grad()
                    outputs = model(inp)
                    loss = loss_fn(outputs, targ)
                    loss = loss * torch.tensor(w, device=inp.device)
                    loss.backward()
                    optimizer.step()

            else:
                # Uniform sampling step
                batch = next(iter(data_loader))
                optimizer.zero_grad()
                outputs = model(batch[0])
                loss = loss_fn(outputs, batch[1])
                loss.backward()
                optimizer.step()

            # Update tau based on importance scores
            tau = alpha_tau * tau + (1 - alpha_tau) * (1 - 1 / np.sum(np.square(importance_scores)))

        print(f"Epoch {epoch + 1} completed")

    # Remove hooks after training is complete
    hook_handle_fwd.remove()
    hook_handle_bwd.remove()

# Example usage
# Define model, loss function, and data loader
model = SimpleModel()
loss_fn = nn.MSELoss()
data_loader = torch.utils.data.DataLoader([(torch.randn(1, 10), torch.randn(1, 1))], batch_size=1)

# Training parameters
epochs = 5
B = 10  # Pre-sampling size
b = 5  # Actual batch size
tau_th = 1.5  # Threshold for importance sampling
alpha_tau = 0.9  # Exponential moving average parameter

# Train the model
train_with_importance_sampling(model, data_loader, loss_fn, epochs, B, b, tau_th, alpha_tau)

