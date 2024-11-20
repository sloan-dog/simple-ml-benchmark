import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define the SimpleCNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 31 * 31, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 31 * 31)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_progress(epochs, total_samples, cur_epoch, cur_idx):
    total_steps = epochs * total_samples
    step = (cur_epoch * total_samples) + cur_idx
    return step / total_steps

def train_network_optimized(device, batch_size = 32, total_samples = 1000):
    # Adjust data size to prevent memory issues
    model = SimpleCNN().to(device)
    inputs = torch.randn(total_samples, 3, 64, 64).to(device)
    labels = torch.randint(0, 10, (total_samples,)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
# Train the model with interrupt handling
    try:
        start_time = time.time()
        print(f"Starting training with [{device}]. Press Ctrl+C to interrupt...")
        for epoch in range(5):  # Reduced number of epochs
            for start_idx in range(0, total_samples, batch_size):
                # Create batches
                end_idx = start_idx + batch_size
                input_batch = inputs[start_idx:end_idx]
                label_batch = labels[start_idx:end_idx]

                # Forward and backward pass
                optimizer.zero_grad()
                outputs = model(input_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Progress: {get_progress(5, total_samples, epoch, start_idx)} ")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        end_time = time.time()
        return end_time - start_time


# Measure time on CPU
cpu_time_optimized = train_network_optimized('cpu', 10000, 10000)

# Measure time on GPU if available
gpu_time_optimized = None
if torch.backends.mps.is_available():
    gpu_time_optimized = train_network_optimized('mps', 10000, 10000)

print(f"CPU Training Time: {cpu_time_optimized:.2f} seconds")
if gpu_time_optimized is not None:
    print(f"GPU Training Time: {gpu_time_optimized:.2f} seconds")
else:
    print("GPU acceleration (MPS) not available on this machine.")
