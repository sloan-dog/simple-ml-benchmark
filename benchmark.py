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


def train_network_optimized(device):
    # Adjust data size to prevent memory issues
    model = SimpleCNN().to(device)
    inputs = torch.randn(100000, 3, 64, 64).to(device)
    labels = torch.randint(0, 10, (100000,)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    for epoch in range(5):  # Reduced number of epochs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()

    return end_time - start_time


# Measure time on CPU
cpu_time_optimized = train_network_optimized('cpu')

# Measure time on GPU if available
gpu_time_optimized = None
if torch.backends.mps.is_available():
    gpu_time_optimized = train_network_optimized('mps')

print(f"CPU Training Time: {cpu_time_optimized:.2f} seconds")
if gpu_time_optimized is not None:
    print(f"GPU Training Time: {gpu_time_optimized:.2f} seconds")
else:
    print("GPU acceleration (MPS) not available on this machine.")
