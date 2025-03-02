# Required Libraries (ensure these are installed first)
# pip install torch torchvision matplotlib numpy sklearn

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# MNIST data loading with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Fully connected NN model with Dropout and Batch Normalization
class VisualNN(nn.Module):
    def __init__(self):
        super(VisualNN, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        layer_outputs = []
        for layer in self.fc_layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                layer_outputs.append(x.detach().cpu())
        return x, layer_outputs

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisualNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adjustable epochs
num_epochs = 5

# Accuracy evaluation function
def evaluate_accuracy(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    epoch_accuracy = evaluate_accuracy(model, test_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}] Test Accuracy: {epoch_accuracy:.2f}%')

# Visualization function for NN layer outputs
def visualize_nn_features(features, layer_name):
    features = features.numpy().squeeze()
    plt.figure(figsize=(10, 4))
    plt.plot(features)
    plt.title(f'Feature activations from {layer_name}')
    plt.xlabel('Neuron Index')
    plt.ylabel('Activation')
    plt.grid(True)
    plt.show()

# Testing and visualization (limited samples)
model.eval()
test_features, test_labels = [], []
sample_count = 0
max_samples_to_visualize = 5

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, layer_outputs = model(images)

        if sample_count < max_samples_to_visualize:
            plt.imshow(images.cpu().squeeze(), cmap='gray')
            plt.title(f'Original Image - Label: {labels.item()}')
            plt.axis('off')
            plt.show()

            for idx, features in enumerate(layer_outputs):
                visualize_nn_features(features, f'Fully Connected Layer {idx+1}')

            sample_count += 1

        test_features.append(layer_outputs[-1].cpu().numpy())
        test_labels.append(labels.cpu().numpy())

        if len(test_features) >= 1000:
            break

test_features = np.concatenate(test_features)
test_labels = np.concatenate(test_labels)

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(test_features)

plt.figure(figsize=(10,8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE Visualization of NN Features (MNIST)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
