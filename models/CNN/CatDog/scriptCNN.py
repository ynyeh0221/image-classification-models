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

# Data loading and preprocessing for CIFAR-10 (cats and dogs)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images from PIL to Tensor format, scaling pixel values to [0, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images to range [-1, 1] for better training
])

# Load CIFAR-10 and filter cat and dog classes
train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Filter function for cats (label=3) and dogs (label=5)
def filter_cats_dogs(dataset):
    indices = [i for i, label in enumerate(dataset.targets) if label in [3, 5]]
    dataset.data = dataset.data[indices]
    dataset.targets = [0 if dataset.targets[i] == 3 else 1 for i in indices]  # cat=0, dog=1
    return dataset

train_dataset = filter_cats_dogs(train_dataset_full)
test_dataset = filter_cats_dogs(test_dataset_full)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        conv_outputs = []
        for layer in self.conv_layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                conv_outputs.append(x.detach().cpu())
        x = self.fc_layers(x)
        return x, conv_outputs

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
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

# Visualization function for CNN feature maps
def visualize_features(features, layer_name):
    num_features = features.shape[1]
    fig, axes = plt.subplots(1, min(num_features, 8), figsize=(15, 15))
    for i in range(min(num_features, 8)):
        axes[i].imshow(features[0, i].numpy(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Feature maps from {layer_name}')
    plt.show()

# Testing and visualization (limited samples)
model.eval()
sample_count = 0
max_samples_to_visualize = 3
test_features, test_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, conv_outputs = model(images)

        if sample_count < max_samples_to_visualize:
            plt.imshow(np.transpose(images.cpu().squeeze().numpy(), (1, 2, 0)) * 0.5 + 0.5)
            plt.title(f'Original Image - Label: {"Cat" if labels.item()==0 else "Dog"}')
            plt.axis('off')
            plt.show()

            for idx, features in enumerate(conv_outputs):
                visualize_features(features, f'Conv Layer {idx+1}')

            sample_count += 1

        test_features.append(conv_outputs[-1].view(-1).numpy())
        test_labels.append(labels.cpu().numpy())

        if len(test_features) >= 500:
            break

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(np.array(test_features))

plt.figure(figsize=(10,8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=test_labels, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE Visualization of CNN Features (Cats vs Dogs)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
