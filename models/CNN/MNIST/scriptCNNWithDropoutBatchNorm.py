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

# MNIST data loading with normalization parameters commonly used for MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean and std dev values precomputed for MNIST
])

train_dataset = torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('../data', train=False, download=True, transform=transform)

# batch_size=64 chosen as a balance between speed and memory usage during training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# batch_size=1 for detailed visualization purposes in the testing phase
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# CNN model with Dropout and Batch Normalization
class VisualCNN(nn.Module):
    def __init__(self):
        super(VisualCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout rate of 0.5 to prevent overfitting
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        conv_out = x.detach().cpu()
        x = self.fc_layers(x)
        return x, conv_out

# Initialize device based on GPU availability for speed optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisualCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adjustable number of epochs to control training duration
num_epochs = 5

# Function to evaluate model accuracy on test data
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

# Training loop with accuracy evaluation after each epoch
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

# Visualization function to illustrate what CNN "sees" at the convolutional layers
def visualize_features(features, layer_name):
    num_features = features.shape[1]
    fig, axes = plt.subplots(1, min(num_features, 8), figsize=(15, 15))
    for i in range(min(num_features, 8)):
        axes[i].imshow(features[0, i, :, :], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Feature maps from {layer_name}')
    plt.show()

# Testing and visualization (limited samples)
model.eval()
test_features, test_labels = [], []
sample_count = 0
max_samples_to_visualize = 5

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, conv_out = model(images)

        if sample_count < max_samples_to_visualize:
            plt.imshow(images.cpu().squeeze(), cmap='gray')
            plt.title(f'Original Image - Label: {labels.item()}')
            plt.axis('off')
            plt.show()

            visualize_features(conv_out, 'Conv Layers')
            sample_count += 1

        test_features.append(conv_out.view(conv_out.size(0), -1).cpu().numpy())
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
plt.title("t-SNE Visualization of CNN Features (MNIST)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
