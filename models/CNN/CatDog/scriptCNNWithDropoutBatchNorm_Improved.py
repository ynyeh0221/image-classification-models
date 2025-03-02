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

# Data loading with Data Augmentation for CIFAR-10 (cats and dogs)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Filter function for cats (label=3) and dogs (label=5)
def filter_cats_dogs(dataset):
    indices = [i for i, label in enumerate(dataset.targets) if label in [3, 5]]
    dataset.data = dataset.data[indices]
    dataset.targets = [0 if dataset.targets[i] == 3 else 1 for i in indices]
    return dataset

train_dataset = filter_cats_dogs(train_dataset_full)
test_dataset = filter_cats_dogs(test_dataset_full)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Improved CNN model with Dropout and BatchNorm
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
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
model = ImprovedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Adjustable epochs
num_epochs = 15

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

# t-SNE visualization preparation
model.eval()
test_features, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs, conv_outputs = model(images)
        test_features.extend(conv_outputs[-1].cpu().view(images.size(0), -1).numpy())
        test_labels.extend(labels.cpu().numpy())

tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(np.array(test_features[:500]))

plt.figure(figsize=(10,8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=test_labels[:500], cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE Visualization of Improved CNN Features (Cats vs Dogs)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()
