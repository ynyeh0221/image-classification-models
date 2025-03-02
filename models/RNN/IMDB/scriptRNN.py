# Required Libraries (ensure these are installed first)
# pip install torch torchtext==0.6.0 matplotlib numpy sklearn

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.datasets import IMDB

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define fields
TEXT = data.Field(tokenize='basic_english', include_lengths=True)
LABEL = data.LabelField(dtype=torch.float)

# Load IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

# Create iterators
train_loader, test_loader = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    sort_within_batch=True,
    device=device)

# RNN Model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), enforce_sorted=False)
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        return self.sigmoid(self.fc(hidden[-1]))

# Initialize model, criterion, optimizer
model = RNNClassifier(len(TEXT.vocab), embed_dim=100, hidden_dim=128, output_dim=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        texts, lengths = batch.text
        labels = batch.label
        optimizer.zero_grad()
        outputs = model(texts, lengths).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Evaluate accuracy
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            texts, lengths = batch.text
            labels = batch.label
            outputs = model(texts, lengths).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

test_accuracy = evaluate_accuracy(model, test_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')
