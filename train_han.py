from han_model import HANModel
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the HeteroData object
data = torch.load('hetero_data_agnews.pt')
data = data.to(device)

# print(data.metadata())
# Extract input dimensions
in_channels_dict = {
    'document': data['document'].x.size(-1),
    'author': data['author'].x.size(-1),
    'tag': data['tag'].x.size(-1)
}
num_classes = data['document'].y.max().item() + 1

model = HANModel(in_channels_dict, out_channels=num_classes, metadata=data.metadata()).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

train_mask = data['document'].train_mask
val_mask = data['document'].val_mask
test_mask = data['document'].test_mask

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        preds = out.argmax(dim=1)
        labels = data['document'].y
        p = preds[mask].cpu().numpy()
        l = labels[mask].cpu().numpy()
        acc = accuracy_score(l, p)
        f1 = f1_score(l, p, average='macro')
    return acc, f1

epochs = 50
train_accuracies = []
train_f1s = []
val_accuracies = []
val_f1s = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out[train_mask], data['document'].y[train_mask])
    loss.backward()
    optimizer.step()

    train_acc, train_f1 = evaluate(train_mask)
    val_acc, val_f1 = evaluate(val_mask)

    train_accuracies.append(train_acc)
    train_f1s.append(train_f1)
    val_accuracies.append(val_acc)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch}: Loss {loss.item():.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}, Val F1 {val_f1:.4f}")

test_acc, test_f1 = evaluate(test_mask)
print(f"Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.plot(train_accuracies, label='Train Acc')
plt.plot(val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.subplot(1,2,2)
plt.plot(train_f1s, label='Train F1')
plt.plot(val_f1s, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score over epochs')

plt.tight_layout()
plt.savefig('training_curves2.png')
