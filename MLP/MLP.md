
---

## **RainNet – Neural Network for Rainfall Prediction**

### **1. Architecture**

```python
class RainNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.model_MLP = nn.Sequential(
            nn.Linear(features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model_MLP(x)
```

* **Type**: Feedforward Multilayer Perceptron (MLP)
* **Input**: Vector of weather-related features
* **Output**: Binary logit for rainfall (0 or 1)
* **Activations**: ReLU
* **Regularization**: Dropout + Batch Normalization
* **Architecture**: \[Input → 32 → 16 → 1]

---

### **2. Training Configuration**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RainNet(features.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
n_epochs = 500
```

* **Loss function**: Binary Cross Entropy with Logits
* **Optimizer**: Adam
* **L2 Regularization**: `weight_decay = 1e-4`
* **Epochs**: 500
* **Device**: GPU if available

---

### **3. Training Loop**

```python
for epoch in range(1, n_epochs + 1):
    model.train()
    sum_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * xb.size(0)
    train_losses.append(sum_loss / len(train_loader.dataset))

    model.eval()
    sum_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            sum_loss += loss.item() * xb.size(0)
    val_losses.append(sum_loss / len(val_loader.dataset))
```

---

### **4. Hyperparameters Summary**

| Hyperparameter  | Description                         |
| --------------- | ----------------------------------- |
| `hidden_layers` | Two layers with 32 and 16 neurons   |
| `activation`    | ReLU for non-linearity              |
| `dropout`       | 0.3 dropout after each hidden layer |
| `optimizer`     | Adam optimizer                      |
| `learning_rate` | 1e-3                                |
| `weight_decay`  | 1e-4 (L2 regularization)            |
| `loss_function` | Binary Cross-Entropy with Logits    |
| `epochs`        | 500                                 |
| `batch_size`    | 32                                  |

---