
import pandas as pd 
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

#imports for KFold
from sklearn.model_selection import KFold

#model :P 

class RainNet(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.model_MLP = nn.Sequential(
            nn.Linear(features, 32), nn.ReLU(),
            nn.BatchNorm1d(32), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(16, 1) 
        )
    def forward(self,x):
        return self.model_MLP(x)
    

# loading data :D  (not PCA)
df = pd.read_csv('data/train_processed.csv')

features = torch.tensor(df.drop(columns=['id','day','rainfall']).values.astype(np.float32))
labels = torch.tensor(df['rainfall'].values.astype(np.float32).reshape(-1,1))
dataset= TensorDataset(features,labels)

train_set, val_set = random_split(
    dataset,
    [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))],
    generator=torch.Generator().manual_seed(42)
)

train_loader=DataLoader(train_set, batch_size=32,shuffle=True)
val_loader=DataLoader(val_set,batch_size=32,shuffle=False)

#Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RainNet(features.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
n_epochs=2000


train_losses, val_losses = [],[]


for epoch in tqdm(range(1,n_epochs+1)):
    #train
    model.train()
    sum_loss=0.0
    for xb,yb in train_loader:

        xb,yb =  xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss=criterion(model(xb),yb)
        loss.backward()
        optimizer.step()
        sum_loss +=loss.item()*xb.size(0)

    epoch_train = sum_loss / len(train_loader.dataset)
    train_losses.append(epoch_train)

    # validation
    model.eval()
    sum_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            sum_loss += loss.item() * xb.size(0)
    epoch_val = sum_loss / len(val_loader.dataset)
    val_losses.append(epoch_val)

    print(f"Epoch {epoch:02d} | train {epoch_train:.4f} | val {epoch_val:.4f}")


import matplotlib.pyplot as plt


plt.figure(figsize=(6,4))
plt.plot(train_losses[10:], label="Train loss")
plt.plot(val_losses[10:],   label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#  ‼️‼️ BELLOW IS GENERATE BY CHAT ‼️‼️

# 📘 Evaluate ROC-AUC on a DataLoader
# > Collect logits → convert to probabilities → compute AUC.  
# > If ton modèle sort déjà des probabilités (Sigmoid intégré), enlève la ligne torch.sigmoid.

import numpy as np
from sklearn.metrics import roc_auc_score

def evaluate_roc_auc(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)                 # (B, 1)  – logits
            probs = torch.sigmoid(logits)      # → (B, 1)  – probs 0-1
            y_pred.append(probs.cpu().numpy().ravel())
            y_true.append(yb.cpu().numpy().ravel())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    return roc_auc_score(y_true, y_pred)

# ――― usage ―――
val_auc = evaluate_roc_auc(model, val_loader, device)
print(f"Validation ROC-AUC : {val_auc:.4f}")


# 1. Charger le jeu de test -------------------------------------------------
df_test = pd.read_csv('data/test_processed.csv')   # adapte le chemin si besoin
ids     = df_test['id'].values                     # colonne id
X_test  = torch.tensor(
    df_test.drop(columns=['id', 'day']).values,    # on enlève id & day
    dtype=torch.float32
)

test_loader = DataLoader(TensorDataset(X_test),
                         batch_size=256, shuffle=False)

# 2. Prédire les probabilités ----------------------------------------------
model.eval()
probs = []

with torch.no_grad():
    for (xb,) in test_loader:          # tuple à un seul élément
        xb = xb.to(device)
        logits = model(xb)             # (B,1) logits
        probs.append(torch.sigmoid(logits).cpu().numpy().ravel())

probs = np.concatenate(probs)          # shape (N,)

# 3. Construire et enregistrer la soumission -------------------------------
submission = pd.DataFrame({
    'id': ids,
    'rainfall': probs
})

submission.to_csv('submission.csv', index=False)
print("✅ submission.csv créé –", submission.shape[0], "lignes")