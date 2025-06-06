
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
    nn.Linear(features, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
    nn.Linear(64, 32),      nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 1)
        )
    def forward(self,x):
        return self.model_MLP(x)
    

# loading data :D  (not PCA)
df = pd.read_csv('data/train_pca.csv')

features = torch.tensor(df.drop(columns=['rainfall']).values.astype(np.float32))
labels = torch.tensor(df['rainfall'].values.astype(np.float32).reshape(-1,1))
dataset= TensorDataset(features,labels)

train_set, val_set = random_split(
    dataset,
    [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))],
    generator=torch.Generator().manual_seed(42)
)

train_loader=DataLoader(train_set, batch_size=32,shuffle=True)
val_loader=DataLoader(val_set,batch_size=32,shuffle=False)

#Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RainNet(features.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
n_epochs=500


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


# --- 1. pertes par epoch --------------------------------------------------
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses,   label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%