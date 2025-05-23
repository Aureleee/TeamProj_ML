import torch.nn as nn



class RainNet(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.model_MLP = nn.Sequential(
            nn.Linear(features,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model_MLP(x)


