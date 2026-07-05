import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# same dataset as Day 63
np.random.seed(42)
n = 2000
data = pd.DataFrame({
    'age':          np.random.randint(18, 70, n),
    'income':       np.random.randint(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'debt_ratio':   np.random.uniform(0, 0.8, n).round(2),
    'loan_amount':  np.random.randint(5000, 100000, n),
    'emp_years':    np.random.randint(0, 30, n),
})
default_prob = (
    0.3 * (data['credit_score'] < 600).astype(int) +
    0.2 * (data['debt_ratio'] > 0.5).astype(int) +
    0.1 * (data['income'] < 40000).astype(int)
)
data['defaulted'] = (default_prob + np.random.uniform(0, 0.3, n) > 0.35).astype(int)

features = ['age', 'income', 'credit_score', 'debt_ratio', 'loan_amount', 'emp_years']
X = data[features].values.astype(np.float32)
y = data['defaulted'].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32)


class CreditNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int],
                 dropout: float = 0.2):
        super().__init__()
        # build network dynamically from hidden_sizes list
        # input → hidden[0] → hidden[1] → ... → 1
        # ReLU + Dropout after each hidden layer
        # Sigmoid on output
        self.netwrok = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class PyTorchTrainer:
    def __init__(self, model: nn.Module, lr: float = 0.001):
        self.model     = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses   = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        for epoch in range(100):
            for X_batch, y_batch in loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss/len(loader):.4f}")

    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        with torch.no_grad():
            y_prob = self.model(X_test_t).squeeze().numpy()
            y_pred = (y_prob >= 0.5).astype(int)

    def fit(self, X_train, y_train, X_val, y_val,
            epochs: int = 100, batch_size: int = 32) -> None:
        # create DataLoaders
        # train for epochs, record train and val loss each epoch
        # print every 10 epochs
        ...

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        # return probabilities as numpy array
        ...

    def predict(self, X: torch.Tensor,
                threshold: float = 0.5) -> np.ndarray:
        ...

    def plot_losses(self, filename: str = 'torch_losses.png') -> None:
        # plot train and val loss on same axes
        ...

    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)