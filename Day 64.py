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

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

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
        total_loss = 0

        for X_batch, y_batch in loader:
            self.optimizer.zero_grad()
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch.unsqueeze(1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(loader)

    def val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch.unsqueeze(1))
                total_loss += loss.item()

            return total_loss / len(loader)

    def fit(self, X_train, y_train, X_val, y_val,
            epochs: int = 100, batch_size: int = 32) -> None:
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.val_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            proba = self.model(X).squeeze().numpy()
        return proba

    def predict(self, X: torch.Tensor,
                threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def plot_losses(self, filename: str = 'torch_losses.png') -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.train_losses, color='#0313A6', lw=2, label='Train Loss')
        ax.plot(self.val_losses, color='#F715AB', lw=2, label='Val Loss', linestyle='--')
        ax.set_title("Training & Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")


    def save(self, filepath: str) -> None:
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

#Example Usage
model = CreditNet(input_size=6, hidden_sizes=[32, 16, 8], dropout=0.2)
trainer = PyTorchTrainer(model, lr=0.001)

trainer.fit(X_train_t, y_train_t, X_test_t, y_test_t,
            epochs=100, batch_size=32)

y_proba = trainer.predict_proba(X_test_t)
y_pred  = trainer.predict(X_test_t)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred))

trainer.plot_losses()
trainer.save('credit_net.pth')
print("Model saved.")