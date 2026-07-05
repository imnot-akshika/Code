import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return (x > 0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], random_state: int = 42):
        np.random.seed(random_state)
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases  = []
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X) -> np.ndarray:
        self.activations = [X]
        self.z_values    = []

        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            self.z_values.append(z)

            if i < len(self.weights) -1:
                current = relu(z)
            else:
                current = sigmoid(z)

            self.activations.append(current)
        return current

    def backward(self, X, y, learning_rate: float) -> None:
        m = X.shape[0]

        delta = self.activations[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ delta / m
            db = np.mean(delta, axis=0, keepdims=True)

            self.weights[i] -= learning_rate * dw
            self.biases[i]  -= learning_rate * db

            if i > 0:
                delta = delta @ self.weights[i].T
                delta *= relu_derivative(self.z_values[i-1])
    
    def _compute_loss(self, y_true, y_pred) -> float:
        return binary_cross_entropy(y_true, y_pred.flatten())

    def train(self, X, y, epochs: int = 500,
              learning_rate: float = 0.01,
              batch_size: int = 32,
              verbose: bool = True) -> list[float]:
        # mini-batch gradient descent
        # return list of losses per epoch
        losses = []
        m = X.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start in range(0, m, batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            y_pred = self.forward(X)
            loss = self._compute_loss(y, y_pred)
            losses.append(loss)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {losses[-1]:.4f}")

        return losses
    

    def predict_proba(self, X) -> np.ndarray:
        self.forward(X)
        return self.activations[-1]

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X).flatten()
        return (proba >= threshold).astype(int)

    def plot_loss(self, losses: list[float],
                  filename: str = 'loss_curve.png') -> None:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize = (10, 5))
        ax.plot(losses, color='#F715AB', lw=2)
        ax.set_title("Training loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Binary Cross Entropy")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")


#Example usage
# generate credit dataset
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
X = data[features].values
y = data['defaulted'].values

# scale features — critical for neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# build and train
nn = NeuralNetwork(layer_sizes=[6, 16, 8, 1])
losses = nn.train(X_train, y_train,
                  epochs=500,
                  learning_rate=0.01,
                  batch_size=32)

# evaluate
y_pred  = nn.predict(X_test)
y_proba = nn.predict_proba(X_test).flatten()

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, y_pred))

nn.plot_loss(losses)

# compare with sklearn
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(16, 8),
                    max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
print(f"\nsklearn MLP accuracy: {mlp.score(X_test, y_test):.4f}")
print(f"sklearn MLP ROC-AUC:  {roc_auc_score(y_test, mlp.predict_proba(X_test)[:,1]):.4f}")