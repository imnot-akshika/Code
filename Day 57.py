import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, accuracy_score, classification_report)

class MLFoundation:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = None

    def explore(self) -> dict:
        df = self.df.copy()
        target_col = df[self.target]

        is_categorical = target_col.dtype == 'object' or str(target_col.dtype) == 'category'

        if is_categorical:
            target_stats = target_col.value_counts().to_dict()
            class_balance = target_stats
        else:
            target_stats = target_col.describe().to_dict()
            class_balance = None
        
        numeric_df = df.select_dtypes(include='number')
        correlations = numeric_df.corr()[self.target].drop(self.target).abs().sort_values(ascending=False).round(2)
        return {
            'shape': df.shape,
            'target_stats': target_stats,
            'correlation': correlations,
            'class_balance': class_balance
        }


    def prepare(self, features: list[str],
                test_size: float = 0.2,
                scale: bool = True) -> tuple:
        X = self.df[features].values
        y = self.df[self.target].values

        scale = scale

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if scale:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.fit_transform(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test
        

    def encode_categoricals(self, columns: list[str]) -> pd.DataFrame:
        self.df = pd.get_dummies(self.df, columns=columns)
        return self.df

    def evaluate_regression(self, y_true, y_pred) -> dict:
        mse = mean_squared_error(y_true, y_pred)
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2_score(y_true, y_pred)
        }

    def evaluate_classification(self, y_true, y_pred) -> dict:
        y_true = y_true
        y_pred = y_pred

        acc = accuracy_score(y_true, y_pred)
        return{
            'accuracy': accuracy_score(y_true, y_pred),
            'report': classification_report(y_true, y_pred)
        }
    def feature_importance_proxy(self, features: list[str]) -> pd.Series:
        df = self.df[features + [self.target]].select_dtypes(include='number')
        return df.corr()[self.target].drop(self.target).abs().sort_values(ascending=False)
    

#Example Usage
# Dataset 1 — regression
np.random.seed(42)
n = 500
reg_df = pd.DataFrame({
    'size':     np.random.randint(500, 3000, n),
    'bedrooms': np.random.randint(1, 6, n),
    'age':      np.random.randint(0, 50, n),
    'location': np.random.choice(['urban', 'suburban', 'rural'], n),
    'price':    np.random.randint(100000, 800000, n)
})
reg_df['price'] = (reg_df['size'] * 200 +
                   reg_df['bedrooms'] * 15000 -
                   reg_df['age'] * 500 +
                   np.random.normal(0, 20000, n))

# Dataset 2 — classification
clf_df = pd.DataFrame({
    'age':          np.random.randint(18, 65, n),
    'income':       np.random.randint(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'debt_ratio':   np.random.uniform(0, 0.8, n).round(2),
    'approved':     np.random.choice([0, 1], n, p=[0.4, 0.6])
})
clf_df['approved'] = (
    (clf_df['credit_score'] > 650).astype(int) &
    (clf_df['debt_ratio'] < 0.4).astype(int)
)

# test regression foundation
reg = MLFoundation(reg_df, 'price')
reg.encode_categoricals(['location'])
info = reg.explore()
print(f"Target range: {info['target_stats']['min']:.0f} — {info['target_stats']['max']:.0f}")

features = ['size', 'bedrooms', 'age',
            'location_urban', 'location_suburban', 'location_rural']
X_train, X_test, y_train, y_test = reg.prepare(features)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

imp = reg.feature_importance_proxy(features)
print(f"Most important feature: {imp.index[0]}")

# test classification foundation
clf = MLFoundation(clf_df, 'approved')
info2 = clf.explore()
print(f"\nClass balance: {info2['class_balance']}")
X_train2, X_test2, y_train2, y_test2 = clf.prepare(
    ['age', 'income', 'credit_score', 'debt_ratio']
)
print(f"Train: {X_train2.shape}, Test: {X_test2.shape}")