import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# reuse the same dataset from Day 59
np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'age':          np.random.randint(18, 70, n),
    'income':       np.random.randint(20000, 150000, n),
    'credit_score': np.random.randint(300, 850, n),
    'debt_ratio':   np.random.uniform(0, 0.8, n).round(2),
    'loan_amount':  np.random.randint(5000, 100000, n),
    'emp_years':    np.random.randint(0, 30, n),
})
default_prob = (
    0.3 * (df['credit_score'] < 600).astype(int) +
    0.2 * (df['debt_ratio'] > 0.5).astype(int) +
    0.1 * (df['income'] < 40000).astype(int) +
    0.1 * (df['emp_years'] < 2).astype(int)
)
df['defaulted'] = (default_prob + np.random.uniform(0, 0.3, n) > 0.35).astype(int)

FEATURES = ['age', 'income', 'credit_score', 'debt_ratio', 'loan_amount', 'emp_years']
X = df[FEATURES]
y = df['defaulted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class TreeModelComparison:
    FEATURES = FEATURES

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.models  = {}     # name → fitted model
        self.results = {}     # name → metrics dict

    def train_all(self) -> None:
        self.dt = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', DecisionTreeClassifier(max_depth=5, random_state=42))
        ])
        self.dt.fit(self.X_train, self.y_train)

        self.dtd = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', DecisionTreeClassifier())
        ])
        self.dt.fit(self.X_train, self.y_train)

        self.rf = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ])
        self.dt.fit(self.X_train, self.y_train)

        self.gb = Pipeline([
            ('scaler', StandardScaler()),
            ('tree', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
        ])
        self.dt.fit(self.X_train, self.y_train)

        self.models = {
            'DecisionTree': self.dt,
            'DecisionTree_deep': self.dtd,
            'RandomForest': self.rf,
            'GradientBoosting': self.gb
        }


    def evaluate_all(self) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                       cv=5, scoring='roc_auc')
            
            rows.append({
                'model': name,
                'accuracy': round(accuracy_score(self.y_test, y_pred), 4),
                'roc_auc':  round(roc_auc_score(self.y_test, y_prob), 4),
                'cv_auc':   round(cv_scores.mean(), 4)
            })
            self.results[name] = rows[-1]
        return pd.DataFrame(rows).set_index('model')

    def plot_feature_importance(self, model_name: str = 'RandomForest',
                                ax=None) -> None:
        model = self.models[model_name]
        importances = model.named_steps['tree'].feature_importances_
        imp_series =pd.Series(importances, index=self.FEATURES).sort_values()

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        imp_series.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_title(f"Feature Importance — {model_name}")
        ax.axvline(0, color='black', linewidth=0.8)
        
    def plot_comparison(self, filename: str = 'tree_comparison.png') -> None:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Plotting of The Data', fontsize=14, fontweight='bold')

        results_df = pd.DataFrame(self.results).T
        results_df['accuracy'].plot(kind='bar', ax=axes[0, 0], color='#F715AB')
        axes[0, 0].set_title("Accuracy by Model")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].tick_params(axis='x', rotation=45)

        results_df['roc_auc'].plot(kind='bar', ax=axes[0, 1], color='#0313A6')
        axes[0, 1].set_title("ROC Score")

        self.plot_feature_importance('RandomForest', ax=axes[1, 0])
        self.plot_feature_importance('GradientBoosting', ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")

    def overfitting_report(self) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            train_acc = model.score(self.X_train, self.y_train)
            test_acc = model.score(self.X_test, self.y_test)
            gap = train_acc - test_acc

            rows.append({
                'name': name,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap
            })

        return pd.DataFrame(rows)
    

#Example Usage
comp = TreeModelComparison(X_train, y_train, X_test, y_test)
comp.train_all()

print("=== Model Comparison ===")
results = comp.evaluate_all()
print(results.to_string())

print("\n=== Overfitting Report ===")
overfit = comp.overfitting_report()
print(overfit.to_string())

comp.plot_comparison('tree_comparison.png')