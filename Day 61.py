from os import name

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_validate,
                                      StratifiedKFold, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              precision_recall_curve, roc_curve,
                              average_precision_score)

# reuse the credit dataset
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
df['defaulted'] = (default_prob +
                   np.random.uniform(0, 0.3, n) > 0.35).astype(int)

FEATURES = ['age', 'income', 'credit_score',
            'debt_ratio', 'loan_amount', 'emp_years']
X = df[FEATURES]
y = df['defaulted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class ModelEvaluator:
    def __init__(self, X_train, y_train, X_test, y_test,
                 feature_names: list[str]):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test  = X_test
        self.y_test  = y_test
        self.feature_names = feature_names
        self.models = {}

    def add_model(self, name: str, model) -> None:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipe.fit(self.X_train, self.y_train)
        self.models[name] = pipe

    def full_report(self, name: str) -> dict:
        model = self.models[name]           # direct lookup, no loop needed
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        return {
            'accuracy':      accuracy_score(self.y_test, y_pred),
            'precision':     precision_score(self.y_test, y_pred),
            'recall':        recall_score(self.y_test, y_pred),
            'f1':            f1_score(self.y_test, y_pred),
            'roc_auc':       roc_auc_score(self.y_test, y_prob),
            'avg_precision': average_precision_score(self.y_test, y_prob),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
    

    def threshold_analysis(self, name: str,
                           thresholds=None) -> pd.DataFrame:
        model = self.models[name]
        y_prob = model.predict_proba(self.X_test)[:, 1]

        if thresholds is None:
                thresholds = np.arange(0.1, 0.9, 0.05)

        results = []
        for t in thresholds:
                y_pred_t = (y_prob >= t).astype(int)
                precision = precision_score(self.y_test, y_pred_t, zero_division=0)
                recall = recall_score(self.y_test, y_pred_t, zero_division=0)
                f1 = f1_score(self.y_test, y_pred_t, zero_division=0)
                results.append({
                    'threshold': round(t, 2),
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
        self.df_thresh = pd.DataFrame(results)
        return self.df_thresh
    
    def cross_validate_all(self, cv: int = 5) -> pd.DataFrame:
        rows = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for model_name, model in self.models.items():

            cv_results = cross_validate(model, self.X_train, self.y_train, cv=skf,
                                        scoring=['accuracy', 'f1', 'roc_auc'],
                                        return_train_score=True)
            
            rows.append ({
                'model': model_name,
                'accuracy': f"{cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}",
                'f1': f"{cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}",
                'roc_auc': f"{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}"
            })

        df_cv = pd.DataFrame(rows).set_index('model')
        return df_cv

    def plot_learning_curve(self, name: str,
                            filename: str = 'learning_curve.png') -> None:
        train_sizes, train_scores, val_scores = learning_curve(
            self.models[name], self.X_train, self.y_train,
            train_sizes = np.linspace(0.1, 1.0, 10),
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_mean, 'o-', color='#0313A6', label='Train')
        ax.fill_between(train_sizes,
                        train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='#0313A6')
        ax.plot(train_sizes, val_mean, 'o-', color='#F715AB', label='Validation')
        ax.fill_between(train_sizes,
                        val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='#F715AB')
        ax.set_xlabel("Training Size")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve")

        plt.tight_layout()
        ax.legend()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_evaluation_dashboard(self, filename: str = 'eval_dashboard.png') -> None:
       plt.style.use('dark_background')
       fig, axes = plt.subplots(2, 3, figsize=(18, 10))
       fig.suptitle("Model Evaluation Dashboard", fontsize=14, fontweight='bold')

       colors = ['#0313A6', '#F715AB', '#34EDF3']

       # [0,0] ROC curves — all models
       for (name, model), color in zip(self.models.items(), colors):
            y_prob = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            auc = roc_auc_score(self.y_test, y_prob)
            axes[0, 0].plot(fpr, tpr, color=color, lw=2, label=f'{name} AUC={auc:.3f}')
       axes[0, 0].plot([0,1],[0,1],'gray',linestyle='--')
       axes[0, 0].set_title("ROC Curves")
       axes[0, 0].set_xlabel("False Positive Rate")
       axes[0, 0].set_ylabel("True Positive Rate")
       axes[0, 0].legend(fontsize=7)

       # [0,1] PR curves — all models
       for (name, model), color in zip(self.models.items(), colors):
           y_prob = model.predict_proba(self.X_test)[:, 1]
           precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
           ap = average_precision_score(self.y_test, y_prob)
           axes[0, 1].plot(recall, precision, color=color, lw=2,
                        label=f'{name} AP={ap:.3f}')
       axes[0, 1].set_title("Precision-Recall Curves")
       axes[0, 1].set_xlabel("Recall")
       axes[0, 1].set_ylabel("Precision")
       axes[0, 1].legend(fontsize=7)

       # [0,2] cross-validated F1 bar chart
       cv_df = self.cross_validate_all()
       f1_means = {name: float(v.split(' ±')[0])
                for name, v in cv_df['f1'].items()}
       axes[0, 2].bar(f1_means.keys(), f1_means.values(), color=colors)
       axes[0, 2].set_title("CV F1 Scores")
       axes[0, 2].set_ylabel("F1")
       axes[0, 2].tick_params(axis='x', rotation=30)

       # find best model by roc_auc
       best_name = max(self.models,
                    key=lambda n: roc_auc_score(
                        self.y_test,
                        self.models[n].predict_proba(self.X_test)[:, 1]))

       # [1,0] confusion matrix — best model
       y_pred_best = self.models[best_name].predict(self.X_test)
       cm = confusion_matrix(self.y_test, y_pred_best)
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])  # Blues not blues
       axes[1, 0].set_title(f"Confusion Matrix — {best_name}")
       axes[1, 0].set_xlabel("Predicted")
       axes[1, 0].set_ylabel("Actual")

       # [1,1] threshold analysis — best model
       thresh_df = self.threshold_analysis(best_name)
       axes[1, 1].plot(thresh_df['threshold'], thresh_df['precision'],
                    label='Precision', color='#0313A6')
       axes[1, 1].plot(thresh_df['threshold'], thresh_df['recall'],
                    label='Recall', color='#F715AB', linestyle='--')
       axes[1, 1].plot(thresh_df['threshold'], thresh_df['f1'],
                    label='F1', color='#34EDF3', linestyle=':')
       axes[1, 1].set_title(f"Threshold Analysis — {best_name}")
       axes[1, 1].set_xlabel("Threshold")
       axes[1, 1].set_ylabel("Score")
       axes[1, 1].legend()

       # [1,2] CV ROC-AUC bar chart
       auc_means = {name: float(v.split(' ±')[0])
                 for name, v in cv_df['roc_auc'].items()}
       axes[1, 2].bar(auc_means.keys(), auc_means.values(), color=colors)
       axes[1, 2].set_title("CV ROC-AUC Scores")
       axes[1, 2].set_ylabel("ROC-AUC")
       axes[1, 2].tick_params(axis='x', rotation=30)

       plt.tight_layout()
       plt.savefig(filename, dpi=150, bbox_inches='tight')
       plt.show()
       print(f"Saved to {filename}")


#Example usage:
evaluator = ModelEvaluator(X_train, y_train, X_test, y_test, FEATURES)

evaluator.add_model('LogisticRegression',
                    LogisticRegression(max_iter=1000, random_state=42))
evaluator.add_model('RandomForest',
                    RandomForestClassifier(n_estimators=100, random_state=42))
evaluator.add_model('GradientBoosting',
                    GradientBoostingClassifier(n_estimators=100, random_state=42))

print("=== Full Report: RandomForest ===")
report = evaluator.full_report('RandomForest')
for k, v in report.items():
    if k != 'confusion_matrix':
        print(f"{k}: {v}")

print("\n=== Threshold Analysis ===")
thresh = evaluator.threshold_analysis('RandomForest')
print(thresh.to_string())

print("\n=== Cross Validation ===")
cv_results = evaluator.cross_validate_all()
print(cv_results.to_string())

evaluator.plot_learning_curve('RandomForest')
evaluator.plot_evaluation_dashboard()