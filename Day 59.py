import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# generate dataset
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

# realistic default probability
default_prob = (
    0.3 * (df['credit_score'] < 600).astype(int) +
    0.2 * (df['debt_ratio'] > 0.5).astype(int) +
    0.1 * (df['income'] < 40000).astype(int) +
    0.1 * (df['emp_years'] < 2).astype(int)
)
df['defaulted'] = (default_prob + np.random.uniform(0, 0.3, n) > 0.35).astype(int)

# also create a regression target — predicted credit score
df['risk_score'] = (
    850 - (df['debt_ratio'] * 200) -
    (df['loan_amount'] / 1000) +
    (df['emp_years'] * 5) +
    np.random.normal(0, 30, n)
).clip(300, 850).round(0)

class CreditRiskModel:
    FEATURES = ['age', 'income', 'credit_score',
                'debt_ratio', 'loan_amount', 'emp_years']

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.clf_pipeline = None
        self.reg_pipeline = None

    def train_classifier(self, X_train, y_train) -> None:
        self.clf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])

        self.clf_pipeline.fit(X_train, y_train)

    def train_regressor(self, X_train, y_train) -> None:
        self.reg_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ])
        
        self.reg_pipeline.fit(X_train, y_train)

    def evaluate_classifier(self, X_test, y_test) -> dict:
        y_pred = self.clf_pipeline.predict(X_test)
        y_prob = self.clf_pipeline.predict_proba(X_test)[:, 1]

        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Roc_auc': roc_auc_score(y_test, y_prob),
            'Classification_report': classification_report(y_test, y_pred),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred)
        }

    def evaluate_regressor(self, X_test, y_test) -> dict:
        y_pred = self.reg_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return {
            'rmse': round(np.sqrt(mse), 2),
            'r2': round(r2_score(y_test, y_pred), 4),
            'mae': round(mean_absolute_error(y_test, y_pred), 2)
        }

    def plot_results(self, X_test, y_test_clf, y_test_reg,
                     filename='credit_risk.png') -> None:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Credit Risk Model Results', fontsize=14, fontweight='bold')

        #[0, 0] confusion matrix
        y_pred = self.clf_pipeline.predict(X_test)
        cm = confusion_matrix(y_test_clf, y_pred)
        sns.heatmap(cm , annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(" Confuion Matrix")
        axes[0, 0].set_xlabel("Predicted")
        axes[0, 0].set_ylabel("Actual")

        #[0 , 1] ROC Curve
        y_prob = self.clf_pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_clf, y_prob)
        auc = roc_auc_score(y_test_clf, y_prob)
        axes[0, 1].plot(fpr, tpr, color='#0313A6', lw=2, label=f'AUC={auc:.3f}')
        axes[0, 1].plot([0,1],[0,1], '#F715AB', linestyle='--')
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].legend()

        #[1, 0] Predicted vs Actual Risk scores
        y_reg_pred = self.reg_pipeline.predict(X_test)
        axes[1, 0].scatter(y_test_reg, y_reg_pred, alpha=0.3, color='teal')
        axes[1, 0].plot([y_test_reg.min(), y_test_reg.max()],
                        [y_test_reg.min(), y_test_reg.max()], 'r--')
        axes[1, 0].set_title("Predicted vs Actual Risk Score")
        axes[1, 0].set_xlabel("Actual")
        axes[1, 0].set_ylabel("Predicted")

        #[1, 1] Feature Coefficients
        coefs = self.clf_pipeline.named_steps['model'].coef_[0]
        coef_series = pd.Series(coefs, index=self.FEATURES).sort_values()
        coef_series.plot(kind='barh', ax=axes[1, 1], color='steelblue')
        axes[1, 1].set_title("Logistic Regression  Coefficients")
        axes[1, 1].axvline(0, color='black', linewidth=0.8)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")

    def predict_risk(self, applicant: dict) -> dict:
        row = pd.DataFrame([applicant])[self.FEATURES]
        default_prob = self.clf_pipeline.predict_proba(row)[0, 1]
        risk_score   = self.reg_pipeline.predict(row)[0]

        if default_prob < 0.3:
            recommendation = 'Approve'
        elif default_prob < 0.5:
            recommendation = 'Review'
        else:
            recommendation = 'Decline'

        return {
            'default_probability': default_prob,
            'risk_score':          risk_score,
            'recommendation':      recommendation
        }
    

#Example Usage
features = CreditRiskModel.FEATURES
X = df[features]

# classification target
y_clf = df['defaulted']
X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# regression target
y_reg = df['risk_score']
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

model = CreditRiskModel(df)
model.train_classifier(X_train, y_train_clf)
model.train_regressor(X_train, y_train_reg)

clf_results = model.evaluate_classifier(X_test, y_test_clf)
print(f"Accuracy: {clf_results['Accuracy']:.4f}")
print(f"ROC-AUC: {clf_results['Roc_auc']:.4f}")
print(clf_results['Classification_report'])

reg_results = model.evaluate_regressor(X_test, y_test_reg)
print(f"RMSE: {reg_results['rmse']:.2f}")
print(f"R²: {reg_results['r2']:.4f}")

model.plot_results(X_test, y_test_clf, y_test_reg)

applicant = {
    'age': 35, 'income': 45000, 'credit_score': 580,
    'debt_ratio': 0.6, 'loan_amount': 25000, 'emp_years': 3
}
risk = model.predict_risk(applicant)
print(f"\nApplicant assessment:")
print(f"Default probability: {risk['default_probability']:.2%}")
print(f"Risk score: {risk['risk_score']:.0f}")
print(f"Recommendation: {risk['recommendation']}")