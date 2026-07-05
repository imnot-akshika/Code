import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split, cross_validate,
                                     GridSearchCV, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_curve, precision_recall_curve,
                             average_precision_score)

np.random.seed(42)
n = 3000

df = pd.DataFrame({
    'customer_id':    range(1001, 1001 + n),
    'age':            np.random.randint(18, 70, n),
    'tenure_months':  np.random.randint(1, 72, n),
    'monthly_charge': np.random.uniform(20, 120, n).round(2),
    'num_products':   np.random.randint(1, 5, n),
    'support_calls':  np.random.randint(0, 10, n),
    'contract_type':  np.random.choice(['monthly', 'annual', 'biennial'], n,
                                        p=[0.5, 0.3, 0.2]),
    'payment_method': np.random.choice(['card', 'bank', 'cash'], n),
    'has_partner':    np.random.choice([True, False], n),
    'has_dependents': np.random.choice([True, False], n, p=[0.3, 0.7]),
})

# introduce missing values
df.loc[np.random.choice(n, 80, replace=False), 'monthly_charge'] = np.nan
df.loc[np.random.choice(n, 50, replace=False), 'support_calls'] = np.nan

# realistic churn logic
churn_prob = (
    0.3 * (df['contract_type'] == 'monthly').astype(int) +
    0.2 * (df['support_calls'].fillna(0) > 5).astype(int) +
    0.15 * (df['tenure_months'] < 12).astype(int) +
    0.1  * (df['monthly_charge'].fillna(70) > 90).astype(int) -
    0.1  * (df['num_products'] > 2).astype(int)
)
df['churned'] = (churn_prob + np.random.uniform(0, 0.3, n) > 0.35).astype(int)

class ChurnPipeline:
    NUMERIC_FEATURES     = ['age', 'tenure_months', 'monthly_charge',
                            'num_products', 'support_calls']
    CATEGORICAL_FEATURES = ['contract_type', 'payment_method']
    BOOL_FEATURES        = ['has_partner', 'has_dependents']    
    TARGET               = 'churned'

    def __init__(self, df: pd.DataFrame):
        self.raw_df    = df.copy()
        self.clean_df  = None
        self.pipeline  = None
        self.best_model = None
        self.results   = {}

    def clean(self) -> pd.DataFrame:
        # fill missing monthly_charge with median
        # fill missing support_calls with 0
        # drop customer_id — not a feature
        # reset index
        df = self.raw_df.copy()
        df['monthly_charge'] = df['monthly_charge'].fillna(df['monthly_charge'].median())
        df['support_calls'] = df['support_calls'].fillna(0)
        df = df.drop(columns=['customer_id'])
        df = df.reset_index(drop=True)
        self.clean_df = df
        return self.clean_df

    def build_preprocessor(self) -> ColumnTransformer:
        # StandardScaler for numeric
        # OneHotEncoder for categorical
        # passthrough for bool
        if self.clean_df is None:
            self.clean()

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.CATEGORICAL_FEATURES),
            ('bool', 'passthrough', self.BOOL_FEATURES)
        ])
        return preprocessor

    def build_pipeline(self, model) -> Pipeline:
        self.pipeline = Pipeline([
            ('processor', self.build_preprocessor()),
            ('model', model)
        ])
        return self.pipeline

    def train_evaluate(self, X_train, y_train,
                       X_test, y_test) -> pd.DataFrame:
        # train and evaluate three models:
        # LogisticRegression, RandomForest, GradientBoosting
        # metrics: accuracy, f1, roc_auc, cv_roc_auc (5-fold)
        # store results, return as DataFrame
        rows = []
        for name, model in {'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
                      'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                      'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)}.items():
            pipeline = self.build_pipeline(model)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            cv_results = cross_validate(pipeline, X_train, y_train,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                        scoring='roc_auc', return_train_score=False)

            rows.append({
                'name': name,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'cv_roc_auc': np.mean(cv_results['test_score'])
            })
            self.results[name] = rows[-1]
        return pd.DataFrame(rows).set_index('name')



    def tune_best_model(self, X_train, y_train) -> dict:
        # GridSearchCV on GradientBoosting with pipeline
        # param grid: n_estimators=[50,100], max_depth=[3,5], learning_rate=[0.05,0.1]
        # return best params and best score
        pipe = self.build_pipeline(GradientBoostingClassifier(random_state=42))
        param_grid = {
            'model__n_estimators': [50,100],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.05, 0.1]
        }

        grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        best_params =grid_search.best_params_
        best_score = grid_search.best_score_
        self.best_model = grid_search.best_estimator_
        return {
            'params': best_params,
            'cv_score': best_score
        }

    def final_evaluation(self, X_test, y_test) -> dict:
        # evaluate self.best_model on test set
        # return accuracy, f1, roc_auc, classification_report, confusion_matrix
        if self.clean_df is None:
            self.clean()
        if self.best_model is None:
            raise ValueError("Run tune_best_model() first.")
        y_pred =self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def plot_dashboard(self, X_test, y_test,
                       filename: str = 'churn_dashboard.png') -> None:
        # 6-panel figure:
        # [0,0] model comparison bar chart (roc_auc)
        # [0,1] ROC curve of best model
        # [0,2] PR curve of best model
        # [1,0] confusion matrix of best model
        # [1,1] feature importance (if tree model)
        # [1,2] churn rate by contract_type (from clean_df)
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize = (10, 6))
        fig.suptitle('Churn Prediction Dashboard', fontsize=14, fontweight='bold')
        if self.best_model is None:
            self.tune_best_model(X_test, y_test)

        results_df = pd.DataFrame(self.results).T
        results_df['roc_auc'].astype(float).plot(
        kind='bar', ax=axes[0, 0], color='#0313A6')
        axes[0, 0].set_title("Model Comparison (ROC-AUC)")
        axes[0, 0].set_ylabel("ROC-AUC")
        axes[0, 0].tick_params(axis='x', rotation=30)

        y_pred =self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0, 1].plot(fpr, tpr, color='#F715AB', lw=2, label=f'AUC={roc_auc_score(y_test, y_prob):.3f}')
        axes[0, 1].set_title("ROC Curve")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].legend(fontsize=7)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        axes[0, 2].plot(recall, precision, color='#34EDF3', lw=2, label=f'AP={ap:.3f}')
        axes[0, 2].set_title("Precision-Recall Curves")
        axes[0, 2].set_xlabel("Recall")
        axes[0, 2].set_ylabel("Precision")
        axes[0, 2].legend(fontsize=7)

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title("Confusion Matrix")
        axes[1, 0].set_xlabel("Predicted")
        axes[1, 0].set_ylabel("Actual")

        if hasattr(self.best_model.named_steps['model'], 'feature_importances_'):
            importances = self.best_model.named_steps['model'].feature_importances_
            feature_names = self.best_model.named_steps['preprocessor'].get_feature_names_out()
            imp = pd.Series(importances, index=feature_names).sort_values().tail(10)
            imp.plot(kind='barh', ax=axes[1, 1], color='#F7A315')
            axes[1, 1].set_title("Feature Importances (Top 10)")

        churn_rate_by_contract = self.clean_df.groupby('contract_type')[self.TARGET].mean().sort_values(ascending=False)
        churn_rate_by_contract.plot(kind='barh', ax=axes[1, 2], color='#15F7A3')
        axes[1, 2].set_title("Churn Rate by Contract Type")
        axes[1, 2].set_xlabel("Churn Rate")
        axes[1, 2].set_ylabel("Contract Type")

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")

    def save(self, filepath: str) -> None:
        joblib.dump(self.best_model, filepath)

    def predict(self, customer: dict) -> dict:
        # takes a dict of customer features
        # returns churn_probability and risk_level:
        # 'High' if prob > 0.6, 'Medium' if > 0.3, 'Low' otherwise
        row = pd.DataFrame([customer])
        churn_probability = self.best_model.predict_proba(row)[:, 1][0]
        if churn_probability > 0.6:
            risk_level = 'High'
        elif churn_probability > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        return {'churn_probability': churn_probability, 'risk_level': risk_level}

#Example Usage
pipeline = ChurnPipeline(df)

# clean
clean = pipeline.clean()
print(f"Clean shape: {clean.shape}")
print(f"Churn rate: {clean['churned'].mean():.2%}")

# prepare features
feature_cols = (ChurnPipeline.NUMERIC_FEATURES +
                ChurnPipeline.CATEGORICAL_FEATURES +
                ChurnPipeline.BOOL_FEATURES)

X = clean[feature_cols]
y = clean[ChurnPipeline.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# train and compare
print("\n=== Model Comparison ===")
results = pipeline.train_evaluate(X_train, y_train, X_test, y_test)
print(results.to_string())

# tune best model
print("\n=== Tuning ===")
tuning = pipeline.tune_best_model(X_train, y_train)
print(f"Best params: {tuning['params']}")
print(f"Best CV ROC-AUC: {tuning['cv_score']:.4f}")

# final evaluation
print("\n=== Final Evaluation ===")
final = pipeline.final_evaluation(X_test, y_test)
print(f"ROC-AUC: {final['roc_auc']:.4f}")
print(f"F1: {final['f1']:.4f}")
print(final['classification_report'])

# dashboard
pipeline.plot_dashboard(X_test, y_test)

# save
pipeline.save('churn_model.pkl')
print("\nModel saved.")

# predict a new customer
new_customer = {
    'age': 28,
    'tenure_months': 3,
    'monthly_charge': 95.0,
    'num_products': 1,
    'support_calls': 7,
    'contract_type': 'monthly',
    'payment_method': 'cash',
    'has_partner': False,
    'has_dependents': False
}
prediction = pipeline.predict(new_customer)
print(f"\nNew customer prediction:")
print(f"Churn probability: {prediction['churn_probability']:.2%}")
print(f"Risk level: {prediction['risk_level']}")