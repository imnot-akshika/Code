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
            ('processor', self.build_preprocessor),
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
            'n_estimators': [50,100],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }

        grid_search = GridSearchCV(
            pipe,
            param_grid,
            cv=5,
            scoring='r2',
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
            self.tune_best_model(X_test, y_test)
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
        ...

    def save(self, filepath: str) -> None:
        joblib.dump(self.best_model, filepath)

    def predict(self, customer: dict) -> dict:
        # takes a dict of customer features
        # returns churn_probability and risk_level:
        # 'High' if prob > 0.6, 'Medium' if > 0.3, 'Low' otherwise
        ...