import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# generate the dataset
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'size':     np.random.randint(500, 3000, n),
    'bedrooms': np.random.randint(1, 6, n),
    'age':      np.random.randint(0, 50, n),
    'location': np.random.choice(['urban', 'suburban', 'rural'], n),
    'condition': np.random.choice(['good', 'fair', 'poor'], n),
    'garage':   np.random.choice([True, False], n, p=[0.7, 0.3]),
})
df['price'] = (df['size'] * 200 +
               df['bedrooms'] * 15000 -
               df['age'] * 500 +
               (df['location'] == 'urban').astype(int) * 50000 +
               (df['condition'] == 'good').astype(int) * 30000 +
               np.random.normal(0, 20000, n)).round(2)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

class HousePriceModel:
    NUMERIC_FEATURES     = ['size', 'bedrooms', 'age']
    CATEGORICAL_FEATURES = ['location', 'condition']
    BOOL_FEATURES        = ['garage']

    def __init__(self):
        self.pipeline = None
        self.results  = {}

    def build_pipeline(self, model) -> Pipeline:

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.CATEGORICAL_FEATURES),
            ('bool', 'passthrough', self.BOOL_FEATURES)
        ])

        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    def train_and_evaluate(self, model, name: str,
                           X_train, y_train,
                           X_test, y_test) -> dict:
        pipe = self.build_pipeline(model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')

        result = {
            'name': name,
            'mae': round(mean_absolute_error(y_test, y_pred), 2),
            'rmse': round(np.sqrt(mse), 2),
            'r2': round(r2_score(y_test, y_pred), 4),
            'cv_r2': round(cv_scores.mean(), 4)
        }

        self.results[name] = result
        self.pipeline = pipe
        return result

    def compare_models(self, X_train, y_train,
                       X_test, y_test) -> pd.DataFrame:
        models = [
            (LinearRegression(), 'LinearRegression'),
            (Ridge(alpha=1.0), 'Ridge'),
            (RandomForestRegressor(n_estimators=100, random_state = 42), 'RandomForest')
        ]
        rows = []
        for model, name in models:
            result = self.train_and_evaluate(model, name,
                                            X_train, y_train,
                                            X_test, y_test)
            rows.append(result)
        return pd.DataFrame(rows).set_index('name')
    
    def tune_random_forest(self, X_train, y_train) -> dict:
        pipe = self.build_pipeline(RandomForestRegressor(random_state=42))
        param_grid = {
            'model__n_estimators': [50,100],
            'model__max_depth': [None,10,20]
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

        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        return{
            'params': best_params,
            'cv_score': best_cv_score
        }

    def save(self, filepath: str) -> None:
        joblib.dump(self.pipeline, filepath)

    def load(self, filepath: str) -> None:
        self.pipeline = joblib.load(filepath)


#Example Usage
hpm = HousePriceModel()

print("=== Model Comparison ===")
comparison = hpm.compare_models(X_train, y_train, X_test, y_test)
print(comparison.to_string())

print("\n=== Tuning Random Forest ===")
best = hpm.tune_random_forest(X_train, y_train)
print(f"Best params: {best['params']}")
print(f"Best CV R²: {best['cv_score']:.4f}")

hpm.save('house_model.pkl')
print("\nModel saved.")