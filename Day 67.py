import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, accuracy_score,
                              roc_auc_score, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

np.random.seed(42)
n = 2000

customers = pd.DataFrame({
    'customer_id':    range(1001, 1001 + n),
    'age':            np.random.randint(18, 70, n),
    'tenure_months':  np.random.randint(1, 72, n),
    'monthly_spend':  np.random.uniform(20, 500, n).round(2),
    'num_products':   np.random.randint(1, 6, n),
    'support_calls':  np.random.randint(0, 15, n),
    'satisfaction':   np.random.randint(1, 6, n),
    'contract':       np.random.choice(['monthly','annual','biennial'], n,
                                        p=[0.5, 0.3, 0.2]),
    'region':         np.random.choice(['North','South','East','West'], n),
})

customers.loc[np.random.choice(n, 60, replace=False), 'monthly_spend'] = np.nan
customers.loc[np.random.choice(n, 40, replace=False), 'satisfaction'] = np.nan

churn_prob = (
    0.3 * (customers['contract'] == 'monthly').astype(int) +
    0.2 * (customers['support_calls'] > 5).astype(int) +
    0.15 * (customers['satisfaction'].fillna(3) < 3).astype(int) +
    0.1  * (customers['tenure_months'] < 12).astype(int) -
    0.1  * (customers['num_products'] > 3).astype(int)
)
customers['churned'] = (
    churn_prob + np.random.uniform(0, 0.3, n) > 0.4
).astype(int)

ticket_templates = [
    ("My bill is incorrect this month", "billing"),
    ("App keeps crashing on my phone", "technical"),
    ("Package not delivered yet", "shipping"),
    ("Can't login to my account", "account"),
    ("Charged twice for same item", "billing"),
    ("Website is very slow today", "technical"),
    ("Wrong item delivered", "shipping"),
    ("Need to reset my password", "account"),
    ("Unexpected charges on invoice", "billing"),
    ("Error message on checkout page", "technical"),
]

tickets = []
for cid in customers['customer_id']:
    n_tickets = np.random.randint(0, 4)
    for _ in range(n_tickets):
        tmpl = ticket_templates[np.random.randint(len(ticket_templates))]
        tickets.append({
            'customer_id': cid,
            'text':        tmpl[0],
            'category':    tmpl[1]
        })

tickets_df = pd.DataFrame(tickets)
print(f"Customers: {len(customers)}")
print(f"Tickets: {len(tickets_df)}")
print(f"Churn rate: {customers['churned'].mean():.2%}")


class CustomerIntelligenceSystem:

    NUMERIC_FEATURES     = ['age', 'tenure_months', 'monthly_spend',
                            'num_products', 'support_calls', 'satisfaction']
    CATEGORICAL_FEATURES = ['contract', 'region']
    TARGET               = 'churned'

    def __init__(self, customers: pd.DataFrame, tickets: pd.DataFrame):
        self.raw_customers = customers.copy()
        self.raw_tickets   = tickets.copy()
        self.clean_df      = None
        self.churn_model   = None
        self.ticket_model  = None
        self.results       = {}

    def clean_and_engineer(self) -> pd.DataFrame:
        df = self.raw_customers.copy()

        # fill missing values
        df['monthly_spend'] = df['monthly_spend'].fillna(df['monthly_spend'].median())
        df['satisfaction']  = df['satisfaction'].fillna(df['satisfaction'].mode()[0])

        # engineered features
        df['avg_spend_per_product'] = df['monthly_spend'] / df['num_products']
        df['support_rate']          = df['support_calls'] / (df['tenure_months'] + 1)
        df['high_value']            = (df['monthly_spend'] > 200).astype(int)

        # merge ticket count per customer
        ticket_counts = (self.raw_tickets.groupby('customer_id')
                         .size()
                         .reset_index(name='ticket_count'))
        df = df.merge(ticket_counts, on='customer_id', how='left')
        df['ticket_count'] = df['ticket_count'].fillna(0)

        df = df.reset_index(drop=True)
        self.clean_df = df
        return df

    def eda_report(self) -> dict:
        df = self.clean_df
        return {
            'churn_rate':            df['churned'].mean(),
            'avg_spend':             df['monthly_spend'].mean(),
            'avg_tenure':            df['tenure_months'].mean(),
            'churn_by_contract':     df.groupby('contract')['churned'].mean().round(3),
            'churn_by_region':       df.groupby('region')['churned'].mean().round(3),
            'top_ticket_categories': self.raw_tickets['category'].value_counts()
        }

    def train_churn_model(self, X_train, y_train) -> None:
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(),
             self.NUMERIC_FEATURES + ['avg_spend_per_product',
                                       'support_rate', 'ticket_count']),
            ('cat', OneHotEncoder(handle_unknown='ignore'),
             self.CATEGORICAL_FEATURES)
        ])
        self.churn_model = Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=4, random_state=42
            ))
        ])
        self.churn_model.fit(X_train, y_train)

    def train_ticket_classifier(self, df: pd.DataFrame) -> None:
        self.ticket_model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf',   LogisticRegression(max_iter=1000, random_state=42))
        ])
        self.ticket_model.fit(df['text'], df['category'])

    def evaluate_churn_model(self, X_test, y_test) -> dict:
        y_pred = self.churn_model.predict(X_test)
        y_prob = self.churn_model.predict_proba(X_test)[:, 1]
        return {
            'accuracy':         accuracy_score(y_test, y_pred),
            'roc_auc':          roc_auc_score(y_test, y_prob),
            'report':           classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def predict_customer_risk(self, customer_id: int) -> dict:
        customer = self.clean_df[self.clean_df['customer_id'] == customer_id]
        if len(customer) == 0:
            raise ValueError(f"Customer {customer_id} not found")

        feature_cols = (self.NUMERIC_FEATURES +
                        ['avg_spend_per_product', 'support_rate', 'ticket_count'] +
                        self.CATEGORICAL_FEATURES)
        X = customer[feature_cols]

        churn_prob = self.churn_model.predict_proba(X)[0, 1]

        if churn_prob > 0.6:
            risk_level = 'High'
        elif churn_prob > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'

        cust_tickets = self.raw_tickets[
            self.raw_tickets['customer_id'] == customer_id
        ]
        if len(cust_tickets) > 0:
            latest_text      = cust_tickets.iloc[-1]['text']
            ticket_category  = self.ticket_model.predict([latest_text])[0]
        else:
            ticket_category  = 'none'

        if risk_level == 'High' and ticket_category == 'billing':
            action = 'Immediate billing review + retention offer'
        elif risk_level == 'High' and ticket_category == 'technical':
            action = 'Priority technical support + account manager call'
        elif risk_level == 'High':
            action = 'Urgent retention call — offer discount'
        elif risk_level == 'Medium':
            action = 'Proactive check-in email + loyalty reward'
        else:
            action = 'Standard engagement — no action needed'

        return {
            'customer_id':            customer_id,
            'churn_probability':      float(churn_prob),
            'risk_level':             risk_level,
            'latest_ticket_category': ticket_category,
            'recommended_action':     action
        }

    def plot_dashboard(self, X_test, y_test,
                       filename: str = 'intelligence_dashboard.png') -> None:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Customer Intelligence Dashboard',
                     fontsize=14, fontweight='bold')

        # [0,0] churn rate by contract
        churn_contract = (self.clean_df.groupby('contract')['churned']
                          .mean().sort_values())
        axes[0, 0].bar(churn_contract.index, churn_contract.values,
                       color='#F715AB')
        axes[0, 0].set_title("Churn Rate by Contract")
        axes[0, 0].set_ylabel("Churn Rate")

        # [0,1] monthly spend by churn status
        sns.boxplot(data=self.clean_df, x='churned',
                    y='monthly_spend', ax=axes[0, 1])
        axes[0, 1].set_title("Spend by Churn Status")
        axes[0, 1].set_xticklabels(['Retained', 'Churned'])

        # [0,2] confusion matrix
        y_pred = self.churn_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
        axes[0, 2].set_title("Confusion Matrix")
        axes[0, 2].set_xlabel("Predicted")
        axes[0, 2].set_ylabel("Actual")

        # [1,0] feature importance
        model_step   = self.churn_model.named_steps['model']
        preprocessor = self.churn_model.named_steps['preprocessor']
        cat_features = list(preprocessor.named_transformers_['cat']
                            .get_feature_names_out())
        feature_names = (self.NUMERIC_FEATURES +
                         ['avg_spend_per_product', 'support_rate', 'ticket_count'] +
                         cat_features)
        importances = pd.Series(
            model_step.feature_importances_, index=feature_names
        ).sort_values().tail(10)
        importances.plot(kind='barh', ax=axes[1, 0], color='#0313A6')
        axes[1, 0].set_title("Top 10 Feature Importances")

        # [1,1] churn rate by region
        churn_region = (self.clean_df.groupby('region')['churned']
                        .mean().sort_values())
        axes[1, 1].bar(churn_region.index, churn_region.values,
                       color='#34EDF3')
        axes[1, 1].set_title("Churn Rate by Region")
        axes[1, 1].set_ylabel("Churn Rate")

        # [1,2] ticket category distribution
        if len(self.raw_tickets) > 0:
            cat_counts = self.raw_tickets['category'].value_counts()
            axes[1, 2].bar(cat_counts.index, cat_counts.values,
                           color='#9201CB')
            axes[1, 2].set_title("Ticket Category Distribution")
            axes[1, 2].set_ylabel("Count")
            axes[1, 2].tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")

    def generate_report(self, X_test, y_test) -> str:
        eval_r = self.evaluate_churn_model(X_test, y_test)
        eda    = self.eda_report()
        report = f"""
╔══════════════════════════════════════════════════════════╗
║         CUSTOMER INTELLIGENCE SYSTEM — REPORT           ║
╚══════════════════════════════════════════════════════════╝

DATASET OVERVIEW
─────────────────
Total customers  : {len(self.clean_df)}
Total tickets    : {len(self.raw_tickets)}
Churn rate       : {eda['churn_rate']:.2%}
Avg monthly spend: ${eda['avg_spend']:.2f}
Avg tenure       : {eda['avg_tenure']:.1f} months

MODEL PERFORMANCE
─────────────────
Churn Model : GradientBoosting
Accuracy    : {eval_r['accuracy']:.4f}
ROC-AUC     : {eval_r['roc_auc']:.4f}

CHURN BY CONTRACT
─────────────────
{eda['churn_by_contract'].to_string()}

CHURN BY REGION
───────────────
{eda['churn_by_region'].to_string()}

TOP TICKET CATEGORIES
─────────────────────
{eda['top_ticket_categories'].to_string()}
"""
        return report

    def save_all(self, prefix: str = 'customer_intelligence') -> None:
        joblib.dump(self.churn_model,  f'{prefix}_churn.pkl')
        joblib.dump(self.ticket_model, f'{prefix}_tickets.pkl')
        print(f"Saved {prefix}_churn.pkl and {prefix}_tickets.pkl")


# ── Test code ────────────────────────────────────────────
system = CustomerIntelligenceSystem(customers, tickets_df)

print("=== Cleaning & Engineering ===")
clean = system.clean_and_engineer()
print(f"Clean shape: {clean.shape}")
print(f"New features: {[c for c in clean.columns if c not in customers.columns]}")

print("\n=== EDA Report ===")
eda = system.eda_report()
print(f"Churn rate: {eda['churn_rate']:.2%}")
print(f"Avg spend: ${eda['avg_spend']:.2f}")
print(f"\nChurn by contract:\n{eda['churn_by_contract']}")
print(f"\nTop ticket categories:\n{eda['top_ticket_categories']}")

feature_cols = (CustomerIntelligenceSystem.NUMERIC_FEATURES +
                ['avg_spend_per_product', 'support_rate', 'ticket_count'] +
                CustomerIntelligenceSystem.CATEGORICAL_FEATURES)

X = clean[feature_cols]
y = clean[CustomerIntelligenceSystem.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== Training Models ===")
system.train_churn_model(X_train, y_train)
system.train_ticket_classifier(tickets_df)
print("Models trained.")

print("\n=== Churn Model Evaluation ===")
eval_results = system.evaluate_churn_model(X_test, y_test)
print(f"Accuracy: {eval_results['accuracy']:.4f}")
print(f"ROC-AUC:  {eval_results['roc_auc']:.4f}")
print(eval_results['report'])

print("\n=== Customer Risk Predictions ===")
for cid in [1001, 1005, 1010, 1050, 1100]:
    try:
        risk = system.predict_customer_risk(cid)
        print(f"Customer {cid}: {risk['risk_level']} risk "
              f"({risk['churn_probability']:.2%}) — {risk['recommended_action']}")
    except Exception as e:
        print(f"Customer {cid}: {e}")

system.plot_dashboard(X_test, y_test)

report = system.generate_report(X_test, y_test)
print("\n=== System Report ===")
print(report)

system.save_all()
print("\nAll models saved.")