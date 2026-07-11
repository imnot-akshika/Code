import pandas as pd
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

np.random.seed(42)

tickets = [
    # billing
    ("My credit card was charged twice for the same order", "billing", "high"),
    ("I need a refund for order #12345", "billing", "medium"),
    ("Wrong amount charged on my invoice", "billing", "high"),
    ("I can't find my payment receipt", "billing", "low"),
    ("Subscription renewal failed payment", "billing", "medium"),
    ("I was overcharged for my monthly plan", "billing", "high"),
    ("Please update my billing address", "billing", "low"),
    ("Why is my bill higher than usual this month", "billing", "medium"),
    # technical
    ("The app keeps crashing when I open it", "technical", "high"),
    ("I can't log into my account", "technical", "high"),
    ("Password reset email never arrived", "technical", "medium"),
    ("The website is down and I can't access anything", "technical", "high"),
    ("My data is not syncing between devices", "technical", "medium"),
    ("Error code 500 appearing on checkout", "technical", "high"),
    ("App is very slow and unresponsive", "technical", "medium"),
    ("Feature X stopped working after the update", "technical", "medium"),
    # shipping
    ("My order hasn't arrived after 2 weeks", "shipping", "high"),
    ("Package was delivered to wrong address", "shipping", "high"),
    ("Tracking number shows delivered but I got nothing", "shipping", "high"),
    ("Can I change my delivery address", "shipping", "medium"),
    ("When will my order ship", "shipping", "low"),
    ("Package arrived damaged", "shipping", "high"),
    ("I need express shipping for urgent order", "shipping", "medium"),
    ("My order is stuck in customs", "shipping", "medium"),
    # account
    ("I want to delete my account permanently", "account", "medium"),
    ("How do I change my email address", "account", "low"),
    ("My account was hacked, please help", "account", "high"),
    ("I forgot my username", "account", "low"),
    ("Please merge my two accounts", "account", "medium"),
    ("I can't update my profile picture", "account", "low"),
    ("My account is locked after too many login attempts", "account", "high"),
    ("How do I upgrade my subscription plan", "account", "low"),
]

# expand dataset with variations
expanded = []
for text, category, urgency in tickets:
    expanded.append((text, category, urgency))
    expanded.append((f"URGENT: {text}", category, "high"))
    expanded.append((f"Hi, {text.lower()}. Please help!", category, urgency))

df = pd.DataFrame(expanded, columns=['text', 'category', 'urgency'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset: {len(df)} tickets")
print(df['category'].value_counts())
print(df['urgency'].value_counts())

class TicketClassifier:

    def __init__(self):
        self.category_model = None
        self.urgency_model  = None
        self.vectorizer     = None
        self.zero_shot      = None
        self.results        = {}

    def train_tfidf_models(self, df: pd.DataFrame) -> dict:
        # TF-IDF + Logistic Regression for category
        # TF-IDF + Random Forest for urgency
        # share the same TfidfVectorizer
        # return dict of classification reports
        self.category_model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

        self.urgency_model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        self.category_model.fit(df['text'], df['category'])
        self.urgency_model.fit(df['text'], df['urgency'])

        cat_pred = self.category_model.predict(df['text'])
        urg_pred = self.urgency_model.predict(df['text'])

        return {
            'category': classification_report(df['category'], cat_pred),
            'urgency':  classification_report(df['urgency'], urg_pred)
        }


    def zero_shot_classify(self, texts: list[str],
                            categories: list[str]) -> pd.DataFrame:
        # use HuggingFace zero-shot pipeline
        # no training needed — works on categories it's never seen
        # return DataFrame: text, predicted_category, confidence
        if self.zero_shot is None:
            self.zero_shot  = pipeline("zero-shot-classification",
                                       model="facebook/bart-large-mnli")
            
        results = self.zero_shot(texts, candidate_label=categories)

        rows = []
        for text, result in zip(texts, results):
            rows.append({
                'text': text[:50] + '...',
                'predicted_category': result['labels'][0],
                'confidence': round(result['scores'][0], 4)
            })
        return pd.DataFrame(rows)
        

    def compare_approaches(self, X_test, y_test_cat,
                            y_test_urg) -> pd.DataFrame:
        # compare TF-IDF model vs zero-shot on category prediction
        # return DataFrame with accuracy scores for both
        tfidf_cat_predict = self.category_model.predict(X_test)
        tfidf_accuracy = accuracy_score(y_test_cat, tfidf_cat_predict)

        categories = list(y_test_cat.unique())
        zs_df = self.zero_shot_classify(X_test.tolist(), categories)
        zs_accuracy = accuracy_score(
            y_test_cat.tolist(),
            zs_df['predcisted_category'].tolist()
        )

        return pd.DataFrame([
            {'approach': 'TF-IDF + LogReg', 'category_accuracy': round(tfidf_accuracy, 4)},
            {'approach': 'Zero-Shot',       'category_accuracy': round(zs_accuracy, 4)}
        ])

    def predict(self, ticket: str) -> dict:
        # predict category and urgency for a single ticket
        # return: category, urgency, confidence, priority_score
        # priority_score: high urgency = 3, medium = 2, low = 1
        #                 multiply by category weight:
        #                 technical=1.5, billing=1.3, shipping=1.2, account=1.0
        category = self.category_model.predict([ticket])[0]
        urgency = self.urgency_model.predict([ticket])[0]

        cat_proba = self.category_model.predict_proba([ticket])
        cat_classes = self.category_model.classes_
        confidence = max(cat_proba[0])

        urgency_score = {'high': 3, "medium": 2, "low": 1}.get(urgency, 1)

        category_weight = {
            'technical': 1.5,
            'billing': 1.3,
            'shipping': 1.2,
            'account': 1.0
        }.get(category, 1.0)

        priority_score = urgency_score * category_weight

        return {
            'category': category,
            'urgency': urgency,
            'confidence': round(float(confidence), 4),
            'priority_score': round(priority_score, 1)
        }

    def plot_results(self, df: pd.DataFrame,
                     filename: str = 'classifier_results.png') -> None:
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize = (14, 10))
        fig.suptitle('Ticket Classifier Results', fontsize = 14, fontweight = 'bold')

        cat_counts = df['category'].value_counts()
        axes[0, 0].bar(cat_counts.index, cat_counts.values, color='#0313A6')
        axes[0, 0].set_title("Tickets by Category")
        axes[0, 0].set_ylabel("Count")  

        urg_counts = df['urgency'].value_counts()
        colors = {'high': '#F715AB', 'medium': '#F7A315', 'low': '#34EDF3'}
        axes[0, 1].bar(urg_counts.index,
                       urg_counts.values,
                       color=[colors.get(u, 'gray') for u in urg_counts.index])
        axes[0, 1].set_title("Tickets by Urgency")
        axes[0, 1].set_ylabel("Count")

        pivot = pd.pivot_table(df, values='text',
                               index='category', columns='urgency',
                               aggfunc='count', fill_value=0)
        sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title("Category × Urgency Heatmap")

        df_plot = df.copy()
        df_plot['length'] = df_plot['text'].str.len()
        sns.boxplot(data=df_plot, x='category', y='length', ax=axes[1, 1])
        axes[1, 1].set_title("Ticket Length by Category")
        axes[1, 1].set_xlabel("Category")
        axes[1, 1].set_ylabel("Character Count")

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")


    def save(self, filepath: str) -> None:
        joblib.dump({
            'category_model': self.category_model,
            'urgency_model':  self.urgency_model,
            'vectorizer':     self.vectorizer
        }, filepath)


#Example Usage
classifier = TicketClassifier()

# split data
X = df['text']
y_cat = df['category']
y_urg = df['urgency']

X_train, X_test, y_train_cat, y_test_cat, y_train_urg, y_test_urg = train_test_split(
    X, y_cat, y_urg, test_size=0.2, random_state=42, stratify=y_cat
)

# train
print("=== Training TF-IDF Models ===")
reports = classifier.train_tfidf_models(
    pd.DataFrame({'text': X_train, 'category': y_train_cat, 'urgency': y_train_urg})
)
print("Category model:")
print(reports['category'])
print("\nUrgency model:")
print(reports['urgency'])

# zero-shot
print("\n=== Zero-Shot Classification ===")
categories = ['billing', 'technical', 'shipping', 'account']
zs_results = classifier.zero_shot_classify(X_test.tolist()[:5], categories)
print(zs_results.to_string())

# predict individual tickets
print("\n=== Predictions ===")
test_tickets = [
    "My account was hacked and I can't get back in",
    "The website is completely down since this morning",
    "I was charged three times for one purchase",
    "My delivery is 3 weeks late",
]
for ticket in test_tickets:
    pred = classifier.predict(ticket)
    print(f"\nTicket: {ticket}")
    print(f"Category: {pred['category']} | Urgency: {pred['urgency']} | Priority: {pred['priority_score']:.1f}")

# plot
classifier.plot_results(df)

# save
classifier.save('ticket_classifier.pkl')
print("\nSaved.")