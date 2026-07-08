from transformers import pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class TextAnalysisTookit:

    def __init__(self):
        # load three pipelines — this will download models on first run
        print("Loading models...")
        self.sentiment  = pipeline("sentiment-analysis")
        self.zero_shot  = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")
        self.ner        = pipeline("ner", grouped_entities=True)
        print("Models loaded.")

    def analyse_sentiment(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        results = self.sentiment(texts)
        for text, result in zip(texts, results):
            rows.append({
                'text': text,
                'label': result['label'],
                'score': round(result['score'], 4),
                'is_positive': result['label'] == 'POSITIVE'
            })
        return pd.DataFrame(rows)


    def classify_topics(self, texts: list[str],
                         topics: list[str]) -> pd.DataFrame:
        rows = []
        results = self.zero_shot(texts, candidate_labels=topics)
        for text, result in zip(texts, results):
            topic_scores = dict(zip(result['label'], result['scores']))
            row = {
                'test': text,
                'top_topic': result['label'][0],
                'confidence': round(result['scores'][0], 4)
            }
            for topic, score in topic_scores.item():
                row[topic] = round(score, 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def extract_entities(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        for text, entities in zip(texts, self.ner(texts)):
            for entity in entities:
                rows.append({
                    'text': text[:40] + '...',
                    'entity': entity['word'],
                    'type': entity['entity_group'],
                    'group': entity['entity_group']
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['text', 'entity', type, 'score']
        )


    def batch_analyse(self, texts: list[str],
                       topics: list[str]) -> dict:
        # run all three analyses
        # return dict with sentiment_df, topics_df, entities_df
        # and a summary dict:
        # positive_pct, most_common_topic, most_common_entity_type
        ...
        sentiment_df = self.analyse_sentiment(texts)
        topics_df = self.classify_topics(texts, topics)
        entities_df = self.extract_entities(texts)

        positive_pct = sentiment_df['is_positive'].mean() * 100
        most_common_topic = topics_df['top_topic'].value_counts().index[0]
        if len(entities_df) > 0:
            most_common_entity_type = entities_df['type'].value_counts().index[0]
        else:
            most_common_entity_type = "None"

        summary = {
            'positive_pct': round(positive_pct, 1),
            'most_common_topic': most_common_topic,
            'most_common_entity_type': most_common_entity_type,
            'total_entities_found': len(entities_df)
        }

        return { 
            'sentiment_df': sentiment_df,
            'topics_df': topics_df,
            'entities_df': entities_df,
            'summary': summary
        }


    def plot_analysis(self, results: dict,
                      filename: str = 'text_analysis.png') -> None:
        sentiment_df = results['sentiment_df']
        topics_df    = results['topics_df']
        entities_df  = results['entities_df']

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Text Analysis Dashboard', fontsize=14, fontweight='bold')

        # [0,0] sentiment distribution — count of POSITIVE vs NEGATIVE
        sentiment_counts = sentiment_df['label'].value_counts()
        axes[0, 0].bar(sentiment_counts.index, sentiment_counts.values,
                       color=['#34EDF3', '#F715AB'])
        axes[0, 0].set_title("Sentiment Distribution")
        axes[0, 0].set_ylabel("Count")

        # [0,1] topic distribution — how many texts per topic
        topic_counts = topics_df['top_topic'].value_counts()
        axes[0, 1].bar(topic_counts.index, topic_counts.values, color='#0313A6')
        axes[0, 1].set_title("Topic Distribution")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].tick_params(axis='x', rotation=30)

        # [1,0] entity type distribution — ORG, PER, LOC etc.
        if len(entities_df) > 0:
            entity_counts = entities_df['type'].value_counts()
            axes[1, 0].bar(entity_counts.index, entity_counts.values,
                           color='#9201CB')
            axes[1, 0].set_title("Entity Type Distribution")
            axes[1, 0].set_ylabel("Count")
        else:
            axes[1, 0].text(0.5, 0.5, 'No entities found',
                            ha='center', va='center', transform=axes[1, 0].transAxes)

        # [1,1] confidence histogram — distribution of sentiment scores
        # shows how confident the model was across all predictions
        axes[1, 1].hist(sentiment_df['score'], bins=10,
                        color='#F7A315', edgecolor='black', alpha=0.8)
        axes[1, 1].set_title("Sentiment Confidence Distribution")
        axes[1, 1].set_xlabel("Confidence Score")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved to {filename}")

# Example Usage
texts = [
    "Apple just released the best iPhone ever, I love it!",
    "The economy is struggling with high inflation and unemployment.",
    "Scientists at MIT discovered a breakthrough in cancer treatment.",
    "The new policy has caused massive protests across the country.",
    "Tesla stock crashed 15% today after poor earnings report.",
    "Manchester United won the championship in an amazing comeback!",
    "This restaurant has terrible service and overpriced food.",
    "The new Python 4.0 release includes incredible new features.",
    "Climate change is causing unprecedented flooding worldwide.",
    "The movie was absolutely brilliant, best film of the year!"
]

topics = ["technology", "finance", "health", "politics", "sports"]

toolkit = TextAnalysisToolkit()
results = toolkit.batch_analyse(texts, topics)

print("Sentiment Results:")
print(results['sentiment_df'].to_string())

print("\nTopic Results:")
print(results['topics_df'][['text', 'top_topic', 'confidence']].to_string())

print("\nEntity Results:")
print(results['entities_df'].head(10).to_string())

print("\nSummary:")
print(results['summary'])

toolkit.plot_analysis(results)
        