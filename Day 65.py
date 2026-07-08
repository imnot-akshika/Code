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
        result = self.sentiment(texts)
        rows.append(result)
        return pd.DataFrame(rows)


    def classify_topics(self, texts: list[str],
                         topics: list[str]) -> pd.DataFrame:
        rows = []
        result = self.zero_shot(texts, candidate_labels=topics)
        rows.append(result)
        return pd.Dataframe(rows)

    def extract_entities(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        result = self.ner(texts)
        for entity in result:
            rows.append({
                'text': entity['text'],
                'entity': entity,
                'type': entity['entity_type'],
                'group': entity['entity_group']
            })
        return pd.DataFrame(rows)


    def batch_analyse(self, texts: list[str],
                       topics: list[str]) -> dict:
        # run all three analyses
        # return dict with sentiment_df, topics_df, entities_df
        # and a summary dict:
        # positive_pct, most_common_topic, most_common_entity_type
        ...
        sentiment_df = []
        result_st = self.sentiment(texts, max_length=50, num_return_swquences=1)

        topics_df = []
        result_tp = self.zero_shot(topics)

        entities_df = []
        result_en = self.ner(texts)

        summary = {
            
        }


    def plot_analysis(self, results: dict,
                      filename: str = 'text_analysis.png') -> None:
        # 2×2 figure:
        # [0,0] sentiment distribution bar chart
        # [0,1] topic distribution bar chart
        # [1,0] entity type distribution
        # [1,1] sentiment confidence histogram
        ...