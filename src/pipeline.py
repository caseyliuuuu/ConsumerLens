"""Shared existing analysis pipeline for uploads and reproducible demo snapshots."""
import io
import pandas as pd
from src.preprocessing import preprocess
from src.topic_modeling import discover_topics
from src.sentiment import analyze_sentiment
from src.trend_analysis import analyze_trends
from src.impact_scoring import score_topics


def analyze(payload: bytes, mode: str, text_sentiment: bool):
    """Run the unchanged analytical steps and return their measured results."""
    raw = pd.read_csv(io.BytesIO(payload))
    clean, quality = preprocess(raw)
    data, topics, topic_meta = discover_topics(clean, mode)
    data, sentiment_meta = analyze_sentiment(data, text_sentiment)
    trends, trend_meta = analyze_trends(data)
    scores = score_topics(data, trends)
    return data, quality, topics, topic_meta, sentiment_meta, trends, trend_meta, scores
