"""Rebuild the instant demo from the real semantic pipeline, never mock metrics."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import analyze

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    payload = (ROOT / 'data/sample_reviews.csv').read_bytes()
    data, quality, topics, topic_meta, sentiment_meta, trends, trend_meta, scores = analyze(payload, 'auto', True)
    snapshot = {
        'schema_version': 1, 'sample_sha256': hashlib.sha256(payload).hexdigest(),
        'data': json.loads(data.to_json(orient='records', date_format='iso')),
        'quality': quality, 'topics': topics, 'topic_meta': topic_meta,
        'sentiment_meta': sentiment_meta,
        'trends': json.loads(trends.to_json(orient='records')),
        'trend_meta': trend_meta, 'scores': json.loads(scores.to_json(orient='records')),
    }
    (ROOT / 'data/demo_analysis.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    print(f'Demo built from {len(data)} reviews: {topic_meta["method"]}, {sentiment_meta["method"]}')
