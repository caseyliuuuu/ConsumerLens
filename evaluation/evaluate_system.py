"""Reproducible, descriptive evaluation of the bundled ConsumerLens demo."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_topics import evaluate as evaluate_topics  # noqa: E402
from src.insight_agent import generate_insights  # noqa: E402
from src.impact_scoring import score_topics  # noqa: E402
from src.trend_analysis import analyze_trends  # noqa: E402


def load_snapshot(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    data = pd.DataFrame(snapshot["data"])
    data["date"] = pd.to_datetime(data.date, utc=True)
    scores = pd.DataFrame(snapshot["scores"])
    topics = {int(key): value for key, value in snapshot["topics"].items()}
    return snapshot, data, scores, topics


def evaluate(snapshot_path: Path, validation_path: Path) -> tuple[dict, pd.DataFrame]:
    snapshot, data, scores, topics = load_snapshot(snapshot_path)
    topic_diagnostics = evaluate_topics(data)
    topic_rows = topic_diagnostics["topics"]
    total = len(data)
    weighted_cohesion = sum(row["lexical_cohesion_proxy"] * row["count"] for row in topic_rows) / total
    unweighted_cohesion = float(np.mean([row["lexical_cohesion_proxy"] for row in topic_rows]))

    known_sentiment = data.sentiment_label.ne("Unknown")
    assigned = data.topic_id.ne(-1)

    first_brief = generate_insights(data, scores, topics, False)
    second_brief = generate_insights(data, scores, topics, False)
    canonical_first = json.dumps(first_brief, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    canonical_second = json.dumps(second_brief, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    known_ids = set(data.review_id)
    topic_by_review = data.set_index("review_id").topic_id.to_dict()
    grounded = []
    cited_ids = []
    for insight in first_brief["insights"]:
        ids = insight["review_ids"]
        cited_ids.extend(ids)
        grounded.append(bool(ids) and set(ids) <= known_ids and all(topic_by_review[review_id] == insight["topic_id"] for review_id in ids))

    trends, _ = analyze_trends(data)
    rescored = score_topics(data, trends).set_index("topic_id").sort_index()
    saved = scores.set_index("topic_id").sort_index()
    max_score_difference = float((rescored.impact_score - saved.impact_score).abs().max())

    validation = pd.read_csv(validation_path, keep_default_na=False)
    labeled = validation.human_topic.astype(str).str.strip().ne("") & validation.human_sentiment.astype(str).str.strip().ne("")

    result = {
        "evaluation_scope": {
            "dataset": "Bundled synthetic Aster Slate 10 reviews",
            "synthetic_data": True,
            "valid_reviews": total,
            "topic_groups_including_outliers": len(scores),
            "topic_method": snapshot["topic_meta"]["method"],
            "sentiment_method": snapshot["sentiment_meta"]["method"],
            "sample_sha256": snapshot["sample_sha256"],
        },
        "topic_quality": {
            "metric": "TF-IDF document-to-topic-centroid cosine similarity proxy",
            "review_weighted_mean": weighted_cohesion,
            "unweighted_topic_mean": unweighted_cohesion,
            "per_topic": topic_rows,
            "interpretation": "A descriptive lexical cohesion proxy, not human topic accuracy or semantic coherence.",
        },
        "coverage": {
            "sentiment_coverage_pct": float(known_sentiment.mean() * 100),
            "topic_assignment_coverage_pct": float(assigned.mean() * 100),
            "outlier_rate_pct": float((~assigned).mean() * 100),
            "outlier_review_count": int((~assigned).sum()),
            "interpretation": "Coverage reports whether a label exists; it does not measure whether that label is correct.",
        },
        "recommendation_grounding": {
            "recommendation_count": len(first_brief["insights"]),
            "recommendations_with_valid_same_topic_citations": int(sum(grounded)),
            "evidence_citation_rate_pct": float(np.mean(grounded) * 100) if grounded else None,
            "unique_cited_review_count": len(set(cited_ids)),
            "interpretation": "Checks citation presence, existence, and same-topic linkage; it does not judge recommendation usefulness.",
        },
        "deterministic_reproducibility": {
            "briefs_identical_across_two_runs": canonical_first == canonical_second,
            "brief_sha256": hashlib.sha256(canonical_first.encode("utf-8")).hexdigest(),
            "max_impact_score_difference_after_recalculation": max_score_difference,
            "interpretation": "Two same-input deterministic brief runs are compared byte-for-byte after canonical JSON serialization; impact scores are recalculated from the saved review assignments.",
        },
        "manual_validation_status": {
            "sample_rows": len(validation),
            "rows_with_both_human_labels": int(labeled.sum()),
            "accuracy_reported": False,
            "interpretation": "Human fields are intentionally blank, so no topic or sentiment accuracy is claimed.",
        },
        "limitations": [
            "The included dataset is synthetic.",
            "Lexical cohesion is a proxy and does not establish topic correctness.",
            "Coverage and citation validity do not measure model accuracy or business value.",
            "BERTopic stability across repeated stochastic fits is not measured here.",
            "No LLM quality benchmark is reported because no human rubric labels are available.",
        ],
    }
    summary = pd.DataFrame(
        [
            {"metric": "Lexical topic cohesion proxy (weighted mean)", "value": weighted_cohesion, "unit": "cosine similarity"},
            {"metric": "Sentiment coverage", "value": known_sentiment.mean() * 100, "unit": "percent"},
            {"metric": "Topic assignment coverage", "value": assigned.mean() * 100, "unit": "percent"},
            {"metric": "BERTopic outlier rate", "value": (~assigned).mean() * 100, "unit": "percent"},
            {"metric": "Recommendation evidence citation rate", "value": np.mean(grounded) * 100 if grounded else np.nan, "unit": "percent"},
            {"metric": "Deterministic brief reproducibility", "value": int(canonical_first == canonical_second), "unit": "boolean (1=true)"},
            {"metric": "Maximum recalculated impact-score difference", "value": max_score_difference, "unit": "score points"},
            {"metric": "Completed human validation rows", "value": int(labeled.sum()), "unit": f"of {len(validation)}"},
        ]
    )
    return result, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/demo_analysis.json")
    parser.add_argument("--validation", type=Path, default=ROOT / "evaluation/sample_validation.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/system_evaluation.json")
    args = parser.parse_args()
    results, summary_table = evaluate(args.snapshot, args.validation)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = args.output.with_suffix(".csv")
    summary_table.to_csv(summary_path, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")
