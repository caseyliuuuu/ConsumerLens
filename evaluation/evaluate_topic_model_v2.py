"""Evaluate template-aware BERTopic assignments without altering V1 baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.impact_scoring import score_topics  # noqa: E402
from src.preprocessing import preprocess  # noqa: E402
from src.topic_modeling import discover_topics  # noqa: E402
from src.trend_analysis import analyze_trends  # noqa: E402


# Product-area vocabulary is defined independently of the validation rows. It is
# used only to name unsupervised clusters after BERTopic has assigned reviews.
BUSINESS_TOPIC_TERMS = {
    "Battery Life": ("battery", "drain", "dies", "idle", "power", "lasts", "holds up"),
    "Charging": ("charging", "charge", "charger", "cable"),
    "Price & Value": ("price", "value", "cost", "worth", "expensive", "cheap", "buy", "refund"),
    "Display": ("display", "screen", "brightness", "colors", "crisp", "reading outside"),
    "Build Quality": ("build quality", "metal body", "sturdy", "assembled", "buttons", "plastic", "hinge"),
    "Customer Service": ("customer service", "customer support", "support", "warranty", "replacement process", "replied"),
    "Accessories": ("accessory", "keyboard", "stylus", "case", "pairs"),
    "Usability & Interface": ("interface", "settings are easy", "navigation", "easy to find"),
    "Software & Updates": ("software", "update", "crash", "restart", "notification", "losing my settings", "freeze"),
    "Software & Performance": ("performance", "processor", "apps open", "multitasking", "games", "stutter", "overheat"),
}

BASELINE_LABELS = {
    "Mixed everyday experiences": "Mixed / contextual",
    "Build quality": "Build Quality",
    "Interface & settings": "Usability & Interface",
    "Software update stability": "Software & Updates",
    "Screen clarity & color": "Display",
    "App freezes & lost settings": "Software & Updates",
    "App speed & multitasking": "Software & Performance",
    "Gaming performance": "Software & Performance",
    "Outdoor screen visibility": "Display",
    "Battery life": "Battery Life",
    "Keyboard accessories": "Accessories",
    "Unassigned / mixed": "Unassigned / mixed",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_unsupervised_clusters(data: pd.DataFrame) -> dict[int, str]:
    """Name clusters from their topic_text only; human labels are not available here."""
    labels: dict[int, str] = {}
    for topic_id, group in data.groupby("topic_id", sort=True):
        if int(topic_id) == -1:
            labels[int(topic_id)] = "Unassigned / mixed"
            continue
        corpus = " ".join(group.topic_text.astype(str)).casefold()
        scores = {
            label: sum(corpus.count(term.casefold()) for term in terms)
            for label, terms in BUSINESS_TOPIC_TERMS.items()
        }
        best_score = max(scores.values())
        labels[int(topic_id)] = max(scores, key=scores.get) if best_score else "Other / mixed"
    return labels


def parse_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().map({"true": True, "false": False})


def evaluate_v2(
    raw_path: Path,
    baseline_snapshot_path: Path,
    validation_path: Path,
    baseline_labels_path: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """Fit V2 before opening held-out labels, then compare on the fixed 50 rows."""
    raw = pd.read_csv(raw_path)
    clean, quality = preprocess(raw)
    original_reviews = clean.review_text.copy(deep=True)
    v2, topics, topic_meta = discover_topics(clean, "auto")
    if not v2.review_text.equals(original_reviews):
        raise AssertionError("V2 changed review_text; only topic_text may be cleaned.")

    cluster_labels = label_unsupervised_clusters(v2)
    v2["topic_model_label"] = v2["topic"]
    v2["topic"] = v2.topic_id.map(cluster_labels)
    for topic_id, details in topics.items():
        details["model_label"] = details["label"]
        details["label"] = cluster_labels[int(topic_id)]

    # Reuse V1 sentiment outputs verbatim so the experiment isolates topic modeling.
    baseline_snapshot = json.loads(baseline_snapshot_path.read_text(encoding="utf-8"))
    baseline_data = pd.DataFrame(baseline_snapshot["data"])
    sentiment_columns = ["review_id", "sentiment_label", "sentiment_score", "sentiment_method"]
    v2 = v2.merge(baseline_data[sentiment_columns], on="review_id", how="left", validate="one_to_one")
    trends, trend_meta = analyze_trends(v2)
    scores = score_topics(v2, trends)

    baseline_display = json.loads(baseline_labels_path.read_text(encoding="utf-8"))
    baseline_data["baseline_display_topic"] = baseline_data.topic.map(baseline_display).fillna(baseline_data.topic)
    baseline_data["baseline_business_topic"] = baseline_data.baseline_display_topic.map(BASELINE_LABELS).fillna(baseline_data.baseline_display_topic)
    assignments = v2[["review_id", "review_text", "topic_text", "topic_id", "topic_model_label", "topic"]].rename(
        columns={"topic_id": "v2_topic_id", "topic": "v2_business_topic"}
    )
    assignments = assignments.merge(
        baseline_data[["review_id", "topic_id", "topic", "baseline_display_topic", "baseline_business_topic"]].rename(
            columns={"topic_id": "baseline_topic_id", "topic": "baseline_model_topic"}
        ),
        on="review_id",
        validate="one_to_one",
    )
    assignments["business_topic_changed"] = assignments.v2_business_topic.ne(assignments.baseline_business_topic)

    # Held-out labels are loaded only after preprocessing, fitting, and cluster naming.
    validation = pd.read_csv(validation_path, keep_default_na=False)
    if len(validation) != 50 or validation.human_topic.astype(str).str.strip().eq("").any():
        raise ValueError("Expected the fixed 50-row, fully labeled validation file.")
    evaluated = validation.merge(
        assignments[["review_text", "baseline_business_topic", "v2_topic_id", "topic_model_label", "v2_business_topic"]],
        on="review_text",
        how="left",
        validate="one_to_one",
    )
    evaluated["baseline_topic_correct"] = parse_bool(evaluated.topic_correct)
    evaluated["v2_topic_correct"] = evaluated.v2_business_topic.eq(evaluated.human_topic)
    evaluated["assignment_changed"] = evaluated.baseline_business_topic.ne(evaluated.v2_business_topic)

    rows = []
    for human_topic, group in evaluated.groupby("human_topic", sort=True):
        predictions = group.v2_business_topic.value_counts().sort_index().to_dict()
        errors = group.loc[~group.v2_topic_correct, "v2_business_topic"].value_counts().sort_index().to_dict()
        rows.append(
            {
                "human_topic": human_topic,
                "reviews": len(group),
                "baseline_correct": int(group.baseline_topic_correct.sum()),
                "baseline_accuracy_pct": float(group.baseline_topic_correct.mean() * 100),
                "v2_correct": int(group.v2_topic_correct.sum()),
                "v2_accuracy_pct": float(group.v2_topic_correct.mean() * 100),
                "v2_prediction_breakdown": json.dumps(predictions, sort_keys=True),
                "v2_error_breakdown": json.dumps(errors, sort_keys=True),
            }
        )
    confusion = pd.DataFrame(rows)

    baseline_accuracy = float(evaluated.baseline_topic_correct.mean() * 100)
    v2_accuracy = float(evaluated.v2_topic_correct.mean() * 100)
    baseline_outlier = float(baseline_data.topic_id.eq(-1).mean() * 100)
    v2_outlier = float(v2.topic_id.eq(-1).mean() * 100)
    corrected = evaluated[~evaluated.baseline_topic_correct & evaluated.v2_topic_correct]
    regressed = evaluated[evaluated.baseline_topic_correct & ~evaluated.v2_topic_correct]

    def examples(frame: pd.DataFrame, limit: int = 8) -> list[dict]:
        return frame[["review_text", "human_topic", "model_topic", "baseline_business_topic", "topic_model_label", "v2_business_topic"]].head(limit).to_dict("records")

    comparison = {
        "methodology": {
            "change": "Known synthetic context templates are removed from topic_text only before MiniLM embedding and BERTopic clustering.",
            "review_text_preserved": True,
            "human_labels_used_for_training_tuning_or_assignment": False,
            "human_labels_loaded_after_cluster_assignment_and_naming": True,
            "sentiment_evaluation_changed": False,
        },
        "input_integrity": {
            "raw_reviews_sha256": file_sha256(raw_path),
            "baseline_snapshot_sha256": file_sha256(baseline_snapshot_path),
            "human_validation_sha256": file_sha256(validation_path),
            "baseline_labels_sha256": file_sha256(baseline_labels_path),
        },
        "results": {
            "validation_reviews": len(evaluated),
            "baseline_topic_accuracy_pct": baseline_accuracy,
            "v2_topic_accuracy_pct": v2_accuracy,
            "absolute_percentage_point_change": v2_accuracy - baseline_accuracy,
            "baseline_outlier_rate_pct": baseline_outlier,
            "v2_outlier_rate_pct": v2_outlier,
            "baseline_assignment_coverage_pct": 100 - baseline_outlier,
            "v2_assignment_coverage_pct": 100 - v2_outlier,
            "all_reviews_with_changed_business_topic": int(assignments.business_topic_changed.sum()),
            "all_valid_reviews": len(assignments),
            "validation_assignments_changed": int(evaluated.assignment_changed.sum()),
            "corrected_validation_assignments": len(corrected),
            "newly_incorrect_validation_assignments": len(regressed),
            "sentiment_accuracy_pct_unchanged": float(parse_bool(evaluated.sentiment_correct).mean() * 100),
            "sentiment_coverage_pct_unchanged": float(baseline_data.sentiment_label.ne("Unknown").mean() * 100),
        },
        "topic_model": topic_meta,
        "cluster_business_labels": {str(key): value for key, value in cluster_labels.items()},
        "confusion_by_human_topic": confusion.to_dict("records"),
        "corrected_examples": examples(corrected),
        "newly_incorrect_examples": examples(regressed),
        "limitations": [
            "The experiment uses synthetic English reviews and removes phrases specific to that generator.",
            "Business labels are assigned after clustering with a fixed product-area vocabulary, not human validation rows.",
            "The 50-row labeled sample is small; percentage changes are descriptive and not a generalization estimate.",
            "BERTopic cluster IDs are arbitrary, so the full-corpus change count compares normalized business-topic labels.",
        ],
    }
    snapshot = {
        "schema_version": 2,
        "sample_sha256": file_sha256(raw_path),
        "data": json.loads(v2.to_json(orient="records", date_format="iso")),
        "quality": quality,
        "topics": topics,
        "topic_meta": topic_meta,
        "sentiment_meta": baseline_snapshot["sentiment_meta"],
        "trends": json.loads(trends.to_json(orient="records")),
        "trend_meta": trend_meta,
        "scores": json.loads(scores.to_json(orient="records")),
    }
    return comparison, evaluated, assignments, snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data/sample_reviews.csv")
    parser.add_argument("--baseline-snapshot", type=Path, default=ROOT / "data/demo_analysis.json")
    parser.add_argument("--validation", type=Path, default=ROOT / "evaluation/sample_validation.csv")
    parser.add_argument("--baseline-labels", type=Path, default=ROOT / "data/demo_labels.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation")
    args = parser.parse_args()
    comparison, evaluated, assignments, snapshot = evaluate_v2(
        args.raw, args.baseline_snapshot, args.validation, args.baseline_labels
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "topic_v2_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    evaluated.to_csv(args.output_dir / "topic_v2_validation.csv", index=False)
    assignments.to_csv(args.output_dir / "topic_v2_assignments.csv", index=False)
    pd.DataFrame(comparison["confusion_by_human_topic"]).to_csv(
        args.output_dir / "topic_v2_confusion.csv", index=False
    )
    (ROOT / "data/demo_analysis_v2.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(comparison["results"], indent=2))
