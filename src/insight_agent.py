"""Evidence-grounded recommendations with a deterministic default and optional LLM synthesis."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import urllib.request

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMERCIAL_TERMS = re.compile(
    r"\b(?:price|value|buy|purchase|return|refund|quality|recommend|worth|expensive|cheap|replacement|subscription)\b",
    re.I,
)
DISALLOWED_LLM_CLAIMS = re.compile(
    r"(?:\d|%|\$|\brevenue\b|\bmarket[- ]wide\b|\bmarket share\b|\bcauses?\b|\bcaused\b|\bguarantee[ds]?\b)",
    re.I,
)
REQUIRED_CATEGORIES = {"Product", "Pricing", "Marketing"}


def _review_ids_for_topic(group: pd.DataFrame, topic: dict, limit: int = 5) -> list[str]:
    """Build a small, deterministic evidence pool from representative and relevant reviews."""
    ids = list(topic.get("representative_ids", []))
    ids.extend(group.loc[group.sentiment_label.eq("Negative"), "review_id"].head(2).tolist())
    ids.extend(group.loc[group.review_text.str.contains(COMMERCIAL_TERMS.pattern, regex=True), "review_id"].head(2).tolist())
    valid = set(group.review_id)
    return list(dict.fromkeys(review_id for review_id in ids if review_id in valid))[:limit]


def build_structured_evidence(data: pd.DataFrame, scores: pd.DataFrame, topics: dict) -> dict:
    """Create the only payload the optional LLM is permitted to consume."""
    topic_rows = []
    ranked = scores.reset_index(drop=True)
    for index, row in ranked.iterrows():
        topic_id = int(row.topic_id)
        group = data[data.topic_id.eq(topic_id)]
        review_ids = _review_ids_for_topic(group, topics[topic_id])
        evidence_reviews = []
        for review_id in review_ids:
            review = group[group.review_id.eq(review_id)].iloc[0]
            evidence_reviews.append(
                {
                    "review_id": review_id,
                    "review_text": review.review_text,
                    "source": str(review.source) if pd.notna(review.source) else "Unavailable",
                    "sentiment": review.sentiment_label,
                }
            )
        topic_rows.append(
            {
                "topic_id": topic_id,
                "topic_name": row.topic,
                "impact_rank": index + 1,
                "review_count": int(row.review_count),
                "share_of_voice_pct": round(float(row.share_pct), 4),
                "sentiment": {
                    "positive_pct": round(float(row.positive_pct), 4),
                    "neutral_pct": round(float(row.neutral_pct), 4),
                    "negative_pct": round(float(row.negative_pct), 4),
                    "unknown_pct": round(float(row.unknown_pct), 4),
                    "average_score": None if pd.isna(row.average_sentiment) else round(float(row.average_sentiment), 4),
                    "coverage_pct": round(float(row.sentiment_coverage), 4),
                },
                "trend": {
                    "direction": row.trend,
                    "change_percentage_points": None if pd.isna(row.change_pp) else round(float(row.change_pp), 4),
                },
                "purchase_relevance": {
                    "score_0_to_10": round(float(row.purchase_relevance_score), 4),
                    "matched_review_pct": round(float(row.purchase_relevance_score) * 10, 4),
                },
                "business_impact": {
                    "score_0_to_10": round(float(row.impact_score), 4),
                    "available_weight": round(float(row.available_weight), 4),
                    "components": {
                        "prevalence": round(float(row.prevalence_score), 4),
                        "negative_intensity": None if pd.isna(row.negative_intensity_score) else round(float(row.negative_intensity_score), 4),
                        "purchase_relevance": round(float(row.purchase_relevance_score), 4),
                        "trend_growth": None if pd.isna(row.trend_growth_score) else round(float(row.trend_growth_score), 4),
                    },
                },
                "evidence_reviews": evidence_reviews,
            }
        )
    return {
        "analysis_scope": {
            "valid_review_count": len(data),
            "topic_count": len(ranked),
            "limitations": [
                "Metrics describe only the supplied review dataset.",
                "Association does not establish causation or financial impact.",
                "Recommended actions are hypotheses that require validation.",
            ],
        },
        "topics": topic_rows,
    }


def _evidence_sentence(row: pd.Series, rank: int, total_topics: int) -> str:
    text = (
        f"Rank {rank} of {total_topics}; {int(row.review_count)} reviews "
        f"({row.share_pct:.1f}% share of voice); {row.negative_pct:.1f}% negative; "
        f"Business Impact Score {row.impact_score:.2f}/10; purchase relevance "
        f"{row.purchase_relevance_score:.2f}/10; sentiment coverage {row.sentiment_coverage:.1f}%."
    )
    if pd.notna(row.change_pp):
        text += f" Topic share changed {row.change_pp:+.1f} percentage points between the two dated periods."
    return text


def _deterministic_brief(data: pd.DataFrame, scores: pd.DataFrame, topics: dict) -> dict:
    candidates = []
    ranked = scores.reset_index(drop=True)

    def add(row: pd.Series, category: str, action: str, negative: bool = False) -> None:
        group = data[data.topic_id.eq(row.topic_id)]
        rank = int(ranked.index[ranked.topic_id.eq(row.topic_id)][0]) + 1
        if negative:
            ids = group.loc[group.sentiment_label.eq("Negative"), "review_id"].head(3).tolist()
        else:
            ids = [review_id for review_id in topics[int(row.topic_id)]["representative_ids"] if review_id in set(group.review_id)]
        if category == "Pricing":
            ids = group.loc[group.review_text.str.contains(COMMERCIAL_TERMS.pattern, regex=True), "review_id"].head(3).tolist()
        if not ids:
            return
        candidates.append(
            {
                "id": f"D{len(candidates) + 1}",
                "category": category,
                "topic_id": int(row.topic_id),
                "topic": row.topic,
                "rank": rank,
                "finding": f"{row.topic}: {row.negative_pct:.1f}% negative across {int(row.review_count)} reviews",
                "evidence": _evidence_sentence(row, rank, len(ranked)),
                "implication": "This review pattern warrants investigation; it does not establish a cause, market-wide prevalence, or revenue impact.",
                "action": action,
                "review_ids": ids,
            }
        )

    pains = ranked[ranked.negative_pct.gt(0)]
    if not pains.empty:
        add(pains.iloc[0], "Pain Points", "Read the cited complaints and verify the reported failure conditions before assigning an owner.", True)
        add(pains.iloc[0], "Product", "Reproduce the reported experience, then test a targeted improvement with affected customers.", True)
    for _, row in ranked.iterrows():
        group = data[data.topic_id.eq(row.topic_id)]
        if group.review_text.str.contains(COMMERCIAL_TERMS.pattern, regex=True).any():
            add(row, "Pricing", "Investigate the price and value language in the cited reviews through interviews and a pricing study before changing price.")
            break
    add(ranked.iloc[0], "Marketing", "Audit product messaging against the cited experiences and test clearer expectation-setting without making unverified performance claims.")
    emerging = ranked[ranked.change_pp.ge(2) & ranked.negative_pct.gt(0)]
    if not emerging.empty:
        add(emerging.sort_values("change_pp", ascending=False).iloc[0], "Emerging Signals", "Monitor the next comparable period and inspect source-mix changes before treating this increase as persistent.", True)
    top = ranked.iloc[0]
    summary = (
        f"Analyzed {len(data):,} reviews across {len(ranked)} topics. {top.topic} ranks first within this dataset "
        f"with an Impact Score of {top.impact_score:.2f}/10. This is a relative investigation priority, not a claim of high absolute impact. "
        "Recommendations are evidence-linked hypotheses for validation."
    )
    return {"summary": summary, "insights": candidates, "method": "Deterministic evidence-grounded brief"}


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()
    return json.loads(content)


def _validated_llm_brief(parsed: dict, evidence: dict, scores: pd.DataFrame) -> dict:
    if set(parsed) != {"executive_synthesis", "recommendations"}:
        raise ValueError("Unexpected LLM response fields")
    summary = parsed["executive_synthesis"]
    if not isinstance(summary, str) or not 20 <= len(summary) <= 900 or DISALLOWED_LLM_CLAIMS.search(summary):
        raise ValueError("Unsupported or unbounded executive synthesis")
    recommendations = parsed["recommendations"]
    if not isinstance(recommendations, list) or len(recommendations) != 3:
        raise ValueError("Exactly three recommendations are required")
    topic_evidence = {item["topic_id"]: item for item in evidence["topics"]}
    score_lookup = scores.set_index("topic_id")
    ranked = scores.reset_index(drop=True)
    seen_categories = set()
    insights = []
    required = {"category", "topic_id", "finding", "business_implication", "recommended_action", "evidence_review_ids"}
    for index, item in enumerate(recommendations, start=1):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("Invalid recommendation fields")
        category = item["category"]
        topic_id = item["topic_id"]
        if category not in REQUIRED_CATEGORIES or category in seen_categories or topic_id not in topic_evidence:
            raise ValueError("Invalid category or topic reference")
        seen_categories.add(category)
        for field in ("finding", "business_implication", "recommended_action"):
            value = item[field]
            if not isinstance(value, str) or not 12 <= len(value) <= 700 or DISALLOWED_LLM_CLAIMS.search(value):
                raise ValueError(f"Unsupported claim in {field}")
        ids = item["evidence_review_ids"]
        allowed_ids = {review["review_id"] for review in topic_evidence[topic_id]["evidence_reviews"]}
        if not isinstance(ids, list) or not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)) or not set(ids) <= allowed_ids:
            raise ValueError("Recommendation citations do not match supplied topic evidence")
        row = score_lookup.loc[topic_id]
        rank = int(ranked.index[ranked.topic_id.eq(topic_id)][0]) + 1
        insights.append(
            {
                "id": f"L{index}",
                "category": category,
                "topic_id": int(topic_id),
                "topic": row.topic,
                "rank": rank,
                "finding": item["finding"],
                "evidence": _evidence_sentence(row, rank, len(scores)),
                "implication": item["business_implication"],
                "action": item["recommended_action"],
                "review_ids": ids,
            }
        )
    if seen_categories != REQUIRED_CATEGORIES:
        raise ValueError("Product, Pricing, and Marketing recommendations are required")
    return {"summary": summary, "insights": insights, "method": "LLM-assisted evidence-grounded brief"}


def generate_insights(data: pd.DataFrame, scores: pd.DataFrame, topics: dict, use_api: bool = False) -> dict:
    """Generate a deterministic brief, or a validated LLM brief when explicitly enabled."""
    fallback = _deterministic_brief(data, scores, topics)
    if not use_api:
        return fallback
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {**fallback, "warning": "No API key is configured; showing the deterministic evidence-grounded brief."}
    try:
        evidence = build_structured_evidence(data, scores, topics)
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": (ROOT / "prompts/consumer_insight_prompt.txt").read_text(encoding="utf-8")},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)},
            ],
        }
        endpoint = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)
        parsed = _parse_llm_json(result["choices"][0]["message"]["content"])
        return _validated_llm_brief(parsed, evidence, scores)
    except Exception as exc:
        return {
            **fallback,
            "warning": f"LLM response was unavailable or failed grounding validation ({type(exc).__name__}); showing the deterministic evidence-grounded brief.",
        }
