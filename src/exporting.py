"""Portable CSV exports with review evidence and spreadsheet-safe text."""
import pandas as pd


def csv_bytes(frame):
    """Escape spreadsheet formula prefixes in exported user-controlled strings."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object', 'string']):
        safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
    return safe.to_csv(index=False).encode('utf-8-sig')


def insight_export(brief, data, dataset, synthetic):
    rows = []
    for item in brief['insights']:
        cited = data.set_index('review_id').loc[item['review_ids']]
        rows.append({'dataset': dataset, 'synthetic_data': synthetic, 'generation_method': brief['method'],
                     'executive_summary': brief['summary'], 'category': item['category'], 'finding': item['finding'],
                     'evidence': item['evidence'], 'business_implication': item['implication'],
                     'recommended_action': item['action'], 'review_ids': ' | '.join(item['review_ids']),
                     'supporting_reviews': ' | '.join(f'{rid}: {r.review_text}' for rid, r in cited.iterrows()),
                     'sources': ' | '.join(cited.source.fillna('Unavailable').astype(str))})
    return csv_bytes(pd.DataFrame(rows))


