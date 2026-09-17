"""Conservative validation that preserves language context and source row IDs."""
import re
import unicodedata
import pandas as pd

ALIASES = {'review_text': ['review', 'text', 'comment', 'content'], 'rating': ['stars', 'score', 'review_score'], 'date': ['review_date', 'created_at', 'timestamp']}

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize, remove blanks/noise/exact normalized duplicates, and report quality."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Column names must be unique after trimming and lowercasing.')
    for canonical, aliases in ALIASES.items():
        if canonical not in df:
            match = next((a for a in aliases if a in df), None)
            if match: df = df.rename(columns={match: canonical})
    if 'review_text' not in df:
        raise ValueError('Add a review_text column (or review, text, comment, content).')
    stats = {'uploaded': len(df), 'missing_values': df.isna().sum().to_dict()}
    df['review_id'] = [f'R{i+1:06d}' for i in range(len(df))]
    df['review_text'] = df.review_text.fillna('').astype(str).map(lambda s: re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', s)).strip())
    missing = df.review_text.eq(''); stats['missing_removed'] = int(missing.sum()); df = df[~missing].copy()
    noise = ~df.review_text.map(lambda s: any(c.isalpha() for c in s)).astype(bool)
    stats['noise_removed'] = int(noise.sum()); df = df[~noise].copy()
    duplicate = df.review_text.str.casefold().duplicated()
    stats['duplicates_removed'] = int(duplicate.sum()); df = df[~duplicate].copy()
    df['review_length'] = df.review_text.str.split().str.len()
    for c in ['rating', 'date', 'product', 'source']:
        if c not in df: df[c] = pd.NA
    ratings = pd.to_numeric(df.rating, errors='coerce')
    stats['invalid_ratings'] = int((df.rating.notna() & (~ratings.between(1,5))).sum())
    df['rating'] = ratings.where(ratings.between(1,5))
    dates = pd.to_datetime(df.date, errors='coerce', utc=True, format='mixed')
    stats['invalid_dates'] = int((df.date.notna() & dates.isna()).sum())
    df['date'] = dates
    stats.update(valid=len(df), average_length=float(df.review_length.mean()) if len(df) else 0, dated_reviews=int(dates.notna().sum()))
    return df.reset_index(drop=True), stats
