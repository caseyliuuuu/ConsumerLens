"""Explainable prioritization index, not an estimate of financial ROI."""
import re
import numpy as np
import pandas as pd

COMMERCIAL=re.compile(r'\b(?:price|value|buy|purchase|return|refund|quality|recommend|worth|expensive|cheap|replacement|subscription)\b',re.I)
WEIGHTS={'prevalence_score':.35,'negative_intensity_score':.30,'purchase_relevance_score':.20,'trend_growth_score':.15}

def score_topics(df: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for tid,g in df.groupby('topic_id'):
        known=g.sentiment_label.ne('Unknown'); coverage=float(known.mean()); neg=g.sentiment_label.eq('Negative')
        fractions={s.lower()+'_pct':float(g.sentiment_label.eq(s).mean()*100) for s in ['Positive','Neutral','Negative','Unknown']}
        # Known-only estimate; explicitly unavailable when no usable sentiment.
        intensity=float((-g.loc[known,'sentiment_score'].clip(upper=0)).mean()*10) if known.any() else np.nan
        row={'topic_id':tid,'topic':g.topic.iloc[0],'review_count':len(g),'share_pct':len(g)/len(df)*100,**fractions,
             'average_sentiment':g.sentiment_score.mean(),'sentiment_coverage':coverage*100,'prevalence_score':len(g)/len(df)*10,
             'negative_intensity_score':intensity,'purchase_relevance_score':g.review_text.str.contains(COMMERCIAL.pattern,case=False,regex=True).mean()*10,
             'trend_growth_score':np.nan,'change_pp':np.nan,'trend':'Unavailable'}
        if not trends.empty:
            t=trends[trends.topic_id.eq(tid)].iloc[0]
            row.update(change_pp=t.change_pp,trend=t.trend,trend_growth_score=float(np.clip(t.change_pp,0,10)))
        weight=sum(w for k,w in WEIGHTS.items() if pd.notna(row[k]))
        row['impact_score']=sum(row[k]*w for k,w in WEIGHTS.items() if pd.notna(row[k]))/weight
        row['available_weight']=weight
        rows.append(row)
    return pd.DataFrame(rows).sort_values('impact_score',ascending=False).reset_index(drop=True)
