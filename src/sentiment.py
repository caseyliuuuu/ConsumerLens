"""Three-class text sentiment, rating fallback, and honest unknown values."""
from functools import lru_cache
import numpy as np
import pandas as pd

@lru_cache(maxsize=1)
def sentiment_model():
    from transformers import pipeline
    return pipeline('text-classification',model='cardiffnlp/twitter-roberta-base-sentiment-latest',device=-1)

def analyze_sentiment(df: pd.DataFrame, use_model: bool = True) -> tuple[pd.DataFrame, dict]:
    """Score is signed polarity [-1,1], not a calibrated satisfaction probability."""
    out=df.copy(); labels=None; warnings=[]
    if use_model:
        try:
            predictions=sentiment_model()(out.review_text.tolist(),batch_size=32,truncation=True,max_length=512,top_k=None)
            labels=[]; scores=[]
            for row in predictions:
                probs={p['label'].lower():p['score'] for p in row}
                probs={ {'label_0':'negative','label_1':'neutral','label_2':'positive'}.get(k,k):v for k,v in probs.items() }
                if not {'negative','neutral','positive'} <= probs.keys(): raise ValueError('Unexpected sentiment labels')
                labels.append(max(probs,key=probs.get).title()); scores.append(probs['positive']-probs['negative'])
            out['sentiment_label']=labels;out['sentiment_score']=scores;out['sentiment_method']='Text model'
        except Exception as exc:
            labels=None;warnings.append(f'Text sentiment unavailable ({type(exc).__name__}); using ratings where valid.')
    if labels is None:
        r=pd.to_numeric(out.rating,errors='coerce')
        out['sentiment_label']=np.select([r>=4,r<=2,r.eq(3)],['Positive','Negative','Neutral'],default='Unknown')
        out['sentiment_score']=(r-3)/2
        out.loc[out.sentiment_label.eq('Unknown'),'sentiment_score']=np.nan
        out['sentiment_method']=np.where(out.sentiment_label.eq('Unknown'),'Unavailable','Rating proxy')
    return out,{'method':'Text model' if labels is not None else 'Rating proxy (missing/ambiguous ratings = Unknown)','warnings':warnings}
