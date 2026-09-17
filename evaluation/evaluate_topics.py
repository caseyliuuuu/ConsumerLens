"""Run deterministic validation sampling and descriptive cluster diagnostics."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocessing import preprocess
from src.topic_modeling import discover_topics
from src.sentiment import analyze_sentiment

def evaluate(data: pd.DataFrame) -> dict:
    """Lexical centroid similarity is a proxy, not human-judged topic coherence."""
    vector=TfidfVectorizer(analyzer='word',token_pattern=r'(?u)\b\w+\b').fit_transform(data.review_text)
    result=[]
    for tid,g in data.groupby('topic_id'):
        idx=np.flatnonzero(data.topic_id.to_numpy()==tid);centroid=np.asarray(vector[idx].mean(axis=0))
        similarities=cosine_similarity(vector[idx],centroid).ravel()
        result.append({'topic_id':int(tid),'topic':g.topic.iloc[0],'count':len(g),'share':len(g)/len(data),'lexical_cohesion_proxy':float(similarities.mean()),'singleton':len(g)==1,'inspect_reviews':g.review_text.head(3).tolist()})
    return {'topics':result,'note':'Singleton cohesion is trivially 1. This diagnostic is not accuracy; inspect evidence and manually label the sample.'}

def validation_sample(data: pd.DataFrame) -> pd.DataFrame:
    sample=data.sample(n=min(50,len(data)),random_state=42)[['review_text','topic','sentiment_label']].rename(columns={'topic':'model_topic','sentiment_label':'model_sentiment'})
    for c in ['human_topic','human_sentiment','topic_correct','sentiment_correct']: sample[c]=''
    return sample

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'data/sample_reviews.csv');parser.add_argument('--offline',action='store_true');parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();data,_=preprocess(pd.read_csv(args.input));data,topics,meta=discover_topics(data,'offline' if args.offline else 'auto');data,sentiment=analyze_sentiment(data,not args.offline)
    args.output.mkdir(parents=True,exist_ok=True)
    validation_sample(data).to_csv(args.output/'sample_validation.csv',index=False)
    (args.output/'topic_diagnostics.json').write_text(json.dumps({'topic_method':meta,'sentiment_method':sentiment,**evaluate(data)},indent=2))
    print(f'Wrote {min(50,len(data))} validation rows and diagnostics to {args.output}')
